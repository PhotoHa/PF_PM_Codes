# -*- coding: utf-8 -*-
"""
Created on Wed Sep 18 11:03:43 2024

@author: david
"""

###################################################
### 미국 섹터 ETF 투자 전략: Trend Following 전략 ###
###################################################

import pandas as pd
import numpy as np
import yfinance as yf
import talib as ta
import matplotlib.pyplot as plt
import matplotlib.dates as mdates


# 주가 데이터
##factor_etfs = ['QUAL','MTUM','USMV','VLUE','SIZE']
sector_etfs = ['XLE','XLK','XLB','XLY','XLC','XLI','XLP','XLU','XLF','XLV','XLRE']
data_prc = yf.download(sector_etfs, start = '2024-01-01')
data_prc = data_prc.tail(90)
cls_prc = data_prc['Close']
hig_prc = data_prc['High']
low_prc = data_prc['Low']


# function
def custom_cummax(series):
    cummax_series = series.copy()
    start_idx = 0
    length = len(series)

    while start_idx < length:
        # NaN이 있는 곳의 인덱스를 찾음
        if series[start_idx:].isna().any():  # NaN이 있는지 확인
            next_nan = series[start_idx:].isna().idxmax() + start_idx
        else:
            next_nan = length  # 더 이상 NaN이 없다면 끝까지 처리

        # 해당 구간에 대해 cummax 적용
        cummax_series.iloc[start_idx:next_nan] = series.iloc[start_idx:next_nan].cummax()

        # NaN을 만나면 그 다음 값부터 다시 cummax 적용
        start_idx = next_nan + 1
        
    return cummax_series


def indiv_bands(target = 'XLB', k = 2, up_range = 20, dw_range = 40):
    '''     # Model Description:
            1) Entry Criteria
            2) Sizing and Position Management
            3) Exit Criteria  '''

    imsi_cls = cls_prc[target]
    imsi_hig = hig_prc[target]
    imsi_low = low_prc[target]


    # 1) Entry Criteria:
    # Keltner Channels: upper = 20days, lower = 40 days
    ema_up = ta.EMA(imsi_cls, up_range)
    atr_up = ta.ATR(imsi_hig, imsi_low, imsi_cls, up_range)
    keltner_up = ema_up + k*atr_up
    
    ema_dw = ta.EMA(imsi_cls, dw_range)
    atr_dw = ta.ATR(imsi_hig, imsi_low, imsi_cls, dw_range)
    keltner_dw = ema_dw - k*atr_dw
    
    # Donchian Channels
    donchian_up = ta.MAX(imsi_cls, up_range)
    donchian_dw = ta.MIN(imsi_cls, dw_range)
    
    # Upper Band
    df_upper = pd.DataFrame(data={'cls':imsi_cls, 'donchian':donchian_up, 'keltner':keltner_up})
    df_upper['upper_band'] = df_upper.apply(lambda x: min(x['donchian'], x['keltner']), axis=1)
    df_upper['upper_band_shift'] = df_upper['upper_band'].shift(1)
    
    # 2) Sizing and Position Management: After the looping process
    # 3) Exit Criteria
    df_lower = pd.DataFrame(data={'cls':imsi_cls, 'donchian':donchian_dw, 'keltner':keltner_dw})
    df_lower['lower_band'] = df_lower.apply(lambda x: max(x['donchian'], x['keltner']), axis=1)
    df_lower['lower_band'] = df_lower['lower_band'].shift(1)
    df_lower['lower_trailing_stop'] = df_lower['lower_band'].shift(1).cummax()
    
    # sum-up
    df_sum_up = pd.DataFrame({'price':df_upper.cls, 'ub':df_upper.upper_band_shift,
                              'lb':df_lower.lower_band})
    
    
    df_sum_up['over_up'] = (df_sum_up['price'] > df_sum_up['ub'])*1
    df_sum_up['over_up'] = df_sum_up['over_up'].replace(0, np.NaN)
    df_sum_up['lb_imsi'] = df_sum_up['over_up'] * df_sum_up['lb'] 
    df_sum_up['group'] = (~df_sum_up['lb_imsi'].isna()).cumsum()
    df_sum_up['lb_new'] = df_sum_up.groupby('group')['lb_imsi'].cummax().ffill()
    
    df_sum_up_1 = df_sum_up[['price','ub','lb_new']]
    df_sum_up_1.columns = [f'price_{target}',f'upper_{target}',f'lower_{target}']

    return df_sum_up_1



# 4x3 subplot을 설정
fig, axs = plt.subplots(4, 3, figsize=(18, 12))  # 4x3 subplot grid
fig.tight_layout(pad=4.0)

# DateFormatter 설정 (월-일만 표시)
date_format = mdates.DateFormatter('%m-%d')

for i, tgt in enumerate(sector_etfs):
    row = i // 3  # 행 번호
    col = i % 3   # 열 번호
    
    ax = axs[row, col]  # 해당 위치의 subplot 선택
    plot_data = indiv_bands(target=tgt, up_range=20, dw_range=40)
    plot_data.plot(ax=ax)  # subplot에 그래프 출력
    
    #ax.set_title(f"Plot for {tgt}")  # 각 subplot에 제목 추가
    
    # x축의 날짜 형식을 월-일로 포맷팅
    ax.xaxis.set_major_formatter(date_format)

    # x축 레이블을 회전하여 가독성을 높임
    plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, ha="right")

# 남은 빈칸은 비활성화
for j in range(len(sector_etfs), 12):
    fig.delaxes(axs[j // 3, j % 3])

plt.show()


