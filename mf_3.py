# -*- coding: utf-8 -*-
"""
Created on Fri May 10 16:32:18 2024

@author: 11149
"""

import pandas as pd
import numpy as np
import pyodbc
import statsmodels.api as sm
from statsmodels.tsa.stattools import coint, adfuller

conn_quant = pyodbc.connect('driver={SQL Server};server=46.2.90.172;database=quant;uid=index;pwd=samsung@00')
conn_wisefn = pyodbc.connect('driver={SQL Server};server=46.2.90.172;database=wisefn;uid=index;pwd=samsung@00')

def invact_prc(dt_beg, dt_end, inv_cd):
    # date import
    query2 = '''
    select CONVERT(CHAR(8),DT,112) 'DT'	from wisefn..TZ_DATE where TRADE_YN = 1 AND YMD BETWEEN {} AND {} 
    '''.format(dt_beg, dt_end)
    me_date = pd.read_sql(query2,conn_quant)
    date_list_d = pd.DataFrame(me_date['DT'].unique())
    date_list_d.columns = ['trd_dt']
    date_list_d['BF_DT'] = date_list_d['trd_dt'].shift(1)
    date_list_d['NXT_DT'] = date_list_d['trd_dt'].shift(-1)
    date_list_d = date_list_d.tail(40)
    
    # update Date    
    dt_beg = int(date_list_d['trd_dt'].min())
    dt_end = int(date_list_d['trd_dt'].max())
    
    # 연기금 수급
    inv_cd = inv_cd
    
    d_sql = '''
    select a.trd_dt, a.stk_cd, b.stk_nm_kor, c.inv_cd, a.val, c.net_buy_amt amt 
    from wisefn..ts_stk_data a
        inner join wisefn..ts_stk_issue b 
            on a.stk_cd = b.stk_cd
            and a.trd_dt = b.trd_dt
            and b.ks200_typ = 1
           
        inner join wisefn..ts_stk_invact c
            on b.stk_cd = c.stk_cd
            and b.trd_dt = c.trd_dt 
            and c.inv_cd = {}
            and isnull(c.net_buy_amt,0) <> 0
    where a.trd_dt between {} and {}    
    and a.item_cd= '100300'
    '''.format(inv_cd, dt_beg, dt_end) 
    data = pd.read_sql(d_sql, conn_quant)
    
    data = data.sort_values(['stk_cd','trd_dt'])
    data = pd.DataFrame(data)
    
    return data


def ma_f(df, window, prc_idx):
    df1 = df.copy()
    df1[f'MA_{window}'] = df1[prc_idx].rolling(window).mean()
    return df1


# regrs_data
def calculate_beta(group):
    from sklearn.linear_model import LinearRegression
    X = group.index.values.reshape(-1, 1)
    y = group['amt'].values
    model = LinearRegression()
    for i in range(len(group)):
        if i >= 20:  # 20일 이전 데이터부터 계산
            X_window = X[i-19:i+1]
            y_window = y[i-19:i+1]
            model.fit(X_window, y_window)
            beta = model.coef_[0]
            group.loc[group.index[i], 'beta'] = beta
    return group


def RSI(rt,term):
    up = rt >= 0
    down = rt < 0
    up = up.replace(True, 1)
    up = up.replace(False,0)
    down = down.replace(True, 1)
    down = down.replace(False,0)
    
    up_df = rt*up; down_df = rt*down
        
    AU = up_df.rolling(14).sum()
    AD = -1 * down_df.rolling(14).sum()
    rsi = 100 * AU / (AU+AD)
    return rsi

# rsi_df = RSI(rt=rt, term = 14).dropna()


def adj_prc(dt_beg, dt_end):
    d_sql = '''
            select a.trd_dt, a.stk_cd, b.stk_nm_kor, a.val
            from wisefn..ts_stk_data a
                inner join wisefn..ts_stk_issue b 
                    on a.stk_cd = b.stk_cd
                    and a.trd_dt = b.trd_dt
                    and b.ks200_typ = 1
            where a.trd_dt between {} and {}    
            and a.item_cd= '100300'
            '''.format(dt_beg, dt_end) 
    data = pd.read_sql(d_sql, conn_quant)

    data = data.sort_values(['stk_cd','trd_dt'])
    data = pd.DataFrame(data)
    
    return data



def find_cointegrated_pairs(data):
    n = data.shape[1]
    score_matrix = np.zeros((n, n))
    pvalue_matrix = np.ones((n, n))
    keys = data.keys()
    pairs = []
    for i in range(n):
        for j in range(i+1, n):
            S1 = data[keys[i]]
            S2 = data[keys[j]]
            result = coint(S1, S2)
            score = result[0]
            pvalue = result[1]
            score_matrix[i, j] = score
            pvalue_matrix[i, j] = pvalue
            if pvalue < 0.05:
                pairs.append((keys[i], keys[j]))
    return score_matrix, pvalue_matrix, pairs



def moving_data(data, window, main_column):
    compressed_dfs = []
    for index, row in data.iterrows():
        start_index = max(0, index - window)
        subset_df = data.iloc[start_index+1:index + 1]
        subset_df.index = main_column[start_index+1:index + 1]
        if len(subset_df) >= window:
            compressed_dfs.append(subset_df)
    return compressed_dfs



def load_prc(how = 'folder', pkl_path = 'T:\\index\\999_quant\\__data_pkl\\kgh\\tmp_daily\\', file_name = 'stock_price_close.pkl', pivot_base = 'CLOSE_PRC', st_date = '20050101', stk_list = tuple(['005930','005935'])):
    
    ''' folder: pickle data. file_name 만 넣으면 됨. 기본 세팅은 종가 데이터.
        그 외: wisefn db. pivot_base, st_date, stk_list 넣으면 됨 '''
    
    def load_pkl(pkl_path, file_name):
        import pickle
        with open(pkl_path + file_name, 'rb') as f:
            data = pickle.load(f)
        data.index = pd.to_datetime(data.index)
        return data


    def load_db(pivot_base, stk_list, st_date):
        '''  pickle data처럼 불러옴  '''
        conn_wisefn = pyodbc.connect('driver={SQL Server};server=46.2.90.172;database=wisefn;uid=index;pwd=samsung@00')
        sql_trd_data = f''' SELECT TRD_DT as date, 'A'+ STK_CD AS code, OPEN_PRC, CLOSE_PRC
                            FROM WISEFN..TS_STK_DAILY
                            WHERE 1=1
                            AND STK_CD IN {stk_list}
                            AND TRD_DT > {st_date}  '''
        data = pd.read_sql(sql_trd_data, conn_wisefn)
        data = data.pivot_table(index='date',values=pivot_base, columns='code')
        data.index = pd.to_datetime(data.index, format="%Y%m%d")
        return data


    if how == 'folder':
        df = load_pkl(pkl_path, file_name)
    else:
        df = load_db(pivot_base, stk_list, st_date)

    return df



def pref_list():
    
    conn_pcor = pyodbc.connect('driver={Oracle in OraClient12Home1};dbq=PCOR;uid=EF0SEL;pwd=EF0SEL#076')
    sql_pref = '''  SELECT CONCAT('A',A.STK_CD) AS pref_code, A.STK_NM_KOR AS pref_nm, CONCAT('A',B.STK_CD) AS comm_code, B.STK_NM_KOR AS comm_nm
                    FROM COROWN.TS_STOCK A
                    LEFT JOIN COROWN.TS_STOCK B
                    ON A.CMP_CD = B.CMP_CD
                    WHERE A.LIST_YN = 1 AND A.ISSUE_TYP = 5 AND B.ISSUE_TYP = 1  '''
    df_pref_list = pd.read_sql(sql_pref, conn_pcor)

    return df_pref_list



def relative_prc(general_paris_list, unique_stk_list, prc):
    ''' general_pairs_list : pair 목록
        unique_stk_list : 중복 제거한 종목 리스트
        prev_close_prc : 전일 종가
        open_prc : 당일 시가 '''
    
    # 페어별 가격
    df_prc_1 = prc[unique_stk_list]
    df_prc_first = df_prc_1[general_paris_list['pair_first']]
    df_prc_second = df_prc_1[general_paris_list['pair_second']]

    # 컬럼 바꾸기 (우선주, 보통주)
    pairs_zip = list(zip(np.array(general_paris_list['pair_first']), np.array(general_paris_list['pair_second'])))
    df_prc_first.columns = pairs_zip
    df_prc_second.columns = pairs_zip

    reltv_prc = df_prc_first / df_prc_second
    
    return reltv_prc



def bb_stdz(prc_data, window):
    prc = prc_data.copy()
    avg  = prc_data.rolling(window).mean()
    stdv = prc_data.rolling(window).std()
    stdz_data = (prc - avg) / stdv

    return stdz_data



def compound(r):
    return np.expm1(np.log1p(r).sum())


def annualize_rets(r, periods_per_year):
    compounded_growth = (1+r).prod()
    n_periods = r.shape[0]
    return compounded_growth**(periods_per_year/n_periods)-1


def annualize_vol(r, periods_per_year):
    return r.std()*(periods_per_year**0.5)


# Trading 정보
def trading_info(stdz_data, df_buy_sign, df_sell_sign):
    # buy/sell dataframe
    buy_df = (stdz_data < df_buy_sign)*1
    buy_df = buy_df * abs(buy_df.shift(1) - 1)
    sell_df = (stdz_data > df_sell_sign)*(-1)
    sell_df = sell_df * abs(sell_df.shift(1) + 1)

    # POSITION (hold하는 경우 1, 그렇지 않은 경우 0)
    pstn_0 = buy_df + sell_df
    pstn_1 = pstn_0.replace(0,np.NaN)
    pstn_2 = pstn_1.ffill()
    pstn_3 = pstn_2 + pstn_2.shift(1)
    pstn_3 = (pstn_3 != 0)*1
    pstn_4 = pstn_2 * pstn_3
    pstn_4 = (pstn_4 == 1)*1
    # 청산일도 포함시킨 테이블
    pstn_bs = pstn_4 + pstn_4.shift(1)
    pstn_bs = (pstn_bs > 0)*1

    # TRADING TIME
    trdt_0 = pstn_4 - pstn_4.shift(1)
    trdt_0 = trdt_0.replace(0,np.NaN)
    trdt_0 = trdt_0.replace( 1,'Buy')
    trdt_0 = trdt_0.replace(-1,'Sell')

    return pstn_4, pstn_bs, trdt_0


def trading_exe(df_prc, pairs_info, hold_info, trade_info, buy_tax, sell_tx):
    # 첫 번째 페어 시초가, 매매수량
    pair_first_prc = df_prc[pairs_info['pair_first']]
    pair_first_qty = 100000000 / pair_first_prc
    pair_first_qty = pair_first_qty
    # 두 번째 페어 시초가, 매매수량
    pair_second_prc = df_prc[pairs_info['pair_second']]
    pair_second_qty = 100000000 / pair_second_prc
    # 칼럼명 정리
    pair_first_prc.columns = hold_info.columns
    pair_first_qty.columns = hold_info.columns
    pair_second_prc.columns = hold_info.columns
    pair_second_qty.columns = hold_info.columns

    # qty_ts : 매수, 매도를 1로 변경한 테이블
    qty_ts = trade_info.replace('Buy',1)
    qty_ts = qty_ts.replace('Sell',-1)

    # 첫 번째 페어
    pair_first_qty_hold = (pair_first_qty * qty_ts).ffill()
    pair_first_qty_hold[pair_first_qty_hold < 0] = np.NaN
    aa = (pair_first_qty_hold.fillna(0).shift(1) - pair_first_qty_hold.fillna(0))
    aa[aa<0] = 0
    pair_first_qty_hold = (pair_first_qty_hold.fillna(0) + aa).replace(0, np.NaN)
    # 두 번째 페어
    pair_second_qty_hold = (pair_second_qty * qty_ts).ffill()
    pair_second_qty_hold[pair_second_qty_hold < 0] = np.NaN
    bb = (pair_second_qty_hold.fillna(0).shift(1) - pair_second_qty_hold.fillna(0))
    bb[bb<0] = 0
    pair_second_qty_hold = (pair_second_qty_hold.fillna(0) + bb).replace(0, np.NaN)

    pair_first_value  = (pair_first_qty_hold * pair_first_prc).replace(0, np.NaN)
    pair_second_value = (pair_second_qty_hold * pair_second_prc).replace(0, np.NaN)


    # 첫째 페어 거래비용
    trdt_0_first_cost = trade_info.replace('Sell', sell_tx)
    trdt_0_first_cost = trdt_0_first_cost.replace('Buy', buy_tax)
    # 둘째 페어 거래비용
    trdt_0_second_cost = trade_info.replace('Sell', buy_tax)
    trdt_0_second_cost = trdt_0_second_cost.replace('Buy', sell_tx)
    # 총 거래비용
    first_total_cost  = (trdt_0_first_cost * pair_first_value).fillna(0)
    second_total_cost = (trdt_0_second_cost * pair_second_value).fillna(0)
    pairs_trd_cost = first_total_cost + second_total_cost

    pairs_stt_cost = pairs_trd_cost * hold_info.replace(0,np.NaN)
    pairs_stt_cost = pairs_stt_cost.fillna(2)
    pairs_stt_cost = pairs_stt_cost.replace(0, np.NaN)
    pairs_stt_cost = pairs_stt_cost.ffill()
    pairs_stt_cost = pairs_stt_cost.replace(2, 0)

    pairs_trd_cost_ = pairs_stt_cost.shift(1)+pairs_trd_cost
    pfmc_ts_0 = pair_first_value - pair_second_value - pairs_trd_cost_

    return pfmc_ts_0








