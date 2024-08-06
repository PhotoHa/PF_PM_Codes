# -*- coding: utf-8 -*-
"""
Created on Tue Aug  6 20:04:55 2024

@author: david
"""

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.decomposition import NMF
from statsmodels.tsa.arima.model import ARIMA
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
import pickle
import matplotlib.pyplot as plt
import os
os.chdir("C:\\Users\\david\\OneDrive\\바탕 화면\\data\\")

with open('daily_analysis.pkl', 'rb') as file:
    df00 = pickle.load(file)

# rsi function
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

def conditional_cumsum(series):
    cumsum = 0
    result = []
    for value in series:
        if value == 1:
            cumsum += value
            result.append(cumsum)
        else:
            cumsum = 0
            result.append(np.nan)
    return pd.Series(result, index=series.index)

def keep_max_in_sequences(series):
    # 자연수 구역을 식별
    mask = series.notna()
    group = (mask != mask.shift()).cumsum()
    
    result = series.copy()
    for g in result[mask].groupby(group):
        max_val = g[1].max()
        result[g[1].index] = np.nan
        result[g[1].idxmax()] = max_val
        
    return result


factor_daily_rtn = df00.shift(1).dropna()
factor_daily_idx = (1+factor_daily_rtn).cumprod()*1000
factor_weekly_idx = factor_daily_idx.resample('W').last()
factor_weekly_rtn = factor_weekly_idx.pct_change(1).dropna()

factor_rsi = RSI(factor_weekly_rtn, 14).dropna()


# 과매수/매도 유지 기간 분석 (주간 변동성 줄이고자 MA(4주))
factor_rsi_ma_4 = factor_rsi.rolling(4).mean().dropna()
factor_rsi_ma_4_over  = (factor_rsi_ma_4 > 70)*1
factor_rsi_ma_4_under = (factor_rsi_ma_4 < 30)*1

imsi = factor_rsi_ma_4_over.transform(lambda x: conditional_cumsum(x))
imsi2 = imsi.transform(lambda x: keep_max_in_sequences(x) )
over_count = imsi2.stack().reset_index()
over_count.groupby('FactorGroup_sub')[0].mean()
over_count.groupby('FactorGroup_sub')[0].describe()

imsi2 = factor_rsi_ma_4_under.transform(lambda x: conditional_cumsum(x))
imsi22 = imsi2.transform(lambda x: keep_max_in_sequences(x) )
under_count = imsi22.stack().reset_index()
under_count.groupby('FactorGroup_sub')[0].mean()
under_count.groupby('FactorGroup_sub')[0].describe()

# 여기에 1개월 모멘텀 효과 사용하면 좋을 듯