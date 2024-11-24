# -*- coding: utf-8 -*-
"""
Created on Wed Nov 13 21:12:04 2024

@author: david
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

#https://github.com/Crypto-toolbox/pandas-technical-indicators/blob/master/technical_indicators.py
import pickle

# 파일 불러오기
with open("C:\\Users\\david\\OneDrive\\바탕 화면\\data\\test_data.pkl", 'rb') as f:
    data = pickle.load(f)


data = np.load("C:\\Users\\david\\OneDrive\\바탕 화면\\data\\test_data.pkl", allow_pickle=True)


    
os.chdir('C:\\Users\\david\\OneDrive\\바탕 화면\\data\\')
import pandas_techinal_indicators as ta




raw_data = data.copy()
raw_data.index = pd.to_datetime(raw_data['DATE'])

raw_data['MSCI WORLD VALUE RS'] = raw_data['MSCI WORLD VALUE INDEX']/raw_data['MSCI WORLD']
raw_data['MSCI WORLD GROWTH RS'] = raw_data['MSCI WORLD GROWTH INDEX']/raw_data['MSCI WORLD']
raw_data['S&P 500 VALUE RS'] = raw_data['S&P 500 VALUE']/raw_data['S&P 500 INDEX']
raw_data['S&P 500 GROWTH RS'] = raw_data['S&P 500 GROWTH']/raw_data['S&P 500 INDEX']
raw_data['RUSSELL 1000 VALUE RS'] = raw_data['RUSSELL 1000 VALUE INDEX']/raw_data['RUSSELL 1000 INDEX']
raw_data['RUSSELL 1000 GROWTH RS'] = raw_data['RUSSELL 1000 GROWTH INDEX']/raw_data['RUSSELL 1000 INDEX']
raw_data['RUSSELL 2000 VALUE RS'] = raw_data['RUSSELL 2000 VALUE INDEX']/raw_data['RUSSELL 2000 INDEX']
raw_data['RUSSELL 2000 GROWTH RS'] = raw_data['RUSSELL 2000 GROWTH INDEX']/raw_data['RUSSELL 2000 INDEX']
raw_data['RUSSELL SIZE RS'] = raw_data['RUSSELL 2000 INDEX']/raw_data['RUSSELL 1000 INDEX']

ratio_data = raw_data[['MSCI WORLD VALUE RS','MSCI WORLD GROWTH RS', 'S&P 500 VALUE RS', 'S&P 500 GROWTH RS', 'RUSSELL 1000 VALUE RS', 'RUSSELL 1000 GROWTH RS', 'RUSSELL 2000 VALUE RS', 'RUSSELL 2000 GROWTH RS', 'RUSSELL SIZE RS' ]].copy()
ratio_data_std = ratio_data/ratio_data.iloc[0]

def feature_extraction(data):
    for x in [5, 14, 26, 44, 66]:
        data = ta.momentum(data, n=x)
        data = ta.rate_of_change(data, n=x)
        data = ta.trix(data, n=x)


    data['ema50'] = data['Close'] / data['Close'].ewm(50).mean()
    data['ema21'] = data['Close'] / data['Close'].ewm(21).mean()
    data['ema14'] = data['Close'] / data['Close'].ewm(14).mean()
    data['ema5'] = data['Close'] / data['Close'].ewm(5).mean()

    #Williams %R is missing
    data = ta.macd(data, n_fast=12, n_slow=26)
    return data





def compute_prediction_int(df, n):
    pred = (df.shift(-n)['Close'] >= df['Close'])
    pred = pred.iloc[:-n]
    return pred.astype(int)

def prepare_data(df, horizon):
    data = feature_extraction(df).dropna().iloc[:-horizon]
    data['pred'] = compute_prediction_int(data, n=horizon)
    del(data['Close'])
    return data.dropna()


ratio_data = ratio_data.reset_index()
ratio_data = ratio_data.drop(columns='DATE', axis=1)


ratio_data['MSCI WORLD VALUE RS'] = ratio_data['MSCI WORLD VALUE RS'].pct_change(periods=20)
ratio_data['MSCI WORLD GROWTH RS'] = ratio_data['MSCI WORLD GROWTH RS'].pct_change(periods=20)
ratio_data['S&P 500 VALUE RS'] = ratio_data['S&P 500 VALUE RS'].pct_change(periods=20)
ratio_data['S&P 500 GROWTH RS'] = ratio_data['S&P 500 GROWTH RS'].pct_change(periods=20)
ratio_data['RUSSELL 1000 VALUE RS'] = ratio_data['RUSSELL 1000 VALUE RS'].pct_change(periods=20)
ratio_data['RUSSELL 1000 GROWTH RS'] = ratio_data['RUSSELL 1000 GROWTH RS'].pct_change(periods=20)
ratio_data['RUSSELL 2000 VALUE RS'] = ratio_data['RUSSELL 2000 VALUE RS'].pct_change(periods=20)
ratio_data['RUSSELL 2000 GROWTH RS'] = ratio_data['RUSSELL 2000 GROWTH RS'].pct_change(periods=20)
ratio_data['RUSSELL SIZE RS'] = ratio_data['RUSSELL SIZE RS'].pct_change(periods=20)

ratio_data = ratio_data.drop(0)
ratio_data = ratio_data.reset_index()

ratio_data['Close'] = ratio_data['MSCI WORLD VALUE RS'].copy()   # TARGET
data = prepare_data(ratio_data, 20)  # 5, 60 
data.reset_index(drop=True, inplace=True)

data = data.iloc[:,1:]


y = data['pred']

#remove the output from the input
features = [x for x in data.columns if x not in ['gain', 'pred']]
X = data[features]
X = X.iloc[:,1:]
X = X.replace(np.inf, 0)
X = X.replace(-np.inf, 0)


train_size = 2*len(X) // 3
for_train_x = X[:train_size]
for_train_y = y[:train_size]


for_valid_x = X[train_size:]
for_valid_y = y[train_size:]