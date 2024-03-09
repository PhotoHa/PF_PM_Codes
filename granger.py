# -*- coding: utf-8 -*-
"""
Created on Sun Aug 28 01:29:00 2022

@author: david
"""

import statsmodels as sm
from statsmodels.tsa.stattools import grangercausalitytests
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv("C:\\Users\\david\\OneDrive\\바탕 화면\\granger_vkspi_short.csv", header=0, index_col=0)
df = df.rolling(30).mean()
df = df.dropna()
gc_res = grangercausalitytests(df, 10)
gc_res2 = grangercausalitytests(df.iloc[:,[1,0]], 10)

#adf test
target1 = df.vkospi.copy()
integ_result1 = pd.Series(sm.tsa.stattools.adfuller(target1)[0:4], 
                         index=['Test Statistics', 'p-value', 'Used Lag', 'Used Observations'])
Y1_integ_order1 = 0
if integ_result1[1] > 0.1:
    Y1_integ_order1 = Y1_integ_order1 + 1






df1 = pd.read_csv("C:\\Users\\david\\OneDrive\\바탕 화면\\granger_vkspi_sector.csv", header=0, index_col=0)
df1 = df1.dropna()
df1.plot()

df11 = df1[["vkospi","one_v"]]
#df11 = df1[["vkospi","one"]]
df11_high_vol = df11[df11["vkospi"]>20] 

#df1 = df1.rolling(30).mean()
#df1 = df1.dropna()
gc_res1 = grangercausalitytests(df11_high_vol, 20)




# 비정상성 차수 추론
# adf테스트로 추세확인 후 1차 차분
target = df11_high_vol.one_v.copy()
integ_result = pd.Series(sm.tsa.stattools.adfuller(target)[0:4], 
                         index=['Test Statistics', 'p-value', 'Used Lag', 'Used Observations'])
Y1_integ_order = 0
if integ_result[1] > 0.1:
    Y1_integ_order = Y1_integ_order + 1

def adf_test(df):
    target = df.iloc[:,0].copy()
    integ_result = pd.Series(sm.tsa.stattools.adfuller(target)[0:4], 
                             index=['Test Statistics', 'p-value', 'Used Lag', 'Used Observations'])
    Y1_integ_order = 0
    if integ_result[1] > 0.1:
        Y1_integ_order = Y1_integ_order + 1
        
    target2 = df.iloc[:,1].copy()  ##ksp200
    integ_result2 = pd.Series(sm.tsa.stattools.adfuller(target2)[0:4], 
                             index=['Test Statistics', 'p-value', 'Used Lag', 'Used Observations'])
    Y2_integ_order = 0
    if integ_result2[1] > 0.1:
        Y2_integ_order = Y2_integ_order + 1

    print('Y1_order: ', Y1_integ_order, 'Y2_order: ', Y2_integ_order)
    
#target = raw.SE_PS.copy()
#integ_result = pd.Series(sm.tsa.stattools.adfuller(target)[0:4], 
#                         index=['Test Statistics', 'p-value', 'Used Lag', 'Used Observations'])
#Y2_integ_order = 0
#if integ_result[1] > 0.1:
#    Y2_integ_order = Y2_integ_order + 1
#print('Y1_order: ', Y1_integ_order, 'Y2_order: ', Y2_integ_order)



granger_result1 = sm.tsa.stattools.grangercausalitytests(df11_high_vol.diff(1).dropna().values, maxlag=20, verbose=True)




#### 무역수지 경상수지와 코스피200 그레인져 테스트 ####
bp_df = pd.read_csv("C:\\Users\\david\\OneDrive\\바탕 화면\\balanceP.csv")
bp_df
bp_ca = bp_df[["current account balance","ksp200"]]
bp_t = bp_df[["trade balance","ksp200"]]


adf_test(bp_ca)
adf_test(bp_t)

gc_ca_test1 = grangercausalitytests(bp_ca, 5)
gc_ca_test2 = grangercausalitytests(bp_ca.iloc[:,[1,0]], 5)

gc_t_test1 = grangercausalitytests(bp_t, 5)
gc_t_test2 = grangercausalitytests(bp_t.iloc[:,[1,0]], 5)






