# -*- coding: utf-8 -*-
"""
Created on Fri Jun 14 13:17:08 2024

@author: 11149
"""


import pandas as pd
import numpy as np
import pyodbc
import os
os.chdir('T:\\index\\95_곽용하\\운용\\코드\\')
import mf_3 as mf

conn_quant = pyodbc.connect('driver={SQL Server};server=46.2.90.172;database=quant;uid=index;pwd=samsung@00')
conn_wisefn = pyodbc.connect('driver={SQL Server};server=46.2.90.172;database=wisefn;uid=index;pwd=samsung@00')

# factor data
sql =  '''  SELECT CONVERT(VARCHAR(8),ScoreDate,112) as TRD_DT, FactorCode AS FACTOR, Code AS STK_CD, Ratio AS VAL
            FROM QUANT..QA_FactorDat_
            WHERE 1=1
            AND FactorCode IN ('231850_FY0', 'ES_NI_FQ0FWD_3')
            AND ScoreDate IN (SELECT DT
            				FROM WISEFN..TZ_DATE
            				WHERE YMD BETWEEN '20140430' AND '20231231' AND MN_END_YN = 1 AND TRADE_YN = 1)
            ORDER BY TRD_DT, STK_CD  '''

data_factor = pd.read_sql(sql, conn_quant)

# daily composition data
sql_dt_comp = '''  SELECT TRD_DT, 'A'+STK_CD AS STK_CD
                    FROM WISEFN..TS_STK_ISSUE
                    WHERE 1=1
                    AND TRD_DT IN (SELECT YMD
                    				FROM WISEFN..TZ_DATE
                    				WHERE YMD BETWEEN '20140430' AND '20231231' AND MN_END_YN = 1 AND TRADE_YN = 1)
                    AND MKT_TYP = 1
                    AND MV_SIZE_TYP != 0 --우선주, ETF, REITS 등 제외
                    AND ADMIN_YN = 0 --관리종목여부
                    AND TRD_STOP_TYP = 0 --거래정지여부
                    AND CAUTION_TYP = 0 --투자유의구분  '''
data_ksp_comp = pd.read_sql(sql_dt_comp, conn_wisefn)

# MktCap data
sql_cap = '''  SELECT TRD_DT, 'A'+STK_CD AS STK_CD, MKT_VAL
                FROM WISEFN..TS_STK_DAILY
                WHERE 1=1
                AND STK_CD IN (SELECT CMP_CD
                				FROM WISEFN..TS_STOCK
                				WHERE MKT_TYP = 1 AND STK_TYP = 1 AND ISSUE_TYP = 1)
                AND TRD_DT IN (SELECT YMD
                				FROM WISEFN..TZ_DATE
                				WHERE YMD BETWEEN '20140430' AND '20231231' AND MN_END_YN = 1 AND TRADE_YN = 1)  '''
data_cap = pd.read_sql(sql_cap, conn_wisefn)
data_cap = data_cap.sort_values(['TRD_DT','MKT_VAL'],ascending=[True,False]).reset_index(drop=True)
data_cap['cum_cap'] = data_cap.groupby('TRD_DT').cumsum()

cap_total = pd.DataFrame(data_cap.groupby('TRD_DT')['MKT_VAL'].sum()).reset_index()
cap_total.columns = ['TRD_DT','CAP_TOTAL']
data_cap = pd.merge(data_cap, cap_total, 'left', 'TRD_DT')
data_cap['cum_ratio'] = data_cap.cum_cap / data_cap.CAP_TOTAL
data_cap['size_1'] = data_cap['cum_ratio']

data_cap['BM'] = data_cap.MKT_VAL / data_cap.CAP_TOTAL



for i in range(len(data_cap)):
    if data_cap['cum_ratio'][i] <= 0.75:
        data_cap['size_1'][i] = 4
    elif data_cap['cum_ratio'][i] <= 0.90:
        data_cap['size_1'][i] = 3
    elif data_cap['cum_ratio'][i] <= 0.97:
        data_cap['size_1'][i] = 2
    else:
        data_cap['size_1'][i] = 1

cap_tile = data_cap[['TRD_DT','STK_CD','MKT_VAL','BM','size_1']].rename(columns={'size_1':'SIZ'})



# 팩터 데이터 분류
df_factor_value = pd.merge(data_ksp_comp, data_factor, 'left', ['TRD_DT','STK_CD'])
df_factor_1 = df_factor_value[df_factor_value['FACTOR']=='231850_FY0']
df_factor_2 = df_factor_value[df_factor_value['FACTOR']=='ES_NI_FQ0FWD_3']


df_factor_1 = df_factor_1.rename(columns={'VAL':'231850_FY0'})
df_factor_2 = df_factor_2.rename(columns={'VAL':'ES_NI_FQ0FWD_3'})

df_factor_1 = df_factor_1.drop('FACTOR', axis=1)
df_factor_2 = df_factor_2.drop('FACTOR', axis=1)

df_factors = pd.merge(df_factor_1, df_factor_2, 'outer', ['TRD_DT','STK_CD'])
df_0 = pd.merge(data_cap, df_factors, 'left', ['TRD_DT','STK_CD'])


#
sql_prc = '''  SELECT TRD_DT, 'A'+STK_CD AS STK_CD, VAL AS ADJPRC
                FROM WISEFN..TS_STK_DATA
                WHERE 1=1
                AND STK_CD IN (SELECT CMP_CD
                				FROM WISEFN..TS_STOCK
                				WHERE MKT_TYP = 1 AND STK_TYP = 1 AND ISSUE_TYP = 1)
                AND TRD_DT IN (SELECT YMD
                				FROM WISEFN..TZ_DATE
                				WHERE YMD BETWEEN '20140430' AND '20231231' AND MN_END_YN = 1 AND TRADE_YN = 1)
                AND ITEM_CD = '100300'  '''

data_prc = pd.read_sql(sql_prc, conn_wisefn)

prc_table = data_prc.pivot_table('ADJPRC', 'TRD_DT', 'STK_CD')
rtn_table = prc_table.pct_change(1).shift(-1) #fwd
fwd_rtn_stack = rtn_table.stack()
fwd_rtn_stack = fwd_rtn_stack.reset_index()
fwd_rtn_stack.columns = ['TRD_DT','STK_CD','Fwd_Rtn']



# Factor Pivot
#df_0 = pd.read_excel("T:\\index\\95_곽용하\\연구\\00_MSCI\\data.xlsx")

def apply_percent_rank(group):
    def percent_rank(arr, x):
        arr = np.sort(arr)
        N = len(arr)
        
        if x < arr[0]:
            return 0.0
        if x > arr[-1]:
            return 1.0
        
        for i in range(N):
            if arr[i] == x:
                break
            elif arr[i] > x:
                break
        rank = i + (x - arr[i-1]) / (arr[i] - arr[i-1])
        percent_rank = rank / (N - 1)
        return percent_rank
    
    return group.apply(lambda x: percent_rank(group.values, x))

# 샘플 데이터 생성
# 그룹별로 percent_rank 계산
df = df_0.copy() #iloc[:,1:]
df['TRD_DT'] = df['TRD_DT'].astype('str')

df['PctRnk_NI_yoy'] = df.groupby('TRD_DT')['231850_FY0'].transform(apply_percent_rank)
df['PctRnk_ES_NI'] = df.groupby('TRD_DT')['ES_NI_FQ0FWD_3'].transform(apply_percent_rank)

df['PctRnk_NI_yoy'] = (df['PctRnk_NI_yoy']-0.5)*2
df['PctRnk_ES_NI']  = (df['PctRnk_ES_NI']-0.5)*2

# nan -> 0
df['PctRnk_NI_yoy'] = df['PctRnk_NI_yoy'].fillna(0)
df['PctRnk_ES_NI']  = df['PctRnk_ES_NI'].fillna(0)


# 사이즈별 가중치 부여
df['loading'] = df['size_1']
df['loading'] = df['loading'].replace(4,0.0014).replace(3,0.0010).replace(2,0.0006).replace(1,0.0004)

# 틸팅
df['tilting_NI_yoy'] = df['PctRnk_NI_yoy']*df['loading']
df['tilting_NI_ES_NI'] = df['PctRnk_ES_NI']*df['loading']
df['tilting_MIX'] = df['tilting_NI_yoy']
for i in range(len(df)):
    if df['size_1'][i] >= 3:
        df['tilting_MIX'][i] = df['tilting_NI_ES_NI'][i]
    else:
        df['tilting_MIX'][i] = df['tilting_NI_yoy'][i]

# TP1
df['MP_NI_yoy'] = df['tilting_NI_yoy'] + df['BM']
df['MP_ES_NI'] = df['tilting_NI_ES_NI'] + df['BM']
df['MP_MIX'] = df['tilting_MIX'] + df['BM']
        



# 중간 정리
df1 = pd.merge(df, fwd_rtn_stack, 'left', ['TRD_DT','STK_CD'])
df1_1 = df1[['TRD_DT','STK_CD','Fwd_Rtn','BM','MP_NI_yoy','MP_ES_NI','MP_MIX']]

# 0보다 작으면 0 부여
df1_1['MP_NI_yoy'][df1_1['MP_NI_yoy']<0] = 0
df1_1['MP_ES_NI'][df1_1['MP_NI_yoy']<0] = 0
df1_1['MP_MIX'][df1_1['MP_NI_yoy']<0] = 0

# 100환산
imsi = df1_1.groupby('TRD_DT')[['MP_NI_yoy','MP_ES_NI','MP_MIX']].transform('sum')
df1_1[['MP_NI_yoy_1','MP_ES_NI_1','MP_MIX_1']] = df1_1[['MP_NI_yoy','MP_ES_NI','MP_MIX']]/ imsi
df1_1 = df1_1.fillna(0)

# 월별 수익률
mp_1 = df1_1.groupby('TRD_DT').apply(lambda x: (x['MP_NI_yoy_1']*x['Fwd_Rtn']).sum()).reset_index(name='MP_NI_yoy_1')
mp_2 = df1_1.groupby('TRD_DT').apply(lambda x: (x['MP_ES_NI']*x['Fwd_Rtn']).sum()).reset_index(name='MP_ES_NI')
mp_3 = df1_1.groupby('TRD_DT').apply(lambda x: (x['MP_MIX']*x['Fwd_Rtn']).sum()).reset_index(name='MP_MIX')
bm   = df1_1.groupby('TRD_DT').apply(lambda x: (x['BM']*x['Fwd_Rtn']).sum()).reset_index(name='BM')

# 데이터프레임 정리
a1 = pd.merge(bm, mp_1, 'left','TRD_DT')
a2 = pd.merge(a1, mp_2, 'left','TRD_DT')
a3 = pd.merge(a2, mp_3, 'left','TRD_DT')

# fin
fin = a3.copy()
fin.TRD_DT = pd.to_datetime(fin.TRD_DT, format='%Y%m%d')
fin.set_index('TRD_DT',drop=True, inplace=True)
fin = fin.shift(1).fillna(0)
((1+fin).cumprod()-1).plot(figsize=(12,6))

fin_summary = mf.summary_stats(fin, 12)


fin_index = ((1+fin).cumprod())*1000

# analysis






