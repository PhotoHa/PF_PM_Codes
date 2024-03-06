# -*- coding: utf-8 -*-
"""
Created on Fri Mar 17 16:06:55 2023

@author: 11149
"""
import pyodbc
import pandas as pd
from datetime import datetime
import numpy as np
import openpyxl as op
import xlwings as xw
import itertools
import math
import time
import os
os.chdir('')
import functions_0 as mf

conn_quant = pyodbc.connect('driver={SQL Server};server=')
conn_wisefn = pyodbc.connect('driver={SQL Server};server=')
conn_pcor = pyodbc.connect('driver={Oracle in OraClient12Home1};')


'''  
1) ChatGPT로 국면 질문
2) 해당 국면과 유사한 기간을 답변 받은 뒤, 해당 기간으로 필터링
3) 팀 DB에서 모든 팩터들에 대해 롱숏 성과 추출
4) 코스피200 지수의 1개월 뒤 수익률(예측력)에 대한 팩터성과들 다중회귀분석
5) 다중회귀분석을 할 때 LASSO 방식을 적용해 5개 팩터 선별
'''

# 특정기간 : chatGPT 에 질문해서 나온 값
#time_pd = ['201007', '201008', '201301', '201302', '201303', '201304', '201507', '201508', '201611', '201612']
time_pd = ['200006', '200111', '200208', '200209', '200811', '200812', '201108', '201109', '201110', '201111',
           '201112', '201201', '201202', '201203', '201204', '201205', '201206']
idx_cd = 'IKS200'
max_features = 3
long_num = 30



########## a) 특정 기간 내 5개 팩터 선별 ##########

sql = """ select a.BASE_D, a.FactorCode, a.TILE_RET - b.TILE_RET as LS
            from RET a
            	left join RET b
            		on a.BASE_D = b.BASE_D and a.FactorCode = b.FactorCode
            where 1=1
            	and a.BASE_D >= '20000101'
            	and a.bm = 'ks200'
            	and b.bm = 'ks200'
            	and a.tile  = 4
            	and b.tile = 1
            order by BASE_D, FactorCode """
df_raw = pd.read_sql(sql, conn_wisefn)

sql_idx = f'''  SELECT TRD_DT AS BASE_D, CLOSE_PRC, LEAD(CLOSE_PRC, 1) OVER (ORDER BY TRD_DT) AS NXT_D
                FROM DAILY
                WHERE 1=1
                	AND TRD_DT IN (SELECT YMD FROM DATE WHERE MN_END_YN = 1)
                	AND TRD_DT >= '20000101'
                	AND SEC_CD = '{idx_cd}'
                ORDER BY TRD_DT  '''
df_idx = pd.read_sql(sql_idx, conn_wisefn)
df_idx['fwd_rt'] = df_idx.NXT_D / df_idx.CLOSE_PRC - 1


# df 합치기 (모양만 바꾸는 것)
df_raw_new = df_raw.pivot(index = 'BASE_D', columns = 'FactorCode', values = 'LS')
df_raw_new.reset_index(inplace = True)

df_raw_new['BASE_D'] = df_raw_new['BASE_D'].astype(str)
df = pd.merge(df_idx, df_raw_new, "inner", 'BASE_D')

time_pd_8 = df['BASE_D'].unique() ## 연월일

df['BASE_D_6'] = df['BASE_D'].astype(str).str[:6]

df = df[df['BASE_D_6'].isin(time_pd)] ##팩터별 롱숏 성과도 여기서 특정 기간으로 자름 (chk')
df = df.dropna(axis=1) ##모든 기간에 값이 존재하여야 함


# lasso
from sklearn.linear_model import LassoCV
from sklearn.feature_selection import SelectFromModel

lasso_0 = LassoCV()
y = df['fwd_rt']
X = df.iloc[:,4:-1]

lasso_0.fit(X, y)

mdl_lasso = SelectFromModel(lasso_0, prefit=True, max_features = max_features)
X_new = mdl_lasso.transform(X)
X_fin = pd.DataFrame(X_new)

slctd = []
for i in range(len(X_fin.columns)):
    col_to_find = X_fin.iloc[0,i]
    cols = df.columns[df.isin([col_to_find]).any()]
    slctd = np.append(slctd, cols)

print(slctd) ## 선정된 5개 팩터



# 팩터 추출
time_pd_80 = tuple(time_pd_8)

facotr_list = tuple(slctd)
f1 = facotr_list[0]; f2 = facotr_list[1]; f3 = facotr_list[2]

sql_200 = f'''  select TRD_DT AS BASE_D, STK_CD as Code
            	from wisefn..ts_stk_issue
            	where 1=1
            		and KS200_TYP = 1 -- 이 부분만 바꾸면 됨. KQ150
            		and TRD_DT in (select ymd from wisefn..tz_date where MN_END_YN = 1)
            		and TRD_DT in {time_pd_80}  '''

sql_fct = f'''  select convert(varchar(08),ScoreDate,112) as BASE_D, FactorCode, right(Code,6) as Code, Ratio
			from quant..QA_FactorDat_
            where 1=1
                and FactorCode in {facotr_list}
                and convert(varchar(08),ScoreDate,112) in {time_pd_80}'''

list_200 = pd.read_sql(sql_200, conn_quant)
fct5_data = pd.read_sql(sql_fct, conn_quant)


########## b-1) 전체 기간에 대한 개별 종목의 팩터 및 1개월 수익률 구하기 ##########

# 코스피200 종목의 수정주가 데이터 가져오
sql_prc = ''' select a.TRD_DT AS BASE_D, a.STK_CD, b.VAL
                from wisefn..ts_stk_issue a
                	left join wisefn..ts_stk_data b
                		on a.TRD_DT = b.TRD_DT and a.STK_CD = b.STK_CD
                where 1=1
                    and a.KS200_TYP = 1 -- 이 부분만 바꾸면 됨. KQ150
                    and a.TRD_DT in (select ymd from wisefn..tz_date where MN_END_YN = 1)
                    and a.TRD_DT >= '20000101'
                	and b.ITEM_CD = '100300' '''

prc_df = pd.read_sql(sql_prc,conn_wisefn)


# 종목을 기준으로 그룹화하고, 그룹 내에서 각 종목별 수익률의 한 달 뒤 수익률을 계산하기
prc_df = prc_df.sort_values(['STK_CD','BASE_D'])
prc_df['fwd_1m'] = prc_df.groupby('STK_CD')['VAL'].shift(-1)
prc_df['fwd_1m_rt'] = prc_df['fwd_1m'] / prc_df['VAL'] - 1

prc_df_lasso_yong = prc_df.copy() ## lasso 로 변수 추리기 할 때 사용할거임

imsi_prc = prc_df.groupby('BASE_D').apply(lambda x: x.rank(pct=True))
prc_df['pct'] = imsi_prc['fwd_1m_rt']
prc_df = prc_df[['BASE_D','STK_CD','pct']]



########## b-2) 개별 팩터 값을 분위 값으로 변경하기 ##########


# df 만들기 (1) : 기본
''' 
n/a 가 있는 행은 지움. 즉 개별 종목의 경우 모든 팩터의 값이 들어 있어야 함
값이 클 수록 높은 숫자의 순위가 매겨지도록 한 뒤, 순위값을 기준으로 표준화하여 값 부여.
이상치를 고려한 것. 랭크로 팩터별 분산의 정도를 고려하지 않을 수 있고, 팩터 5개의 영향력을 보다 편중되지 않게 하고자.
'''
df2 = pd.merge(list_200, fct5_data[fct5_data['FactorCode'] == f1], 'left', ['BASE_D', 'Code'])
df2 = pd.merge(df2, fct5_data[fct5_data['FactorCode'] == f2], 'left', ['BASE_D', 'Code'])
df2 = pd.merge(df2, fct5_data[fct5_data['FactorCode'] == f3], 'left', ['BASE_D', 'Code'])


df2 = df2.iloc[:,[0,1,3,5,7]]
df2.columns = ['BASE_D', 'Code', f1, f2, f3]

df3 = df2.copy()
df3 = df3.dropna(axis = 0)

df4 = df3.iloc[:,[0,2,3,4]]
df4 = df4.groupby('BASE_D').rank(ascending = True) ## 랭크화
df4['BASE_D'] = df3['BASE_D']

df5 = df4.groupby('BASE_D').apply(lambda x: (x-x.mean()) / x.std() ) ## 표준화
df5['BASE_D'] = df3['BASE_D']
df5['STK_CD'] = df3['Code']


########## c) 로지스틱 리그레션 ##########

# Logistic Regression
from sklearn.linear_model import LogisticRegression
from sklearn import metrics
from sklearn.model_selection import cross_val_score
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn import metrics


def lgst_reg(df, target1 = 0.6, target21 = 0.65, target22 = 0.35, target31 = 0.75, target32 = 0.50, target33 = 0.25, prc_df = prc_df, num = 'binomial', kan = 3, time_pd = time_pd):
    ''' target1 : binomial할 때. 상위40%(백분위 60%) 이상일 때 1 부여
        target21 : multinomial, 3개로 나눌 때, 가장 수익률이 좋은 분위에 속하는 기준
        target31 : multinomial, 4개로 나눌 때, 가장 수익률이 좋은 분위에 속하는 기준 
        num : 'binomial' 또는 'multinomial
        kan : 3 또는 4 '''
    df5 = df

    # learning : binomial 방식
    df_mid = pd.merge(df5, prc_df, 'left', ['BASE_D','STK_CD']).dropna()
    df_mid['rst'] = 0
    df_mid.loc[df_mid['pct'] >= target1, 'rst'] = 1
    
    df_mid['BASE_D'] = df_mid['BASE_D'].astype(str).str[:6]
    df_mid = df_mid[df_mid['BASE_D'].isin(time_pd)]  ## 특정 기간에 해당하는 값들로 필터링하는 것
    mal_X = df_mid.iloc[:,0:3] 
    mal_Y = df_mid["rst"]
    model_1 = LogisticRegression()
    model_1.fit(X = mal_X, y = mal_Y)
    prb1 = model_1.predict_proba(mal_X)
    #예측치
    y_pred = model_1.predict(X = mal_X)
    # 분류 정확도
    acc = metrics.accuracy_score(mal_Y, y_pred)
    # confusion matrix
    con_mat = metrics.confusion_matrix(y_true = mal_Y, y_pred = y_pred)
    
    
    # learning2 : multinomial 방식. 3칸
    df_mid2 = pd.merge(df5, prc_df, 'left', ['BASE_D','STK_CD']).dropna()
    df_mid2['rst'] = 1
    df_mid2.loc[df_mid2['pct'] >= target21, 'rst'] = 2  # 가장 좋은 것
    df_mid2.loc[df_mid2['pct'] < target22, 'rst'] = 0 
    
    mal_X2 = df_mid2.iloc[:,0:3] 
    mal_Y2 = df_mid2["rst"]
    model_2 = LogisticRegression(random_state=0, solver='lbfgs', multi_class='multinomial')
    model_2.fit(X = mal_X2, y = mal_Y2) #학습수행
    prb2 = model_2.predict_proba(mal_X2)
    
    y_pred2 = model_2.predict(X = mal_X2)
    acc2 = metrics.accuracy_score(mal_Y2, y_pred2)
    con_mat2 = metrics.confusion_matrix(y_true = mal_Y2, y_pred = y_pred2)


    # learning3 : multinomial 방식. 4칸
    df_mid3 = pd.merge(df5, prc_df, 'left', ['BASE_D','STK_CD']).dropna()
    df_mid3['rst'] = 1
    df_mid3.loc[df_mid3['pct'] >= target31, 'rst'] = 3 # 가장 좋은 것
    df_mid3.loc[(df_mid3['pct'] < target32) & (df_mid3['pct'] >= target33), 'rst'] = 2 
    df_mid3.loc[df_mid3['pct'] < target33, 'rst'] = 0 

    
    mal_X3 = df_mid3.iloc[:,0:3] 
    mal_Y3 = df_mid3["rst"]
    model_3 = LogisticRegression(random_state=0, solver='lbfgs', multi_class='multinomial')
    model_3.fit(X = mal_X3, y = mal_Y3) #학습수행
    prb3 = model_3.predict_proba(mal_X3)
    
    y_pred3 = model_3.predict(X = mal_X3)
    acc3 = metrics.accuracy_score(mal_Y3, y_pred3)
    con_mat3 = metrics.confusion_matrix(y_true = mal_Y3, y_pred = y_pred3)


    if num == 'binomial':
        con_mat_f = con_mat
        acc_f = acc
        coef = model_1.coef_
        intercept = model_1.intercept_
        y_exp = y_pred
        raw_df = df_mid
        mal_X = mal_X
        mal_Y = mal_Y
        prb = prb1
        
    elif (num == 'multinomial') & (kan == 3):
        con_mat_f = con_mat2
        acc_f = acc2
        coef = model_2.coef_
        intercept = model_2.intercept_   
        y_exp = y_pred2
        raw_df = df_mid2
        mal_X = mal_X2
        mal_Y = mal_Y2  
        prb = prb2
        
    elif (num == 'multinomial') & (kan == 4):
        con_mat_f = con_mat3
        acc_f = acc3
        coef = model_3.coef_
        intercept = model_3.intercept_
        y_exp = y_pred3  
        raw_df = df_mid3  
        mal_X = mal_X3
        mal_Y = mal_Y3   
        prb = prb3
        
    else:
        pass
    
    return con_mat_f, acc_f, coef, intercept, y_exp, raw_df, mal_X, mal_Y, prb


### binomial(sell), mutinomial(buy) 방식

## BUY
trial_bin_buy = lgst_reg(df5, target1 = 0.50, target21 = 0.65, target22 = 0.35, target31 = 0.55, target32 = 0.50, target33 = 0.45, num = 'multinomial', kan = 3)
print(trial_bin_buy[0], trial_bin_buy[1]) ## 각각 예측-실제 행렬, 전체 정확도

# 정확도 체크
chk_bin_buy = trial_bin_buy[0]
col_sum = chk_bin_buy.sum(axis = 0)
rst_con = chk_bin_buy/col_sum * 100 ## 예측 대비 정확도 (%)
print(rst_con)


# 시각화 (정확도)
#plt.figure(figsize=(6,6))
#plt.xlabel('예측')
#plt.ylabel('실제')
#sns.heatmap(rst_con, annot=True, fmt=".3f", linewidths=.5, square=True)


# 상위 하위 30종목 추출
new_df = trial_bin_buy[5]
#new_df = new_df.astype(int)
probt = pd.DataFrame(trial_bin_buy[8])
probt[['BASE_D','STK_CD']] = new_df[['BASE_D','STK_CD']]
#probt = probt.astype(int)


new_df2 = pd.merge(new_df, probt, 'inner', ['BASE_D','STK_CD'])
#new_df2 = new_df2.astype(int)
#new_df2[['BASE_D','STK_CD']] = new_df2[['BASE_D','STK_CD']].astype(str)

new_df2.rename(columns = {0:'sell', 1:'ntrl', 2:'buy'}, inplace = True)
new_df2['BASE_D'] = new_df2['BASE_D'].str[0:6]
prc_df_lasso_yong['BASE_D'] = prc_df_lasso_yong['BASE_D'].str[0:6]



new_df3 = pd.merge(new_df2, prc_df_lasso_yong, 'inner', ['BASE_D','STK_CD']) ####################################################





sang30 = new_df3.sort_values(['BASE_D','buy']).groupby('BASE_D').tail(long_num)
hawi30 = new_df3.sort_values(['BASE_D','sell']).groupby('BASE_D').tail(long_num)

sang_rtn = sang30.groupby('BASE_D').mean()
hawi_rtn = hawi30.groupby('BASE_D').mean()

ddff = pd.DataFrame({'sang':sang_rtn['fwd_1m_rt'], 'hawi': hawi_rtn['fwd_1m_rt']})
ddff['ls'] = ddff.sang - ddff.hawi
print(ddff)





