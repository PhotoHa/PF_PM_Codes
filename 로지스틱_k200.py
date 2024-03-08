# -*- coding: utf-8 -*-
"""
Created on Tue Nov 15 10:50:27 2022

@author: 11149
"""

import numpy as np
import pandas as pd
import scipy.stats as ss
import itertools

import matplotlib.pyplot as plt
import seaborn as sn #heatmap-Accuracy Score

from sklearn.linear_model import LogisticRegression
from sklearn import metrics
from sklearn.model_selection import cross_val_score

### data input ###
k200 = pd.read_csv("k200.csv", index_col=0)
prc = pd.read_csv("prc.csv", index_col=0)
vol = pd.read_csv("vol.csv", index_col=0)
pns = pd.read_csv("pns.csv", index_col=0)
vksp = pd.read_csv("csv", index_col=0)

 

k200.index = pd.to_datetime(k200.index, format="%Y%m%d")
prc.index = pd.to_datetime(prc.index, format="%Y%m%d")
vol.index = pd.to_datetime(vol.index, format="%Y%m%d")
pns.index = pd.to_datetime(pns.index, format="%Y%m%d")
vksp.index = pd.to_datetime(vksp.index, format="%Y%m%d")

vksp = vksp['2004-12':]
### vkospi 신호 정리하기 ###
''' 4:고상  3:고하  2:저상  1:저하 '''
v_signal = vksp.copy()
v_signal['HL'] = v_signal['vkospi'] >= 18
v_signal['UD'] = v_signal['vkospi'].pct_change(1) >= 0

v_sign = pd.DataFrame(index=vksp.index, columns=['v_sign'])
for i in range(v_signal.shape[0]):
    if (v_signal.iloc[i,1] == True) & (v_signal.iloc[i,2] == True):
        v_sign.iloc[i] = 4
    elif (v_signal.iloc[i,1] == True) & (v_signal.iloc[i,2] == False):
        v_sign.iloc[i] = 3
    elif (v_signal.iloc[i,1] == False) & (v_signal.iloc[i,2] == True):
        v_sign.iloc[i] = 2
    elif (v_signal.iloc[i,1] == False) & (v_signal.iloc[i,2] == False):
        v_sign.iloc[i] = 1
    else:
        pass

v_sign = v_sign['2005':'2022-09'] 

### k200 여부 ###

''' 1이면 200, 0이면 nan  '''
k200 = k200.replace(0,np.NaN)

 

### 1개월  후 수익률 ###
fwd_M12 = prc.pct_change(1).shift(-1)

 
### 12개월, 1개월 모멘텀 ###
M12 = prc.pct_change(12)
M1 = prc.pct_change(1)

### 60일 변동성의 직전 달 대비 변화율 ###
vol_diff = vol.pct_change(1)

### 2005년 이후로 슬라이싱 ###
fwd_M12 = fwd_M12['2005':]
M12 = M12['2005':]
M1 = M1['2004-12':]
vol_diff = vol_diff['2005':]


### k200인 경우로 걸러내기 ###
k200_1 = k200['2005':]
k200_2 = k200['2004-12':]

M12 = M12 * k200_1
M1 = M1 * k200_2
fwd_M12 = fwd_M12 * k200_1
vol_diff = vol_diff * k200_1


 

#####################
### 순위 및 정규화 ###
#####################
M12_rank = M12.rank(ascending = True, axis = 1)
M12_rank_z = ss.zscore(M12_rank, axis= 1, nan_policy='omit')
M12_rank_z_df = pd.DataFrame(M12_rank_z)
M12_rank_z_df.index = M12_rank.index
 
M1_rank = M1.rank(ascending = True, axis = 1)
M1_rank_z = ss.zscore(M1_rank, axis= 1, nan_policy='omit')
M1_rank_z_df = pd.DataFrame(M1_rank_z)
M1_rank_z_df.index = M1_rank.index

M2 = M1_rank_z_df.diff(1)
M2_rank_z = ss.zscore(M2, axis= 1, nan_policy='omit')
M2_rank_z_df = pd.DataFrame(M2_rank_z)
M2_rank_z_df.index = M1_rank.index
 
vol_diff_rank = vol_diff.rank(ascending = False, axis = 1)
vol_diff_rank_z = ss.zscore(vol_diff_rank, axis= 1, nan_policy='omit')
vol_diff_rank_z_df = pd.DataFrame(vol_diff_rank_z)
vol_diff_rank_z_df.index = vol_diff_rank.index
 

fwd_M12_rank = fwd_M12.rank(ascending = False, axis = 1)
fwd_M12_rank_z = ss.zscore(fwd_M12_rank, axis= 1, nan_policy='omit')
fwd_M12_rank_z_df = pd.DataFrame(fwd_M12_rank_z)
fwd_M12_rank_z_df.index = fwd_M12_rank.index
 

#####################
### 4분위   프레임 ###
#####################
def tile5(df):
    t5 = pd.DataFrame(data=None, index = df.index, columns=[['1Q','2Q','3Q']])
    t5['1Q'] = df.quantile(q=0.20,axis=1)
    t5['2Q'] = df.quantile(q=0.40,axis=1)
    t5['3Q'] = df.quantile(q=0.60,axis=1)
    t5['4Q'] = df.quantile(q=0.80,axis=1)
    return t5


def tile4(df):
    t4 = pd.DataFrame(data=None, index = df.index, columns=[['1Q','2Q','3Q']])
    t4['1Q'] = df.quantile(q=0.25,axis=1)
    t4['2Q'] = df.quantile(q=0.5,axis=1)
    t4['3Q'] = df.quantile(q=0.75,axis=1)
    return t4

def tile3(df):
    t3 = pd.DataFrame(data=None, index = df.index, columns=[['1Q','2Q']])
    t3['1Q'] = df.quantile(q=0.333,axis=1)
    t3['2Q'] = df.quantile(q=0.666,axis=1)
    return t3

def tile_alct(df):
    t_val = tile4(df)
    c_df = df.copy()
    for i in range(c_df.shape[1]):
        for j in range(c_df.shape[0]):
            if df.iloc[j,i] < t_val.iloc[j,0]:
                c_df.iloc[j,i] = 1
            elif df.iloc[j,i] < t_val.iloc[j,1]:
                c_df.iloc[j,i] = 2
            elif df.iloc[j,i] < t_val.iloc[j,2]:
                c_df.iloc[j,i] = 3
            elif df.iloc[j,i] >= t_val.iloc[j,2]:
                c_df.iloc[j,i] = 4
            else:
                pass
    return c_df

def tile_alct_3(df):
    t_val = tile3(df)
    c_df = df.copy()
    for i in range(c_df.shape[1]):
        for j in range(c_df.shape[0]):
            if df.iloc[j,i] < t_val.iloc[j,0]:
                c_df.iloc[j,i] = 1
            elif df.iloc[j,i] < t_val.iloc[j,1]:
                c_df.iloc[j,i] = 2
            elif df.iloc[j,i] >= t_val.iloc[j,1]:
                c_df.iloc[j,i] = 3
            else:
                pass
    return c_df
 
def tile_alct_5(df):
    t_val = tile5(df)
    c_df = df.copy()
    for i in range(c_df.shape[1]):
        for j in range(c_df.shape[0]):
            if df.iloc[j,i] < t_val.iloc[j,0]:
                c_df.iloc[j,i] = 1
            elif df.iloc[j,i] < t_val.iloc[j,1]:
                c_df.iloc[j,i] = 2
            elif df.iloc[j,i] < t_val.iloc[j,2]:
                c_df.iloc[j,i] = 3
            elif df.iloc[j,i] < t_val.iloc[j,3]:
                c_df.iloc[j,i] = 4
            elif df.iloc[j,i] >= t_val.iloc[j,3]:
                c_df.iloc[j,i] = 5
            else:
                pass
    return c_df
 
### 더미 ###
d_pansion = pns > 0
M1_rank_z_tile = tile_alct(M1_rank_z_df) #중간단계
M_tile_chg = (M1_rank_z_tile == 2) & (M1_rank_z_tile.shift(1) == 1)

### y ###
fwd_M12_rank_z_tile = tile_alct_3(fwd_M12_rank_z_df)
fwd_M12_rank_z_tile = fwd_M12_rank_z_df >= 0  ####################################################################

# 그냥 등수를 활용하는 것
fwd_M12_rank_tile = fwd_M12_rank.copy()
fwd_M12_rank_tile = fwd_M12_rank_tile['2005':'2022-09']
fwd_M12_rank_tile = fwd_M12_rank_tile <= 70 ## 등수 값을 직접 사용하는 것




##########################################################################
########################### index 컬럼 만들기 #############################
M_imsi = M12['2005':'2022-09'] 

index_col = pd.DataFrame(index = range(M_imsi.shape[0]*M_imsi.shape[1]),columns = [['date','vsign','comp']])
#index_col = pd.DataFrame(index = range(M12.shape[0]*M12.shape[1]),columns = [['date','comp']])

######################################################################################### 여기에 에러 있음
date_list = M_imsi.index.unique()
comp_list = M_imsi.columns
v_sign = v_sign['2005':'2022-09']
v_list = v_sign.values.tolist()


comp_list2 = np.tile(comp_list, len(date_list))
date_list2 = np.repeat(date_list,len(comp_list))
v_list2 = np.repeat(v_list,len(comp_list))

index_col['date'] = date_list2
index_col['comp'] = comp_list2
index_col['vsign'] = v_list2


##########################################################################
''' M12_rank_z, M2_rank_z, vol_diff_rank_z, d_pansion, M_tile_chg '''

M12_rank_z_df = M12_rank_z_df['2005':'2022-09']
M2_rank_z_df = M2_rank_z_df['2005':'2022-09']
vol_diff_rank_z_df = vol_diff_rank_z_df['2005':'2022-09']
d_pansion = d_pansion['2005':'2022-09']
M_tile_chg = M_tile_chg['2005':'2022-09']

c1 = M12_rank_z_df.copy()
c1 = c1.values.tolist()
c1 = list(itertools.chain.from_iterable(c1))

c2 = M2_rank_z_df.copy()
c2 = c2.values.tolist()
c2 = list(itertools.chain.from_iterable(c2))

c3 = vol_diff_rank_z_df.copy()
c3 = c3.values.tolist()
c3 = list(itertools.chain.from_iterable(c3))

c4 = d_pansion.copy()
c4 = c4.values.tolist()
c4 = list(itertools.chain.from_iterable(c4))
 
c5 = M_tile_chg.copy()
c5 = c5.values.tolist()
c5 = list(itertools.chain.from_iterable(c5))

 
y = fwd_M12_rank_tile.copy()
y = y.values.tolist()
y = list(itertools.chain.from_iterable(y))

 
index_col['M12_rank_z'] = c1
index_col['M2_rank_z'] = c2
index_col['vol_diff_rank_z'] = c3
index_col['d_pansion'] = c4
index_col['M_tile_chg'] = c5
index_col['y'] = y


fin1 = index_col.copy()
fin2 = fin1.dropna()

fin2.columns = ['date','vsign','comp','12m_rt','mom_chg','vol','d_pansion','1to2_mom','y']
fin2 = fin2.replace(True,1); fin2 = fin2.replace(False,0)
##########################################################################
##########################################################################
############################# Logistic Model #############################
##########################################################################
##########################################################################

real = pd.read_csv("T:\\index\\95_곽용하\\연구\\logit\\real_test_data_150.csv", index_col = 0)
real_ = real.dropna() ## 사용할 데이터


md1 = fin2.copy() ################################################################################################
md2 = fin2.copy() ################################################################################################

md1 = md1[(md1['vsign']==4) | (md1['vsign']==3)]
md2 = md2[(md2['vsign']==4) | (md2['vsign']==3)]

md1 = md1[md1['vsign']==3]
md2 = md2[md2['vsign']==3]

## 데이터 정리
#model_1
md1['win'] = md1['y'] == 1 ##3
md1 = md1.replace(False, 0)
md1 = md1.replace(True, 1)

df_mal = md1.copy()
mal_X = df_mal.iloc[:,3:8] 
mal_Y = df_mal["win"]

model_1 = LogisticRegression()
model_1.fit(X = mal_X, y = mal_Y)


########################################################################
#회귀계수 출력
print('logistic coef:\n', model_1.coef_)
print('\nlogistic intercept:\n', model_1.intercept_)

### 3. model 평가
#예측치
y_pred = model_1.predict(X = mal_X)

#분류 정확도
acc = metrics.accuracy_score(mal_Y, y_pred)
print('accuracy=', acc)

#confusion matrix
con_mat = metrics.confusion_matrix(y_true = mal_Y, y_pred = y_pred)
con_mat

acc1 = (con_mat[0,0]+con_mat[1,1]+con_mat[2,2])/len(mal_Y)
print('accuracy=',acc1)


### 4. 시각화 (정확도)
plt.figure(figsize=(6,6))
sn.heatmap(con_mat, annot=True, fmt=".3f", linewidths=.5, square=True)

########################################################################
########################################################################



#model_2
md2['2'] = md2['y'] == 3
md2['0'] = md2['y'] == 1

md2['2'] = md2['2'].replace(False, 0)
md2['2'] = md2['2'].replace(True, 2)

md2['0'] = md2['0'].replace(True, 1)
md2['0'] = md2['0'].replace(False, 0)

md2['tile'] = md2['2']  + md2['0']
md2['tile'] =md2['tile'].replace(1,3) #3을 0으로 바꿔야 함
md2['tile'] =md2['tile'].replace(0,-1) #-1을 1로 바꿔야 함

md2['tile'] =md2['tile'].replace(3,0) #3을 0으로 바꿔야 함
md2['tile'] =md2['tile'].replace(-1,1) #-1을 1로 바꿔야 함




md2['tile'] = md2['y']
md2['tile'] = md2['tile'].replace(1,0)
md2['tile'] = md2['tile'].replace(2,1)
md2['tile'] = md2['tile'].replace(3,2)
md2['tile'] = md2['tile'].replace(4,3)
md2['tile'] = md2['tile'].replace(5,4)





#model_2 learning
df_mal = md2.copy()
mal_X = df_mal.iloc[:,3:8] 
mal_Y = df_mal["tile"]
model_2 = LogisticRegression(random_state=0, solver='lbfgs', multi_class='multinomial')
model_2.fit(X = mal_X, y = mal_Y) #학습수행

########################################################################
#회귀계수 출력
print('logistic coef:\n', model_2.coef_)
print('\nlogistic intercept:\n', model_2.intercept_)

### 3. model 평가
#예측치
y_pred = model_2.predict(X = mal_X)

#분류 정확도
acc = metrics.accuracy_score(mal_Y, y_pred)
print('accuracy=', acc)

#confusion matrix
con_mat = metrics.confusion_matrix(y_true = mal_Y, y_pred = y_pred)
con_mat

acc1 = (con_mat[0,0]+con_mat[1,1]+con_mat[2,2])/len(mal_Y)
print('accuracy=',acc1)


### 4. 시각화 (정확도)
plt.figure(figsize=(6,6))
sn.heatmap(con_mat, annot=True, fmt=".3f", linewidths=.5, square=True)

########################################################################


#model_1 prediction
######## 실제 데이타
aaa = model_1.predict_proba(real_)

#model_2 prediction
bbb = model_2.predict_proba(real_)
bbb1 = model_2.predict(real_)











