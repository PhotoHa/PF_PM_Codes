# -*- coding: utf-8 -*-
"""
Created on Wed Sep 13 13:41:15 2023

@author: 11149
"""


import numpy as np
import pandas as pd
import os
os.chdir("T:\\index\\95_곽용하\\코드\\")
import functions_1 as mf1
import pickle
import datetime as dt

#########################################################################
############################# 초기  setting #############################
#########################################################################
tile_n = 10
tgt_siz = 100000 #(100 kb 이상)
nn = 20 #종목 수 20개 미만인 경우 수익률 0으로 바꿀거임


#########################################################################
############################### daily prc ###############################
#########################################################################
path_prc = 'T:\\index\\88_권구황\\Python_server\\tdpm_server\\data_pkl\\tmp_daily\\'
with open(path_prc +'stock_price_close.pkl', 'rb') as f:
    stk_prc = pickle.load(f)
stk_prc.index = pd.to_datetime(stk_prc.index) # datetime



#########################################################################
############################## Factor Loop ##############################
#########################################################################
## factor data
path = "T:\\index\88_권구황\\Python_server\\tdpm_server\\data_pkl\\tmp_monthly_dix_factor_size_industry\\"
file_name = os.listdir(path)

## file list
file_list = []
for name in file_name:
    name = path + name
    file_list = np.append(file_list, name)



df_imsi_file = pd.DataFrame({'name':file_name, 'path':file_list})

## file size filtering
def size_is_bigger(x, siz = tgt_siz):
    return os.path.getsize(x) > siz

filtered_files = list(filter(size_is_bigger,file_list))
df_imsi_file = df_imsi_file[df_imsi_file['path'].isin(filtered_files)]

file_list = np.array(df_imsi_file['path']) #경로포함
file_name = np.array(df_imsi_file['name'])#이름만



################################################################################### loop 시작점 #########
for i in range(len(file_list)):
    
    run_file_name = file_name[i]
    imsi = mf1.load_pkl(file_list[i], drop=False)
    data = imsi.reset_index(drop=True)
    factor_data = data.copy() #데이터 필터링 용도
    
    ##########################################################################
    ########################### fwd Rt Calculation ###########################
    ##########################################################################
   
    # 기간 중 종목 수 20개 미만인 경우 확인
    rt_zero = data.groupby('date').count()
    rt_zero[rt_zero['size_f'] < nn] = 0
    rt_zero[rt_zero['size_f'] >= nn] = 1
    
    rt_zero.index = pd.to_datetime(rt_zero.index)
    
    rt_zero['imsi_date'] = rt_zero.index
    rt_zero['M'] = rt_zero['imsi_date'].dt.month
    rt_zero['Y'] = rt_zero['imsi_date'].dt.year
    rt_zero.reset_index(drop=True,inplace=True)
    
    # factor data에 있는 종목들로 필터링
    prc = stk_prc[factor_data['code'].unique()]
    fwd_rt = prc.pct_change(1).shift(-1)
    fwd_rt['imsi_date'] = fwd_rt.index.to_series().dt.to_period('M').dt.to_timestamp() - pd.DateOffset(days=1) 
    fwd_rt['M'] = fwd_rt['imsi_date'].dt.month
    fwd_rt['Y'] = fwd_rt['imsi_date'].dt.year
        
    
    # 기간 중 종목 수 20개 미만인 경우 수익률 0으로 변경
    fwd_rt = pd.merge(fwd_rt, rt_zero[['code','M','Y']],'left',['M','Y'])
    fwd_rt.set_index(prc.index,inplace=True)
    
    fwd_rt['code'] = fwd_rt['code'].fillna(0)
    #fwd_rt.set_index('imsi_date',drop=True,inplace=True)
    for stk in fwd_rt.columns[:-4]:
        fwd_rt[stk] = fwd_rt[stk]*fwd_rt['code']
    
    
    # Stack
    fwd_rt = fwd_rt.iloc[:,:-4].stack().reset_index()
    fwd_rt.columns = ['raw_date','code','fwd_rt']
    fwd_rt['imsi_date'] = fwd_rt['raw_date'].dt.to_period('M').dt.to_timestamp() - pd.DateOffset(days=1) 
    fwd_rt['M'] = fwd_rt['imsi_date'].dt.month
    fwd_rt['Y'] = fwd_rt['imsi_date'].dt.year
    
    
    
    if abs(fwd_rt['fwd_rt']).sum() == 0: ##종목수는 20개가 되지만 수익률이나 팩터 값이 없는 경우들
        pass
    else:
        
        ##########################################################################
        ############################ Tile Calculation ############################
        ##########################################################################
        # 분위 수 부여
        tile_df = data.copy()
        tile_df.date = pd.to_datetime(tile_df.date)
        
        grouped = tile_df.groupby('date')
        for col in tile_df.columns[2:]:
            tile_df[col] = grouped[col].transform(mf1.custom_qcut, tile = tile_n)
        tile_df.iloc[:,2:] += 1
        tile_df['M'] = tile_df['date'].dt.month
        tile_df['Y'] = tile_df['date'].dt.year
        
        
        rt_and_tile = pd.merge(fwd_rt,tile_df,'left',['code','Y','M']).dropna()
        
        
        ############
        ### 성과 ###
        ############
        list_col = tile_df.columns[2:-2]
        list_col = np.array(list_col)
        list_dt = rt_and_tile['raw_date'].unique()
        
        d0 = pd.DataFrame()
        for factor in list_col:
            colmn = np.append(['fwd_rt','raw_date'], factor)
            a = rt_and_tile[colmn]
            #print(a)
            g_a = a.groupby([colmn[1],colmn[2]]).mean()
            g_a = g_a.reset_index()
            g_a.columns = ['raw_date','tile',colmn[2]]
            
            if list_col[0] == factor:
                d0 = g_a
            else:
                d0 = pd.merge(d0, g_a,'outer',['raw_date','tile'])
        
        
        ## factor별로 테이블 만들기
        ## tile list (Q_1 ~ Q_10)
        tt = []
        for i in range(tile_n):
            t = f'{run_file_name}_Q_{i+1}'
            tt = np.append(tt,t)
            
        Series_factor_time = pd.Series([0]*len(list_col))
        for j in range(len(list_col)):
            factor1 = list_col[j]
            a = d0[np.append(['raw_date','tile'],factor1)]
            a.set_index(['raw_date','tile'],inplace=True)
            a = a.unstack()
            
            aaa = [item + factor1 for item in tt]
            a.columns = aaa
            
            a = mf1.compound_idx(a)
            Series_factor_time[j] = a
        
        ##########################################################################
        ## size_industry_simulation 저장!!
        ##########################################################################
        for k in range(len(list_col)):
            dtdt = Series_factor_time[k]
            col_name = list_col[k]
            
            path_save = "T:\\index\\999_quant\\__data_pkl\\kyh\\size_industry_simulation\\"
            with open(path_save + f'{run_file_name}_{col_name}.pkl', 'wb') as f:
                pickle.dump(dtdt,f)
        
        ##########################################################################
        ## tmp_factor_fractile 저장!!
        ##########################################################################
        tile_df = tile_df.iloc[:,:-2]
        tile_df['universe'] = run_file_name
        
        path_save_mid = "T:\\index\\999_quant\\__data_pkl\\kyh\\tmp_factor_fractile\\"
        with open(path_save_mid + f'{run_file_name}.pkl', 'wb') as f:
            pickle.dump(tile_df,f)
            
        print(f'Done_{run_file_name}')


