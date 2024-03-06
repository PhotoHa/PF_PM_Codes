# -*- coding: utf-8 -*-
"""
Created on Wed Feb 28 08:43:16 2024

"""

import pandas as pd
import numpy as np
import pyodbc
import openpyxl as op
import os
os.chdir('')
import schedule
import time

pd.set_option("display.max_columns", 50) #최대 50개의 컬럼까지 줄바꿈으로 보여줌

# 0. 데이터베이스 연결
conn_pcor = pyodbc.connect('driver={Oracle in OraClient12Home1};')
conn_quant = pyodbc.connect('driver={SQL Server};')
conn_wisefn = pyodbc.connect('driver={SQL Server};')



RBDT = ['20230630','20220630','20210630','20200630','20190628'] #'20240131'는 2023AS가 안 나와서 bs_date를 바꿔야 함 


# 1. 초기 세팅
# 수기 세팅
rebalancing_dt = '20200630'  #
beta_up_lmt = 2
beta_ud_lmt = 0.5
roe_set = 0.08
up_priority = 25
sub_target_number = 15 #75
pbr_set = 1


sql_q_date =  f'''  select top 2 ymd
                    from DATE
                    where 1=1
                    and ymd <= {rebalancing_dt}
                    and MNO_OF_YR = 6
                    and MN_END_YN = 1
                    and TRADE_YN = 1
                    order by dt desc   '''

q_date_list = pd.read_sql(sql_q_date, conn_wisefn)



# FY-0
sql_tgt_date_1 =  f'''  select top 1 ymd
                        from DATE
                        where 1=1
                        and ymd < {rebalancing_dt}
                        and MNO_OF_YR = 12
                        and MN_END_YN = 1
                        and TRADE_YN = 1
                        order by dt desc  '''
tgt_date = pd.read_sql(sql_tgt_date_1, conn_wisefn)
tgt_date = tgt_date.iloc[0,0] #'20231228'

# FY-1
sql_tgt_date_2 =  f'''  select top 1 ymd
                        from DATE
                        where 1=1
                        and ymd < {tgt_date}
                        and MNO_OF_YR = 12
                        and MN_END_YN = 1
                        and TRADE_YN = 1
                        order by dt desc  '''
tgt_date_1 = pd.read_sql(sql_tgt_date_2, conn_wisefn)
tgt_date_1 = tgt_date_1.iloc[0,0] #'20221229'

# FY-2 : 쓸 일 없음
sql_tgt_date_3 =  f'''  select top 1 ymd
                        from DATE
                        where 1=1
                        and ymd < {tgt_date_1}
                        and MNO_OF_YR = 12
                        and MN_END_YN = 1
                        and TRADE_YN = 1
                        order by dt desc  '''
tgt_date_2 = pd.read_sql(sql_tgt_date_3, conn_wisefn)
tgt_date_2 = tgt_date_2.iloc[0,0] #'20211230'

#set_date   = 20231231
# 자동 세팅
end_of_year = tgt_date
bs_date = [tgt_date, tgt_date_1]


# 2. 유니버스      
sql_universe = f'''
                with AvgCalc as (
                    SELECT STK_CD, TRD_DT, 
                    [5일평균거래대금] = AVG(TRD_AMT) OVER (PARTITION BY STK_CD ORDER BY STK_CD, TRD_DT ROWS BETWEEN 4 PRECEDING AND CURRENT ROW),
                    [20일평균거래대금] = AVG(TRD_AMT) OVER (PARTITION BY STK_CD ORDER BY STK_CD, TRD_DT ROWS BETWEEN 19 PRECEDING AND CURRENT ROW)
                    FROM DAILY
                    where trd_dt <= '{tgt_date}'
                    and trd_dt >= Dateadd(day, -19, '{tgt_date}')
                    )
        
               SELECT A.TRD_DT 기준일, A.STK_CD 종목코드, A.STK_NM_KOR 종목이름, A.MKT_TYP 시장구분, A.KS200_TYP K200WLTN,
               A.MV_SIZE_TYP 시총규모, B.STK_TYP 유형, A.CAUTION_TYP 투자유의, A.ADMIN_YN 관리, A.TRD_STOP_TYP 거래정지,
               B.CMP_CD 기업코드, D.CLOSE_PRC 종가, D.MKT_VAL 시가총액,
               D.TRD_AMT 거래대금, E.SECTOR_NM 섹터이름, E2.SECTOR_NM 산업이름, F.[5일평균거래대금],F.[20일평균거래대금] 
        
               FROM ISSUE A
               LEFT OUTER JOIN STOCK B
                   ON A.STK_CD = B.STK_CD
        
               LEFT OUTER JOIN COMPANY C
                   ON B.CMP_CD = C.CMP_CD
        
               LEFT OUTER JOIN DAILY D
                   ON A.STK_CD = D.STK_CD
                   AND A.TRD_DT = D.TRD_DT
        
               LEFT OUTER JOIN (
                   SELECT SEC_CD SECTOR_CD, SEC_N_KOR SECTOR_NM
                   FROM SECTOR
                   WHERE SEC_TYP = \'G\'
                       AND LEN(SEC_CD) = 3
                   ) E
                   ON LEFT(C.GICS_CD,3) = E.SECTOR_CD
        
               LEFT OUTER JOIN (
                   SELECT SEC_CD SECTOR_CD, SEC_N_KOR SECTOR_NM
                   FROM SECTOR
                   WHERE SEC_TYP = \'G\'
                       AND LEN(SEC_CD) = 5
                   ) E2
                   ON LEFT(C.GICS_CD,5) = E2.SECTOR_CD
                
                LEFT OUTER JOIN (
                        select
                        stk_cd,
                        '{tgt_date}' as trd_dt,
                        [5일평균거래대금],
                        [20일평균거래대금]
                        from AvgCalc
                        where 
                        trd_dt = '{tgt_date}'
                ) F
                   ON A.STK_CD = F.STK_CD
                   AND A.TRD_DT = F.TRD_DT
        
               WHERE A.TRD_DT = '{tgt_date}'
                   AND A.LIST_TYP != 7
                   AND B.ISSUE_TYP IN (1,5,6)
               '''

universe_0 = pd.read_sql(sql_universe, conn_wisefn)

# 시총 순위 300위 이내
universe_1 = universe_0.copy()

# 코스피 보통주
universe_1 = universe_1[universe_1['시장구분'] == 1]
universe_1 = universe_1[universe_1['시총규모'] != 0]
universe_1 = universe_1[universe_1['유형']==1]
universe_11 = universe_1.copy() # 코스피 전체 저장용
# 관리
universe_1 = universe_1[universe_1['투자유의'] == 0]
universe_1 = universe_1[universe_1['관리']==0]
universe_1 = universe_1[universe_1['거래정지'] == 0]


universe_1 = universe_1.sort_values('시가총액', ascending=False)
universe_1 = universe_1.head(300)


universe_1.insert(1, 'Rebal_dt', rebalancing_dt)



# 3. 재무제표 데이터 및 마켓 데이터 (2018년 이후)
path = ""

beta_data = pd.read_excel(path + "kor_ver_data.xlsx", skiprows = list(range(14)), header = None, sheet_name = 'Beta')
beta_data.columns = ['STK_CD','종목이름','결산','20181228','20191230','20201230','20211230','20221229','20231228']
# 3-1) beta 조건 설정
for col in beta_data.columns[3:]:
    beta_data.loc[beta_data[col] > beta_up_lmt, col] = beta_up_lmt
for col in beta_data.columns[3:]:
    beta_data.loc[beta_data[col] < beta_ud_lmt, col] = beta_ud_lmt

beta_data['종목코드'] = [ x[1:] for x in beta_data['STK_CD'] ]


# 단위: %
roe_data = pd.read_excel(path + "kor_ver_data.xlsx", skiprows = list(range(14)), header = None, sheet_name = 'ROE')
roe_data.columns = ['STK_CD','종목이름','결산','20181228','20191230','20201230','20211230','20221229'] #,'20231231'
roe_data['종목코드'] = [ x[1:] for x in beta_data['STK_CD'] ]

mkt_data = pd.read_excel(path + "kor_ver_data.xlsx", header = 0, sheet_name = 'macro_')
mkt_data['TRD_DT'] = mkt_data['TRD_DT'].astype(str)



# 4. 필요 데이터 (2023년)
in_need = ['종목코드', end_of_year]
in_need_1 = ['종목코드', bs_date[0]]

# 1) Equity Spread
# 1-1) beta 당기
beta_in = beta_data[in_need]
beta_in['mkt-0']   = mkt_data[mkt_data['TRD_DT']==end_of_year]['kospi'].iloc[0]
beta_in['R_mkt-0'] = mkt_data[mkt_data['TRD_DT']==end_of_year]['10y_bond'].iloc[0]
beta_in['cost_0'] = beta_in['R_mkt-0'] + (beta_in['mkt-0'] - beta_in['R_mkt-0'])*beta_in[end_of_year]
beta_in.columns = ['종목코드','beta-0','mkt-0','R_mkt-0','cost_0']

# 1-2) beta 전기
beta_in_1 = beta_data[in_need_1]
beta_in_1['mkt-1']   = mkt_data[mkt_data['TRD_DT']==bs_date[0]]['kospi'].iloc[0]
beta_in_1['R_mkt-1'] = mkt_data[mkt_data['TRD_DT']==bs_date[0]]['10y_bond'].iloc[0]
beta_in_1['cost_1'] = beta_in_1['R_mkt-1'] + (beta_in_1['mkt-1'] - beta_in_1['R_mkt-1'])*beta_in_1[bs_date[0]]
beta_in_1.columns = ['종목코드','beta-1','mkt-1','R_mkt-1','cost_1']

# 1-3) beta 당기, 전기 정리
beta_in_2 = pd.merge(beta_in, beta_in_1, 'inner', '종목코드' ) #.iloc[:,[0,-1]]

# 1-4) roe 당기, 전기
bs_date.append('종목코드')
roe_in = roe_data[bs_date]
roe_in.iloc[:,:2] = roe_in.iloc[:,:2] /100
roe_in.columns = ['ROE_FY-0','ROE_FY-1','종목코드']


es_0 = pd.merge(beta_in_2, roe_in, 'inner', '종목코드')
es_0['ES-0'] = es_0['ROE_FY-0'] - es_0['cost_0']
es_0['ES-1'] = es_0['ROE_FY-1'] - es_0['cost_1']



# 2) PBR (1/BTM)
# 2-1) 당기
sql_date = q_date_list.iloc[0,0]
sql_date = sql_date[0:4] + '-' + sql_date[4:6] + '-' + sql_date[6:8]
sql_pbr =  f''' SELECT CONVERT(VARCHAR(8), ScoreDate, 112) AS '기준일', Code AS '종목코드', Ratio AS PBR_0
                FROM Dat_
                WHERE 1=1
                AND FactorCode = 'BTM_QW'
                AND ScoreDate =  '{sql_date}' '''

pbr_data = pd.read_sql(sql_pbr, conn_quant)
pbr_data['PBR_0'] = 1 / pbr_data['PBR_0']
pbr_data['종목코드'] = [x[1:] for x in pbr_data['종목코드']]
# 2-2) 전기
sql_date_1 = q_date_list.iloc[1,0]
sql_date_1 = sql_date_1[0:4] + '-' + sql_date_1[4:6] + '-' + sql_date_1[6:8]
sql_pbr_1 =  f''' SELECT CONVERT(VARCHAR(8), ScoreDate, 112) AS '기준일', Code AS '종목코드', Ratio AS PBR_1
                FROM Dat_
                WHERE 1=1
                AND FactorCode = 'BTM_QW'
                AND ScoreDate =  '{sql_date_1}' '''

pbr_data_1 = pd.read_sql(sql_pbr_1, conn_quant)
pbr_data_1['PBR_1'] = 1 / pbr_data_1['PBR_1']
pbr_data_1['종목코드'] = [x[1:] for x in pbr_data_1['종목코드']]
# 2-3) 당기,전기
pbr_data_2 = pd.merge(pbr_data.iloc[:,1:], pbr_data_1.iloc[:,1:], 'left', '종목코드')


# 5. 데이터 합치기
imsi = pd.merge(es_0, pbr_data_2, 'left', '종목코드') # inner로 하는게 맞나??
df0 = pd.merge(universe_1, imsi, 'left', '종목코드')


# 6. ROE/ES 기준
df_1 = df0.copy()
df_1 = df_1[df_1['ROE_FY-0'] > roe_set ]
df_1 = df_1[df_1['ES-0'] > 0 ]
df_1 = df_1[df_1['ES-1'] > 0 ]

# 6-1) 75개
df_1_1 = df_1.copy()
df_1_1 = df_1_1.sort_values(['ES-0','ES-1'],ascending=False)
df_roe_es = df_1_1.head(sub_target_number)
df_roe_es['기준'] = 'ROE_ES'
roe_es_stk_list = list(df_roe_es['종목코드'])


# 7. PBR 기준
df_2 = df0[~df0['종목코드'].isin(roe_es_stk_list)]
df_2_1 = df_2.copy()
df_2_1 = df_2_1[ ( df_2_1['PBR_0'] > -100 ) & ( df_2_1['PBR_1'] > -100 ) ]
df_2_1 = df_2_1[ ( df_2_1['PBR_0'] > pbr_set ) & ( (df_2_1['PBR_0']+df_2_1['PBR_1'])/2 > pbr_set) ]

df_2_2 = df_2_1.sort_values(['시가총액','PBR_1'],ascending=False)
df_pbr = df_2_2.head(sub_target_number)
df_pbr['기준'] = 'PBR'



# 8. FINAL PORTFOLIO
df_3 = pd.concat([df_roe_es, df_pbr])
df_3 = df_3.sort_values(['시가총액','ROE_FY-0','PBR_0'],ascending=False)

# 9. excel 저장
df0.to_excel(path+ f'\\universe\\{rebalancing_dt}_basket_{len(df_3)}_.xlsx', encoding='utf-8-sig')
df_3.to_excel(path+ f'\\basket_150\\{rebalancing_dt}_basket_{len(df_3)}_pf.xlsx', encoding='utf-8-sig')




print(f'리밸런싱 일자: {rebalancing_dt}')
print(f'ROE FY-0 일자: {tgt_date}')
print(f'ROE FY-1 일자: {tgt_date_1}')
print(f'PBR FY-0 일자: {sql_date}')
print(f'PBR FY-1 일자: {sql_date_1}')









