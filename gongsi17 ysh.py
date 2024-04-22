# -*- coding: utf-8 -*-
"""
Created on Wed Mar 20 14:56:15 2024
https://opendart.fss.or.kr/intro/main.do
https://opendart.fss.or.kr/guide/detail.do?apiGrpCd=DS001&apiId=2019001
@author: 10684
"""
import requests as rq
from io import BytesIO
import zipfile
import xmltodict
import json
import pandas as pd
from datetime import date
from dateutil.relativedelta import relativedelta
import re
import dart_fss as dart
from dart_fss import get_corp_list
import math
from pandas.tseries.offsets import BDay, Day
import pyodbc
import numpy as np
import urllib
from sqlalchemy import create_engine
import os
os.chdir('T:\\index\\92_유서훈\\03 Coding\\파이썬\\모듈') #functions_0의 위치
import functions_0 as mf



days_back = -1 #과거 n일 설정, 음의 정수로 설정

api_key = '26bfe94b268c8ae8dbe76bfabd20d4548da2e65e'
dart.set_api_key(api_key = api_key)

##### 최근n일동안의 공시 불러오기 위한 날짜 설정
bgn_date = (date.today() + relativedelta(days = days_back)).strftime("%Y%m%d")#최근n일동안
end_date = (date.today()).strftime("%Y%m%d")


##### 반복횟수 찾기 위한 페이지수 설정을 위해 공시개수 찾기
reports01 = dart.filings.search(corp_cls='Y',bgn_de='20240405',end_de='20240405',page_no=1, page_count=10000 )
df_reports01 = pd.json_normalize(reports01.to_dict()['report_list'])#데이터프레임으로 전환
df_reports011 = pd.json_normalize(reports01.to_dict())#데이터프레임으로 전환
#page_count:페이지당 건수(1~100) 기본값 : 10, default : 100
df_reports012 = pd.DataFrame(pd.json_normalize(reports01.to_dict())['total_count'])

reports02 = dart.filings.search(corp_cls='K',bgn_de='20240405',end_de='20240405',page_no=1, page_count=10000 )
df_reports02 = pd.json_normalize(reports02.to_dict()['report_list'])#데이터프레임으로 전환
df_reports021 = pd.json_normalize(reports02.to_dict())#데이터프레임으로 전환
#page_count:페이지당 건수(1~100) 기본값 : 10, default : 100
df_reports022 = pd.DataFrame(pd.json_normalize(reports02.to_dict())['total_count'])

kospi_c = df_reports012.iloc[0,0] 
kosdaq_c = df_reports022.iloc[0,0]
t_c = kospi_c + kosdaq_c

##### 최대 페이지수 설정
max_pages = math.ceil(t_c / 50) 


##### 종목코드 받기
codezip_url = f'''http://opendart.fss.or.kr/api/corpCode.xml?crtfc_key={api_key}'''
codezip_data = rq.get(codezip_url)
#codezip_data.headers #data확인용 코드
#codezip_data.headers['Content-Disposition'] #data확인용 코드
codezip_file = zipfile.ZipFile(BytesIO(codezip_data.content))
#codezip_file.namelist() #data확인용 코드
code_data = codezip_file.read('CORPCODE.xml').decode('utf-8')


##### 받아온 xml 형태의 code_data를 dictionary 형태로 변경후 json으로 변경후 dataframe으로 변경
data_odict = xmltodict.parse(code_data) #dictionary 형태로 변경
data_dict = json.loads(json.dumps(data_odict)) #josn으로 변경
data = data_dict['result']['list'] #get 함수를 통해 result내에서 list 부분만 불러옴
corp_list = pd.DataFrame(data)#dataframe으로 변경


##### stock_code열이 'None'이 아닌 데이터만 선택후 인덱스 초기화
corp_list_l = corp_list[~corp_list['stock_code'].isna()].reset_index(drop=True)


##### dataframe으로 초기화
notice_data_dft = pd.DataFrame() 


##### 공시불러오기
for i in range(1, max_pages + 1):
#for i in range(1, 200):
    for corp_cl in ['Y','K']:
    #for corp_cl in ['Y']:
        notice_url = f'''https://opendart.fss.or.kr/api/list.json?crtfc_key={api_key}
        &bgn_de={bgn_date}&end_de={end_date}&corp_cls={corp_cl}&page_no={i}&page_count=100'''
        #corp_cls 시장구분 : Y(유가), K(코스닥), N(코넥스), E(기타)        
        notice_data = rq.get(notice_url.strip()) #url에서 줄바뀜제거, 주소수정
        if notice_data.status_code == 200:
            notice_data_df = notice_data.json().get('list')#list부분만 뽑아냄
            if notice_data: #data가 있으면
                notice_data_dft = pd.concat([notice_data_dft, pd.DataFrame(notice_data_df)], ignore_index=True)#data프레임에 추가    

notice_data_dft_0 = notice_data_dft


##### 중복된 행 삭제
notice_data_dft_0.drop_duplicates(inplace = True)

            
##### 최근일자 공시 dataframe에 url 주소 리스트를 열 추가
notice_urls = [] #초기화

for rcp_no in notice_data_dft_0.index: #dataframe의 실제 길이만큼 반복
    notice_url_exam = notice_data_dft_0.loc[rcp_no,'rcept_no']#dataframe의 첫번째 공시번호
    #특정공시번호에 따른 url 생성
    notice_dart_url = f'http://dart.fss.or.kr/dsaf001/main.do?rcpNo={notice_url_exam}'
    notice_urls.append(f'=hyperlink("{notice_dart_url}")')

notice_data_dft_0['notice_url'] = notice_urls # 열추가


##### 특정이벤트체크 함수
def check_event(report_name):
    if '유상증자' in report_name: return '유상증자'
    elif '무상증자' in report_name: return '무상증자'
    elif '감자' in report_name: return '감자'
    elif '공개매수' in report_name: return '공개매수'
    elif '액면분할' in report_name: return '액면분할'
    elif '분할' in report_name: return '분할'
    elif '합병' in report_name: return '합병'
    elif '병합' in report_name: return '병합'
    elif '거절' in report_name: return '거절'
    elif '비적정' in report_name: return '비적정'
    elif '지연' in report_name: return '지연'
    elif '불성실' in report_name: return '불성실'
    elif '상장적격성' in report_name: return '상정적격성'
    elif '상장폐지' in report_name: return '상장폐지'
    elif '관리종목' in report_name: return '관리종목'
    elif '거래정지' in report_name: return '거래정지'
    elif '횡령' in report_name: return '횡령'
    elif '배임' in report_name: return '배임'
    elif '영업양수도' in report_name: return '영업양수도'
    elif '영업정지' in report_name: return '영업정지'
    elif '주식소각' in report_name: return '주식소각'
    elif '회생' in report_name: return '회생'
    elif '생산중단' in report_name: return '생산중단'
    elif '조회공시요구' in report_name: return '조회공시요구'
    else: return '' #해당되지 않는 경우 빈 문자열 반환


##### 특정이벤트인지 체크되는 열 추가
notice_data_dft_0['event'] = notice_data_dft_0['report_nm'].apply(check_event)


##### 데이터베이스 연결
conn_pcor = pyodbc.connect('driver={Oracle in OraClient12home1};dbq=PCOR;uid=EF0SEL;pwd=EF0SEL#076')
conn_quant = pyodbc.connect('driver={SQL Server};server=46.2.90.172;database=quant;uid=index;pwd=samsung@00')
conn_wisefn = pyodbc.connect('driver={SQL Server};server=46.2.90.172;database=wisefn;uid=index;pwd=samsung@00')

if conn_pcor:
    print("Server PCOR OK")
    
if conn_quant:
    print("Server QUANT OK")
    

##### 전일자 코스피 200 지수비중
sql_index1 = '''
SELECT FILE_DATE, INDEX_ISIN, INDEX_NAME_KR, CONSTITUENT_ISIN, CONSTITUENT_CODE, CONSTITUENT_Name_KR,  
LISTED_SHARES, PRICE, MARKET_CAP, FREE_FLOAT_FACTOR, INDEX_MARKET_CAP, INDEX_WEIGHT, FILE_GB
FROM COROWN.TB_FXN_IDX_CONS_ITEM_INF_01
WHERE INDEX_NAME_KR ='코스피 200' -- 코스닥 150 // 코스피 200
AND FILE_DATE = '20240404' 
AND FILE_GB = 'NXT' 
order by INDEX_WEIGHT desc
'''.format()
df_index1 = pd.read_sql(sql_index1,conn_pcor)
print(df_index1)


##### 전일자 코스닥 150 지수비중
sql_index2 = '''
SELECT FILE_DATE, INDEX_ISIN, INDEX_NAME_KR, CONSTITUENT_ISIN, CONSTITUENT_CODE, CONSTITUENT_Name_KR,  
LISTED_SHARES, PRICE, MARKET_CAP, FREE_FLOAT_FACTOR, INDEX_MARKET_CAP, INDEX_WEIGHT, FILE_GB
FROM COROWN.TB_FXN_IDX_CONS_ITEM_INF_01
WHERE INDEX_NAME_KR ='코스닥 150'  -- 코스닥 150 // 코스피 200
AND FILE_DATE = '20240404' 
AND FILE_GB = 'NXT' 
order by INDEX_WEIGHT desc
'''.format()
df_index2 = pd.read_sql(sql_index2,conn_pcor)
print(df_index2)


##### 코스피 200지수, 코스닥150 지수 구성종목 df를 합쳐서 하나의 df로
df_index3 = pd.concat([df_index1, df_index2], ignore_index=True)


##### 칼럼명 변경
df_index3 = df_index3.rename(columns={
    'CONSTITUENT_CODE':'stock_code'
})

#종목코드에서 A삭제
df_index3['stock_code'] = df_index3['stock_code'].str.replace('A', '', regex=False)


#레프트 아우터 조인
notice_data_dft_0_1 = pd.merge(notice_data_dft_0, df_index3, on = 'stock_code', how = 'left')


##### 거래소 시장이름 변경 
notice_data_dft_0_1['corp_cls'] = notice_data_dft_0_1['corp_cls'].replace('Y', '유가증권')
notice_data_dft_0_1['corp_cls'] = notice_data_dft_0_1['corp_cls'].replace('K', '코스닥')


##### 종목코드에 A추가 
notice_data_dft_0_1['stock_code'] = 'A' + notice_data_dft_0_1['stock_code'].astype(str) #astype는 문자열이 아닌경우대비


##### 열삭제
notice_data_dft_1 = notice_data_dft_0_1.drop(['flr_nm','rm'], axis=1)


##### 열이름 변경
notice_data_dft_2 = notice_data_dft_1.rename(columns={
    'corp_code':'Dart종목코드',
    'stock_code':'종목코드',
    'corp_name':'종목명',
    'rcept_no':'공시번호',
    'rcept_dt':'공시일',
    'report_nm':'공시제목',
    'corp_cls':'거래소',
    'notice_url':'url',
    'INDEX_NAME_KR':'지수',
    'INDEX_WEIGHT':'지수비중'
})


##### 선택된 열만 출력
notice_data_dft_3 = notice_data_dft_2.reindex([
    '공시일',
    '종목코드',
    '종목명',
    '거래소',
    '공시제목',
    '공시번호',
    'url',
    'event',
    '지수',
    '지수비중'
    ], 
    axis = 1
)#열순서바꾸기
print(notice_data_dft_3)


##### nan값을 공백으로 변경
notice_data_dft_3.fillna('', inplace=True)


##### event에 해당되지 않는 행 삭제
notice_data_dft_4 = notice_data_dft_3[notice_data_dft_3['event'] != ''].reset_index(drop = True)


##### 두개의 공백이 문자열에 있을때 하나의 공백으로 대체하는 함수
def remove_multiple_spaces(str):        
    return re.sub(' +', ' ',str)


##### 많은 공백 제거
notice_data_dft_4['공시제목'] = notice_data_dft_4['공시제목'].apply(remove_multiple_spaces)


##### 펀드별 보유종목 불러오기
fund_list = ('790001','704007','700187','700182','700178','700045','500001','450018',
                            '450017','441003','441002','430003','430002','430001','420012','420011',
                            '420010','420005','420004','420003','420002','420001','411013','411012',
                            '403102','403101','402716','402616','402615','402609','402415','402315',
                            '401319','2T4101','2T0201','2P0400','2MF180','2MF170','2MF150','2MF123',
                            '2MF111','2MF100','290001','234701','234601','234202','218483','218478',
                           '218451','203007','200806','210555','500002','287001','700187','2P1801',
                           '219939','704004','423001','423003','422010','494004','494005','790003',
                           '420013','710001','129950')
holding = mf.get_fund_hold(fund_list,conn_pcor)

##### 해당종목 보유하고 있는 펀드리스트 열 추가
stk_cd = notice_data_dft_4['종목코드'].unique()
holding1 = holding[holding['GICODE'].isin(stk_cd)]
holding1 = holding1[['F_ID','GICODE','SEC_NM']]
notice_data_dft_4['stks'] = notice_data_dft_4['종목코드'] ##임시용

for i in range(len(notice_data_dft_4)):
    stk = notice_data_dft_4['종목코드'].iloc[i]
    filterd_hd = holding1[holding1['GICODE'] == stk]
    fnd = filterd_hd['F_ID'].unique()
    fnd_string = ', '.join(map(str, fnd))
    notice_data_dft_4['stks'].iloc[i] = fnd_string

path = os.path.dirname(__file__)
##### 엑셀로 저장

notice_data_dft_0.to_excel(path+'\\notice_dft_0.xlsx')
notice_data_dft_4.to_excel(path+'\\notice_dft_4.xlsx')


