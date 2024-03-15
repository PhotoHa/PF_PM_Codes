# -*- coding: utf-8 -*-
"""
Created on Wed Feb 14 11:11:25 2024

@author: 11149
"""

import pandas as pd
import numpy as np
import pyodbc
import openpyxl as op
import os
os.chdir('T:\\index\\95_곽용하\\운용\\코드')
import functions_0 as mf0
import functions_1 as mf1

import schedule
import time
import matplotlib.pyplot as plt

pd.set_option("display.max_columns", 50) #최대 50개의 컬럼까지 줄바꿈으로 보여줌
############################ 0. 데이터베이스 연결
conn_pcor = pyodbc.connect('driver={Oracle in OraClient12Home1};dbq=PCOR;uid=EF0SEL;pwd=EF0SEL#076')
conn_quant = pyodbc.connect('driver={SQL Server};server=46.2.90.172;database=quant;uid=index;pwd=samsung@00')
conn_wisefn = pyodbc.connect('driver={SQL Server};server=46.2.90.172;database=wisefn;uid=index;pwd=samsung@00')



# PBR
sql_pbr = '''   SELECT CONVERT(char(8), ScoreDate,112) AS TRD_DT , Code, Ratio AS INV_PBR
                FROM QUANT..QA_FactorDat_
                WHERE FactorCode = 'BTM_QW' AND ScoreDate > '2005-01-01'
                AND Code IN (SELECT 'A'+STK_CD COLLATE Korean_Wansung_CI_AS   AS Code 
                				FROM WISEFN..TS_STOCK
                				WHERE 1=1
                				AND MKT_TYP = 1
                				AND STK_TYP = 1
                				AND ISSUE_TYP = 1
                				AND STK_CD NOT IN (SELECT STK_CD FROM WISEFN..TS_STOCK WHERE MKT_TYP = 1 AND STK_TYP = 1 AND DELIST_DT < '20050101' AND ISSUE_TYP = 1)  )  '''
    
df_inv_pbr = pd.read_sql(sql_pbr, conn_wisefn)
df_pbr = df_inv_pbr.copy()
df_pbr['INV_PBR'] = 1 / df_pbr['INV_PBR']



# ROE
sql_roe = '''   SELECT CONVERT(char(8), ScoreDate,112) AS TRD_DT , Code, Ratio AS ROE
                FROM QUANT..QA_FactorDat_
                WHERE FactorCode = '211500_FQ0' AND ScoreDate > '2005-01-01'
                AND Code IN (SELECT 'A'+STK_CD COLLATE Korean_Wansung_CI_AS   AS Code 
                				FROM WISEFN..TS_STOCK
                				WHERE 1=1
                				AND MKT_TYP = 1
                				AND STK_TYP = 1
                				AND ISSUE_TYP = 1
                				AND STK_CD NOT IN (SELECT STK_CD FROM WISEFN..TS_STOCK WHERE MKT_TYP = 1 AND STK_TYP = 1 AND DELIST_DT < '20050101' AND ISSUE_TYP = 1)  )  '''
                    
df_roe = pd.read_sql(sql_roe, conn_wisefn)



# PRC : 분류 모델 학습에는 사용하지 않음
# sql_prc = '''   SELECT TRD_DT, STK_CD, VAL AS PRC
#                 FROM WISEFN..TS_STK_DATA
#                 WHERE ITEM_CD = '100300'
#                 AND TRD_DT IN (SELECT YMD FROM WISEFN..TZ_DATE WHERE MN_END_YN = 1 AND YMD > '20050101' AND TRADE_YN = 1)
#                 AND STK_CD IN (SELECT STK_CD FROM WISEFN..TS_STK_ISSUE WHERE KS200_TYP = 1 AND TRD_DT IN (SELECT YMD FROM WISEFN..TZ_DATE WHERE MN_END_YN = 1 AND YMD > '20050101' AND TRADE_YN = 1))  '''

# df_adj_prc = pd.read_sql(sql_prc, conn_wisefn)


# 
# df_0 = pd.merge(df_pbr, df_adj_prc, 'left', ('TRD_DT','STK_CD'))
# df_0 = pd.merge(df_0, df_roe, 'left', ('TRD_DT','STK_CD'))


df_0 = pd.merge(df_pbr, df_roe, 'left', ('TRD_DT','Code'))
df_0.columns = ['TRD_DT','STK_CD','PBR','ROE']
# df_0['STK_CD'] = 'A'+ df_0['STK_CD']
# df_0.to_csv("T:\\index\\95_곽용하\\연구\\8_pbr_roe\\pbr_roe.csv")


##############################################################################
##############################################################################
##############################################################################

df_0 = pd.read_csv("T:\\index\\95_곽용하\\연구\\8_pbr_roe\\pbr_roe.csv")
df_0 = df_0.iloc[:,1:]


stk_list = df_0['STK_CD'].unique()
path_0 = "T:\\index\\95_곽용하\\연구\\8_pbr_roe\\img_files2\\"


from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()


for stk in stk_list:
    case1 = df_0[df_0['STK_CD'] == stk]
    case1 = case1.sort_values('TRD_DT')
    case1.dropna(inplace=True)

    if len(case1) > 20:      
        case11 = case1.copy()
        # case11['PBR'] = scaler.fit_transform(case11[['PBR']].values)
        # case11['ROE'] = scaler.fit_transform(case11[['ROE']].values)    
    
        plt.figure(figsize=(6,6))
        plt.scatter(case11['PBR'], case11['ROE'])
        plt.plot(case11['PBR'], case11['ROE'], marker='o')
        plt.xlabel('PBR')
        plt.ylabel('ROE')
        plt.savefig(path_0 + f'img_{stk}.png')  

    else:
        pass
    
    print(f'end of {stk}')



    # if len(case1) > 20:
        
    #     for sub in range(len(case1)-20+1):
    #         case11 = case1.iloc[sub:sub+20]
    #         #case11['PBR'] = scaler.fit_transform(case11[['PBR']].values)
    #         #case11['ROE'] = scaler.fit_transform(case11[['ROE']].values)    
        
    #         plt.figure(figsize=(6,6))
    #         plt.scatter(case11['PBR'], case11['ROE'])
    #         plt.plot(case11['PBR'], case11['ROE'], marker='o')
    #         plt.xlabel('PBR')
    #         plt.ylabel('ROE')
    #         plt.savefig(path_0 + f'img_{stk}_{sub+1}.png')  
    
    #     else:
    #         pass
        
    #     print(f'end of {stk}_{sub+1}')




##############################################################################
# K-means Clustering
from PIL import Image





image_directory = "T:\\index\\95_곽용하\\연구\\8_pbr_roe\\img_files2\\"   # 이미지가 저장된 디렉토리 경로
image_vectors = []  # 이미지 벡터를 저장할 리스트 초기화

# 디렉토리 내 모든 이미지 파일에 대해 순회
for filename in os.listdir(image_directory):
    if filename.endswith(".jpg") or filename.endswith(".png"): # 이미지 파일 필터링
        # 이미지 불러오기
        image_path = os.path.join(image_directory, filename)
        image = Image.open(image_path)
        image = image.resize((100,100))  # 이미지 크기 조정 100
        
        # 이미지를 벡터로 변환하여 리스트에 추가
        image_array = np.array(image)
        image_vector = image_array.flatten()
        image_vector_nml = scaler.fit_transform(image_vector.reshape(-1,1))
        image_vectors.append(image_vector_nml)


image_vectors = np.array(image_vectors)  # 리스트를 배열로 변환



from sklearn.cluster import KMeans
image_matrix = image_vectors.reshape((923,40000))  # 이미지 벡터를 하나의 행렬로 합치기 40000
num_clusters = 6  # 클러스터 개수 설정
kmeans = KMeans(n_clusters=num_clusters)  # K-means 모델 초기화
kmeans.fit(image_matrix)  # K-means 모델 학습
cluster_labels = kmeans.labels_  # 각 이미지에 대한 클러스터 할당 확인

# 결과 출력
for i in range(num_clusters):
    cluster_i_indices = np.where(cluster_labels == i)[0]
    print(cluster_i_indices)
    print(f"Cluster {i+1}: {len(cluster_i_indices)} images")




filtered_stk = df_0.dropna().groupby('STK_CD').count().mean(axis=1)
filtered_stk = filtered_stk[filtered_stk>20]

rst_1 = pd.DataFrame({'STK_CD':filtered_stk.index, 'cluster':cluster_labels})





# step 3 : 안 하는게 좋을지도



cluster_means = []
for i in range(num_clusters):
    cluster_i_indices = np.where(cluster_labels == i)[0]
    cluster_i_vectors = image_vectors[cluster_i_indices]
    cluster_i_mean = np.mean(cluster_i_vectors, axis=0)
    cluster_i_mean = scaler.inverse_transform(cluster_i_mean)
    cluster_means.append(cluster_i_mean)



# 각 클러스터의 평균 이미지를 다시 이미지로 변환하여 저장
path_1 = "T:\\index\\95_곽용하\\연구\\8_pbr_roe\\rst_files\\"

for i, mean_vector in enumerate(cluster_means):
    # 이미지 벡터를 2차원 배열로 변환
    mean_array = mean_vector.reshape((100, 100, 4)) 
    # 2차원 배열을 이미지로 변환
    mean_image = Image.fromarray(mean_array.astype(np.uint8))
    
    # 이미지를 파일로 저장
    mean_image.save(path_1+f"cluster_{i+1}_mean.png")

















case1 = df_0[df_0['STK_CD']=='A000660']
case1 = case1.sort_values('TRD_DT')

case1['PBR'] = scaler.fit_transform(case1[['PBR']].values)
case1['ROE'] = scaler.fit_transform(case1[['ROE']].values)

plt.figure(figsize=(6,6))
plt.scatter(case1['PBR'], case1['ROE'])
plt.plot(case1['PBR'], case1['ROE'], marker='.')
plt.xlabel('PBR')
plt.ylabel('ROE')
plt.grid(True)
plt.show()
plt.savefig("T:\\index\\95_곽용하\\연구\\8_pbr_roe\\img_files\\")


# 이미지 크기 조정
# image0 = Image.open("T:\\index\\95_곽용하\\연구\\8_pbr_roe\\img_files\\img_A000660.png")
# image0 = image0.resize((100,100))
# image_array = np.array(image0)
# image_vector = image_array.flatten()

# # 데이터 정규화

# scaler = MinMaxScaler()
# normalized_image_vector = scaler.fit_transform(image_vector.reshape(-1,1))

