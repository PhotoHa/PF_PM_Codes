# -*- coding: utf-8 -*-
"""
Created on Sat Jun 22 10:05:33 2024

@author: david
"""

#### 스타일 팩터를 활용해 시장 상태를 판단
# QUAL: MSCI 퀄리티 팩터
# MTUM: MSCI 모멘텀 팩터
# USMV: MSCI 로우볼 팩터
# VLUE: MSCI 밸류   팩터
# SIZE: MSCI 사이즈 팩터


import pandas as pd
import numpy as np
import yfinance as yf


# 주가 데이터
data_prc = yf.download(['QUAL','MTUM','USMV','VLUE','SIZE'], start = '2022-01-01')
data_adj = data_prc['Adj Close']
data_rtn = data_adj.pct_change(1).dropna()


# 행렬 인수분해 함수
def matrix_factorization(R_df, K, steps=5000, alpha=0.002, beta=0.02):
    # 사용자와 아이템의 고유 값 리스트
    user_ids = R_df.index.tolist()
    item_ids = R_df.columns.tolist()
    
    # 사용자와 아이템의 고유 값 개수
    num_users = len(user_ids)
    num_items = len(item_ids)
    
    # 임의의 값으로 사용자 및 아이템 잠재 요인 행렬 초기화
    P = np.random.rand(num_users, K)
    Q = np.random.rand(num_items, K)
    
    # 사용자 및 아이템 인덱스 매핑
    user_id_map = {user_id: idx for idx, user_id in enumerate(user_ids)}
    item_id_map = {item_id: idx for idx, item_id in enumerate(item_ids)}
    
    # 인덱스를 통해 접근하여 행렬 인수분해 수행
    for step in range(steps):
        for user_id, row in R_df.iterrows():
            for item_id, rating in row.items():
                if rating != 0:
                    i = user_id_map[user_id]
                    j = item_id_map[item_id]
                    eij = rating - np.dot(P[i, :], Q[j, :])
                    for k in range(K):
                        P[i, k] = P[i, k] + alpha * (2 * eij * Q[j, k] - beta * P[i, k])
                        Q[j, k] = Q[j, k] + alpha * (2 * eij * P[i, k] - beta * Q[j, k])
        
        eR = np.dot(P, Q.T)
        e = 0
        for user_id, row in R_df.iterrows():
            for item_id, rating in row.items():
                if rating != 0:
                    i = user_id_map[user_id]
                    j = item_id_map[item_id]
                    e = e + pow(rating - np.dot(P[i, :], Q[j, :]), 2)
                    for k in range(K):
                        e = e + (beta / 2) * (pow(P[i, k], 2) + pow(Q[j, k], 2))
        if e < 0.001:
            break
    
    return P, Q



# 행렬 인수분해 적용
K = 4  # 잠재 요인 수 설정
P, Q = matrix_factorization(data_rtn, K)
nR = np.dot(P, Q.T)



print("원본 행렬:\n", R)
print("복원된 행렬:\n", nR)
