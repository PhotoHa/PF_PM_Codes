# -*- coding: utf-8 -*-
"""
Created on Mon Apr 24 12:57:15 2023
"""

#pip install requests beautifulsoup4
from bs4 import BeautifulSoup
import numpy as np
import pandas as pd
import requests

rss_url = 'http://kind.krx.co.kr:80/disclosure/rsstodaydistribute.do?method=searchRssTodayDistribute&repIsuSrtCd=&mktTpCd=0&searchCorpName=&currentPageSize=200'

response = requests.get(rss_url)
xml = response.content
soup = BeautifulSoup(xml, 'xml')

print("피드 제목:", soup.title.text)
print("피드 설명:", soup.description.text)

items = soup.find_all('item')
   
ca_in_time = []

for item in items:
    if item.title:
        ca = item.title.text
        ca_in_time = np.append(ca_in_time, ca)
    else:
        pass

df_ca = pd.DataFrame(ca_in_time)
df_ca.columns = ['ca']

#df_ca[df_ca['ca'].str.contains('공개매수신고서')]
df_ca_1 = df_ca[df_ca['ca'].str.contains('공개매수신고서')]
print(df_ca_1)
