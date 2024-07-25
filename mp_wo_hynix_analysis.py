# -*- coding: utf-8 -*-
"""
Created on Wed Jul 17 17:06:20 2024

@author: 11149
"""

import numpy as np
import pandas as pd
import pyodbc
import os
os.chdir('T:\\index\\95_곽용하\\운용\\코드\\')
import mf_3 as mf

data = pd.read_excel("T:\\index\\95_곽용하\\연구\\12_TEAM\\mp_hynix\\tp_idx_all.xlsx",index_col=0, header=0)

# plot
data.plot(figsize=(12,6),legend=False)

# rtn analysis
rtn = data.pct_change().dropna()

df_summary = mf.summary_stats(rtn, 250)
