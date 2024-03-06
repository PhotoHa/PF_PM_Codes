# -*- coding: utf-8 -*-
"""
Created on Fri Nov 25 09:57:41 2022

@author: 11149
"""

import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
import requests
import statsmodels.api as sm



df = pd.read_csv("T:\\index\\95_곽용하\\연구\\국면_기준 탐구\\변동성_markov switching model\\200_rt.csv", index_col = 0)
df.index = pd.to_datetime(df.index, format="%Y%m%d")
# Fit the model
mod_kns = sm.tsa.MarkovRegression(
    df, k_regimes=2, trend="n", switching_variance=True
)
res_kns = mod_kns.fit()

res_kns.summary()


fig, axes = plt.subplots(2, figsize=(10, 7))

ax = axes[0]
ax.plot(res_kns.smoothed_marginal_probabilities[0])
ax.set(title="Smoothed probability of a low-variance regime for stock returns")

ax = axes[1]
ax.plot(res_kns.smoothed_marginal_probabilities[1])
ax.set(title="Smoothed probability of a high-variance regime for stock returns")

fig.tight_layout()


a = res_kns.smoothed_marginal_probabilities
