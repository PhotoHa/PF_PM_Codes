# -*- coding: utf-8 -*-
"""
Created on Wed Oct  5 11:13:30 2022
"""

%matplotlib inline
import numpy as np
import pandas as pd

def annualize_rets(r, periods_per_year):
    """
    Annualizes a set of returns
    We should infer the periods per year
    but that is currently left as an exercise
    to the reader :-)
    """
    compounded_growth = (1+r).prod()
    n_periods = r.shape[0]
    return compounded_growth**(periods_per_year/n_periods)-1


def annualize_vol(r, periods_per_year):
    """
    Annualizes the vol of a set of returns
    We should infer the periods per year
    but that is currently left as an exercise
    to the reader :-)
    """
    return r.std()*(periods_per_year**0.5)

########################### 150 #################################
df_rt = pd.read_csv("150_rt.csv")
rt = df_rt.iloc[:,0]

mu_150 = annualize_rets(rt, periods_per_year=252)
sigma_150 = annualize_vol(rt, periods_per_year=252)

########################### 2배 200 #################################
df_rt = pd.read_csv("2_200_rt.csv")
rt = df_rt.iloc[:,0]

mu_150 = annualize_rets(rt, periods_per_year=252)
sigma_150 = annualize_vol(rt, periods_per_year=252)

########################### 인버스 200 #################################
df_rt = pd.read_csv("in_200_rt.csv")
rt = df_rt.iloc[:,0]

mu_150 = annualize_rets(rt, periods_per_year=252)
sigma_150 = annualize_vol(rt, periods_per_year=252)

########################### 중국 #################################
df_rt = pd.read_csv("china_rt.csv")
rt = df_rt.iloc[:,0]

mu_150 = annualize_rets(rt, periods_per_year=252)
sigma_150 = annualize_vol(rt, periods_per_year=252)




def gbm0(n_years = 10, n_scenarios=1000, mu=0.07, sigma=0.15, steps_per_year=12, s_0=100.0):
    # Derive per-step Model Parameters from User Specifications
    dt = 1/steps_per_year
    n_steps = int(n_years*steps_per_year)
    xi = np.random.normal(size=(n_steps, n_scenarios))
    rets = mu*dt + sigma*np.sqrt(dt)*xi
    # convert to a DataFrame
    rets = pd.DataFrame(rets)
    # convert to prices
    prices = s_0*(rets+1).cumprod()
    return prices

#p = gbm0(n_years=10, n_scenarios=10000, mu=0.07)
#p.shape
#p.iloc[-1].mean(), 100*1.07**10

rst = gbm0(n_years=1, n_scenarios=10000, mu = mu_150, sigma = sigma_150, steps_per_year=252)
#gbm0(n_years=1, n_scenarios=10000, mu = mu_150, sigma = sigma_150, steps_per_year=252).plot(figsize=(12,5), legend=False)
imsi = rst.copy()
imsi = imsi.iloc[[0,251],:]
imsi_rt = imsi.pct_change(1)
fin_rt = imsi_rt.iloc[1,:]


# 분석
fin_rt_loss = fin_rt[fin_rt<0]
avg = fin_rt_loss.mean()
max_loss = min(fin_rt)

print(avg, max_loss)
