# -*- coding: utf-8 -*-
"""
Created on Tue Nov 22 10:38:03 2022
"""

import numpy as np
import pandas as pd
import statsmodels.api as sm
from statsmodels.tsa.api import VAR
from statsmodels.tsa.stattools import adfuller


df = pd.read_csv("gt_spd_vksp.csv",index_col=0)
mdata = df
data = np.log(mdata).diff().dropna()

model = VAR(data)

results = model.fit(2)
results.summary()

results.plot()
results.plot_acorr()

irf = results.irf(10)
irf.plot(orth=False)


'''  https://www.statsmodels.org/dev/vector_ar.html  '''


## UDF for ADF test
def adf_test(timeseries):
    print("Results of Dickey-Fuller Test:")
    dftest = adfuller(timeseries, autolag="AIC")
    dfoutput = pd.Series(
        dftest[0:4],
        index=[
            "Test Statistic",
            "p-value",
            "#Lags Used",
            "Number of Observations Used",
        ],
    )
    for key, value in dftest[4].items():
        dfoutput["Critical Value (%s)" % key] = value
    print(dfoutput)
    
adf_test(df['sprd_idx'])
