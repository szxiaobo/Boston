import pandas as pd
df_boston = pd.read_csv('Boston_house_price.csv')
df_boston.shape

import statsmodels.api as sm

y = df_boston['Value'] # dependent variable
x = df_boston['Rooms'] # independent variable
print(y.head())
print(x.head())