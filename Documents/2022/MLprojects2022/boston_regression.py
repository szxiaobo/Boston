import pandas as pd
df_boston = pd.read_csv('Boston_house_price.csv')
df_boston.shape

import statsmodels.api as sm

y = df_boston['Value'] # dependent variable
x = df_boston['Rooms'] # independent variable
print(y.head())
print(x.head())

x1=5
x = sm.add_constant(x1) # adding a constant
lm = sm.OLS(y,x).fit() # fitting the model

lm.predict(x)


#part 3
lm.summary()
# Rooms coef: 9.1021
# Constant coef: - 34.6706
# Linear equation: 𝑦 = 𝑎𝑥 + 𝑏
y_pred = 9.1021 * x['Rooms'] - 34.6706

import seaborn as sns
import matplotlib.pyplot as plt
# plotting the data points
sns.scatterplot(x=x['Rooms'], y=y, z=None) # 1 update this line first
#plotting the line
sns.lineplot(x=x['Rooms'],y=y_pred, color='red', color2='yellow2') #2 adding the color
#axes
plt.xlim(10) # 3a chage it again 
plt.ylim(110)  # 3b chage it again 
#<<<<<<< branch_modeling
plt.show()

#part 4
lm = linear_model.LinearRegression()
lm.fit(X, y) # fitting the model
lm.predict(X)
lm.score(X, y)

lm.coef_

lm.intercept_
plt.show()

