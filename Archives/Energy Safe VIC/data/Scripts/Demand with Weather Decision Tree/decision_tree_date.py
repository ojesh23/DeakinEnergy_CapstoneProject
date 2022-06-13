import matplotlib.pyplot as plt
import pandas as pd

df=pd.read_csv(r'C:/Users/Jaspreet Singh/Documents/deakinenergyt12020/data/PRICE_AND_DEMAND_201811_VIC1.csv')
df = df[['SETTLEMENTDATE', 'TOTALDEMAND']]
df['SETTLEMENTDATE'] =  pd.to_datetime(df['SETTLEMENTDATE'], dayfirst=True)

df['Date']=pd.DatetimeIndex(df['SETTLEMENTDATE']).day
df['Month']=pd.DatetimeIndex(df['SETTLEMENTDATE']).month
df['Hour']=pd.DatetimeIndex(df['SETTLEMENTDATE']).hour
df['Minute']=pd.DatetimeIndex(df['SETTLEMENTDATE']).minute

X=df.iloc[:,2:6]
Y=df.iloc[:,1:2]

X_train = X[0:1100]
X_test = X[1100:]
Y_train = Y[0:1100]
Y_test = Y[1100:]

from sklearn import preprocessing
minmaxscaler=preprocessing.MinMaxScaler()

X_train = minmaxscaler.fit_transform(X_train)
X_test = minmaxscaler.fit_transform(X_test)
Y_train = minmaxscaler.fit_transform(Y_train)
Y_test = minmaxscaler.fit_transform(Y_test)
  
from sklearn.tree import DecisionTreeRegressor
regressor = DecisionTreeRegressor(max_depth=25,
  		                           max_leaf_nodes=None)
regressor.fit(X_train, Y_train)
Y_predict=regressor.predict(X_test)

Y_predict = minmaxscaler.inverse_transform(Y_predict.reshape(-1,1))
Y_test = minmaxscaler.inverse_transform(Y_test.reshape(-1,1))

plt.figure(figsize=(9,6))
plt.plot(Y_test, color='red', label = 'Actual Energy Demand')
plt.plot(Y_predict, color='blue', label = 'Predicted Energy Demand')
plt.title('Energy demand- Actual vs Predicted ')
plt.ylabel('Demand')
plt.legend()

from sklearn.metrics import mean_absolute_error
mean_absolute_error(Y_test, Y_predict)

#-----------------------------------------------------------------------------------------

from sklearn import svm
import numpy as np
SVR_model = svm.SVR(kernel='rbf',C=100,gamma=.001).fit(X_train,np.ravel(Y_train,order='C'))

# Use SVR model to calculate predicted next-hour usage
Y_predict = SVR_model.predict(X_test.reshape(-1,1))

Y_predict = minmaxscaler.inverse_transform(Y_predict.reshape(-1,1))
Y_test = minmaxscaler.inverse_transform(Y_test.reshape(-1,1))

plt.figure(figsize=(9,6))
plt.plot(Y_test, color='red', label = 'Actual Energy Demand')
plt.plot(Y_predict, color='blue', label = 'Predicted Energy Demand')
plt.title('Energy demand- Actual vs Predicted ')
plt.ylabel('Demand')
plt.legend()