import matplotlib.pyplot as plt
import pandas as pd

df=pd.read_csv(r'C:/Users/Jaspreet Singh/Documents/Combined-Weather-Energy-Mel-Gee-201905.csv')
df['DATETIME'] =  pd.to_datetime(df['DATETIME'], dayfirst=True)

df['Date']=pd.DatetimeIndex(df['DATETIME']).day
df['Month']=pd.DatetimeIndex(df['DATETIME']).month
df['Hour']=pd.DatetimeIndex(df['DATETIME']).hour
df['Minute']=pd.DatetimeIndex(df['DATETIME']).minute

X=df.iloc[:,[2,3,4,5,6,7,9,10,11,12]]
Y=df.iloc[:,8:9]

X_train = X[0:367]
X_test = X[367:]
Y_train = Y[0:367]
Y_test = Y[367:]

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