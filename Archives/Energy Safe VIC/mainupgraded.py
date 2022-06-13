import os, sys
#import magic
import urllib.request
from app import app

from flask import Flask, flash, request, redirect, render_template, url_for
from werkzeug.utils import secure_filename

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import LSTM
from keras.layers import Dense
from sklearn import preprocessing as pre
import random
import glob

from tensorflow.python.keras.layers import Input, GRU, Embedding
from tensorflow.python.keras.optimizers import RMSprop
from tensorflow.python.keras.callbacks import EarlyStopping, ModelCheckpoint, TensorBoard, ReduceLROnPlateau
from sklearn.metrics import mean_squared_error

# import warnings filter
from warnings import simplefilter
# ignore all future warnings
simplefilter(action='ignore', category=FutureWarning)

val = 1
image_name ="default"
selected_algorithm = ""
selected_optimizer = ""
RMSE = 0

"__________________________________________________________________________________________________"

ALLOWED_EXTENSIONS = set(['csv'])

def allowed_file(filename):
	return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS
"__________________________________________________________________________________________________"

@app.route('/', methods=['GET','POST'])

def upload_file():

	if request.method == 'POST':
        # check if the post request has the files part
		if 'files[]' not in request.files:
			flash('No file part')
			return redirect(request.url)
		files = request.files.getlist('files[]')
		upload_success = True

		for file in files:
			if file and allowed_file(file.filename):
				filename = secure_filename(file.filename)
				file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
			else:
				upload_success = False

		if upload_success:
			flash('File(s) uploaded successfully')
		else:
			flash("Invalid file format. Acceptable format '.csv' ")
		return redirect('/')
	else:
		print("image name: {}".format(image_name))
		filelist = os.listdir( "data" )
		return render_template('upload2.html', filenames = filelist, image_name=image_name, selected_algorithm=selected_algorithm, RMSE=RMSE, selected_optimizer= selected_optimizer)

"__________________________________________________________________________________________________"

def univariate_data(dataset, start_index, end_index, history_size, target_size):
	data = []
	labels = []

	start_index = start_index + history_size
	if end_index is None:
	   end_index = len(dataset) - target_size

	for i in range(start_index, end_index):
	  indices = range(i-history_size, i)
	  # Reshape data from (history_size,) to (history_size, 1)
	  data.append(np.reshape(dataset[indices], (history_size, 1)))
	  labels.append(dataset[i+target_size])
	return np.array(data), np.array(labels)

def create_time_steps(length):
	time_steps = []
	for i in range(-length, 0, 1):
	    time_steps.append(i)
	return time_steps

def baseline(history):
	return np.mean(history)

"__________________________________________________________________________________________________"

@app.route('/forecast', methods= ['POST'])

def forecast():

	global selected_algorithm, selected_optimizer, image_name
	selected_algorithm = request.form.get('alist')
	selected_optimizer = request.form.get('olist')
	image_name = "default"

	chart_list=glob.glob("static/images/*.png")
	for chart in chart_list:
  		os.remove(chart)

	selected_file = request.form.get('flist')
	if selected_file == "default":
		#selected_algorithm = " "
		filelist = os.listdir( "data" )
		return render_template('upload2.html', filenames = filelist, image_name= image_name, selected_algorithm=selected_algorithm, selected_optimizer=selected_optimizer, error = True)
	print(selected_file)

	file_path = "data\\" + selected_file
	print("file path - {}".format(file_path))
	energyData=pd.read_csv(file_path,header=0)

	def lstm(energyData):

		energyData.head()

		energyData=energyData.drop(['REGION'],axis=1)

		energyData=energyData.drop(['PERIODTYPE'],axis=1)
		energyData=energyData.drop(['RRP( Regional reference price)'],axis=1)
		#energyData=energyData.drop(['TOTALDEMAND'],axis=1)

		energyData=energyData.rename(columns={"RRP( Regional reference price)" : "RRP"})

		energyData.tail()

		def univariate_data(dataset, start_index, end_index, history_size, target_size):
		  data = []
		  labels = []

		  start_index = start_index + history_size
		  if end_index is None:
		    end_index = len(dataset) - target_size

		  for i in range(start_index, end_index):
		    indices = range(i-history_size, i)
		    # Reshape data from (history_size,) to (history_size, 1)
		    data.append(np.reshape(dataset[indices], (history_size, 1)))
		    labels.append(dataset[i+target_size])
		  return np.array(data), np.array(labels)

		TRAIN_SPLIT = 1339

		tf.random.set_random_seed(13)

		uni_data = energyData['TOTALDEMAND']
		uni_data.index = energyData['SETTLEMENTDATE']
		uni_data.head()
		uni_data = uni_data.values
		uni_train_mean = uni_data[:TRAIN_SPLIT].mean()
		uni_train_std = uni_data[:TRAIN_SPLIT].std()

		uni_data = (uni_data-uni_train_mean)/uni_train_std

		univariate_past_history = 20
		univariate_future_target = 0

		x_train_uni, y_train_uni = univariate_data(uni_data, 0, TRAIN_SPLIT,
		                                           univariate_past_history,
		                                           univariate_future_target)
		x_val_uni, y_val_uni = univariate_data(uni_data, TRAIN_SPLIT, None,
		                                       univariate_past_history,
		                                       univariate_future_target)
		print(x_train_uni.shape[-2:])

		print ('Single window of past history')
		print (x_train_uni[0])
		print ('\n Target Demand to predict')
		print (y_train_uni[0])

		from keras.layers import LSTM
		simple_lstm_model = Sequential()
		simple_lstm_model.add(LSTM(50, input_shape=x_train_uni.shape[-2:]))
		simple_lstm_model.add(Dense(1))

		simple_lstm_model.compile(optimizer='adam', loss='mae')

		print(simple_lstm_model.predict(x_val_uni).shape)

		EVALUATION_INTERVAL = 200
		EPOCHS = 10

		simple_lstm_model.fit(x_train_uni,y_train_uni, epochs=EPOCHS,
		                      steps_per_epoch=EVALUATION_INTERVAL,
		                      validation_data=(x_val_uni,y_val_uni), validation_steps=50)

		y_pred=simple_lstm_model.predict(x_val_uni)

		plt.plot(y_val_uni, color = 'red', label = 'Actual Energy Demand')
		plt.plot(y_pred, color = 'blue', label = 'Predicted Energy Demand')
		plt.title('Energy demand- Actual vs Predicted ')
		plt.ylabel('Demand')
		plt.legend()

		val = random.randrange(1,500)
		global image_name
		image_name = "chart"+str(val)+'.png'
		plt.savefig(os.path.join(os.getcwd(), 'static/images',image_name), format='png')

		from sklearn.metrics import mean_squared_error
		from math import sqrt

		rms = sqrt(mean_squared_error(y_val_uni, y_pred))
		global RMSE
		RMSE=rms
		print(rms)

	def lstm2(energyData):

		energyData.head()

		energyData=energyData.drop(['REGION'],axis=1)

		energyData=energyData.drop(['PERIODTYPE'],axis=1)
		energyData=energyData.drop(['RRP( Regional reference price)'],axis=1)
		#energyData=energyData.drop(['TOTALDEMAND'],axis=1)

		energyData=energyData.rename(columns={"RRP( Regional reference price)" : "RRP"})

		energyData.tail()

		def univariate_data(dataset, start_index, end_index, history_size, target_size):
		  data = []
		  labels = []

		  start_index = start_index + history_size
		  if end_index is None:
		    end_index = len(dataset) - target_size

		  for i in range(start_index, end_index):
		    indices = range(i-history_size, i)
		    # Reshape data from (history_size,) to (history_size, 1)
		    data.append(np.reshape(dataset[indices], (history_size, 1)))
		    labels.append(dataset[i+target_size])
		  return np.array(data), np.array(labels)

		TRAIN_SPLIT = 1339

		tf.random.set_random_seed(13)

		uni_data = energyData['TOTALDEMAND']
		uni_data.index = energyData['SETTLEMENTDATE']
		uni_data.head()
		uni_data = uni_data.values
		uni_train_mean = uni_data[:TRAIN_SPLIT].mean()
		uni_train_std = uni_data[:TRAIN_SPLIT].std()

		uni_data = (uni_data-uni_train_mean)/uni_train_std

		univariate_past_history = 20
		univariate_future_target = 0

		x_train_uni, y_train_uni = univariate_data(uni_data, 0, TRAIN_SPLIT,
		                                           univariate_past_history,
		                                           univariate_future_target)
		x_val_uni, y_val_uni = univariate_data(uni_data, TRAIN_SPLIT, None,
		                                       univariate_past_history,
		                                       univariate_future_target)
		print(x_train_uni.shape[-2:])

		print ('Single window of past history')
		print (x_train_uni[0])
		print ('\n Target Demand to predict')
		print (y_train_uni[0])

		from keras.layers import LSTM
		simple_lstm_model = Sequential()
		simple_lstm_model.add(LSTM(50, input_shape=x_train_uni.shape[-2:]))
		simple_lstm_model.add(Dense(1))

		simple_lstm_model.compile(optimizer=selected_optimizer, loss='mae')

		print(simple_lstm_model.predict(x_val_uni).shape)

		EVALUATION_INTERVAL = 200
		EPOCHS = 10

		simple_lstm_model.fit(x_train_uni,y_train_uni, epochs=EPOCHS,
		                      steps_per_epoch=EVALUATION_INTERVAL,
		                      validation_data=(x_val_uni,y_val_uni), validation_steps=50)

		y_pred=simple_lstm_model.predict(x_val_uni)

		plt.plot(y_val_uni, color = 'red', label = 'Actual Energy Demand')
		plt.plot(y_pred, color = 'blue', label = 'Predicted Energy Demand')
		plt.title('Energy demand- Actual vs Predicted ')
		plt.ylabel('Demand')
		plt.legend()

		val = random.randrange(1,500)
		global image_name
		image_name = "chart"+str(val)+'.png'
		plt.savefig(os.path.join(os.getcwd(), 'static/images',image_name), format='png')

		from sklearn.metrics import mean_squared_error
		from math import sqrt

		rms = sqrt(mean_squared_error(y_val_uni, y_pred))
		global RMSE
		RMSE=rms
		print(rms)

	def Gradient_booster(energyData):

		df = energyData[['SETTLEMENTDATE', 'TOTALDEMAND']]

		df['SETTLEMENTDATE'] =  pd.to_datetime(df['SETTLEMENTDATE'], dayfirst=True)

		print(df.head)
		print (df.dtypes)

		from sklearn import preprocessing
		minmaxscaler=preprocessing.MinMaxScaler()

		array_y=np.array(df['TOTALDEMAND'])

		normalized_y=minmaxscaler.fit_transform(array_y.reshape(-1,1))

		X = np.array(df['SETTLEMENTDATE'])
		X=X.astype(float)

		from sklearn.model_selection import train_test_split
		xTrain, xTest, yTrain, yTest = train_test_split(X, normalized_y, test_size = 0.2, random_state = 0)

		from sklearn.ensemble import GradientBoostingRegressor
		gbrt=GradientBoostingRegressor(n_estimators=150,learning_rate=0.02,subsample=.5,max_depth=8)
		gbrt.fit(xTrain.reshape(-1, 1), yTrain.reshape(-1, 1))

		y_pred=gbrt.predict(xTest.reshape(-1, 1))

		plt.figure(figsize=(9,6))
		plt.plot(minmaxscaler.inverse_transform(yTest.reshape(-1,1)), color='red', label = 'Actual Energy Demand')
		plt.plot(minmaxscaler.inverse_transform(y_pred.reshape(-1,1)), color='blue', label = 'Predicted Energy Demand')
		plt.title('Energy demand- Actual vs Predicted ')
		plt.ylabel('Demand')
		plt.legend()

		val = random.randrange(1,500)
		global image_name
		image_name = "chart"+str(val)+'.png'
		plt.savefig(os.path.join(os.getcwd(), 'static/images',image_name), format='png')

		from sklearn.metrics import mean_squared_error
		from math import sqrt
		Grms = sqrt(mean_squared_error(yTest, y_pred))
		print("Root Mean Square error : {}".format(Grms))
		global RMSE
		RMSE=Grms

		from sklearn.metrics import mean_absolute_error
		mean_absolute_error(yTest, y_pred)

	def decision_tree(energyData):
		df = energyData[['SETTLEMENTDATE', 'TOTALDEMAND']]
		df['SETTLEMENTDATE'] =  pd.to_datetime(df['SETTLEMENTDATE'], dayfirst=True)

		from sklearn import preprocessing
		minmaxscaler=preprocessing.MinMaxScaler()
		array_y=np.array(df['TOTALDEMAND'])
		normalized_y=minmaxscaler.fit_transform(array_y.reshape(-1,1))

		X = np.array(df['SETTLEMENTDATE'])
		X=X.astype(float)

		from sklearn.model_selection import train_test_split
		xTrain, xTest, yTrain, yTest = train_test_split(X, normalized_y, test_size = 0.2, random_state = 0)

		from sklearn.tree import DecisionTreeRegressor
		regressor = DecisionTreeRegressor(random_state = 0, max_depth=75,min_samples_split=2,
		                           max_leaf_nodes=None)
		regressor.fit(xTrain.reshape(-1,1), yTrain.reshape(-1,1))
		y_predict=regressor.predict(xTest.reshape(-1,1))

		plt.figure(figsize=(9,6))
		plt.plot(minmaxscaler.inverse_transform(yTest.reshape(-1,1)), color='red', label = 'Actual Energy Demand')
		plt.plot(minmaxscaler.inverse_transform(y_predict.reshape(-1,1)), color='blue', label = 'Predicted Energy Demand')
		plt.title('Energy demand- Actual vs Predicted ')
		plt.ylabel('Demand')
		plt.legend()

		val = random.randrange(1,500)
		global image_name
		image_name = "chart"+str(val)+'.png'
		plt.savefig(os.path.join(os.getcwd(), 'static/images',image_name), format='png')

		from math import sqrt
		Drms= sqrt(mean_squared_error(yTest, y_predict))
		print("Root Mean Square Error : {}".format(Drms))

		global RMSE
		RMSE=Drms

		from sklearn.metrics import mean_absolute_error
		mean_absolute_error(yTest, y_predict)

	def svm(file_path):
		fontsize = 18
		data_file = pd.read_csv(file_path,parse_dates=True, index_col=1)
		data_file=data_file.drop(['REGION'],axis=1)
		data_file=data_file.drop(['PERIODTYPE'],axis=1)

		## Set weekends and holidays to 1, otherwise 0
		data_file['Atypical_Day'] = np.zeros(len(data_file['TOTALDEMAND']))

		# Weekends
		data_file['Atypical_Day'][(data_file.index.dayofweek==5)|(data_file.index.dayofweek==6)] = 1

		data_file.head(50)

		# Create new column for each hour of day, assign 1 if index.hour is corresponding hour of column, 0 otherwise

		for i in range(0,48):
		    data_file[i] = np.zeros(len(data_file['TOTALDEMAND']))
		    data_file[i][data_file.index.hour==i] = 1

		# Example 3am
		data_file[3][:6]

		# Add historic usage to each X vector

		# Set number of hours prediction is in advance
		n_hours_advance = 1

		# Set number of historic hours used
		n_hours_window = 6


		for k in range(n_hours_advance,n_hours_advance+n_hours_window):

		    data_file['TOTALDEMAND_t-%i'% k] = np.zeros(len(data_file['TOTALDEMAND']))


		for i in range(n_hours_advance+n_hours_window,len(data_file['TOTALDEMAND'])):

		    for j in range(n_hours_advance,n_hours_advance+n_hours_window):

		        data_file['TOTALDEMAND_t-%i'% j][i] = data_file['TOTALDEMAND'][i-j]


		# Define training and testing periods
		train_start = '1-march-2019'
		train_end = '23-march-2019'
		test_start = '24-march-2019'
		test_end = '1-april-2019'

		# Split up into training and testing sets (still in Pandas dataframes)

		X_train_df = data_file[train_start:train_end]
		y_train_df = data_file['TOTALDEMAND'][train_start:train_end]

		X_test_df = data_file[test_start:test_end]
		y_test_df = data_file['TOTALDEMAND'][test_start:test_end]

		N_train = len(X_train_df[0])
		print ('Number of observations in the training set: ', N_train)

		# Numpy arrays for sklearn
		X_train = np.array(X_train_df)
		X_test = np.array(X_test_df)
		y_train = np.array(y_train_df)
		y_test = np.array(y_test_df)

		from sklearn import preprocessing as pre
		from sklearn import svm
		scaler = pre.StandardScaler().fit(X_train)
		X_train_scaled = scaler.transform(X_train)
		X_test_scaled = scaler.transform(X_test)

		SVR_model = svm.SVR(kernel='rbf',C=100,gamma=.001).fit(X_train_scaled,y_train)
		print ('Testing R^2 =', round(SVR_model.score(X_test_scaled,y_test),3))

		# Use SVR model to calculate predicted next-hour usage
		predict_y_array = SVR_model.predict(X_test_scaled)

		# Put it in a Pandas dataframe for ease of use
		predict_y = pd.DataFrame(predict_y_array,columns=['TOTALDEMAND'])
		predict_y.index = X_test_df.index

		### Plot daily total kWh over testing period
		y_test_barplot_df = pd.DataFrame(y_test_df,columns=['TOTALDEMAND'])
		y_test_barplot_df['Predicted'] = predict_y['TOTALDEMAND']

		fig = plt.figure(figsize=[11,7])
		ax = fig.add_subplot(111)
		y_test_barplot_df.plot(kind='line',ax=ax,color=['red','blue'])
		ax.grid(False)
		ax.set_ylabel('Electricity Demand(kWh)', fontsize=fontsize)
		ax.set_xlabel('')
		# Pandas/Matplotlib bar graphs convert xaxis to floats, so need a hack to get datetimes back
		ax.set_xticklabels([dt.strftime('%b %d') for dt in y_test_df.index.to_pydatetime()],rotation=0, fontsize=fontsize)
		plt.title('Energy demand- Actual vs Predicted ')
		plt.legend(fontsize=fontsize)

		val = random.randrange(1,500)
		global image_name
		image_name = "chart"+str(val)+'.png'
		plt.savefig(os.path.join(os.getcwd(), 'static/images',image_name), format='png')

		from sklearn.metrics import mean_squared_error
		from math import sqrt

		SvmRms = sqrt(mean_squared_error(y_test_df,predict_y))
		global RMSE
		RMSE=SvmRms

		print("Root Mean Square Error : {}".format(SvmRms))

	if selected_algorithm == "LSTM":
		lstm(energyData)
	elif selected_algorithm == "Gradient_Booster":
		Gradient_booster(energyData)
	elif selected_algorithm == "Decision_Tree":
		decision_tree(energyData)
	elif selected_algorithm == "SVM":
		svm(file_path)
	elif selected_algorithm == "LSTM2":
		lstm2(energyData)
	return redirect('/')


if __name__ == "__main__":
    app.run()
