import os, sys
#import magic
import urllib.request
from app import app

from flask import Flask, flash, request, redirect, render_template, url_for
from werkzeug.utils import secure_filename

import requests
from io import StringIO
import datetime
import numpy as np
import calendar
from datetime import date
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM
from tensorflow.keras.layers import Dense
from sklearn import preprocessing as pre
import random
import glob
import pygal

from datetime import datetime as dt
from tensorflow.keras.layers import Input, GRU, Embedding
from tensorflow.keras.optimizers import RMSprop
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, TensorBoard, ReduceLROnPlateau
from sklearn.metrics import mean_squared_error
import sklearn.preprocessing
from keras.layers import Dense,Dropout,LSTM,SimpleRNN
from keras.models import Sequential

# import warnings filter
from warnings import simplefilter
# ignore all future warnings
simplefilter(action='ignore', category=FutureWarning)

val = 1
image_name ="default"
line_chart = "";
festival_image = "";
monthly_image = "";

selected_algorithm = ""
RMSE = 0
ExecutionTime=0
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
		global filelist
		filelist = os.listdir(app.config['UPLOAD_FOLDER'])
		return render_template('upload.html', filenames = filelist, monthly_image = monthly_image, festival_image = festival_image,line_chart= line_chart, image_name=image_name, selected_algorithm=selected_algorithm, RMSE=RMSE, ExecutionTime=ExecutionTime)

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
	# data.append(np.reshape(dataset[indices], (history_size, 1)))
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

	global line_chart
	global selected_algorithm, image_name
	selected_algorithm = request.form.get('alist')
	image_name = "default"

	chart_list=glob.glob("static/images/*.png")
	for chart in chart_list:
		os.remove(chart)

	selected_file = request.form.get('flist')
	if selected_file == "default":
		#selected_algorithm = " "
		filelist = os.listdir(app.config["UPLOAD_FOLDER"])
		return render_template('upload.html',filenames = filelist, monthly_image = monthly_image, festival_image = festival_image, line_chart= line_chart, image_name= image_name, selected_algorithm=selected_algorithm, error = True)
	print(selected_file)

	file_path = app.config["UPLOAD_FOLDER"]+"/" + selected_file
	print("file path - {}".format(file_path))
	energyData=pd.read_csv(file_path,header=0)

	def lstm(energyData):
	
		now1 = dt.now()
		df=energyData
		df = pd.read_csv(file_path,parse_dates=True, index_col=1)
		df=df.drop(['REGION'],axis=1)
		df=df.drop(['PERIODTYPE'],axis=1)
		df=df.drop(['RRP'],axis=1)
		
		def normalize_data(df):
			scaler = sklearn.preprocessing.MinMaxScaler()
			df['TOTALDEMAND']=scaler.fit_transform(df['TOTALDEMAND'].values.reshape(-1,1))
			return df
		df_norm = normalize_data(df)
		df_norm.shape

		def load_data(stock, seq_len):
			X_train = []
			y_train = []
			for i in range(seq_len, len(stock)):
				X_train.append(stock.iloc[i-seq_len : i, 0])
				y_train.append(stock.iloc[i, 0])
			
			#1 last 6189 days are going to be used in test
			X_test = X_train[1300:]             
			y_test = y_train[1300:]
			
			#2 first 110000 days are going to be used in training
			X_train = X_train[:1300]           
			y_train = y_train[:1300]
			
			#3 convert to numpy array
			X_train = np.array(X_train)
			y_train = np.array(y_train)
			
			X_test = np.array(X_test)
			y_test = np.array(y_test)
			
			#4 reshape data to input into RNN models
			X_train = np.reshape(X_train, (1300, seq_len, 1))
			X_test = np.reshape(X_test, (X_test.shape[0], seq_len, 1))
			
			return [X_train, y_train, X_test, y_test]
		
		#create train, test data
		seq_len = 20 #choose sequence length
		
		X_train, y_train, X_test, y_test = load_data(df, seq_len)
		
		print('X_train.shape = ',X_train.shape)
		print('y_train.shape = ', y_train.shape)
		print('X_test.shape = ', X_test.shape)
		print('y_test.shape = ',y_test.shape)
		
		rnn_model = Sequential()
		
		rnn_model.add(SimpleRNN(40,activation="tanh",return_sequences=True, input_shape=(X_train.shape[1],1)))
		rnn_model.add(Dropout(0.15))
		
		rnn_model.add(SimpleRNN(40,activation="tanh",return_sequences=True))
		rnn_model.add(Dropout(0.15))
		
		rnn_model.add(SimpleRNN(40,activation="tanh",return_sequences=False))
		rnn_model.add(Dropout(0.15))
		
		rnn_model.add(Dense(1))
		
		rnn_model.summary()
		
		rnn_model.compile(optimizer="adam",loss="MSE")
		rnn_model.fit(X_train, y_train, epochs=10, batch_size=1000)
		
		rnn_predictions = rnn_model.predict(X_test)
		
		def plot_predictions(test, predicted, title):
			global line_chart
			line_chart = pygal.Line()
			line_chart.x_label = 'Demand'
			line_chart.title = title
			line_chart.add('Actual Energy Demand',(y_test.reshape(-1,1)).flatten())
			line_chart.add('Predicted Energy Demand',(rnn_predictions.reshape(-1,1)).flatten())
			line_chart = line_chart.render_data_uri()
		plot_predictions(y_test, rnn_predictions, "Predictions made by LSTM model")
		from sklearn.metrics import mean_squared_error
		from math import sqrt
		lrms = sqrt(mean_squared_error(y_test, rnn_predictions))
		print("Root Mean Square error : {}".format(lrms))
		global RMSE
		RMSE=lrms
		now2 = dt.now()
		diff=now2-now1
		global ExecutionTime # for execution time of each algorithm
		ExecutionTime=str(diff.seconds)+" seconds"
		print("Execution time =",ExecutionTime, "Seconds" )
		print(line_chart)

	def Gradient_booster(energyData):
		now1 = dt.now()
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

		from sklearn.ensemble import GradientBoostingRegressor
		gbrt=GradientBoostingRegressor(n_estimators=150,learning_rate=0.02,subsample=.5,max_depth=8)
		gbrt.fit(xTrain.reshape(-1, 1), yTrain.reshape(-1, 1))

		y_pred=gbrt.predict(xTest.reshape(-1, 1))

		global line_chart
		line_chart = pygal.Line()
		line_chart.x_label = 'Demand'
		line_chart.title = 'Energy demand- Actual vs Predicted'
		line_chart.add('Actual Energy Demand', minmaxscaler.inverse_transform(yTest.reshape(-1,1)).flatten())
		line_chart.add('Predicted Energy Demand',  minmaxscaler.inverse_transform(y_pred.reshape(-1,1)).flatten() )
		line_chart = line_chart.render_data_uri()

		from sklearn.metrics import mean_squared_error
		from math import sqrt
		Grms = sqrt(mean_squared_error(yTest, y_pred))
		print("Root Mean Square error : {}".format(Grms))
		global RMSE
		RMSE=Grms

		now2 = dt.now()
		diff=now2-now1
		global ExecutionTime # for execution time of each algorithm
		ExecutionTime=str(diff.microseconds/1000000)+" seconds"
		print("Execution time =",ExecutionTime, "Seconds" )
		
		from sklearn.metrics import mean_absolute_error
		mean_absolute_error(yTest, y_pred)

	def decision_tree(energyData):
		now1 = dt.now()
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

		global line_chart
		line_chart = pygal.Line()
		line_chart.y_label = 'Demand'
		line_chart.title = 'Energy demand- Actual vs Predicted'
		line_chart.add('Actual Energy Demand', minmaxscaler.inverse_transform(yTest.reshape(-1,1)).flatten())
		line_chart.add('Predicted Energy Demand',  minmaxscaler.inverse_transform(y_predict.reshape(-1,1)).flatten() )
		line_chart = line_chart.render_data_uri()
		
		from math import sqrt
		Drms= sqrt(mean_squared_error(yTest, y_predict))
		print("Root Mean Square Error : {}".format(Drms))

		global RMSE
		RMSE=Drms

		global ExecutionTime # for execution time of each algorithm
		now2 = dt.now()
		diff=now2-now1
		ExecutionTime=str(diff.microseconds/1000000)+" seconds"
		print("Execution time =",ExecutionTime)

		from sklearn.metrics import mean_absolute_error
		mean_absolute_error(yTest, y_predict)

	def svm(file_path):
		now1 = dt.now()
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
		
		#-- 19/05 commit
		          
 
        #Split up into training and testing sets (still in Pandas dataframes)
		X_train_df = data_file.head(int(len(data_file)*(70/100)))
		
		y_train_df = data_file['TOTALDEMAND'].head(int(len(data_file)*(70/100)))

		
		X_test_df = data_file.tail(int(len(data_file)*(30/100)))
		
		y_test_df = data_file['TOTALDEMAND'].tail(int(len(data_file)*(30/100)))



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
		
		###Actual dataframe saved to var
		y_test_barplot_df_actual=y_test_barplot_df
		
		y_test_barplot_df['Predicted'] = predict_y['TOTALDEMAND']

		#New plot 19/05/20 dynamic svm
		global line_chart
		line_chart = pygal.Line()
		line_chart.y_label = 'Demand'
		line_chart.title = 'Energy demand- Actual vs Predicted'
		line_chart.add('Actual Energy Demand', np.array(y_test_barplot_df_actual['TOTALDEMAND']))
		line_chart.add('Predicted Energy Demand', np.array(y_test_barplot_df['Predicted']))
		line_chart = line_chart.render_data_uri()
		

		from sklearn.metrics import mean_squared_error
		from math import sqrt
		
		p_normalised = predict_y.max()
		t_normalised = y_test_df.max()
				
		p = predict_y/p_normalised
		t = y_test_df/t_normalised
		
		SvmRms = sqrt(mean_squared_error(p,t))
		global RMSE
		RMSE=SvmRms
		now2 = dt.now()
		diff=now2-now1
		global ExecutionTime # for execution time of each algorithm
		ExecutionTime=str(diff.seconds)+" seconds"
		print("Execution time =",ExecutionTime)
		print("Root Mean Square Error : {}".format(SvmRms))

	if selected_algorithm == "LSTM":
		lstm(energyData)    
	elif selected_algorithm == "Gradient_Booster":
		Gradient_booster(energyData)
	elif selected_algorithm == "Decision_Tree":
		decision_tree(energyData)
	elif selected_algorithm == "SVM":
		svm(file_path)
	return redirect('/')

@app.route('/festival', methods= ['POST'])
def festival():
	global festival_image
	chart_list=glob.glob("static/images/festival*.png")
	for chart in chart_list:
		os.remove(chart)
		
	#these two options will be linked with the drop down list on the webpage
	festival = str(request.form.get('festival'))
	year = str(request.form.get('fyear'))

	#main algo starts from here

	#headers for fixing error 403
	headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 6.1; WOW64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/56.0.2924.76 Safari/537.36'}

	#code.activestate.com
	def calc_easter(year):
		"Returns Easter as a date object."
		a = year % 19
		b = year // 100
		c = year % 100
		d = (19 * a + b - b // 4 - ((b - (b + 8) // 25 + 1) // 3) + 15) % 30
		e = (32 + 2 * (b % 4) + 2 * (c // 4) - d - (c % 4)) % 7
		f = d + e - 7 * ((a + 11 * d + 22 * e) // 451) + 114
		month = f // 31
		day = f % 31 + 1    
		return date(year, month, day)

	#checking input to decide day and month
	if(festival == 'New Year'):
		month = '01'
		day=date(int(year), int(month), 1)
	if(festival == 'Christmas'):
		month = '12'
		day=date(int(year), int(month), 25)
	if(festival == 'Australia Day'):
		month='01'
		day=date(int(year), int(month), 26)
	if(festival == 'Anzac Day'):
		month='04'
		day=date(int(year), int(month), 25)
	if(festival == 'Easter'):
		day=calc_easter(int(year))
		month=str(0)+str(day.month)
	if(festival == 'Good Friday'):
		day=calc_easter(int(year))-datetime.timedelta(days=2)
		month=str(0)+str(day.month)

	#setting prev and next day from the festival
	day1=day-datetime.timedelta(days=1)
	day2=day+datetime.timedelta(days=1)

	#states array
	states = ['VIC1',
			  'NSW1',
			  'SA1',
			  'QLD1',
			  'TAS1']

	#demands array initially empty
	demands = []
	demands1 = []
	demands2 = []
	average = []

	#calculating the demands
	for state in states:
		#adding month and year to the link
		url = 'https://aemo.com.au/aemo/data/nem/priceanddemand/PRICE_AND_DEMAND_{}{}_{}.csv'.format(year,month,state)
		s = requests.get(url, headers = headers).text
		df=pd.read_csv(StringIO(s), sep=',')
		
		df[['Date','Time']] = df.SETTLEMENTDATE.str.split(' ',expand=True)
		df['Date'] = pd.to_datetime(df['Date'])
		
		#filtering data based on day
		average.append(df['TOTALDEMAND'].sum()/(df.shape[0]/2));
		
		#total demand on festival
		x=df.query('Date == @day')
		count=x.shape[0]/2
		#averaging for each hour
		demands.append(x['TOTALDEMAND'].sum()/count)
		
		#total demand a day before
		x=df.query('Date == @day1')
		count=x.shape[0]/2
		#averaging for each hour
		demands1.append(x['TOTALDEMAND'].sum()/count)
		
		#total demand a day after
		x=df.query('Date == @day2')
		count=x.shape[0]/2
		#averaging for each hour
		demands2.append(x['TOTALDEMAND'].sum()/count)

	#array for states
	objects = ('Victoria', 'New South Wales', 'South Australia', 'Queensland', 'Tasmania')
	
	#plotting using pygal
	chart=pygal.Bar(legend_at_bottom=True,title=u'Energy Demand per Hour on '+festival+' '+str(year), y_title='Demand Per Hour (MWh)')
	chart.x_labels = objects
	
	#if demand is null dont plot (else it will throw)
	if(pd.notnull(demands1[1])):
		chart.add('Previous Day ('+str(day1.day)+' '+day1.strftime("%B")+' '+calendar.day_name[day1.weekday()]+')',demands1)
	chart.add(festival+' ('+str(day.day)+' '+day.strftime("%B")+' '+calendar.day_name[day.weekday()]+')',demands)
	chart.add('Next Day ('+str(day2.day)+' '+day2.strftime("%B")+' '+calendar.day_name[day2.weekday()]+')',demands2)
	chart.add('Average for Month',average)
	
	#converting and storing data 
	festival_image = chart.render_data_uri()
	
	return redirect('/?tab=festival&year='+year+'&festival='+festival)

@app.route('/monthly', methods= ['POST'])
def monthly():

	global monthly_image
	
	#these two options will be linked with the drop down list on the webpage
	month = str(request.form.get('month'))
	state = str(request.form.get('mstate'))

	#main algo starts from here

	#headers for fixing error 403
	headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 6.1; WOW64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/56.0.2924.76 Safari/537.36'}

	objects = ('Victoria', 'New South Wales', 'South Australia', 'Queensland', 'Tasmania')
	
	#states array
	states = ['VIC1',
			  'NSW1',
			  'SA1',
			  'QLD1',
			  'TAS1']

	#user options (can be changes)
	months = ['January',
			  'February',
			  'March',
			  'April',
			  'May',
			  'June',
			  'July',
			  'August',
			  'September',
			  'October',
			  'November',
			  'December']

	#empty demands array initially
	demands = []

	#calculating demand of that month every year
	for year in range(2000, 2020):
	#adding month and year to the link
		url = 'https://aemo.com.au/aemo/data/nem/priceanddemand/PRICE_AND_DEMAND_{}{}_{}.csv'.format(year, month, state)
		s = requests.get(url, headers = headers).text
		df = pd.read_csv(StringIO(s), sep = ",")
		
		#adding monthly demand
		demands.append(df['TOTALDEMAND'].sum()/1000)

	#plotting
	label = range(2000, 2020)
	chart=pygal.Line(x_label_rotation=-45,title=u'Energy Demand in ' + months[int(month)-1] + ' for '+objects[states.index(state)], y_title='Demand per Month (GWh)', x_title='Year')
	chart.x_labels = label
	chart.add(months[int(month)-1],demands)
	
	#converting data and storing
	monthly_image = chart.render_data_uri()
	
	return redirect('/?tab=monthly&state='+state+'&month='+month)
	
@app.route('/showall',methods= ['POST'])
def showall():

	global monthly_image
	
	#months list
	months = ['January',
			  'February',
			  'March',
			  'April',
			  'May',
			  'June',
			  'July',
			  'August',
			  'September',
			  'October',
			  'November',
			  'December']
			  
	#demands list for vic month wise
	a=[[7248.77467005, 7908.355003289999, 7544.283648429999, 8006.315524930001, 7776.386873359998, 8185.8860318, 8502.56477499, 8613.72533, 8793.52838, 8799.58015, 8528.18606, 8448.27724, 8497.38604, 8213.4666, 8418.090530000001, 7480.38133, 7457.264630000001, 6994.94767, 7733.49734, 7552.06083], [7554.06616344, 7551.61419114, 7189.0111533, 7506.10974656, 7843.588768209999, 7558.7946066800005, 7702.436583410001, 8388.113650000001, 8219.59357, 8085.87066, 8173.95082, 7805.555040000001, 8184.28518, 7846.2019900000005, 7647.3168700000015, 7210.296109999999, 7236.92699, 6418.8393, 6747.991140000001, 6505.218900000001], [7803.67143255, 7850.522989110001, 7709.6250432, 7957.55604661, 8214.88871158, 8200.358946620001, 8586.52765824, 8611.46466, 8784.95274, 8452.69684, 8581.459270000001, 8286.26521, 8097.88691, 8451.10321, 7925.40523, 7299.57326, 7722.035140000001, 7272.77175, 6968.72111, 7229.24025], [7090.91619462, 7372.28313483, 7473.389423419999, 7618.71114855, 7774.24650999, 7902.44766154, 7996.2581349, 7905.578619999999, 8268.62445, 8123.28766, 7897.96053, 7798.95839, 7760.09354, 7646.1043899999995, 7454.952730000001, 7176.851009999999, 7003.37582, 6463.712769999999, 6788.757820000001, 6517.67316], [8082.69120575, 8136.490126520001, 8188.50317979, 8333.02805329, 8586.332086580001, 8600.287000080001, 9005.6999882, 8600.44385, 9012.073380000002, 8824.809860000001, 8664.315530000002, 8821.49995, 8761.87078, 8426.653380000002, 8097.52954, 7754.270759999999, 7438.0401600000005, 7468.75323, 7561.98694, 7390.375029999999], [8102.89144896, 8067.19477773, 8239.748894999999, 8350.78852665, 8647.704388459999, 8575.89868654, 9070.725900000001, 9057.50309, 8839.34512, 8973.62575, 8998.010769999999, 8756.916969999998, 8691.42225, 8425.968620000001, 8127.541439999999, 8048.85726, 8022.7639500000005, 7886.218289999999, 7941.72583, 7778.61351], [8333.731960430001, 8455.00501171, 8718.06440824, 8833.69295499, 9026.78919502, 8968.73823151, 9224.64692, 9545.669629999999, 9623.441420000001, 9159.392800000001, 9413.982759999997, 9064.597399999999, 8932.5885, 8646.956460000001, 8574.53508, 8546.93163, 8159.86759, 8046.374390000001, 8073.91421, 7995.7842200000005], [8225.005630860001, 8502.65447501, 8573.36603497, 8721.68640338, 8880.107711630002, 8934.5625117, 9062.989850000002, 9161.91596, 9575.70322, 8738.52659, 9237.193190000002, 8781.22957, 8762.01049, 8372.711299999999, 8141.52729, 8191.05655, 8133.453, 8032.30351, 7884.42327, 7927.09778], [7618.84348159, 7692.092806740001, 7912.5567281, 8122.8510900599995, 8250.102096609999, 8235.82802984, 8318.29911, 8439.24899, 8578.15739, 8203.90135, 8522.654580000002, 8162.29019, 7872.70435, 7540.85955, 7302.0696, 7405.5406299999995, 7358.935530000001, 7169.97511, 6978.71068, 6977.245440000001], [7762.755300119999, 7832.86973509, 7984.40730826, 8258.46482995, 8152.43417497, 8098.61662475, 8468.32019, 8599.532959999999, 8668.027390000001, 8430.123210000002, 8258.28575, 8241.8138, 8114.0965, 7806.02709, 7367.99557, 7261.50502, 7131.541480000001, 6888.85797, 6717.606470000001, 6878.1761400000005], [7437.7495401099995, 7440.7350248600005, 7664.44725003, 7662.61144176, 8013.20192173, 7907.09486499, 8176.9457, 8329.313590000002, 8165.72999, 8453.68712, 8060.79134, 8005.413479999999, 7876.05975, 7524.02279, 7015.2801500000005, 6955.8427, 6914.62597, 7171.90317, 6427.270189999999, 6503.339249999999],[7470.55410919, 7249.09872163, 7823.92149313, 8050.2636316, 7996.289184970001, 8167.840404940001, 8303.19126, 8416.04419, 8110.73598, 8365.27598, 8030.8893100000005, 8110.90271, 7836.54907, 7666.15341, 7211.421300000001, 7677.80929, 6716.3223499999995, 7021.17167, 6907.14875, 6840.97934]]
	
	#plotting
	label = range(2000, 2020)
	chart=pygal.Line(x_label_rotation=-45,title=u'Energy Demand Trend for Victoria', y_title='Demand in (GWh)', x_title='Year')
	chart.x_labels = label
	for i in range(0,12):
		chart.add(months[i],a[i])
		
	#converting and storing data 
	monthly_image = chart.render_data_uri()
	
	return redirect('/?tab=monthly')

	
if __name__ == "__main__":
	app.run()
