# prepare data

from numpy import loadtxt
from numpy import nan
from numpy import isnan
from numpy import count_nonzero
from numpy import unique
from numpy import array
from numpy import nanmedian
from numpy import save
from numpy import load
from numpy import loadtxt
from numpy import nan
from numpy import isnan
from numpy import count_nonzero
from numpy import unique
from numpy import array
from numpy import delete
import pandas as pd
import numpy as np
import numpy.ma as ma
from datetime import date, timedelta 
import calendar
from sklearn.base import clone
from sklearn.metrics import mean_squared_log_error
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Lasso
from sklearn.linear_model import Ridge
from sklearn.linear_model import ElasticNet
from sklearn.linear_model import HuberRegressor
from sklearn.linear_model import LassoLars
from sklearn.linear_model import PassiveAggressiveRegressor
from sklearn.linear_model import SGDRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.tree import ExtraTreeRegressor
from sklearn.svm import SVR
from sklearn.ensemble import AdaBoostRegressor
from sklearn.ensemble import BaggingRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.ensemble import GradientBoostingRegressor
 
# split the dataset by 'chunkID', return a list of chunks
def to_chunks(df, predictor_ix=0, diagnosis_ix =0):
	chunks = list()
	time_index = pd.DataFrame(pd.date_range('2019-05-01', periods=11, freq='MS')+ pd.DateOffset(days=12), columns=['Date'])
	# get the unique chunk ids
	predictor_ids = unique(df.loc[:, predictor_ix])
	#diagnosis_ids = unique(df.loc[:, diagnosis_ix])
	# group rows by chunk id
	for predictor_id in predictor_ids:
		selection = df.loc[:, predictor_ix] == predictor_id
		diagnosis_ids = unique(df.loc[selection, diagnosis_ix])
		for diagnosis_id in diagnosis_ids:
			selection = (df.loc[:, predictor_ix] == predictor_id)&(df.loc[:, diagnosis_ix] == diagnosis_id)
			if len(df.loc[selection, :]) == 11:
				chunks.append(df.loc[selection, :])
			else:
				imputed = pd.merge(time_index,df.loc[selection, :],left_on='Date', right_on='Date',how='left')
				imputed.set_index('Date',inplace=True)
				imputed['PARAMETER_ESTIMATE'] = imputed['PARAMETER_ESTIMATE'].astype(float).interpolate(limit_direction ='forward')
				imputed['PARAMETER_ESTIMATE'] = imputed['PARAMETER_ESTIMATE'].astype(float).interpolate(limit_direction ='backward')
				chunks.append(imputed)
	return chunks
 
# return a list of relative forecast lead times
def get_lead_times():
	return [1]

# return true if the array has any non-nan values
def has_data(data):
	return count_nonzero(isnan(data)) < len(data)
 

 
# layout a variable with breaks in the data for missing positions
def variable_to_series(chunk_train, col_ix, n_steps=10):
	# lay out whole series
	if n_steps < len(chunk_train):
		data = [nan for _ in range(len(chunk_train))]
	else:	
		data = [nan for _ in range(n_steps)]
	# mark all available data
	for i in range(len(chunk_train)):
		# get position in chunk
		position = i
		# store data
		data[position] = chunk_train.iloc[i, col_ix]
	return data
 
# created input/output patterns from a sequence
def supervised_for_steps(series, n_lag, lead_time):
    data = list()
    for i in range(n_lag, len(series)):
        end_ix = i + (lead_time - 1)
        if end_ix >= len(series):
            break
        start_ix = i - n_lag
        ne = []
        for item in series[start_ix:i]:
            ne.append(item)
        ne.append([series[end_ix]][0])        
        row = ne        
        data.append(row)	
    return array(data[:-1]),array(data[-1])
 
# create supervised learning data for each lead time for this target
def target_to_supervised(rows, col_ix, n_lag):
	train_lead_times = list()
	# get series
	series = variable_to_series(rows, col_ix)
	#if not has_data(series.astype(float)):
		#return None, [nan for _ in range(n_lag)]
	# impute
	imputed = array(series).astype(float)
	# prepare test sample for chunk-variable
	actual = array(imputed[-1])
	# enumerate lead times
	lead_times = get_lead_times()
	for lead_time in lead_times:
		# make input/output data from series
		train_samples,test_sample = supervised_for_steps(imputed, n_lag, lead_time)
		train_lead_times.append(train_samples)
	return train_lead_times, test_sample,actual
 
# prepare training [var][lead time][sample] and test [chunk][var][sample]
def data_prep(chunks, n_lag, n_vars=1):
	lead_times = get_lead_times()
	#train_data = [[list() for _ in range(len(lead_times))] for _ in range(n_vars)]
	train_data = [[list() for _ in range(n_vars)] for _ in range(len(chunks))]
	test_data = [[list() for _ in range(n_vars)] for _ in range(len(chunks))]
	actual_data = [[list() for _ in range(n_vars)] for _ in range(len(chunks))]
	# enumerate targets for chunk
	for var in range(n_vars):
		# convert target number into column number
		col_ix = 3 + var
		# enumerate chunks to forecast
		for c_id in range(len(chunks)):
			rows = chunks[c_id]
			# prepare sequence of hours for the chunk
			#hours = variable_to_series(rows, 2)
			# interpolate hours
			#interpolate_hours(hours)
			# check for no data
			if not has_data(rows.iloc[:, col_ix].astype(float)):
				continue
			# convert series into training data for each lead time
			train, test_sample,actual = target_to_supervised(rows, col_ix, n_lag)
			# store test sample for this var-chunk
			test_data[c_id][var] = test_sample
			train_data[c_id][var] = train
			actual_data[c_id][var] = actual
			""" if train is not None:
				# store samples per lead time
				for lead_time in range(len(lead_times)):
					# add all rows to the existing list of rows
					train_data[var][lead_time].extend(train[lead_time])
		# convert all rows for each var-lead time to a numpy array
		for lead_time in range(len(lead_times)):
			train_data[var][lead_time] = array(train_data[var][lead_time]) """
	return array(train_data), array(test_data), array(actual_data)
 
# fit a single model
def fit_model(model, X, y):
	# clone the model configuration
	local_model = clone(model)
	# fit the model
	local_model.fit(X, y)
	return local_model
 
# fit one model for each variable and each forecast lead time [var][time][model]
def fit_models(model, train):
	# prepare structure for saving models
	models = [[list() for _ in range(train.shape[1])] for _ in range(train.shape[0])]
	# enumerate vars
	for i in range(train.shape[0]):
		# enumerate lead times
		for j in range(train.shape[1]):
			# get data
			data = train[i, j]
			#print(i,j)
			#print(data[0])
			X, y = data[0][:, :-1], data[0][:, -1]
			# fit model
			local_model = fit_model(model, X, y)
			models[i][j].append(local_model)
	return models


 
# return forecasts as [chunks][var][time]
def make_predictions(models, test):
	lead_times = get_lead_times()
	predictions = list()
	# enumerate chunks
	for i in range(test.shape[0]):
		# enumerate variables
		chunk_predictions = list()
		for j in range(test.shape[1]):
			# get the input pattern for this chunk and target
			pattern = test[i,j]
			pattern = delete(pattern,len(pattern)-1)
			# assume a nan forecast
			forecasts = array([nan for _ in range(len(lead_times))])
			# check we can make a forecast
			if has_data(pattern):
				pattern = pattern.reshape((1, len(pattern)))
				# forecast each lead time
				forecasts = list()
				for k in range(len(lead_times)):
					yhat = models[j][k][0].predict(pattern)
					forecasts.append(yhat[0])
				forecasts = array(forecasts)
			# save forecasts for each lead time for this variable
			chunk_predictions.append(forecasts)
		# save forecasts for this chunk
		chunk_predictions = array(chunk_predictions)
		predictions.append(chunk_predictions)
	return array(predictions)
 
# convert the test dataset in chunks to [chunk][variable][time] format
def prepare_test_forecasts(test_chunks):
	predictions = list()
	# enumerate chunks to forecast
	for rows in test_chunks:
		# enumerate targets for chunk
		chunk_predictions = list()
		for j in range(3, rows.shape[1]):
			yhat = rows[:, j]
			chunk_predictions.append(yhat)
		chunk_predictions = array(chunk_predictions)
		predictions.append(chunk_predictions)
	return array(predictions)
 
# calculate the error between an actual and predicted value
def calculate_error(actual, predicted):
	# give the full actual value if predicted is nan
	if isnan(predicted):
		return abs(actual)
	# calculate abs difference	
	return abs(actual - predicted)
 
# evaluate a forecast in the format [chunk][variable][time]
def evaluate_forecasts(predictions, testset):
	lead_times = get_lead_times()
	total_mae, times_mae = 0.0, [0.0 for _ in range(len(lead_times))]
	total_c, times_c = 0, [0 for _ in range(len(lead_times))]
	# enumerate test chunks
	for i in range(len(testset)):
		# convert to forecasts
		actual = testset[i]
		predicted = predictions[i]
		# enumerate target variables
		for j in range(predicted.shape[0]):
			# enumerate lead times
			for k in range(len(lead_times)):
				# skip if actual in nan
				#print(actual[j])
				if isnan(actual[j]):
					continue
				# calculate error
				error = calculate_error(actual[j], predicted[j, k])
				#print(error)
				# update statistics
				total_mae += error
				times_mae[k] += error
				total_c += 1
				times_c[k] += 1
		#print(actual, predicted)	
		#print(mean_squared_log_error(actual, predicted))
	total_mae /= total_c
	times_mae = [times_mae[i]/times_c[i] for i in range(len(times_mae))]
	return total_mae, times_mae
 
# summarize scores
def summarize_error(name, total_mae):
	print('%s: %.3f MAE' % (name, total_mae))
 
# prepare a list of ml models
def get_models(models=dict()):
	# linear models
	models['lr'] = LinearRegression()
	models['lasso'] = Lasso()
	models['ridge'] = Ridge()
	models['en'] = ElasticNet()
	models['huber'] = HuberRegressor()
	models['llars'] = LassoLars()
	models['pa'] = PassiveAggressiveRegressor(max_iter=1000, tol=1e-3)
	models['sgd'] = SGDRegressor(max_iter=1000, tol=1e-3)
	print('Defined %d models' % len(models))
	return models
 
# evaluate a suite of models
def evaluate_models(models, train, test, actual):
	for name, model in models.items():
		# fit models
		fits = fit_models(model, train)
		# make predictions
		
		predictions = make_predictions(fits, test)
		# evaluate forecast
		#print(actual)
		total_mae, _ = evaluate_forecasts(predictions, actual)
		# summarize forecast
		
		summarize_error(name, total_mae)

def fit_predict_refit(model,history_data,col_ix,n_lag,steps,future_time_index,predictor_id,diagnosis_id):
	predicted = []
	abnormal_list = []
	train, test_sample,actual = target_to_supervised(history_data, 0, n_lag)
	#print(train[0])
	#print(test_sample)
	whole = np.concatenate((train[0],[test_sample]))
	#print(whole[-1, 1:])						
	for i in range(steps):
		try:
			model_fit = model.fit(whole[:, :-1], whole[:, -1])
		except:
			print(predicted,predictor_id,diagnosis_id,i, whole)
		last_value = whole[-1, -1]
		pattern = whole[-1, 1:]
		pattern = pattern.reshape((1, len(pattern)))
		#print(pattern)
		yhat = model_fit.predict(pattern)
		
		#print(predictor_id,diagnosis_id)
		if abs((yhat.astype(float)-last_value.astype(float))/last_value.astype(float)) > 0.5:			
			pre = yhat
			mask = np.logical_or(whole[:,-1] == whole[:,-1].max(), whole[:,-1] == whole[:,-1].min())
			a_masked = ma.masked_array(whole[:,-1], mask = mask)
			average_diff = np.average(np.diff(a_masked))
			yhat = array([last_value+average_diff])
			if isnan(yhat.astype(float)):
				yhat = array([last_value])
			abnormal_list.append([future_time_index['Date'].iloc[i],future_time_index['SHMIPREDICTOR'].iloc[i],predictor_id,diagnosis_id,abs(yhat.astype(float)-last_value.astype(float)),average_diff, yhat,pre])
		predicted.append([future_time_index['Date'].iloc[i],future_time_index['SHMIPREDICTOR'].iloc[i],predictor_id,diagnosis_id,yhat[0]])
		#print(whole[-1, 1:],yhat)
		new_row = np.concatenate((whole[-1, 1:],yhat),axis=0)
		#print(new_row)
		#print(whole)
		whole = np.concatenate((whole, [new_row]))
	return predicted,abnormal_list


def make_future_prediction(model,final_df,steps, n_lag):
	
	time_index = pd.DataFrame(pd.date_range('2019-05-01', periods=11, freq='MS')+ pd.DateOffset(days=12), columns=['Date'])
	future_time_index = pd.DataFrame(pd.date_range(time_index['Date'].iloc[-1], periods=steps, freq='MS')+ pd.DateOffset(days=12), columns=['Date'])
	_list = []
	start = date(int(final_df['SHMIPREDICTOR'].unique()[-1][0:4]),int(final_df['SHMIPREDICTOR'].unique()[-1][4:6]),1)
	end = date(int(final_df['SHMIPREDICTOR'].unique()[-1][6:10]),int(final_df['SHMIPREDICTOR'].unique()[-1][10:12]),1)

	for i in range(steps):        
		days_in_month_start = calendar.monthrange(start.year,start.month)[1]
		start = start+timedelta(days=days_in_month_start)
		days_in_month_end = calendar.monthrange(end.year,end.month)[1]
		end = end+timedelta(days=days_in_month_end)
		_list.append(str(start.year)+"{:02d}".format(start.month)+str(end.year)+"{:02d}".format(end.month))
		
	future_time_index['SHMIPREDICTOR'] = pd.DataFrame(_list)

	predictor_ids = unique(final_df.loc[:, 'PREDICTOR'])
	#print(predictor_ids)
	#diagnosis_ids = unique(df.loc[:, diagnosis_ix])
	# group rows by chunk id
	whole_predicted = []
	whole_abnormal_list = []	
	for predictor_id in predictor_ids:
		selection = final_df.loc[:, 'PREDICTOR'] == predictor_id
		diagnosis_ids = unique(final_df.loc[selection, 'DIAGNOSIS_GROUP'])
		for diagnosis_id in diagnosis_ids:
			selection = (final_df.loc[:, 'PREDICTOR'] == predictor_id)&(final_df.loc[:, 'DIAGNOSIS_GROUP'] == diagnosis_id)
			if len(final_df.loc[selection, :]) == 11:
				history_data = final_df.loc[selection, :][['Date','PARAMETER_ESTIMATE']]
				history_data.set_index('Date', inplace=True)
				predicted,abnormal = fit_predict_refit(model,history_data,0,n_lag,steps,future_time_index,predictor_id,diagnosis_id)
				for item in predicted:
					whole_predicted.append(item)
				if len(abnormal) > 0:
					for item in abnormal:
						whole_abnormal_list.append(item)
			else:
				imputed = pd.merge(time_index,final_df.loc[selection, :],left_on='Date', right_on='Date',how='left')
				imputed.set_index('Date',inplace=True)
				imputed['PARAMETER_ESTIMATE'] = imputed['PARAMETER_ESTIMATE'].astype(float).interpolate(limit_direction ='forward')
				imputed['PARAMETER_ESTIMATE'] = imputed['PARAMETER_ESTIMATE'].astype(float).interpolate(limit_direction ='backward')
				history_data = imputed[['PARAMETER_ESTIMATE']]
				predicted,abnormal = fit_predict_refit(model,history_data,0,n_lag,steps,future_time_index,predictor_id,diagnosis_id)
				for item in predicted:
					whole_predicted.append(item)
				if len(abnormal) > 0:
					for item in abnormal:
						whole_abnormal_list.append(item)	
	predictors = pd.DataFrame(whole_predicted, columns=['Date','SHMIPREDICTOR','PREDICTOR','DIAGNOSIS_GROUP','PARAMETER_ESTIMATE'])
	predictors['INDICATOR_CODE'] = 'FUTURE'
	predictors['STANDARD_ERROR'] = 'FUTURE'
	predictors['WALD_CHI_SQUARE'] = 'FUTURE'
	predictors['WALD_DF'] = 'FUTURE'
	predictors['WALD_P_VALUE'] = 'FUTURE'
	today = date.today()
	predictors['PUBLISHED_DATE'] = today.strftime("%Y%m%d")
	predictors = predictors[['INDICATOR_CODE','DIAGNOSIS_GROUP','PREDICTOR','PARAMETER_ESTIMATE','STANDARD_ERROR','WALD_CHI_SQUARE','WALD_DF','WALD_P_VALUE','PUBLISHED_DATE','SHMIPREDICTOR']]
	abnormal_list = pd.DataFrame(whole_abnormal_list, columns=['Date','SHIMIPREDICTOR','Predictor','Diagnosis_Group','PRED_DIFF','Average_Diff','New_Pred','Pre_Predict'])
	return predictors,abnormal_list