# -*- coding: utf-8 -*-
"""
Created on Fri Dec  9 09:48:27 2016

@author: GunjanPandya
"""

import numpy as np
import pandas as pd
import datetime as dt
from scipy.stats import skew

def feature_engg():
	#Function for feature engineering.
	## load data
	df_train = pd.read_csv('C:\\Users\\GunjanPandya\\Downloads\\Telstra\\Data\\train.csv')
	df_test = pd.read_csv('C:\\Users\\GunjanPandya\\Downloads\\Telstra\\Data\\test.csv')
	df_event_type = pd.read_csv('C:\\Users\\GunjanPandya\\Downloads\\Telstra\\Data\\event_type.csv')
	df_log_feature = pd.read_csv('C:\\Users\\GunjanPandya\\Downloads\\Telstra\\Data\\log_feature.csv')
	df_resource_type = pd.read_csv('C:\\Users\\GunjanPandya\\Downloads\\Telstra\\Data\\resource_type.csv')
	df_severity_type = pd.read_csv('C:\\Users\\GunjanPandya\\Downloads\\Telstra\\Data\\severity_type.csv')
	## pre-processing
	y = df_train['fault_severity'].values
	num_class = max(y)+1
	df_train.drop('fault_severity', axis=1, inplace=True)
	ids = df_test['id'].values
	n_train = df_train.shape[0]
	df_all = pd.concat([df_train, df_test], axis=0, ignore_index=True)
	df_severity_type_all = df_severity_type.merge(df_all, on='id')
	loc = 'location 1'
	time = []
	norm_time =[]
	st = 0
	t = 0
	for i in range(df_severity_type_all.shape[0]):
		if df_severity_type_all['location'][i]==loc:
			t += 1
			time += [t]
		else:
			norm_time += [float(time[j])/t for j in range(st, i)]
			st = i
			loc = df_severity_type_all['location'][i]
			t = 1
			time += [t]
	norm_time += [float(time[j])/t for j in range(st, i+1)]
	df_time = pd.DataFrame({'id': df_severity_type['id'], 'time': time, 'norm_time': norm_time})
	
	# feature for location
	loc_diff = np.setdiff1d(df_test['location'].values, df_train['location'].values)
	df_loc_table = pd.pivot_table(df_all, index='id', columns='location', aggfunc=len, 
		fill_value=0)
	# drop the locations contained in df_test but not in df_train
	df_loc_table.drop(loc_diff, axis=1, inplace=True)
    # high risk or low risk?
	loc_freq = df_all['location'].value_counts()
	loc_freq = pd.DataFrame({'loc_freq': loc_freq})
	loc_freq.index.name = 'location'
	df_all = df_all.join(loc_freq, on='location')
	# location as a numeric variable
	func = lambda x: int(x.strip('location '))
	df_all['location'] = df_all['location'].apply(func)
	# event_type
	df_event_type_table = pd.pivot_table(df_event_type, index='id', columns='event_type', aggfunc=len,
		fill_value=0)
	# max, min, std, skew
	grouped = df_event_type[['id', 'event_type']].groupby('id')
	
	func = lambda x: len(np.unique(x))
	df_event_type_num = grouped.aggregate(func)
	func = lambda x: max(x.apply(lambda x: x.strip('event_type ')).astype(int)) 
	df_event_type_max = grouped.aggregate(func)
	df_event_type_max.rename(columns={'event_type': 'eve_max'}, inplace=True)
	func = lambda x: min(x.apply(lambda x: x.strip('event_type ')).astype(int)) 
	df_event_type_min = grouped.aggregate(func)
	df_event_type_min.rename(columns={'event_type': 'eve_min'}, inplace=True)
	func = lambda x: np.std(x.apply(lambda x: x.strip('event_type ')).astype(int)) 
	df_event_type_std = grouped.aggregate(func)
	df_event_type_std.rename(columns={'event_type': 'eve_std'}, inplace=True)
	func = lambda x: skew(x.apply(lambda x: x.strip('event_type ')).astype(int)) 
	df_event_type_skew = grouped.aggregate(func)
	df_event_type_skew.rename(columns={'event_type': 'eve_skew'}, inplace=True)
	# log_feature
	df_log_feature_table = pd.pivot_table(df_log_feature, values='volume', index='id', columns='log_feature', aggfunc=np.sum,
		fill_value=0)
	
	df_log_feature_vol_sum = pd.DataFrame(data={'log_vol_sum': df_log_feature_table.sum(axis=1)})
	# number of log_features
	grouped = df_log_feature[['id', 'log_feature']].groupby('id')
	func = lambda x: len(np.unique(x))
	df_log_feature_feat_num = grouped.aggregate(func)
	df_log_feature_feat_num.rename(columns={'log_feature': 'log_feat_num'}, inplace=True)
	# max, min, std, skew
	func = lambda x: max(x.apply(lambda x: x.strip('feature ')).astype(int))
	df_log_feature_feat_max = grouped.aggregate(func)
	df_log_feature_feat_max.rename(columns={'log_feature': 'log_feat_max'}, inplace=True)
	func = lambda x: min(x.apply(lambda x: x.strip('feature ')).astype(int))
	df_log_feature_feat_min = grouped.aggregate(func)
	df_log_feature_feat_min.rename(columns={'log_feature': 'log_feat_min'}, inplace=True)
	func = lambda x: np.std(x.apply(lambda x: x.strip('feature ')).astype(int))
	df_log_feature_feat_std = grouped.aggregate(func)
	df_log_feature_feat_std.rename(columns={'log_feature': 'log_feat_std'}, inplace=True)
	func = lambda x: skew(x.apply(lambda x: x.strip('feature ')).astype(int))
	df_log_feature_feat_skew = grouped.aggregate(func)
	df_log_feature_feat_skew.rename(columns={'log_feature': 'log_feat_skew'}, inplace=True)
   
	grouped = df_log_feature[['id', 'volume']].groupby('id')
	func = lambda x: len(np.unique(x))
	df_log_feature_vol_num = grouped.aggregate(func)
	df_log_feature_vol_num.rename(columns={'volume': 'vol_num'}, inplace=True)
	
	func = lambda x: min(x)
	df_log_feature_vol_min = grouped.aggregate(func)
	df_log_feature_vol_min.rename(columns={'volume': 'vol_min'}, inplace=True)
    # high freq or low freq
	grouped = df_log_feature[['log_feature', 'volume']].groupby('log_feature')
	func = lambda x: np.sum(x)
	log_feat_freq = grouped.aggregate(func)
	log_feat_freq.rename(columns={'volume': 'log_feat_freq'}, inplace=True)
	df_log_feature_feat_freq = df_log_feature.join(log_feat_freq, on='log_feature')
	
	grouped = df_log_feature_feat_freq[['id', 'log_feat_freq']].groupby('id')
	func = lambda x: max(x)
	df_log_feature_feat_freq_max = grouped.aggregate(func)
	df_log_feature_feat_freq_max.rename(columns={'log_feat_freq': 'log_feat_freq_max'}, inplace=True)
	func = lambda x: min(x)
	df_log_feature_feat_freq_min = grouped.aggregate(func)
	df_log_feature_feat_freq_min.rename(columns={'log_feat_freq': 'log_feat_freq_min'}, inplace=True)
	# resource_type
	df_resource_type_table = pd.pivot_table(df_resource_type, index='id', columns='resource_type', aggfunc=len,
		fill_value=0)
	# num, std
	# max, min will not work since the range of resource_type is too small
	grouped = df_resource_type[['id', 'resource_type']].groupby('id')
	func = lambda x: len(np.unique(x))
	df_resource_type_num = grouped.aggregate(func)
	df_resource_type_num.rename(columns={'resource_type': 'res_num'}, inplace=True)
	func = lambda x: np.std(x.apply(lambda x: x.strip('resource_type ')).astype(int))
	df_resource_type_std = grouped.aggregate(func)
	df_resource_type_std.rename(columns={'resource_type': 'res_std'}, inplace=True)
	# severity_type - categorical. No ordering.
	df_severity_type_table = pd.pivot_table(df_severity_type, index='id', columns='severity_type', aggfunc=len,
		fill_value=0)
	# checking interactions
	# sum of volume for each location
	df_log_feature_loc = df_log_feature.merge(df_all[['id', 'location']], on='id')
	grouped = df_log_feature_loc[['volume', 'location']].groupby('location')
	func = lambda x: np.sum(x)
	df_loc_log_vol_sum = grouped.aggregate(func)
	df_loc_log_vol_sum.rename(columns={'volume': 'loc_log_vol_sum'}, inplace=True)
	# num, max of log_features for each location
	grouped = df_log_feature_loc[['log_feature', 'location']].groupby('location')
	func = lambda x: len(np.unique(x))
	df_loc_log_feat_num = grouped.aggregate(func)
	df_loc_log_feat_num.rename(columns={'log_feature': 'loc_log_feat_num'}, inplace=True)
	func = lambda x: max(x.apply(lambda x: x.strip('feature ')).astype(int))
	df_loc_log_feat_max = grouped.aggregate(func)
	df_loc_log_feat_max.rename(columns={'log_feature': 'loc_log_feat_max'}, inplace=True)
    # max of event_types for each location
	df_event_type_loc = df_event_type.merge(df_all[['id', 'location']], on='id')
	grouped = df_event_type_loc[['event_type', 'location']].groupby('location')
	func = lambda x: max(x.apply(lambda x: x.strip('event_type ')).astype(int)) 
	df_loc_eve_max = grouped.aggregate(func)
	df_loc_eve_max.rename(columns={'event_type': 'loc_eve_max'}, inplace=True)
    # num of resource_types for each location
	df_resource_type_loc = df_resource_type.merge(df_all[['id', 'location']], on='id')
	grouped = df_resource_type_loc[['resource_type', 'location']].groupby('location')
	func = lambda x: len(np.unique(x))
	df_loc_res_num = grouped.aggregate(func)
	df_loc_res_num.rename(columns={'resource_type': 'loc_res_num'}, inplace=True)
    # num of log_feat_num for each location
	df_lfn_loc = df_all[['id', 'location']].join(df_log_feature_feat_num, on='id')
	grouped = df_lfn_loc[['location', 'log_feat_num']].groupby('location')
	func = lambda x: len(np.unique(x))
	df_loc_lfn_num = grouped.aggregate(func)
	df_loc_lfn_num.rename(columns={'log_feat_num': 'loc_lfn_num'}, inplace=True)
	# freq of (location, log_feat_max)
	df_loc_lfm = df_all[['id', 'location']].join(df_log_feature_feat_max, on='id')
	df_loc_lfm_freq = pd.DataFrame(data={'loc_lfm_freq': df_loc_lfm.groupby(['location', 'log_feat_max']).size()})
	df_loc_lfm_freq = df_loc_lfm.join(df_loc_lfm_freq, on=['location', 'log_feat_max'])[['id', 'loc_lfm_freq']]
    # freq of (location, log_feat_num)
	df_loc_lfn = df_all[['id', 'location']].join(df_log_feature_feat_num, on='id')
	df_loc_lfn_freq = pd.DataFrame(data={'loc_lfn_freq': df_loc_lfn.groupby(['location', 'log_feat_num']).size()})
	df_loc_lfn_freq = df_loc_lfn.join(df_loc_lfn_freq, on=['location', 'log_feat_num'])[['id', 'loc_lfn_freq']]
	# combine
	df_data = df_all.join(df_loc_log_vol_sum, on='location')
	df_data = df_data.join(df_loc_log_feat_num, on='location')
	df_data = df_data.join(df_loc_log_feat_max, on='location')
	df_data = df_data.join(df_loc_eve_max, on='location')
	df_data = df_data.join(df_loc_res_num, on='location')
	df_data = df_data.join(df_loc_lfn_num, on='location')
	df_data = df_data.join(df_event_type_max, on='id')
	df_data = df_data.join(df_event_type_min, on='id')
	df_data = df_data.join(df_event_type_std, on='id')
	df_data = df_data.join(df_event_type_skew, on='id')
	df_data = df_data.join(df_log_feature_vol_sum, on='id')
	df_data = df_data.join(df_log_feature_vol_num, on='id')
	df_data = df_data.join(df_log_feature_vol_min, on='id')
	df_data = df_data.join(df_log_feature_feat_num, on='id')
	df_data = df_data.join(df_log_feature_feat_max, on='id')
	df_data = df_data.join(df_log_feature_feat_min, on='id')
	df_data = df_data.join(df_log_feature_feat_std, on='id')
	df_data = df_data.join(df_log_feature_feat_skew, on='id')
	df_data = df_data.join(df_log_feature_feat_freq_max, on='id')
	df_data = df_data.join(df_log_feature_feat_freq_min, on='id')
	df_data = df_data.join(df_resource_type_num, on='id')
	df_data = df_data.join(df_resource_type_std, on='id')
	df_data = df_data.merge(df_loc_lfm_freq, on='id')
	df_data = df_data.merge(df_loc_lfn_freq, on='id')
	df_data = df_data.merge(df_time, on='id')
	df_data = df_data.join(df_event_type_table, on='id')
	df_data = df_data.join(df_log_feature_table, on='id')
	df_data = df_data.join(df_resource_type_table, on='id')
	n_feat2 = df_data.shape[1]-1
	df_data = df_data.join(df_severity_type_table, on='id')
	n_feat = df_data.shape[1]-1
	df_data = df_data.join(df_loc_table, on='id')
	# check NaN
	df_data.isnull().any().any()
	# drop id
	df_data_no_id = df_data.drop('id', axis=1, inplace=False)
	X_all = df_data_no_id.values
	return (X_all, y, num_class, n_train, n_feat, n_feat2, ids, df_data['location'].values)
 
if __name__=="__main__":
	(X_all, y, num_class, n_train, n_feat, n_feat2, ids, df_data['location'].values) = feature_engg()