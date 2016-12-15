# -*- coding: utf-8 -*-
"""
Created on Fri Dec  9 13:32:36 2016

@author: GunjanPandya
"""

import numpy as np
import xgboost as xgb
from sklearn.cross_validation import KFold

#xgboost classifier=
class xgboost_clf(object):
	
	def __init__(self, obj, eval_met, number_class, nthread, silent, eta, colsample_bytree, subsample, max_depth, max_delta_step, gamma, alpha, param_lambda, n_fold, seed_value):
		self.obj = obj
		self.eval_met = eval_met
		self.number_class = number_class
		self.nthread = nthread
		self.silent = silent
		self.eta = eta
		self.colsample_bytree = colsample_bytree
		self.subsample = subsample
		self.max_depth = max_depth
		self.max_delta_step = max_delta_step
		self.gamma = gamma
		self.alpha = alpha
		self.param_lambda = param_lambda
		self.n_fold = n_fold
		self.seed_value = seed_value
	# predict
	def predict(self, X_train, y_train, X_test, stage_val):
		np.random.seed(self.seed)
		n_train = X_train.shape[0]
		kf = KFold(n_train, n_folds=self.n_fold, shuffle=True)
		param = {}
		param['objective'] = self.obj
		param['eval_met'] = self.eval_met
		param['number_class'] = self.number_class
		param['nthread'] = self.nthread
		param['silent'] = self.silent
		param['eta'] = self.eta
		param['colsample_bytree'] = self.colsample_bytree
		param['subsample'] = self.subsample
		param['max_depth'] = self.max_depth
		param['max_delta_step'] = self.max_delta_step
		param['gamma'] = self.gamma
		param['alpha'] = self.alpha
		param['lambda'] = self.param_lambda
		num_round = 10000
		best_score = []
		best_iter = []
		y_pred_sum = np.zeros((X_test.shape[0], self.num_class))
		if stage_val=='base':
			meta_feat = np.zeros((n_train+X_test.shape[0], self.num_class))
		xg_test = xgb.DMatrix(X_test)
		i = 0
		for train, val in kf:
			i += 1
			print(i)
			X_train_new, X_vali, y_train_new, y_vali = X_train[train], X_train[val], y_train[train], y_train[val]
			xg_train = xgb.DMatrix(X_train_new, y_train_new)
			xg_val = xgb.DMatrix(X_vali, y_vali)
			evallist  = [(xg_train,'train'), (xg_val,'eval')]
			## CV sets
			# train
			bst = xgb.train(param, xg_train, num_round, evallist, early_stopping_rounds=100)
			best_score += [bst.best_score]
			best_iter += [bst.best_iteration]
			# predict
			if stage_val=='base':
				meta_feat[val, :] = bst.predict(xg_val, ntree_limit=bst.best_iteration)
			else:
				y_predict = bst.predict(xg_test, ntree_limit=bst.best_iteration)
				y_pred_sum = y_pred_sum+y_predict
		print(np.mean(best_score), np.std(best_score))
		## test set
		if stage_val=='base':
			# train
			xg_train = xgb.DMatrix(X_train, y_train)
			evallist  = [(xg_train,'train')]
			bst = xgb.train(param, xg_train, int(np.mean(best_iter)), evallist)
			# predict
			meta_feat[n_train:, :] = bst.predict(xg_test)
			return meta_feat
		else:
			y_predict = y_pred_sum/self.n_fold
			return y_predict