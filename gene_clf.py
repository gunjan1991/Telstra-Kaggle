# -*- coding: utf-8 -*-
"""
Created on Fri Dec  9 09:36:23 2016

@author: GunjanPandya
"""

import numpy as np
from sklearn.metrics import log_loss
from sklearn.cross_validation import KFold

class my_classifier(object):
	'''Class: my_classifier'''
	# init
	def __init__(self, number_class, number_fold, number_seed):
		self.number_class = number_class
		self.number_fold = number_fold
		self.number_seed = number_seed
	# predict
	def predict(self, classifier, X_train, y_train, X_test, stage_val):
		np.random.seed(self.number_seed)
		n_train = X_train.shape[0]
		kf = KFold(n_train, n_folds=self.number_fold, shuffle=True)
		best_sc = []
		y_predict_sum = np.zeros((X_test.shape[0], self.number_class))
		if stage_val=='base':
			meta_feat = np.zeros((n_train+X_test.shape[0], self.number_class))
		i = 0
		for train, val in kf:
			i += 1
			print(i)
			X_train_new, X_vali, y_train_new, y_vali = X_train[train], X_train[val], y_train[train], y_train[val]
			## CV sets
			# train
			classifier.fit(X_train_new, y_train_new)
			curr_pred = classifier.predict_proba(X_vali)
			curr_best_sc = log_loss(y_vali, curr_pred)
			print(curr_best_sc)
			best_sc += [curr_best_sc]
			# predict
			if stage_val=='base':
				meta_feat[val, :] = curr_pred
			else:
				y_predict = classifier.predict_proba(X_test)
				y_predict_sum = y_predict_sum+y_predict
		print(np.mean(best_sc), np.std(best_sc))
		## test set
		if stage_val=='base':
			# train
			classifier.fit(X_train, y_train)
			# predict
			meta_feat[n_train:, :] = classifier.predict_proba(X_test)
			return meta_feat
		else:
			y_predict = y_predict_sum/self.number_fold
			return y_predict