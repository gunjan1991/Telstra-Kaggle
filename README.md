# Telstra-Kaggle

# Feature engineering

As always, feature engineering is the first and the most important step in participating a Kaggle competition. The most important three features are:

Treating the location as numeric. The intuition is that records with similar location numbers probably have similar fault severity. It could be that these locations are close to each other and thus their behaviors are similar. Including this feature reduces my mlogloss by ~0.02.
Adding the frequency of each location that appears in both training and testing data sets as a feature. This works especially well for high cardinality categorical variables. Including this feature reduces my mlogloss by ~0.01. The variable log feature can be processed following the same idea.

The time information, which gives the biggest reduction as ~0.06. It is very clear from the description of the competition that The goal of the problem is to predict Telstra network's fault severity at a time at a particular location based on the log data available. Another interesting finding is that after joining severity_type.csv and the data frame (which is concatenated by train.csv and test.csv) on id, the order of the records contains time information (Check sev_loc.csv in the repo). For each location, the neighboring records tend to have the same fault severity. This implies these records are arranged in the order of time. It is reasonable that the network continues its status for a while and then changes to another status. There are two ways to encode this time information. One is for each location, use the row number, which starts from 1 to the total number of rows for that location. The other is to normalize the time information such that they fall within [0, 1]. I have tried to use either or both of them, and the predictions can be bagged in the end.

There are other features I have tried, and they are all written in get_raw_feat.py. Basically, they are one-hot-encoding, using various summary statistics to reduce one-to-many relationship to one-to-one, and two-way or three-way interaction among multiple variables. I didn't include the one-hot-encoding of location directly as a feature given the number of unique locations is very large. Tree-based classifiers such as Random Forests and Gradient Tree Boosting are not good at handling this huge sparse feature matrix. I encoded this sparse feature matrix by stacking. Specifically, I used Logistic regression to fit the data with this sparse matrix as predictors. The (class) predictions based on the fitted Logistic regression model are used as meta features.

# Classifier

I used Xgboost, which seems to work better than the other classifiers I have tried. The best single model using 10-fold cross-validation averaging gives scores: 0.41351 (Public LB) and 0.41136 (Private LB).

my_xgb = xgb_clf.my_xgb(obj='multi:softprob', eval_metric='mlogloss', num_class=num_class, nthread=20, silent=1, eta=0.02, colsample_bytree=0.6, subsample=0.9, max_depth=8, max_delta_step=1, gamma=0.1, alpha=0, param_lambda=1, n_fold=10, seed=0)

# Ensembling

I am not experienced in ensembling. So what I did is just averaging Xgboost's with colsample_bytree=0.5, 0.6 and feature sets including time or time_norm or both of them. This final model gives scores: 0.41038 (Public LB) and 0.40742 (Private LB). But unfortunately I didn't choose it as my final submissions since its public score is worse than the other model.
