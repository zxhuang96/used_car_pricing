#!/usr/bin/env python
# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np
import time
import xgboost as xgb
import utils
from sklearn.model_selection import StratifiedKFold, KFold, RepeatedKFold
from sklearn import metrics


def train_model_regression(X, X_test, y, params, folds, eval_metric='mae', columns=None,  verbose=500, early_stopping_rounds=200):
    """
    A function to train a variety of regression models.
    Returns dictionary with oof predictions, test predictions, scores and, if necessary, feature importances.

    :params: X - training data, can be pd.DataFrame or np.ndarray (after normalizing)
    :params: X_test - test data, can be pd.DataFrame or np.ndarray (after normalizing)
    :params: y - target
    :params: folds - folds to split data
    :params: model_type - type of model to use
    :params: eval_metric - metric to use
    :params: columns - columns to use. If None - use all columns
    :params: plot_feature_importance - whether to plot feature importance of LGB
    :params: model - sklearn model, works only for "sklearn" model type

    """
    columns = X.columns if columns is None else columns
    X_test = X_test[columns]

    # to set up scoring parameters
    metrics_dict = {'mae': {'lgb_metric_name': 'mae',
                            'catboost_metric_name': 'MAE',
                            'sklearn_scoring_function': metrics.mean_absolute_error},
                    # 'group_mae': {'lgb_metric_name': 'mae',
                    #               'catboost_metric_name': 'MAE',
                    #               'scoring_function': utils.group_mean_log_mae},
                    'mse': {'lgb_metric_name': 'mse',
                            'catboost_metric_name': 'MSE',
                            'sklearn_scoring_function': metrics.mean_squared_error}
                    }
    result_dict = {}

    # out-of-fold predictions on train data
    oof = np.zeros(len(X))

    # averaged predictions on train data
    prediction = np.zeros(len(X_test))

    # list of scores on folds
    scores = []
    feature_importance = pd.DataFrame()

    # split and train on folds
    for fold_n, (train_index, valid_index) in enumerate(folds.split(X)):
        print(f'Fold {fold_n + 1} started at {time.ctime()}')
        if type(X) == np.ndarray:
            X_train, X_valid = X[columns][train_index], X[columns][valid_index]
            y_train, y_valid = y[train_index], y[valid_index]
        else:
            X_train, X_valid = X[columns].iloc[train_index], X[columns].iloc[valid_index]
            y_train, y_valid = y.iloc[train_index], y.iloc[valid_index]

        train_data = xgb.DMatrix(data=X_train, label=y_train, feature_names=X.columns)
        valid_data = xgb.DMatrix(data=X_valid, label=y_valid, feature_names=X.columns)

        watchlist = [(train_data, 'train'), (valid_data, 'valid_data')]
        model = xgb.train(dtrain=train_data, num_boost_round=200, evals=watchlist, early_stopping_rounds=early_stopping_rounds, verbose_eval=verbose, params=params)
        y_pred_valid = model.predict(xgb.DMatrix(X_valid, feature_names=X.columns),
                                     ntree_limit=model.best_ntree_limit)
        y_pred = model.predict(xgb.DMatrix(X_test, feature_names=X.columns), ntree_limit=model.best_ntree_limit)

        oof[valid_index] = y_pred_valid.reshape(-1, )
        if eval_metric != 'group_mae':
            scores.append(metrics_dict[eval_metric]['sklearn_scoring_function'](y_valid, y_pred_valid))
        else:
            scores.append(metrics_dict[eval_metric]['scoring_function'](y_valid, y_pred_valid, X_valid['type']))

        prediction += y_pred


    prediction /= folds.n_splits

    print('CV mean score: {0:.4f}, std: {1:.4f}.'.format(np.mean(scores), np.std(scores)))

    result_dict['oof'] = oof
    result_dict['prediction'] = prediction
    result_dict['scores'] = scores

    return result_dict

features_colname = ['year','odometer','income_rank','median_income',
       'cylinders_onehot_1', 'cylinders_onehot_2', 'cylinders_onehot_3',
       'cylinders_onehot_4', 'cylinders_onehot_5', 'cylinders_onehot_6',
       'cylinders_onehot_7', 'cylinders_onehot_8', 'type_onehot_1',
       'type_onehot_2', 'type_onehot_3', 'type_onehot_4', 'type_onehot_5',
       'type_onehot_6', 'type_onehot_7', 'type_onehot_8', 'type_onehot_9',
       'type_onehot_10', 'type_onehot_11', 'type_onehot_12', 'type_onehot_13',
       'type_onehot_14','drive_onehot_1', 'drive_onehot_2', 'drive_onehot_3',
       'drive_onehot_4', 'title_status_mean_price', 'title_status_median_price', 'title_status_price_std','manufacturer_mean_price','manufacturer_median_price',
       'manufacturer_price_std','manufacturer_make_mean_price','manufacturer_make_median_price', 'manufacturer_make_price_std','transmission_encoded', 'fuel_encoded', 'year_mean_price', 'year_median_price', 'year_price_std']



file_folder = '/root/4741project'
data_all_IS = pd.read_csv(f'{file_folder}/IS/data_with_grouping_operations_IS.csv')
data_all_OS = pd.read_csv(f'{file_folder}/OS/data_with_grouping_operations_OS.csv')

X = data_all_IS[features_colname]
# scaling
zscore_cols = ['odometer', 'median_income', 'manufacturer_mean_price', 'manufacturer_median_price',
       'manufacturer_price_std', 'manufacturer_make_mean_price', 'manufacturer_make_median_price', 'manufacturer_make_price_std', 'title_status_mean_price', 'title_status_median_price', 'title_status_price_std','year_mean_price', 'year_median_price', 'year_price_std']
positive_cols = ['income_rank']

X = utils.scaling(X,X,zscore_cols,positive_cols)

X_os = data_all_OS[features_colname]

#scaling
X_os = utils.scaling(X_os,X,zscore_cols,positive_cols)

y = data_all_IS['price']

n_fold = 5
folds = KFold(n_splits=n_fold, shuffle=True, random_state=11)
params_xgb = {'num_leaves': 128, # adjust
          'min_child_weight': 10, # adjust 
          'max_depth': 6, # adjust
          'eta': 0.2,
          'gamma':0.1,
          'objective': 'reg:squarederror',
          #"boosting_type": "gbdt",
          #"subsample_freq": 1,
          "subsample": 0.9,
          #"bagging_seed": 11,
          "metric": 'mae',
          "verbosity": 3,
          'alpha': 1,
          'lambda': 1,
          'silent': 1, 
          'colsample_bytree': 1.0
         }

result_dict_xgb = train_model_regression(X=X, X_test=X_os, y=y, params=params_xgb, folds=folds,  eval_metric='mse', verbose=300, early_stopping_rounds=100)

print("Writing files...")
out_xgb = pd.DataFrame([])
out_xgb['OS_predicted_price'] = result_dict_xgb['prediction']
out_xgb.to_csv(f'{file_folder}/output/OS_prediction_xgb_2.csv', index=False)

is_xgb = pd.DataFrame([])
is_xgb['IS_predicted_price'] = result_dict_xgb['oof']
is_xgb.to_csv(f'{file_folder}/output/IS_prediction_xgb_2.csv', index=False)
print("len(OS)",len(out_xgb))
print("len(IS)",len(is_xgb))

print(f'Completed at {time.ctime()}')
