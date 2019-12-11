#!/usr/bin/env python
# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np
import time
import catboost
from catboost import CatBoostRegressor
from sklearn.model_selection import StratifiedKFold, KFold, RepeatedKFold
import sys
sys.path.append("..")
import utils
from sklearn import metrics


def train_model_regression(X, X_test, y, params, folds, eval_metric='mae', columns=None,  iterations=200):
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

        model = CatBoostRegressor(iterations=iterations, eval_metric=metrics_dict[eval_metric]['catboost_metric_name'],
                                  **params,
                                  loss_function=metrics_dict[eval_metric]['catboost_metric_name'])
        model.fit(X_train, y_train, eval_set=(X_valid, y_valid), cat_features=[], use_best_model=True,
                  verbose=500)

        y_pred_valid = model.predict(X_valid)
        y_pred = model.predict(X_test)

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
       'type_onehot_14','drive_onehot_1', 'drive_onehot_2', 'drive_onehot_3','condition_mean_price','condition_median_price', 'condition_price_std', 
       'drive_onehot_4', 'title_status_mean_price', 'title_status_median_price', 'title_status_price_std','manufacturer_mean_price','manufacturer_median_price',
       'manufacturer_price_std','manufacturer_make_mean_price','manufacturer_make_median_price', 'manufacturer_make_price_std','transmission_encoded', 'fuel_encoded', 'year_mean_price', 'year_median_price', 'year_price_std']


file_folder = '../data'
data_all_IS = pd.read_csv(f'{file_folder}/IS/data_with_grouping_operations_IS.csv')
data_all_OS = pd.read_csv(f'{file_folder}/OS/data_with_grouping_operations_OS.csv')
X = data_all_IS[features_colname]
# scaling
zscore_cols = ['odometer', 'median_income', 'manufacturer_mean_price', 'manufacturer_median_price','condition_mean_price','condition_median_price', 'condition_price_std',
       'manufacturer_price_std', 'manufacturer_make_mean_price', 'manufacturer_make_median_price', 'manufacturer_make_price_std', 'title_status_mean_price', 'title_status_median_price', 'title_status_price_std','year_mean_price', 'year_median_price', 'year_price_std']
positive_cols = ['income_rank']

X = utils.scaling(X,data_all_IS,zscore_cols,positive_cols)

X_os = data_all_OS[features_colname]

# scaling
X_os = utils.scaling(X_os,data_all_IS,zscore_cols,positive_cols)

y = data_all_IS['price']

n_fold = 5
folds = KFold(n_splits=n_fold, shuffle=True, random_state=11)
params_cat = {'depth': 10,
          'learning_rate': 0.2,
          "random_seed": 11,
          'reg_lambda': 50,
          'thread_count': -1,
          #'l2_leaf_reg': 5
         }

result_dict_cat = train_model_regression(X=X, X_test=X_os, y=y, params=params_cat, folds=folds,  eval_metric='mae', iterations=1000)

print("Writing files...")
out_cat = pd.DataFrame([])
out_cat['OS_predicted_price'] = result_dict_cat['prediction']
out_cat.to_csv(f'{file_folder}/output/OS_prediction_cat_10,50.csv', index=False)

is_cat = pd.DataFrame([])
is_cat['IS_predicted_price'] = result_dict_cat['oof']
is_cat.to_csv(f'{file_folder}/output/IS_prediction_cat_10,50.csv', index=False)
print("len(OS)",len(out_cat))
print("len(IS)",len(is_cat))

print(f'Completed at {time.ctime()}')
print("INFO:",'depth',params_cat['depth'],";","learning_rate",params_cat['learning_rate'],";",'reg_lambda',params_cat['reg_lambda'])
