#!/usr/bin/env python
# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np
import time
import lightgbm as lgb
from sklearn.model_selection import StratifiedKFold, KFold, RepeatedKFold
import utils
from sklearn import metrics


def train_model_regression(X, X_test, y, params, folds, eval_metric='mse', columns=None,  verbose=500, early_stopping_rounds=200, n_estimators=1500):
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

        model = lgb.LGBMRegressor(**params, n_estimators=n_estimators, n_jobs=-1)
        model.fit(X_train, y_train,
                  eval_set=[(X_train, y_train), (X_valid, y_valid)],
                  eval_metric=metrics_dict[eval_metric]['lgb_metric_name'],
                  verbose=verbose, early_stopping_rounds=early_stopping_rounds)

        y_pred_valid = model.predict(X_valid)
        y_pred = model.predict(X_test, num_iteration=model.best_iteration_)

        booster = model.booster_
        importance = booster.feature_importance(importance_type='split')
        feature_name = booster.feature_name()
        # feature_importance = pd.DataFrame({'feature_name': feature_name, 'importance': importance})
        # feature_importance.to_csv(f'{file_folder}/feature_importance_{fold_n + 1}.csv', index=False)

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

features_colname=['year','odometer','income_rank','median_income',
       'cylinders_onehot_1', 'cylinders_onehot_2', 'cylinders_onehot_3',
       'cylinders_onehot_4', 'cylinders_onehot_5', 'cylinders_onehot_6',
       'cylinders_onehot_7', 'cylinders_onehot_8', 'type_onehot_1',
       'type_onehot_2', 'type_onehot_3', 'type_onehot_4', 'type_onehot_5',
       'type_onehot_6', 'type_onehot_7', 'type_onehot_8', 'type_onehot_9',
       'type_onehot_10', 'type_onehot_11', 'type_onehot_12', 'type_onehot_13',
       'type_onehot_14','drive_onehot_1', 'drive_onehot_2', 'drive_onehot_3',
       'drive_onehot_4', 'title_status_mean_price','manufacturer_mean_price','manufacturer_make_mean_price','transmission_encoded', 'fuel_encoded']


file_folder = '/root/4741project'
data_all_IS = pd.read_csv(f'{file_folder}/IS/data_with_grouping_operations_IS.csv')
data_all_OS = pd.read_csv(f'{file_folder}/OS/data_with_grouping_operations_OS.csv')
X = data_all_IS[features_colname]
# scaling
zscore_cols=['odometer','median_income','manufacturer_mean_price','manufacturer_make_mean_price']
positive_cols=['income_rank']

X = utils.scaling(X,X,zscore_cols,positive_cols)

X_os = data_all_OS[features_colname]

#scaling
X_os = utils.scaling(X_os,X,zscore_cols,positive_cols)

y = data_all_IS['price']

n_fold = 5
folds = KFold(n_splits=n_fold, shuffle=True, random_state=11)
params_lgb = {'num_leaves': 64,
          'min_child_samples': 40,
          'objective': 'regression',
          'max_depth': 6,
          'learning_rate': 0.25,
          "boosting_type": "gbdt",
          "subsample_freq": 1,
          "subsample": 0.9,
          "bagging_seed": 11,
          "metric": 'mse',
          "verbosity": -1,
          'reg_alpha': 0.3,
          'reg_lambda': 0.3,
          'colsample_bytree': 1.0,
         }


result_dict_lgb = train_model_regression(X=X, X_test=X_os, y=y, params=params_lgb, folds=folds,  eval_metric='mse', verbose=500, early_stopping_rounds=200, n_estimators=1000)

print("Writing files...")
out_lgb = pd.DataFrame([])
out_lgb['OS_predicted_price'] = result_dict_lgb['prediction']
out_lgb.to_csv(f'{file_folder}/output/OS_prediction_lgb_1.csv', index=False)

is_lgb = pd.DataFrame([])
is_lgb['IS_predicted_price'] = result_dict_lgb['oof']
is_lgb.to_csv(f'{file_folder}/output/IS_prediction_lgb_1.csv', index=False)
print("len(OS)",len(out_lgb))
print("len(IS)",len(is_lgb))

print(f'Completed at {time.ctime()}')
print("INFO:","num_leaves",params_lgb['num_leaves'],";","min_child_samples",params_lgb['min_child_samples'],";",'max_depth',params_lgb['max_depth'])






