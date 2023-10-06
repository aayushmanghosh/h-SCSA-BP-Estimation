'''Author: Aayushman Ghosh
   Department of Electrical and Computer Engineering
   University of Illinois Urbana-Champaign
   (aghosh14@illinois.edu)
   
   Version: v1.0
''' 

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import catboost
import lightgbm as lgb
import xgboost
from xgboost import XGBRegressor
from catboost import CatBoostRegressor
from sklearn.svm import SVR
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import RandomizedSearchCV
from sklearn.model_selection import GridSearchCV
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import r2_score
from sklearn import metrics
from scipy import stats
from scipy.stats import zscore
from scipy.stats import mode
from scipy.stats import pearsonr
from pprint import pprint
import time
from hyperopt import fmin, tpe, hp, anneal, Trials
from hyperopt.pyll.stochastic import sample

scaler = StandardScaler()

# Declaring Global variables
outer_M = list()
outer_SD = list()
outer_err_five = list()
outer_err_fifeteen = list()
outer_err_ten = list()
outer_SDAE = list()
outer_MAE = list()

# Loading the necessary metadata to perform analysis on. -- Subject to change depending on the experiment
# General rule for naming convention: (Dataset_name)-(Number-of-features)-(Type, eg. Noisy/Normal/PCA/ICA/SCSA)
expt = r'\MIMIC-II-21-Normal.csv' 
absolute_path = os.path.dirname(os.path.abspath('__file__'))
file_dir = os.path.join(absolute_path, expt) 
data = pd.read_csv(file_dir, header=0)
data.dropna()
data.drop_duplicates(inplace=True)
data.loc[:,:] = data[data['fast_SI_diff']<0]
data['map'] = (1/3)*data.sbp + (2/3)*data.dbp
data = data[(data.sbp>80) & (data.sbp<200) & (data.dbp>60) & (data.dbp<120)] 

X = data.iloc[:,:-3]
y = data.loc[:,'sbp'] # Subject to change depending on what results you prefer (Choice: sbp, dbp, map)
X_norm = scaler.fit_transform(X)
y_norm = scaler.fit_transform(y.values.reshape(-1,1))


# Support Vector Machine (SVM) Evaluation 
# Configuring the bayesian function to perform the hyperparameter optimization

def bayesian_hp_svr(cv, X, y):
    # the function gets a set of variable parameters in "params" and then conduct the cross validation with the outer cv_folds.
    # the params needs to be updated with each set of evaluation and for different algorithms (SVR, CatBoost, XGBoost, LightGBM)
    params = {'C': int(params['C']), 
              'kernel': params['kernel'],
              'epsilon': params['epsilon'],
              'gamma': int(params['gamma']),
              'tol': params['tol']}
    
    # we use this params to create a new SVR
    model = SVR(random_state=42, **params)
    score = -cross_val_score(model, X, y, cv=cv, scoring="neg_mean_squared_error", n_jobs=-1).mean()

    return score

# Configuring the nested-cross validation procedure
cv_outer = KFold(n_splits=10, shuffle=True, random_state=1)
for train_ix, test_ix in cv_outer.split(X):
     X_train, X_test = X_norm[train_ix, :], X_norm[test_ix, :]
     y_train, y_test = y_norm[train_ix], y_norm[test_ix]
     cv_inner = KFold(n_splits=10, shuffle=True, random_state=1)
     space={'C': hp.quniform('C', 10, 1000, 10),
            'kernel' : hp.choice('kernel',['linear', 'poly', 'rbf']),
            'epsilon': hp.choice('epsilon',[0.1, 0.001, 0.0001]),
            'gamma': hp.quniform('gamma', 1e-8, 1000, 10),
            'tol': hp.choice('tol',[0.1, 0.01, 0.001, 0.0001])
            }
     trials = Trials()
     best = fmin(fn = bayesian_hp_svr(cv_inner, X_train, y_train),
                 space=space,
                 algo=tpe.suggest,
                 max_evals=100,
                 trials=trials,
                 rstate=np.random.RandomState(42))
     model = SVR(random_state=42, C=int(best['C']), kernel=best['kernel'], epsilon=best['epsilon'],
                 gamma=int(best['gamma']), tol=best['tol'])
     result = model.fit(X_train,y_train.ravel())
     best_model = result.best_estimator_
     y_pred = best_model.predict(X_test)
     y_pred_actual = np.array(scaler.inverse_transform(np.asnumpy(y_pred).reshape(-1,1)))
     y_test_actual = np.array(scaler.inverse_transform(np.asnumpy(y_test).reshape(-1,1)))
     
     err_abs = np.absolute(np.subtract(y_test_actual, y_pred_actual.reshape(-1,1)))
     err_actual = np.subtract(y_test_actual, y_pred_actual.reshape(-1,1))
     
     # Statistics to be reported in the paper (Ghosh et al. Elsevier CBM 2023)
     MAE = metrics.mean_absolute_error(y_test_actual, y_pred_actual)
     RMS = np.sqrt(metrics.mean_squared_error(y_test_actual, y_pred_actual))
     R2Score = metrics.r2_score(y_test_actual, y_pred_actual)
     M = np.mean(err_actual)
     sd = np.std(err_actual)
     sdae = np.std(err_abs)
     r,_ = stats.pearsonr(y_test_actual.ravel(), y_pred_actual)

     # Error less than 5, 10 and 15 mmHg (BHS standards)
     err_five = np.ceil(int(sum(err_abs<=5))/err_abs.shape[0]*100)
     err_ten = np.ceil(int(sum(err_abs<=10))/err_abs.shape[0]*100)
     err_fifteen = np.ceil(int(sum(err_abs<=15))/err_abs.shape[0]*100)

     # Appending the results to create a list of results
     outer_MAE.append(MAE)
     outer_M.append(M)
     outer_SDAE.append(sdae)
     outer_SD.append(sd)
     outer_err_five.append(err_five)
     outer_err_ten.append(err_ten)
     outer_err_fifeteen.append(err_fifteen)

dict = {'MAE': outer_MAE, 'M': outer_M, 'SDAE': outer_SDAE, 
        'SD': outer_SD, 'err_five': outer_err_five, 'err_ten': outer_err_ten, 'err_fifteen': outer_err_fifeteen} 
df_SVM = pd.DataFrame(dict)
print(df_SVM.to_string())


# Extreme Gradient Boosting (XGB) Evaluation 
# Configuring the bayesian function to perform the hyperparameter optimization

def bayesian_hp_xgb(cv, X, y):
    # the function gets a set of variable parameters in "params" and then conduct the cross validation with the outer cv_folds.
    # the params needs to be updated with each set of evaluation and for different algorithms (SVR, CatBoost, XGBoost, LightGBM)
    params = {'max_leaves': params['max_leaves'], 
              'max_bin': params['max_bin'],
              'eta': params['eta'],
              'reg_lambda': params['reg_lambda'],
              'grow_policy': params['grow_policy'],
              'min_child_weight': int(params['min_child_weight']),
              'reg_alpha': int(params['reg_alpha'])}
    
    # we use this params to create a new XGB
    model = XGBRegressor(random_state=42, **params)
    score = -cross_val_score(model, X, y, cv=cv, scoring="neg_mean_squared_error", n_jobs=-1).mean()

    return score

# Configuring the nested-cross validation procedure
cv_outer = KFold(n_splits=10, shuffle=True, random_state=1)
for train_ix, test_ix in cv_outer.split(X):
     X_train, X_test = X_norm[train_ix, :], X_norm[test_ix, :]
     y_train, y_test = y_norm[train_ix], y_norm[test_ix]
     cv_inner = KFold(n_splits=10, shuffle=True, random_state=1)
     space={'max_leaves': hp.choice('max_leaves',[16, 64, 256, 1024]),
            'max_bin' : hp.choice('max_bin',[64, 256, 1024, 2048, 5096]),
            'eta': hp.uniform('eta', 0.01, 0.1),
            'reg_lambda': hp.quniform('reg_lambda', 0.01, 10, 1),
            'grow_policy': hp.choice('grow_policy',['depthwise', 'lossguide']),
            'min_child_weight': hp.quniform('min_child_weight', 0, 100, 10),
            'reg_alpha': hp.quniform('reg_alpha', 0, 50, 5)
            }
     trials = Trials()
     best = fmin(fn = bayesian_hp_xgb(cv_inner, X_train, y_train),
                 space=space,
                 algo=tpe.suggest,
                 max_evals=100,
                 trials=trials,
                 rstate=np.random.RandomState(42))
     model = XGBRegressor(random_state=42, max_leaves=best['max_leaves'], max_bin=best['max_bin'], eta=best['eta'], 
                          reg_lambda = best['reg_lambda'], grow_policy=best['grow_policy'], min_child_weight=int(best['min_child_weight']), reg_alpha=int(best['reg_alpha']))
     result = model.fit(X_train,y_train.ravel())
     best_model = result.best_estimator_
     y_pred = best_model.predict(X_test)
     y_pred_actual = np.array(scaler.inverse_transform(np.asnumpy(y_pred).reshape(-1,1)))
     y_test_actual = np.array(scaler.inverse_transform(np.asnumpy(y_test).reshape(-1,1)))
     
     err_abs = np.absolute(np.subtract(y_test_actual, y_pred_actual.reshape(-1,1)))
     err_actual = np.subtract(y_test_actual, y_pred_actual.reshape(-1,1))
     
     # Statistics to be reported in the paper (Ghosh et al. Elsevier CBM 2023)
     MAE = metrics.mean_absolute_error(y_test_actual, y_pred_actual)
     RMS = np.sqrt(metrics.mean_squared_error(y_test_actual, y_pred_actual))
     R2Score = metrics.r2_score(y_test_actual, y_pred_actual)
     M = np.mean(err_actual)
     sd = np.std(err_actual)
     sdae = np.std(err_abs)
     r,_ = stats.pearsonr(y_test_actual.ravel(), y_pred_actual)

     # Error less than 5, 10 and 15 mmHg (BHS standards)
     err_five = np.ceil(int(sum(err_abs<=5))/err_abs.shape[0]*100)
     err_ten = np.ceil(int(sum(err_abs<=10))/err_abs.shape[0]*100)
     err_fifteen = np.ceil(int(sum(err_abs<=15))/err_abs.shape[0]*100)

     # Appending the results to create a list of results
     outer_MAE.append(MAE)
     outer_M.append(M)
     outer_SDAE.append(sdae)
     outer_SD.append(sd)
     outer_err_five.append(err_five)
     outer_err_ten.append(err_ten)
     outer_err_fifeteen.append(err_fifteen)

dict = {'MAE': outer_MAE, 'M': outer_M, 'SDAE': outer_SDAE, 
        'SD': outer_SD, 'err_five': outer_err_five, 'err_ten': outer_err_ten, 'err_fifteen': outer_err_fifeteen} 
df_XGB = pd.DataFrame(dict)
print(df_XGB.to_string())


# Category Boosting (CatBoost) Evaluation 
# Configuring the bayesian function to perform the hyperparameter optimization

def bayesian_hp_cb(cv, X, y):
    # the function gets a set of variable parameters in "params" and then conduct the cross validation with the outer cv_folds.
    # the params needs to be updated with each set of evaluation and for different algorithms (SVR, CatBoost, XGBoost, LightGBM)
    params = {'depth': params['depth'], 
              'min_data_in_leaf': int(params['min_data_in_leaf']),
              'random_strength': params['random_strength'],
              'learning_rate': params['learning_rate'],
              'grow_policy': params['grow_policy'],
              'l2_leaf_reg': int(params['l2_leaf_reg'])}
    
    # we use this params to create a new XGB
    model = CatBoostRegressor(random_state=42, **params)
    score = -cross_val_score(model, X, y, cv=cv, scoring="neg_mean_squared_error", n_jobs=-1).mean()

    return score

# Configuring the nested-cross validation procedure
cv_outer = KFold(n_splits=10, shuffle=True, random_state=1)
for train_ix, test_ix in cv_outer.split(X):
     X_train, X_test = X_norm[train_ix, :], X_norm[test_ix, :]
     y_train, y_test = y_norm[train_ix], y_norm[test_ix]
     cv_inner = KFold(n_splits=10, shuffle=True, random_state=1)
     space={'depth': hp.choice('depth',[6, 7, 8, 9, 10, 11, 12]),
            'min_data_in_leaf' : hp.uniform('min_data_in_leaf', 1, 100),
            'random_strength': hp.quniform('random_strength', 0.1, 20, 1),
            'learning_rate': hp.quniform('learning_rate', 0.01, 0.1, 1),
            'grow_policy': hp.choice('grow_policy',['SymmetricTree', 'Depthwise', 'Lossguide']),
            'l2_leaf_reg': hp.quniform('l2_leaf_reg', 1, 100, 10)
            }
     trials = Trials()
     best = fmin(fn = bayesian_hp_cb(cv_inner, X_train, y_train),
                 space=space,
                 algo=tpe.suggest,
                 max_evals=100,
                 trials=trials,
                 rstate=np.random.RandomState(42))
     model = CatBoostRegressor(random_state=42, depth=best['depth'], min_data_in_leaf=best['min_data_in_leaf'], random_strength=best['random_strength'], 
                               learning_rate=best['learning_rate'], grow_policy=best['grow_policy'], l2_leaf_reg=int(best['l2_leaf_reg']))
     result = model.fit(X_train,y_train.ravel())
     best_model = result.best_estimator_
     y_pred = best_model.predict(X_test)
     y_pred_actual = np.array(scaler.inverse_transform(np.asnumpy(y_pred).reshape(-1,1)))
     y_test_actual = np.array(scaler.inverse_transform(np.asnumpy(y_test).reshape(-1,1)))
     
     err_abs = np.absolute(np.subtract(y_test_actual, y_pred_actual.reshape(-1,1)))
     err_actual = np.subtract(y_test_actual, y_pred_actual.reshape(-1,1))
     
     # Statistics to be reported in the paper (Ghosh et al. Elsevier CBM 2023)
     MAE = metrics.mean_absolute_error(y_test_actual, y_pred_actual)
     RMS = np.sqrt(metrics.mean_squared_error(y_test_actual, y_pred_actual))
     R2Score = metrics.r2_score(y_test_actual, y_pred_actual)
     M = np.mean(err_actual)
     sd = np.std(err_actual)
     sdae = np.std(err_abs)
     r,_ = stats.pearsonr(y_test_actual.ravel(), y_pred_actual)

     # Error less than 5, 10 and 15 mmHg (BHS standards)
     err_five = np.ceil(int(sum(err_abs<=5))/err_abs.shape[0]*100)
     err_ten = np.ceil(int(sum(err_abs<=10))/err_abs.shape[0]*100)
     err_fifteen = np.ceil(int(sum(err_abs<=15))/err_abs.shape[0]*100)

     # Appending the results to create a list of results
     outer_MAE.append(MAE)
     outer_M.append(M)
     outer_SDAE.append(sdae)
     outer_SD.append(sd)
     outer_err_five.append(err_five)
     outer_err_ten.append(err_ten)
     outer_err_fifeteen.append(err_fifteen)

dict = {'MAE': outer_MAE, 'M': outer_M, 'SDAE': outer_SDAE, 
        'SD': outer_SD, 'err_five': outer_err_five, 'err_ten': outer_err_ten, 'err_fifteen': outer_err_fifeteen} 
df_CB = pd.DataFrame(dict)
print(df_CB.to_string())


# Light Gradient Boosting (LightGBM) Evaluation 
# Configuring the bayesian function to perform the hyperparameter optimization

def bayesian_hp_lgb(cv, X, y):
    # the function gets a set of variable parameters in "params" and then conduct the cross validation with the outer cv_folds.
    # the params needs to be updated with each set of evaluation and for different algorithms (SVR, CatBoost, XGBoost, LightGBM)
    params = {'num_leaves': params['num_leaves'], 
              'max_bin': params['max_bin'],
              'learning_rate': params['learning_rate'],
              'bagging_freq': params['bagging_freq'],
              'min_data_in_leaf': int(params['min_data_in_leaf'])}
    
    # we use this params to create a new XGB
    model = CatBoostRegressor(random_state=42, **params)
    score = -cross_val_score(model, X, y, cv=cv, scoring="neg_mean_squared_error", n_jobs=-1).mean()

    return score

# Configuring the nested-cross validation procedure
cv_outer = KFold(n_splits=10, shuffle=True, random_state=1)
for train_ix, test_ix in cv_outer.split(X):
     X_train, X_test = X_norm[train_ix, :], X_norm[test_ix, :]
     y_train, y_test = y_norm[train_ix], y_norm[test_ix]
     cv_inner = KFold(n_splits=10, shuffle=True, random_state=1)
     space={'num_leaves': hp.choice('num_leaves',[16, 64, 256, 1024]),
            'max_bin' : hp.choice('max_bin',[64, 256, 1024, 2048, 5096]),
            'learning_rate': hp.uniform('learning_rate', 0.01, 0.1),
            'bagging_freq': hp.uniform('bagging_freq', 0, 1),
            'min_data_in_leaf': hp.quniform('min_data_in_leaf', 0, 20, 1)
            }
     trials = Trials()
     best = fmin(fn = bayesian_hp_lgb(cv_inner, X_train, y_train),
                 space=space,
                 algo=tpe.suggest,
                 max_evals=100,
                 trials=trials,
                 rstate=np.random.RandomState(42))
     model = lgb(random_state=42, num_leaves=best['num_leaves'], max_bin=best['max_bin'], learning_rate=best['learning_rate'], 
                 bagging_freq=best['bagging_freq'], min_data_in_leaf=best['min_data_in_leaf'])
     result = model.fit(X_train,y_train.ravel())
     best_model = result.best_estimator_
     y_pred = best_model.predict(X_test)
     y_pred_actual = np.array(scaler.inverse_transform(np.asnumpy(y_pred).reshape(-1,1)))
     y_test_actual = np.array(scaler.inverse_transform(np.asnumpy(y_test).reshape(-1,1)))
     
     err_abs = np.absolute(np.subtract(y_test_actual, y_pred_actual.reshape(-1,1)))
     err_actual = np.subtract(y_test_actual, y_pred_actual.reshape(-1,1))
     
     # Statistics to be reported in the paper (Ghosh et al. Elsevier CBM 2023)
     MAE = metrics.mean_absolute_error(y_test_actual, y_pred_actual)
     RMS = np.sqrt(metrics.mean_squared_error(y_test_actual, y_pred_actual))
     R2Score = metrics.r2_score(y_test_actual, y_pred_actual)
     M = np.mean(err_actual)
     sd = np.std(err_actual)
     sdae = np.std(err_abs)
     r,_ = stats.pearsonr(y_test_actual.ravel(), y_pred_actual)

     # Error less than 5, 10 and 15 mmHg (BHS standards)
     err_five = np.ceil(int(sum(err_abs<=5))/err_abs.shape[0]*100)
     err_ten = np.ceil(int(sum(err_abs<=10))/err_abs.shape[0]*100)
     err_fifteen = np.ceil(int(sum(err_abs<=15))/err_abs.shape[0]*100)

     # Appending the results to create a list of results
     outer_MAE.append(MAE)
     outer_M.append(M)
     outer_SDAE.append(sdae)
     outer_SD.append(sd)
     outer_err_five.append(err_five)
     outer_err_ten.append(err_ten)
     outer_err_fifeteen.append(err_fifteen)

dict = {'MAE': outer_MAE, 'M': outer_M, 'SDAE': outer_SDAE, 
        'SD': outer_SD, 'err_five': outer_err_five, 'err_ten': outer_err_ten, 'err_fifteen': outer_err_fifeteen} 
df_LGB = pd.DataFrame(dict)
print(df_LGB.to_string())
