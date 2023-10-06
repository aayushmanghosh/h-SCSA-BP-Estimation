'''Author: Aayushman Ghosh
   Department of Electrical and Computer Engineering
   University of Illinois Urbana-Champaign
   (aghosh14@illinois.edu)
   
   Version: v1.0
''' 

import shap
import os
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import seaborn as sns
import catboost
from catboost import CatBoostRegressor
from sklearn.feature_selection import mutual_info_regression
from sklearn.model_selection import train_test_split
from sklearn.feature_selection import SelectKBest
from sklearn.svm import SVR
from sklearn.feature_selection import RFE
from sklearn.linear_model import RidgeCV, Ridge
from sklearn.linear_model import LassoCV, Lasso
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
from scipy.special import softmax


def print_feature_importances_shap_values(shap_values, features):
    '''
    Prints the feature importances based on SHAP values in an ordered way
    shap_values -> The SHAP values calculated from a shap.Explainer object
    features -> The name of the features, on the order presented to the explainer
    '''
    importances = []
    for i in range(shap_values.values.shape[1]):
        importances.append(np.mean(np.abs(shap_values.values[:, i])))

    importances_norm = softmax(importances)
    feature_importances = {fea: imp for imp, fea in zip(importances, features)}
    feature_importances_norm = {fea: imp for imp, fea in zip(importances_norm, features)}
    feature_importances = {k: v for k, v in sorted(feature_importances.items(), key=lambda item: item[1], reverse = True)}
    feature_importances_norm= {k: v for k, v in sorted(feature_importances_norm.items(), key=lambda item: item[1], reverse = True)}
    for k, v in feature_importances.items():
        print(f"{k} -> {v:.4f} (softmax = {feature_importances_norm[k]:.4f})")


# Loading the necessary metadata to perform analysis on -- Subject to change depending on the experiment
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
y = data.loc[:,'map'] # Subject to change depending on what results you prefer (Choice: sbp, dbp, map)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30, random_state=0)


# Filter Method 1 (Pearson Correlation Coefficient)
Corr_data = pd.DataFrame(data=data.corr().iloc[:-3, -1])
Corr_data.reset_index(inplace=True)
Corr_data.rename({'map':'Pearson', 'index':'Features'}, axis=1, inplace=True)
Corr_data.Pearson = np.divide(np.abs(Corr_data.Pearson), np.max(np.abs(Corr_data.Pearson)))
Corr_data = Corr_data.sort_values(by = "Pearson", ascending=False)
print(Corr_data)

# Filter Method 2 (Mutual Information)
f_selector = SelectKBest(score_func=mutual_info_regression, k='all')
X_reduced = f_selector.fit(X_train, y_train)
MI_data = pd.DataFrame(data=[X.columns,np.divide(X_reduced.scores_, np.max(X_reduced.scores_))],index=['Features', 'MI']).T
MI_data = MI_data.sort_values(by="MI",ascending=False)
print(MI_data)

# Regressor 1 (XGBoost)
model_xgb = XGBRegressor(objective='reg:squarederror')
model_xgb.fit(X_train, y_train)
explainer = shap.Explainer(model_xgb.predict, X_test)
shap_values_xgb = explainer(X_test)
print_feature_importances_shap_values(shap_values_xgb, X.columns)

# Regressor 2 (CatBoost)
model_cb = CatBoostRegressor(verbose=False, boost_from_average=True)
model_cb.fit(X_train, y_train)
explainer = shap.Explainer(model_cb.predict, X_test)
shap_values_cb = explainer(X_test)
print_feature_importances_shap_values(shap_values_cb, X.columns)

# Regressor 3 (LightGBM)
model_lgb = LGBMRegressor()
model_lgb.fit(X_train, y_train)
explainer = shap.Explainer(model_lgb.predict, X_test)
shap_values_lgb = explainer(X_test)
print_feature_importances_shap_values(shap_values_lgb, X.columns)