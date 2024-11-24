#!/usr/bin/env python
# coding: utf-8

# Student Performance Regression
# * Dataset: https://www.kaggle.com/datasets/devansodariya/student-performance-data/data
# * Original data from UCI: https://archive.ics.uci.edu/dataset/320/student+performance

#!pip install tqdm

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import mutual_info_score
from sklearn.feature_extraction import DictVectorizer
import xgboost as xgb
import pickle


output_file = 'model.bin'


# 1. Loading the data

df = pd.read_csv("student_data.csv")

df.columns = df.columns.str.lower()

categorical = list(df.dtypes[df.dtypes == 'object'].index)
numerical = list(df.dtypes[df.dtypes == 'int64'].index)
numerical.remove("g3")

# 2. Setting up the validation framework

# Perform the train/validation/test split with Scikit-Learn
df_full_train, df_test = train_test_split(df, test_size=0.2, random_state=1)
df_train, df_val = train_test_split(df_full_train, test_size=0.25, random_state=1)

df_train = df_train.reset_index(drop=True)
df_val = df_val.reset_index(drop=True)
df_test = df_test.reset_index(drop=True)

y_train = df_train.g3.values
y_val = df_val.g3.values
y_test = df_test.g3.values

del df_train['g3']
del df_val['g3']
del df_test['g3']

# 3. Data preparation

dv = DictVectorizer(sparse=False)

train_dict = df_train[categorical + numerical].to_dict(orient='records')
X_train = dv.fit_transform(train_dict)

val_dict = df_val[categorical + numerical].to_dict(orient='records')
X_val = dv.transform(val_dict)

dicts_test = df_test[categorical + numerical].to_dict(orient='records')
X_test = dv.transform(dicts_test)

features = list(dv.get_feature_names_out())
dtrain = xgb.DMatrix(X_train, label=y_train, feature_names=features)
dval = xgb.DMatrix(X_val, label=y_val, feature_names=features)
dtest = xgb.DMatrix(X_test, label=y_test, feature_names=features)

# 4. XGB model

xgb_params = {
        'eta': 0.3,
        'max_depth': 10,
        'min_child_weight': 1,
        'objective': 'reg:squarederror',
        'eval_metric': 'rmse',
        'nthread': 8,
        'seed': 1,
        'verbosity': 1,
    }

model = xgb.train(xgb_params, dtrain, num_boost_round=200)

y_pred = model.predict(dtest)

def rmse(y, y_pred):
    se = (y - y_pred) ** 2
    mse = se.mean()
    return np.sqrt(mse)

print(rmse(y_test, y_pred))

with open(output_file, 'wb') as f_out:
    pickle.dump((dv, model), f_out)

