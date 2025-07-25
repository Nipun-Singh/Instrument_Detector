# -*- coding: utf-8 -*-
"""FModel.ipynb

Automatically generated by Colab.

Original file is located at
    https://colab.research.google.com/drive/1nz01wISL2j4kqWVJIJ9_W76I6xoTeT_X
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

dataset = pd.read_csv(r"OrgDataset.csv")
X_train = dataset.iloc[:, 1:-1 ].values
y_train = dataset.iloc[:, 0].values
print(dataset.head(5))
print(len(y_train))

testset = pd.read_csv(r"C:\Users\HP\Desktop\Flask\OrgTest.csv")
X_test = dataset.iloc[:, 1:-1 ].values
y_test = dataset.iloc[:, 0].values
print(testset.head(5))
print(X_test)

from sklearn.impute import SimpleImputer
imputer = SimpleImputer(missing_values=np.nan, strategy='mean')
imputer.fit(X_train)
X_train = imputer.transform(X_train)
X_test = imputer.transform(X_test)

from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)
print(X_train)
print(X_test)

import pickle
with open('scaler.pkl', 'wb') as f:
    pickle.dump(sc, f)

"""# **Ensemble Learning**"""

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler, LabelEncoder
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.ensemble import StackingRegressor, HistGradientBoostingRegressor, RandomForestRegressor, AdaBoostRegressor
from sklearn.linear_model import RidgeCV, LassoCV, LinearRegression, SGDRegressor
from sklearn.neural_network import MLPRegressor
from xgboost import XGBRegressor
from scipy.stats import randint, uniform
from time import time

print(y_train)
print(y_test)
Le=LabelEncoder();
y_train=Le.fit_transform(y_train)
y_test=Le.fit_transform(y_test)
print(y_train)
print(y_test)

final_layer = StackingClassifier(
    estimators=[
        ('gbrt', GradientBoostingClassifier(random_state=42, n_estimators=50)),
        ('rf', RandomForestClassifier(n_estimators=50, max_depth=5, random_state=42)),
        ('adbc', AdaBoostClassifier(n_estimators=50, random_state=42)),
    ],
    final_estimator=LogisticRegression(max_iter=1000),
    passthrough=True,
    n_jobs=-1
)


multi_layer_classifier = StackingClassifier(
    estimators=[
        ('lr', LogisticRegression(max_iter=1000)),
        ('mlp', MLPClassifier(hidden_layer_sizes=(64, 32), max_iter=500, solver='adam', random_state=42)),
        ('xgb', XGBClassifier(use_label_encoder=False, eval_metric='mlogloss', random_state=42, learning_rate=np.float64(0.018236166437629313), max_depth=7, n_estimators=147)),
        ('etc', ExtraTreesClassifier(n_estimators=50, max_depth=10, random_state=42))
    ],
    final_estimator=final_layer,
    passthrough=False,
    n_jobs=-1
)

multi_layer_classifier.fit(X_train, y_train)

import pickle

with open('instrument_model.pkl', 'wb') as f:
    pickle.dump(multi_layer_classifier, f)
