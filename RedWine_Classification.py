#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Sep 20 11:31:00 2017

@author: SwatzMac
@Purpose: Study Red Wine Data using scikit 
@Reference: https://elitedatascience.com/python-machine-learning-tutorial-scikit-learn
"""

import numpy as np
import pandas as pd

# Import sampling helper
from sklearn.model_selection import train_test_split

# Import preprocessing module which contains utilities for scaling, transformation and wrangling data 
from sklearn import preprocessing 

# Import Random Forest family
from sklearn.ensemble import RandomForestRegressor

# Import cross-validation pipeline
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import GridSearchCV 

# Import some metrics to evaluate model performance 
from sklearn.metrics import mean_squared_error, r2_score

# Import module for saving scikit-learn models 
# joblib is used here because its an efficient way of storing large numpy arrays
from sklearn.externals import joblib

## LOAD RED WINE DATA ## 
#dataset_url = 'http://mlr.cs.umass.edu/ml/machine-learning-databases/wine-quality/winequality-red.csv'
#data = pd.read_csv(dataset_url)

#print (data.head())

data = pd.read_csv(dataset_url, sep = ";")
print (data.head())

print (data.shape)
print (data.describe())

# All the features are numerical, but their scales are very different, standardize them later

## Split data into Training and Testing Data ## 
# Separate Targey (Y) features from our input (X) features: QUALITY - Target Variable 
y = data.quality 
x = data.drop('quality',axis=1)

x_train, x_test, y_train, y_test = train_test_split(x,y,test_size = 0.2, random_state = 123,stratify = y)

## Data Preprocessing Steps ## 
# Standardize the features 
# Standardization is the process of subtracting means from each feature and then divide the feature by std. deviation 
# Many algorithms assume that all features are centered around zero and have approx. same variance
#x_train_scaled = preprocessing.scale(x_train)
#print (x_train_scaled.mean(axis=0))

# Using Transformer API to fit the "preprocessing" data
scaler = preprocessing.StandardScaler().fit(x_train)
# the scalar object has several means and std. deviations from each feature in training set 

x_train_scaled = scaler.transform(x_train)
print (x_train_scaled.mean(axis=0))
print (x_train_scaled.std(axis=0))

x_test_scaled = scaler.transform(x_test)
print (x_test_scaled.mean(axis=0))
print (x_test_scaled.std(axis=0))

# Pipeline with preprocessing and model 
pipeline = make_pipeline(preprocessing.StandardScaler(),RandomForestRegressor(n_estimators=100))
#This is exactly what it looks like: a modeling pipeline that first transforms the data using 
#StandardScaler() and then fits a model using a random forest regressor.

## Declare HYPERPARAMETERS To Tune ## 
print (pipeline.get_params())
#Hyperparameters express "higher-level" structural information about the model, 
#and they are typically set before training the model.

# Declare hyperparameters to tune 
hyperparameters = { 'randomforestregressor__max_features' : ['auto', 'sqrt', 'log2'],
                  'randomforestregressor__max_depth': [None, 5, 3, 1]}

# Tune model using a cross validation pipeline 
# This is one of the most important skills of all in ML because it helps maximise model performance
# while reducing chance of overfitting

#Cross-validation is a process for reliably estimating the performance of a method for 
#building a model by training and evaluating your model multiple times using the same method.

clf = GridSearchCV(pipeline, hyperparameters, cv=10)
 
# Fit and tune model
clf.fit(x_train, y_train)

# GridSearchCV essentially performs cross-validation across the entire "grid" 
# (all possible permutations) of hyperparameters.

#print (clf.best_params_)

# Confirm model will be retrained
print (clf.refit)

## EVALUATE MODEL PIPELINE ON TEST DATA ## 
# Predict a new set of data 
y_pred = clf.predict(x_test)

print (r2_score(y_test,y_pred))
print (mean_squared_error(y_test,y_pred))

### Is this performance good enough ### 
# Well, the rule of thumb is that your very first model probably won't be the best possible 
# model. However, we recommend a combination of three strategies to decide if you're satisfied 
# with your model performance.

## How to Improve your model 
# 1. Try other regression model families (regularized regression, boosted trees)
# 2. Collect more data if possible
# 3. Engineer smarter features after spending more time on exploratory analysis.
# 4. Speak to a domain expert to get more context 

## SAVE MODEL FOR FUTURE USE ## 
# Save model to .pkl file 
joblib.dump(clf,"rf_regressor.pkl")

## Reuse the model 
clf2 = joblib.load ("rf_regressor.pkl")
# Predict dataset using loaded model
clf2.predict(x_test)





























