# -*- coding: utf-8 -*-
"""
Created on Mon Mar 24 12:39:12 2025

Baseline models

@author: Marcin
"""
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

import pandas as pd
data = pd.read_excel("data/no_missing_data.xlsx")

# Split the data (80% train, 20% test) by index 
train_idx, test_idx = train_test_split(data.index, test_size=0.2, random_state=123)
train_data = data.loc[train_idx]
test_data = pd.read_excel("data/test_data.xlsx")

def train_and_evaluate(train, test, target, label):
    
    # Handle categorical variables
    train = pd.get_dummies(train, columns=['stage'])
    test = pd.get_dummies(test, columns=['stage'])
    
    # Separate predictors (X) and outcome (y)
    X_train = train.drop(columns=[target]) 
    X_test = test.drop(columns=[target]) 
    
    y_train = train[target]
    y_test = test[target]

    # Train the regression model
    model = LinearRegression()
    model.fit(X_train, y_train)

    # Make predictions
    y_pred = model.predict(X_test)

    # Evaluate performance
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    print(f"Performance for {label}:")
    print(f"  - Mean Squared Error: {mse:.4f}")
    print(f"  - R² Score: {r2:.4f}\n")

    return model, mse, r2

## predicting continuous outcome bloodpressure

model, mse, r2 = train_and_evaluate(train_data, test_data, 'bp', label = 'Baseline (original data)')


