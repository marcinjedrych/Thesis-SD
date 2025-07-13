# -*- coding: utf-8 -*-
"""
Created on Sun Jul 13 17:12:59 2025

@author: Marcin
"""

import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, recall_score, precision_score, confusion_matrix
from functions.other import results_to_excel, format_and_sample
from sklearn.calibration import calibration_curve
from sklearn.metrics import brier_score_loss
from sklearn.metrics import roc_auc_score
from sklearn.metrics import roc_curve

def logistic_regression(train, test, target, label=None, exclude_vars=None):
    
    if exclude_vars is None:
        exclude_vars = []

    # Handle categorical variables
    train = pd.get_dummies(train, columns=['stage'])
    test = pd.get_dummies(test, columns=['stage'])

    # Align columns
    train, test = train.align(test, join='left', axis=1, fill_value=0)

    # Define predictors and target
    X_train = train.drop(columns=[target] + exclude_vars)
    X_test = test.drop(columns=[target] + exclude_vars)
    y_train = train[target]
    y_test = test[target]

    # Train logistic regression model
    model = LogisticRegression(max_iter=1000, random_state=123)
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]

    # Metrics
    acc = accuracy_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred, zero_division=0)
    precision = precision_score(y_test, y_pred, zero_division=0)
    auc = roc_auc_score(y_test, y_prob)
    brier = brier_score_loss(y_test, y_prob)

    result_row = {
        'Accuracy': acc,
        'Recall': recall,
        'Precision': precision,
        'AUC': auc,
        'Brier Score': brier
    }

    return result_row