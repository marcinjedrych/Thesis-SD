# -*- coding: utf-8 -*-
"""
Created on Mon Mar 24 12:39:12 2025

Baseline models

@author: Marcin
"""

import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, recall_score, precision_score, confusion_matrix

no_missing = pd.read_excel("data/original/no_missing.xlsx")
no_missing = no_missing.rename(columns={'Unnamed: 0': 'Index'})

test_data = pd.read_excel("data/test_data.xlsx")
test_data = test_data.rename(columns={'Unnamed: 0': 'Index'})


def logistic_regression(train, test, target, label):
    
    # Handle categorical variables
    train = pd.get_dummies(train, columns=['stage'])
    test = pd.get_dummies(test, columns=['stage'])
    
    # Separate predictors (X) and outcome (y)
    X_train = train.drop(columns=[target])
    X_test = test.drop(columns=[target])
    
    y_train = train[target]
    y_test = test[target]

    # Train logistic regression model
    model = LogisticRegression(max_iter=1000)
    model.fit(X_train, y_train)
    
    # Make predictions
    y_pred = model.predict(X_test)
    
    # Evaluate performance
    acc = accuracy_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    precision = precision_score(y_test,y_pred)
    conf_matrix = confusion_matrix(y_test, y_pred)
    
    print(f"\nPerformance for {label}:")
    print(f"  - Accuracy: {acc:.4f}")
    print(f"  - Recall: {recall:.4f}")
    print(f"  - Precision: {precision:.4f}")
    print("  - Confusion Matrix:")
    
    # Plot confusion matrix
    plt.figure(figsize=(6,5))
    sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues')
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.title(f'Confusion Matrix for {label}')
    plt.show()
    
    
    return model, acc, recall, precision, conf_matrix


### ---- ORIGINAL BASELINE ----

log_model, acc, recall, precision, cm = logistic_regression(no_missing, test_data, 'hospitaldeath', label='Original Baseline (binary outcome)')

### ---- SYNTHETIC BASELINE ----

synthetic_no_missing = pd.read_excel("data/Synthetic/synthetic_no_missing.xlsx")
log_model, acc, recall, precision, cm = logistic_regression(synthetic_no_missing, test_data, 'hospitaldeath', label='Synthetic Baseline (binary outcome)')

