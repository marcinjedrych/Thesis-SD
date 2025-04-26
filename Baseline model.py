# -*- coding: utf-8 -*-
"""
Created on Mon Mar 24 12:39:12 2025

Baseline models

@author: Marcin
"""

import matplotlib.pyplot as plt
import os
import seaborn as sns
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, recall_score, precision_score, confusion_matrix
from functions.other import results_to_excel

no_missing = pd.read_excel("data/original/no_missing.xlsx")
no_missing = no_missing.rename(columns={'Unnamed: 0': 'Index'})

test_data = pd.read_excel("data/test_data.xlsx")
test_data = test_data.rename(columns={'Unnamed: 0': 'Index'})

# Initialize an empty list to store results
results = []
def logistic_regression(train, test, target, label):
    
    # Handle categorical variables
    train = pd.get_dummies(train, columns=['stage'])
    test = pd.get_dummies(test, columns=['stage'])
    
    # Separate predictors (X) and outcome (y)
    X_train = train.drop(columns=[target])
    X_train = X_train.drop(columns=['Index', 'weight', 'stage_I', 'stage_II', 'stage_III', 'stage_IV'])
    print(X_train.columns)
    X_test = test.drop(columns=[target])
    X_test = X_test.drop(columns=['Index', 'weight', 'stage_I', 'stage_II', 'stage_III', 'stage_IV'])
    
    y_train = train[target]
    y_test = test[target]

    # Train logistic regression model
    model = LogisticRegression(max_iter=1000, random_state=123)
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

    # Save results
    results.append({
        'Model': label,
        'Accuracy': acc,
        'Recall': recall,
        'Precision': precision
    })
    
    # Plot confusion matrix
    plt.figure(figsize=(6,5))
    sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues')
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.title(f'Confusion Matrix for {label}')
    plt.show()
    
    
    return model, acc, recall, precision, conf_matrix


### ---- ORIGINAL BASELINE ----

log_model, acc, recall, precision, cm = logistic_regression(no_missing, test_data, 'hospitaldeath', label='Original Baseline (no missingness)')

### ---- SYNTHETIC BASELINE ----

synthetic_no_missing = pd.read_excel("data/Synthetic/synthetic_no_missing.xlsx")
log_model, acc, recall, precision, cm = logistic_regression(synthetic_no_missing, test_data, 'hospitaldeath', label='Synthetic Baseline (no missingness)')

# model performance summary
results_to_excel(results)


