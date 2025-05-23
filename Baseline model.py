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
from sklearn.calibration import calibration_curve
from sklearn.metrics import brier_score_loss
from sklearn.metrics import roc_auc_score  # ADD THIS IMPORT
from sklearn.metrics import roc_curve


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
    X_train = X_train.drop(columns=['Index'])
    print(X_train.columns)
    X_test = test.drop(columns=[target])
    X_test = X_test.drop(columns=['Index'])
    
    y_train = train[target]
    y_test = test[target]

    # Train logistic regression model
    model = LogisticRegression(max_iter=1000, random_state=123)
    model.fit(X_train, y_train)

    # Predictions
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]  # probability of class 1

    # discrimination metrics 
    acc = accuracy_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    precision = precision_score(y_test,y_pred)
    auc = roc_auc_score(y_test, y_prob)  
    
    fpr, tpr, _ = roc_curve(y_test, y_prob)  # ROC curve values
    conf_matrix = confusion_matrix(y_test, y_pred)
    
    # Calibration metrics
    brier = brier_score_loss(y_test, y_prob) # mean squared difference between predicted probabilities and actual binary outcomes.
    prob_true, prob_pred = calibration_curve(y_test, y_prob, n_bins=10)
    
    # Save results
    results.append({
        'Model': label,
        'Accuracy': acc,
        'Recall': recall,
        'Precision': precision,
        'AUC': auc,
        'Brier Score': brier
    })
    
    # Output
    print(f"\nPerformance for {label}:")
    print(f"  - Accuracy: {acc:.4f}")
    print(f"  - Recall: {recall:.4f}")
    print(f"  - Precision: {precision:.4f}")
    print(f"  - AUC: {auc:.4f}")
    print(f"  - Brier Score: {brier:.4f}")
    print("  - Confusion Matrix:")
        
    # Plot confusion matrix
    plt.figure(figsize=(6,5))
    sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues')
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.title(f'Confusion Matrix for {label}')
    plt.show()
    
    # ROC Curve
    plt.figure(figsize=(6,5))
    plt.plot(fpr, tpr, label=f'AUC = {auc:.2f}')
    plt.plot([0, 1], [0, 1], linestyle='--', color='gray')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(f'ROC Curve for {label}')
    plt.legend(loc='lower right')
    plt.grid(True)
    plt.show()

    
    # Calibration plot: predicted probabilities vs. observed frequencies, perfectly calibrated model lies on the diagonal (y = x).
    plt.figure(figsize=(6,5))
    plt.plot(prob_pred, prob_true, marker='o', label='Model')
    plt.plot([0, 1], [0, 1], linestyle='--', color='gray', label='Perfectly Calibrated')
    plt.xlabel('Mean Predicted Probability')
    plt.ylabel('Actual hospitaldeath = 1')
    plt.title(f'Calibration Curve for {label}')
    plt.legend()
    plt.grid(True)
    plt.show()

    return model, acc, recall, precision, brier, conf_matrix


### ---- ORIGINAL BASELINE ----

log_model, acc, recall, precision, brier, cm = logistic_regression(no_missing, test_data, 'hospitaldeath', label='Original Baseline (no missingness)')

### ---- SYNTHETIC BASELINE ----

synthetic_no_missing = pd.read_excel("data/Synthetic/synthetic_no_missing.xlsx")
log_model, acc, recall, precision, brier, cm = logistic_regression(synthetic_no_missing, test_data, 'hospitaldeath', label='Synthetic Baseline (no missingness)')

# model performance summary
results_to_excel(results)


