# -*- coding: utf-8 -*-
"""
Created on Sat Apr 26 16:16:50 2025

MCAR models

@author: Marcin
"""

import matplotlib.pyplot as plt
import os
import seaborn as sns
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, recall_score, precision_score, confusion_matrix
from functions.other import results_to_excel, format_dataframe

# Load Data

# MCAR - Original
original_mcar_cca = pd.read_excel("Data/Original/Complete Case Analysis/bp_mcar_cca.xlsx")
original_mcar_cca = format_dataframe(original_mcar_cca)
original_mcar_mi = pd.read_excel("Data/Original/Multiple Imputation/bp_mcar_mi.xlsx")
original_mcar_mi = format_dataframe(original_mcar_mi)
original_mcar_na = pd.read_excel("Data/Original/Learned NA/bp_mcar.xlsx")
original_mcar_na = format_dataframe(original_mcar_na)

# MCAR - Synthetic
syn_mcar_cca = pd.read_excel("Data/Synthetic/Complete Case Analysis/bp_mcar_cca.xlsx")
syn_mcar_cca = format_dataframe(syn_mcar_cca)
syn_mcar_mi = pd.read_excel("Data/Synthetic/Multiple Imputation/bp_mcar_mi.xlsx")
syn_mcar_mi = format_dataframe(syn_mcar_mi)
syn_mcar_na = pd.read_excel("Data/Synthetic/Learned NA/bp_mcar.xlsx")
syn_mcar_na = format_dataframe(syn_mcar_na)

# MAR - Original
original_mar_cca = pd.read_excel("Data/Original/Complete Case Analysis/bp_mar_cca.xlsx")
original_mar_cca = format_dataframe(original_mar_cca)
original_mar_mi = pd.read_excel("Data/Original/Multiple Imputation/bp_mar_mi.xlsx")
original_mar_mi = format_dataframe(original_mar_mi)
original_mar_na = pd.read_excel("Data/Original/Learned NA/bp_mar.xlsx")
original_mar_na = format_dataframe(original_mar_na)

# MAR - Synthetic
syn_mar_cca = pd.read_excel("Data/Synthetic/Complete Case Analysis/bp_mar_cca.xlsx")
syn_mar_cca = format_dataframe(syn_mar_cca)
syn_mar_mi = pd.read_excel("Data/Synthetic/Multiple Imputation/bp_mar_mi.xlsx")
syn_mar_mi = format_dataframe(syn_mar_mi)
syn_mar_na = pd.read_excel("Data/Synthetic/Learned NA/bp_mar.xlsx")
syn_mar_na = format_dataframe(syn_mar_na)

# MNAR - Original
original_mnar_cca = pd.read_excel("Data/Original/Complete Case Analysis/bp_mnar_cca.xlsx")
original_mnar_cca = format_dataframe(original_mnar_cca)
original_mnar_mi = pd.read_excel("Data/Original/Multiple Imputation/bp_mnar_mi.xlsx")
original_mnar_mi = format_dataframe(original_mnar_mi)
original_mnar_na = pd.read_excel("Data/Original/Learned NA/bp_mnar.xlsx")
original_mnar_na = format_dataframe(original_mnar_na)

# MNAR - Synthetic
syn_mnar_cca = pd.read_excel("Data/Synthetic/Complete Case Analysis/bp_mnar_cca.xlsx")
syn_mnar_cca = format_dataframe(syn_mnar_cca)
syn_mnar_mi = pd.read_excel("Data/Synthetic/Multiple Imputation/bp_mnar_mi.xlsx")
syn_mnar_mi = format_dataframe(syn_mnar_mi)
syn_mnar_na = pd.read_excel("Data/Synthetic/Learned NA/bp_mnar.xlsx")
syn_mnar_na = format_dataframe(syn_mnar_na)

# Test data
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
    X_test = test.drop(columns=[target])
    
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
    precision = precision_score(y_test, y_pred)
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

# Define your target variable
target_variable = 'hospitaldeath'

# Run logistic regression for each dataset

# MCAR
logistic_regression(original_mcar_cca, test_data, target=target_variable, label="Original CCA (MCAR)")
logistic_regression(original_mcar_mi, test_data, target=target_variable, label="Original MI (MCAR)")
#logistic_regression(original_mcar_na, test_data, target=target_variable, label="Original Learned NA (MCAR)")

logistic_regression(syn_mcar_cca, test_data, target=target_variable, label="Synthetic CCA (MCAR)")
logistic_regression(syn_mcar_mi, test_data, target=target_variable, label="Synthetic MI (MCAR)")
#logistic_regression(syn_mcar_na, test_data, target=target_variable, label="Synthetic Learned NA (MCAR)")

# MAR
logistic_regression(original_mar_cca, test_data, target=target_variable, label="Original CCA (MAR)")
logistic_regression(original_mar_mi, test_data, target=target_variable, label="Original MI (MAR)")
#logistic_regression(original_mar_na, test_data, target=target_variable, label="Original Learned NA (MAR)")

logistic_regression(syn_mar_cca, test_data, target=target_variable, label="Synthetic CCA (MAR)")
logistic_regression(syn_mar_mi, test_data, target=target_variable, label="Synthetic MI (MAR)")
#logistic_regression(syn_mar_na, test_data, target=target_variable, label="Synthetic Learned NA (MAR)")

# MNAR
logistic_regression(original_mnar_cca, test_data, target=target_variable, label="Original CCA (MNAR)")
logistic_regression(original_mnar_mi, test_data, target=target_variable, label="Original MI (MNAR)")
#logistic_regression(original_mnar_na, test_data, target=target_variable, label="Original Learned NA (MNAR)")

logistic_regression(syn_mnar_cca, test_data, target=target_variable, label="Synthetic CCA (MNAR)")
logistic_regression(syn_mnar_mi, test_data, target=target_variable, label="Synthetic MI (MNAR)")
#logistic_regression(syn_mnar_na, test_data, target=target_variable, label="Synthetic Learned NA (MNAR)")

# Save all results to Excel
results_to_excel(results)
