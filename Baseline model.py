# -*- coding: utf-8 -*-
"""
Created on Mon Mar 24 12:39:12 2025

Baseline models

@author: Marcin
"""
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, recall_score, precision_score, confusion_matrix

train_data = pd.read_excel("data/no_missing_data.xlsx")
train_data = train_data.rename(columns={'Unnamed: 0': 'Index'})

test_data = pd.read_excel("data/test_data.xlsx")
test_data = test_data.rename(columns={'Unnamed: 0': 'Index'})

def  linear_regression(train, test, target, label):
    
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

def logistic_regression(train, test, target, label):
    
    # Handle categorical variables
    train = pd.get_dummies(train, columns=['stage'])
    test = pd.get_dummies(test, columns=['stage'])
    
    # Separate predictors (X) and outcome (y)
    X_train = train.drop(columns=[target, 'bp'])
    X_test = test.drop(columns=[target, 'bp'])
    
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
    
    print(f"Performance for {label}:")
    print(f"  - Accuracy: {acc:.4f}")
    print(f"  - Recall: {recall:.4f}")
    print(f"  - Precision: {precision:.4f}")
    print("  - Confusion Matrix:")
    
    # Plot confusion matrix
    plt.figure(figsize=(6,5))
    sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', xticklabels=['Low BP', 'High BP'], yticklabels=['Low BP', 'High BP'])
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.title(f'Confusion Matrix for {label}')
    plt.show()
    
    
    return model, acc, recall, precision, conf_matrix


### ---- ORIGINAL BASELINE ----

""" Predicting continuous outcome bloodpressure """

model, mse, r2 = linear_regression(train_data, test_data, 'bp', label = 'Original Baseline (continuous outcome)')

""" ¨Predicting categorical outcome bloodpressure """

#creating cat version of bp
train_data2 = train_data.copy()
test_data2 = test_data.copy()
train_data2['bp_high'] = np.where(train_data['bp'] >= 140, 1, 0)
test_data2['bp_high'] = np.where(test_data['bp'] >= 140, 1, 0)
log_model, acc, recall, precision, cm = logistic_regression(train_data2, test_data2, 'bp_high', label='Original Baseline (binary outcome)')

### ---- SYNTHETIC BASELINE ----

from sdv.single_table import CTGANSynthesizer


metadata = {
    "primary_key": None,  # No explicit primary key in the dataset
    "columns": {
        "age": {"sdtype": "numerical", "computer_representation": "Float"},
        "weight": {"sdtype": "numerical", "computer_representation": "Float"},
        "stage": {"sdtype": "categorical"},
        "therapy": {"sdtype": "categorical"},
        "bp": {"sdtype": "numerical", "computer_representation": "Float"},
    }
}


from sdv.metadata import SingleTableMetadata

index = train_data['Index']
train_data = train_data.drop(columns=['Index'])

# Convert metadata dictionary to SingleTableMetadata object
metadata_obj = SingleTableMetadata.load_from_dict(metadata)

# Initialize the synthesizer with the metadata object
synthesizer = CTGANSynthesizer(metadata_obj)

# Fit the synthesizer
print('\n fitting synthesizer ...')
synthesizer.fit(train_data)

# Generate synthetic data
print('\n generating synthetic data ...')
synthetic_train_data = synthesizer.sample(num_rows=len(train_data))
synthetic_train_data.insert(0, 'Index', index)
                            
""" Predicting continuous outcome bloodpressure (on synthetic data) """

model, mse, r2 = linear_regression(synthetic_train_data, test_data, 'bp', label = 'Synthetic Baseline (continuous outcome)')

""" ¨Predicting categorical outcome bloodpressure """

#creating cat version of bp
synthetic_train_data2 = synthetic_train_data.copy()
synthetic_train_data2['bp_high'] = np.where(synthetic_train_data['bp'] >= 140, 1, 0)
log_model, acc, recall, precision, cm = logistic_regression(synthetic_train_data2, test_data2, 'bp_high', label='Synthetic Baseline (binary outcome)')

