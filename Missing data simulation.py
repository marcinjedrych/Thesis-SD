# -*- coding: utf-8 -*-
"""
Created on Sun Mar  2 14:32:57 2025

Missingness scenarios

@author: Marcin
"""
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

def generate_patient_data(n=1000, seed=42):
    np.random.seed(seed)
    age = np.random.normal(50, 15, n).astype(int)  # Age in years
    blood_pressure = np.random.normal(120, 15, n)  # Systolic blood pressure
    cholesterol = np.random.normal(200, 30, n)  # Cholesterol level
    weight = np.random.normal(70, 10, n)  # Weight in kg
    
    df = pd.DataFrame({
        'Age': age,
        'BloodPressure': blood_pressure,
        'Cholesterol': cholesterol,
        'Weight': weight
    })
    return df

def introduce_mcar(df, missing_rate=0.1, seed=42):
    np.random.seed(seed)
    df_mcar = df.copy()
    mask = np.random.rand(*df.shape) < missing_rate
    df_mcar[mask] = np.nan
    return df_mcar

def introduce_mar(df, missing_rate=0.1, seed=42):
    np.random.seed(seed)
    df_mar = df.copy()
    
    # Let's say missingness in cholesterol depends on age (e.g., older patients are less likely to report it)
    age_threshold = np.percentile(df['Age'], 50)  # Median age
    mask = (df['Age'] > age_threshold) & (np.random.rand(len(df)) < missing_rate)
    df_mar.loc[mask, 'Cholesterol'] = np.nan
    
    return df_mar

def introduce_mnar(df, missing_rate=0.1, seed=42):
    np.random.seed(seed)
    df_mnar = df.copy()
    
    # Missingness in blood pressure depends on its own value (e.g., very high BP is not reported)
    bp_threshold = np.percentile(df['BloodPressure'], 75)  # Upper quartile
    mask = (df['BloodPressure'] > bp_threshold) & (np.random.rand(len(df)) < missing_rate)
    df_mnar.loc[mask, 'BloodPressure'] = np.nan
    
    return df_mnar

def plot_missingness(df, title):
    plt.figure(figsize=(8, 6))
    sns.heatmap(df.isnull(), cmap='viridis', cbar=False, yticklabels=False)
    plt.title(title)
    plt.show()

def plot_dependency_scatter(df, col_missing, col_dependent, title):
    plt.figure(figsize=(8, 6))
    missing_mask = df[col_missing].isnull()

    plt.scatter(df.loc[~missing_mask, col_dependent], np.ones(sum(~missing_mask)), label="Observed", alpha=0.5)
    plt.scatter(df.loc[missing_mask, col_dependent], np.zeros(sum(missing_mask)), label="Missing", color='red', alpha=0.5)

    plt.xlabel(col_dependent)
    plt.ylabel("Missing Status (1=Observed, 0=Missing)")
    plt.title(title)
    plt.legend()
    plt.show()

# Generate patient data
data = generate_patient_data()

# Introduce different missingness mechanisms
data_mcar = introduce_mcar(data)
data_mar = introduce_mar(data)
data_mnar = introduce_mnar(data)

# Display examples
print("Original Data:\n", data.head())
print("\nMCAR Data:\n", data_mcar.head())
print("\nMAR Data:\n", data_mar.head())
print("\nMNAR Data:\n", data_mnar.head())

# Plot missingness patterns
plot_missingness(data_mcar, "MCAR Missingness Pattern")
plot_missingness(data_mar, "MAR Missingness Pattern")
plot_missingness(data_mnar, "MNAR Missingness Pattern")

# Scatter plots to visualize missingness dependency
plot_dependency_scatter(data_mar, 'Cholesterol', 'Age', "MAR: Missing Cholesterol vs. Age")
