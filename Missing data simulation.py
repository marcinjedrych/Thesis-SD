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
        'ID': range(1, n + 1),
        'Age': age,
        'BloodPressure': blood_pressure,
        'Cholesterol': cholesterol,
        'Weight': weight
    })
    return df

def plot_missingness(df, title):
    plt.figure(figsize=(8, 6))
    sns.heatmap(df.isnull(), cmap='viridis', cbar=False, yticklabels=False)
    plt.title(title)
    plt.show()

# Generate patient data
data = generate_patient_data()

## MISSING COMPLETELY AT RANDOM

def mcar(df, column, missing_rate=0.1, seed=42):
    
    print('\n _____________________________________________ \n')
    print("      MISSING COMPLETELY AT RANDOM")
    
    print(f'\n Target column: {column}')
    print('\n _____________________________________________ \n')
    
    np.random.seed(seed)
    df_mcar = df.copy()
    mask = np.random.rand(df.shape[0]) < missing_rate  
    df_mcar.loc[mask, column] = np.nan 
    return df_mcar

data_mcar = mcar(data, 'Cholesterol')
plot_missingness(data_mcar, "MCAR Missingness Pattern")

## MISSING AT RANDOM

def mar(df, target_column, predictor_column, base_missing_rate=0.05, max_missing_rate=0.3, noise_level=0.1, seed=42):
    np.random.seed(seed)
    df_mar = df.copy()

    # Normalize the predictor column between 0 and 1
    min_predictor, max_predictor = df[predictor_column].min(), df[predictor_column].max()
    normalized_predictor = (df[predictor_column] - min_predictor) / (max_predictor - min_predictor)

    # Missing probability with noise
    missing_prob = base_missing_rate + (max_missing_rate - base_missing_rate) * normalized_predictor
    missing_prob += np.random.normal(0, noise_level, len(df))  
    missing_prob = np.clip(missing_prob, 0, 1)  # Probabilities stay within [0,1]

    # For each row, if the random number is smaller than the calculated probability, set the target column to NaN
    mask = np.random.rand(len(df)) < missing_prob
    df_mar.loc[mask, target_column] = np.nan
    
    print('\n _____________________________________________ \n')
    print("      MISSING AT RANDOM")
    
    print(f'\n Predictor column: {predictor_column}')
    print(f'\n Target column: {target_column}')
    print('\n _____________________________________________ \n')
    

    # First plot 
    plt.figure(figsize=(10, 6))
    plt.scatter(df_mar[predictor_column], missing_prob)
    sns.regplot(x=df_mar[predictor_column], y=missing_prob, lowess=True, scatter=False, color='red')
    plt.xlabel(f'{predictor_column}')  
    plt.ylabel(f'Probability of missing {target_column}')
    plt.title(f'MAR: Probability of missingness influenced by {predictor_column}')
    plt.ylim(0, 1)
    plt.show()

    # Second plot
    plt.figure(figsize=(10, 6))
    plt.scatter(df_mar['ID'], df_mar[predictor_column], c=df_mar[target_column].isna(), cmap='coolwarm', edgecolor='k')
    plt.xlabel('ID')
    plt.ylabel(f'{predictor_column}')
    plt.title(f'{predictor_column} vs. ID with missing data indicated in red')
    plt.show()
    
    return df_mar

data_mar = mar(data, target_column='Cholesterol', predictor_column='Age')
plot_missingness(data_mar, "MAR Missingness Pattern")

## MISSING NOT AT RANDOM

def mnar(df, target_column, base_missing_rate=0.05, max_missing_rate=0.3, noise_level=0.02, seed=42):
    np.random.seed(seed)
    df_mnar = df.copy()
    
    # Store the original values of the target column
    df_mnar[f'Original_{target_column}'] = df_mnar[target_column]

    # Normalize the target column between 0 and 1
    min_target, max_target = df[target_column].min(), df[target_column].max()
    normalized_target = (df[target_column] - min_target) / (max_target - min_target)

    # Missing probability with noise
    missing_prob = base_missing_rate + (max_missing_rate - base_missing_rate) * normalized_target
    missing_prob += np.random.normal(0, noise_level, len(df))  
    missing_prob = np.clip(missing_prob, 0, 1)  # Probabilities stay within [0,1]

    # For each row, if the random number is smaller than the calculated probability, set the target column to NaN
    mask = np.random.rand(len(df)) < missing_prob
    df_mnar.loc[mask, target_column] = np.nan
    
    print('\n _____________________________________________ \n')
    print("      MISSING NOT AT RANDOM")
    
    print(f'\n Predictor column: {target_column}')
    print(f'\n Target column: {target_column}')
    print('\n _____________________________________________ \n')
      
    # First plot 
    plt.figure(figsize=(10, 6))
    plt.scatter(df_mnar[target_column], missing_prob)
    sns.regplot(x=df_mnar[target_column], y=missing_prob, lowess=True, scatter=False, color='red')
    plt.xlabel(f'{target_column}')
    plt.ylabel(f'Probability of missing {target_column}')
    plt.title(f'MNAR: Probability of missingness influenced by {target_column}')
    plt.ylim(0, 1)
    plt.show()

    # Second plot
    plt.figure(figsize=(10, 6))
    
    # Plot non-missing values
    plt.scatter(df_mnar['ID'][~df_mnar[target_column].isna()], 
                df_mnar[target_column][~df_mnar[target_column].isna()], 
                c='blue', label='Non-missing', edgecolor='k')
    
    # Plot missing values using original values
    plt.scatter(df_mnar['ID'][df_mnar[target_column].isna()], 
                df_mnar[f'Original_{target_column}'][df_mnar[target_column].isna()], 
                c='red', label='Missing', edgecolor='k')
    
    plt.xlabel('ID')
    plt.ylabel(f'{target_column}')
    plt.title(f'{target_column} vs. ID with missing data indicated in red')
    plt.legend()
    plt.show()
    
    return df_mnar

data_mnar = mnar(data, target_column='Cholesterol')
plot_missingness(data_mnar, "MNAR Missingness Pattern")


