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
from scipy.special import expit 

def generate_patient_data(n=5000, seed=42):
    np.random.seed(seed)
    age = np.random.normal(40, 15, n).astype(int)  # Age in years
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

def mcar(df, target_column, missing_rate=0.2, seed=42):
    
    np.random.seed(seed)
    df_mcar = df.copy()
    
    # Create missing values randomly
    mask = np.random.binomial(1, missing_rate, size=len(df)).astype(bool)
    df_mcar.loc[mask, target_column] = np.nan
    
    missingpr = (df_mcar[target_column].isna().sum() / len(df_mcar[target_column])) *100
    print('\n _____________________________________________ \n')
    print("      MISSING COMPLETELY AT RANDOM (MCAR)")
    print(f'\nTarget column: {target_column}')
    print(f'Target missing rate: {missing_rate}')
    print(f'\n Actual missing: {missingpr:.2f}')
    print('\n _____________________________________________ \n')
    
    df_mcar['missing'] = df_mcar[target_column].isna()

    # Violin plot - age
    plt.figure(figsize=(8, 6))
    sns.violinplot(x=df_mcar['missing'], y=df_mcar['Age'], inner="quart")
    plt.xticks([0, 1], ['Observed', 'Missing'])
    plt.xlabel(f'{target_column} Missingness')
    plt.ylabel('Age')
    plt.title('MCAR Violin plot')
    plt.show()

    return df_mcar

data_mcar = mcar(data, target_column='Cholesterol')
plot_missingness(data_mcar, "MCAR Missingness Pattern")

## BETA 0

#calculate analytically -> to much missingness (more then target)
# def find_beta_0(df, predictor_column, target_missing_rate, beta_1):
    
#     # logistic inverse
#     logit_target_missing_rate = np.log((1 - target_missing_rate) / target_missing_rate)
    
#     beta_0 = -beta_1 * np.mean(df[predictor_column]) - logit_target_missing_rate
    
#     return beta_0

#find optimal beta0
def find_beta_0(df, predictor_column, target_missing_rate, beta_1):
    
    beta_0 = 0  # initial guess
    step_size = 0.01  # Stepsize
    tolerance = 0.001  # Allowable difference
    
    while True:
        
        # Calculate missingness probability
        logit_prob = beta_0 + beta_1 * df[predictor_column]
        missing_prob = expit(logit_prob)
        
        # Simulate
        mask = np.random.rand(len(df)) < missing_prob
        actual_missing_rate = mask.sum() / len(df)
        
        # Check if the missing rate is close to the target
        if abs(actual_missing_rate - target_missing_rate) < tolerance:
            break
        
        # Adjust beta_0 based on the difference between the actual and target missing rate
        if actual_missing_rate < target_missing_rate:
            beta_0 += step_size  # Increase beta_0
        else:
            beta_0 -= step_size  # Decrease beta_0
    
    return beta_0

## MISSING AT RANDOM (MAR) 

def mar(df, target_column, predictor_column, target_missing_rate=0.2, beta_1=0.2, seed=42):
    np.random.seed(seed)
    df_mar = df.copy()
    
    # get beta_0 
    beta_0 = find_beta_0(df, predictor_column, target_missing_rate, beta_1)
    
    # get missing probability
    logit_prob = beta_0 + beta_1 * df[predictor_column]
    missing_prob = expit(logit_prob)  # Apply sigmoid function
    
    # Create missing values based on probability
    mask = np.random.rand(len(df)) < missing_prob
    df_mar.loc[mask, target_column] = np.nan
    
    missingpr = (df_mar[target_column].isna().sum() / len(df_mar[target_column])) *100
    print('\n _____________________________________________ \n')
    print("      MISSING AT RANDOM (LOGISTIC MODEL)")
    print(f'\n Predictor column: {predictor_column}')
    print(f' Target column: {target_column}')
    print(f' Target missing rate: {target_missing_rate}')
    print(f' Beta_1: {beta_1}')
    
    print(f'\n Computed beta_0: {beta_0:.2f}')
    print(f' Actual missing: {missingpr:.2f} %')
    print('\n _____________________________________________ \n')
    
    # Plot missingness probability function
    plt.figure(figsize=(10, 6))
    plt.scatter(df_mar[predictor_column], missing_prob, alpha=0.5)
    sns.regplot(x=df_mar[predictor_column], y=missing_prob, logistic=True, scatter=False, color='red')
    plt.xlabel(f'{predictor_column}')  
    plt.ylabel(f'Probability of missing {target_column}')
    plt.title(f'MAR (Logistic): Probability of missingness by {predictor_column}')
    plt.ylim(0, 1)
    plt.show()
    
    # Violin plot 
    df_mar['missing'] = df_mar[target_column].isna()
    plt.figure(figsize=(8, 6))
    sns.violinplot(x=df_mar['missing'], y=df_mar['Age'], inner="quart")
    plt.xticks([0, 1], ['Observed', 'Missing'])
    plt.xlabel(f'{target_column} Missingness')
    plt.ylabel('Age')
    plt.title('MAR Violin plot')
    plt.show()
    
    return df_mar

data_mar = mar(data, target_column='Cholesterol', predictor_column='Age')
plot_missingness(data_mar, "MAR Missingness Pattern")

# MISSINGNESS NOT AT RANDOM

def mnar(df, target_column, target_missing_rate=0.2, beta_1=0.2, seed=42):
    np.random.seed(seed)
    df_mnar = df.copy()
    
    # get beta_0 
    beta_0 = find_beta_0(df, target_column, target_missing_rate, beta_1)
    
    # get missing probability
    logit_prob = beta_0 + beta_1 * df[target_column]
    missing_prob = expit(logit_prob)  # Apply sigmoid function
    
    # Create missing values based on probability
    mask = np.random.rand(len(df)) < missing_prob
    df_mnar.loc[mask, target_column] = np.nan
    
    missingpr = (df_mnar[target_column].isna().sum() / len(df_mnar[target_column])) * 100
    print('\n _____________________________________________ \n')
    print("      MISSING NOT AT RANDOM (LOGISTIC MODEL)")
    print(f'\n Target column: {target_column}')
    print(f' Target missing rate: {target_missing_rate}')
    print(f' Beta_1: {beta_1}')
    
    print(f'\n Computed beta_0: {beta_0:.2f}')
    print(f' Actual missing: {missingpr:.2f} %')
    print('\n _____________________________________________ \n')
    
    # Plot missingness probability function
    plt.figure(figsize=(10, 6))
    plt.scatter(df_mnar[target_column], missing_prob, alpha=0.5)
    sns.regplot(x=df_mnar[target_column], y=missing_prob, logistic=True, scatter=False, color='red')
    plt.xlabel(f'{target_column}')  
    plt.ylabel(f'Probability of missing {target_column}')
    plt.title(f'MNAR (Logistic): Probability of missingness by {target_column}')
    plt.ylim(0, 1)
    plt.show()
    
    # Violin plot 
    df_mnar['missing'] = df_mnar[target_column].isna()
    plt.figure(figsize=(8, 6))
    sns.violinplot(x=df_mnar['missing'], y=df[target_column], inner="quart") #use old df for y axis! (before missingness)
    plt.xticks([0, 1], ['Observed', 'Missing'])
    plt.xlabel(f'{target_column} Missingness')
    plt.ylabel(f'{target_column}')
    plt.title('MNAR Violin plot')
    plt.show()
    
    return df_mnar

data_mnar = mnar(data, target_column='Cholesterol')
plot_missingness(data_mnar, "MNAR Missingness Pattern")

