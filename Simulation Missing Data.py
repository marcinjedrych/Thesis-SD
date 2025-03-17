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

def generate_patient_data(nsamples=10000, seed=123):
    np.random.seed(seed)  # Set seed for reproducibility
    
    # Generate Age and Weight
    age = np.random.normal(loc=50, scale=10, size=nsamples)
    weight = np.random.normal(loc=70, scale=15, size=nsamples)  # New predictor
    
    # Disease Stage (Ordinal, affected by Age)
    stage_tags = ['I', 'II', 'III', 'IV']
    stage_intercepts = [2, 3, 4]  # Thresholds for stages
    stage_beta_age = -0.1  # Older individuals less likely in higher stages
    
    stage_logodds = np.array(stage_intercepts).reshape(len(stage_intercepts),1) + np.vstack([stage_beta_age * age] * len(stage_intercepts))
    stage_cumprob = expit(stage_logodds)
    stage_probs = np.hstack([stage_cumprob[0].reshape(-1,1), 
                              stage_cumprob[1].reshape(-1,1) - stage_cumprob[0].reshape(-1,1), 
                              stage_cumprob[2].reshape(-1,1) - stage_cumprob[1].reshape(-1,1),
                              1 - stage_cumprob[2].reshape(-1,1)])
    
    stage = np.array([np.random.choice(stage_tags, p=prob) for prob in stage_probs])
    
    # Therapy (Binary, Random)
    therapy = np.random.choice([False, True], size=nsamples, p=[0.5, 0.5])
    
    # Blood Pressure (BP) influenced by Stage, Therapy, and Weight
    bp_intercept = 120
    bp_beta_stage = [0, 10, 20, 30]  # Increasing severity increases BP
    bp_beta_therapy = -20  # Therapy lowers BP
    bp_beta_weight = 0.5  # Higher weight increases BP
    
    stage_to_bp_beta = dict(zip(stage_tags, bp_beta_stage))
    bp_betas = bp_intercept + np.array([stage_to_bp_beta[s] for s in stage]) + bp_beta_therapy * therapy + bp_beta_weight * weight
    
    bp = np.random.normal(loc=bp_betas, scale=10, size=nsamples)
    
    # Create DataFrame
    data = pd.DataFrame({'age': age, 'weight': weight, 'stage': stage, 'therapy': therapy, 'bp': bp})
    return data


def plot_relationships(data):
    sns.set(style="whitegrid")
    
    # Effect of Age on Disease Stage (Violin Plot)
    plt.figure(figsize=(6, 5))
    sns.violinplot(x='stage', y='age', data=data, order=['I', 'II', 'III', 'IV'])
    plt.title("Effect of Age on Disease Stage")
    plt.show()
    
    # Effect of Weight on Bloodpressure (scatterplot with loess)
    sns.regplot(x='weight', y='bp', data=data, lowess=True, scatter_kws={'alpha':0.5})
    plt.title("Effect of Weight on Blood Pressure")
    plt.show()
    
    # Barplots for categorical variables
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    sns.barplot(x='stage', y='bp', data=data, ax=axes[0], order=['I', 'II', 'III', 'IV'])
    axes[0].set_title("Effect of Disease Stage on Blood Pressure")
    sns.barplot(x='therapy', y='bp', data=data, ax=axes[1])
    axes[1].set_title("Effect of Therapy on Blood Pressure")
    plt.show()

data = generate_patient_data(500)
plot_relationships(data)

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
    sns.violinplot(x=df_mcar['missing'], y=df_mcar['age'], inner="quart")
    plt.xticks([0, 1], ['Observed', 'Missing'])
    plt.xlabel(f'{target_column} Missingness')
    plt.ylabel('Age')
    plt.title('MCAR Violin plot')
    plt.show()

    return df_mcar

data_mcar = mcar(data, target_column='bp')
plot_missingness(data_mcar, "MCAR Missingness Pattern")


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

def mar(df, target_column, predictor_column, target_missing_rate=0.2, beta_1=0.1, seed=42):
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
    sns.violinplot(x=df_mar['missing'], y=df_mar['age'], inner="quart")
    plt.xticks([0, 1], ['Observed', 'Missing'])
    plt.xlabel(f'{target_column} Missingness')
    plt.ylabel('Age')
    plt.title('MAR Violin plot')
    plt.show()
    
    return df_mar

data_mar = mar(data, target_column='bp', predictor_column='age')
plot_missingness(data_mar, "MAR Missingness Pattern")

# MISSINGNESS NOT AT RANDOM

def mnar(df, target_column, target_missing_rate=0.2, beta_1=0.1, seed=42):
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

data_mnar = mnar(data, target_column='bp')
plot_missingness(data_mnar, "MNAR Missingness Pattern")

