# -*- coding: utf-8 -*-
"""
Created on Tue Apr  8 16:34:34 2025

@author: Marcin
"""

import numpy as np
import pandas as pd
from scipy.special import expit
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder


def plot_missingness(df, title):
    plt.figure(figsize=(8, 6))
    sns.heatmap(df.isnull(), cmap='viridis', cbar=False, yticklabels=False)
    plt.title(title)
    plt.show()

## MISSING COMPLETELY AT RANDOM

def mcar(df, target_column, target_missing_rate=0.45, seed=123):
    
    np.random.seed(seed)
    df_mcar = df.copy()
    
    # Create missing values randomly
    mask = np.random.binomial(1, target_missing_rate, size=len(df)).astype(bool)
    df_mcar.loc[mask, target_column] = np.nan
    
    missingpr = (df_mcar[target_column].isna().sum() / len(df_mcar[target_column])) *100
    
    # print('\n _____________________________________________ \n')
    # print("      MISSING COMPLETELY AT RANDOM (MCAR)")
    # print(f'\nTarget column: {target_column}')
    # print(f'Target missing rate: {target_missing_rate}')
    # print(f'\n Actual missing: {missingpr:.2f}')
    # print('\n _____________________________________________ \n')
    
    # df_mcar['missing'] = df_mcar[target_column].isna()
    # plt.figure(figsize=(8, 6))
    # sns.countplot(x='stage', hue='missing', data=df_mcar)
    # plt.xlabel('stage')
    # plt.ylabel('Count')
    # plt.title('MCAR Count Plot')
    # plt.show()
    # df_mcar = df_mcar.drop(columns=['missing'])

    return df_mcar, missingpr




def find_beta_0(df, predictor_col, target_missing_rate, beta_1, 
                tol=1e-4, max_iter=1000):
    beta_0 = 0
    step_size = 0.1
    
    for i in range(max_iter):
        logits = beta_0 + beta_1 * df[predictor_col]
        probs = expit(logits)
        current_missing_rate = probs.mean()
        
        diff = current_missing_rate - target_missing_rate
        
        if abs(diff) < tol:
            return beta_0  # Converged
        
        # Adjust beta_0
        beta_0 -= step_size * diff
    
    # no convergence warning
    print(
        f"WARNING: find_beta_0 did not converge after {max_iter} iterations.\n"
        f"   Target missing rate = {target_missing_rate:.4f}, "
        f"achieved = {current_missing_rate:.4f} "
        f"(diff = {diff:.4f})"
    )
    return beta_0  # Return the best we could get


## MISSING AT RANDOM (MAR) 

def mar(df, target_column, target_missing_rate=0.45, beta_1=0.6, seed=124,  predictor_column = "stage"):
    
    np.random.seed(seed)
    df_mar = df.copy()

    # Convert to numerical if necessary
    if df_mar[predictor_column].dtype.name in ['category', 'object']:
        df_mar['predictor_num'] = pd.Categorical(df_mar[predictor_column]).codes + 1
    else:
        df_mar['predictor_num'] = df_mar[predictor_column]

    # Calculate beta0 to get approximately the desired proportion of missingness
    beta_0 = find_beta_0(df_mar, 'predictor_num', target_missing_rate, beta_1)

    # Calculate missingness probability
    logits = beta_0 + beta_1 * df_mar['predictor_num']
    probs = expit(logits)

    # Create missing values based on probability
    mask = np.random.rand(len(df_mar)) < probs
    df_mar.loc[mask, target_column] = np.nan
    
    actual_missing = df_mar[target_column].isna().mean() * 100

    # print('\n _____________________________________________ \n')
    # print("      MISSING AT RANDOM (LOGISTIC MODEL)")
    # print(f'\n Predictor column: {predictor_column}')
    # print(f' Target column: {target_column}')
    # print(f' Target missing rate: {target_missing_rate*100:.2f}%')
    # print(f' Beta_1: {beta_1}')
    # print(f' Computed beta_0: {beta_0:.2f}')
    # print(f' Actual missing: {actual_missing:.2f}%')
    # print(' _____________________________________________ \n')

    # Plot
    # df_mar['missing'] = df_mar[target_column].isna()
    # plt.figure(figsize=(8, 6))
    # sns.countplot(x=predictor_column, hue='missing', data=df_mar)
    # plt.xlabel(predictor_column)
    # plt.ylabel('Count')
    # plt.title('MAR Count Plot')
    # plt.show()
    # df_mar = df_mar.drop(columns=['missing'])

    return df_mar, actual_missing


# MISSINGNESS NOT AT RANDOM

def mnar(df, target_column, target_missing_rate=0.45, beta_1=0.1, seed=123):
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
    
    # print('\n _____________________________________________ \n')
    # print("      MISSING NOT AT RANDOM (LOGISTIC MODEL)")
    # print(f'\n Target column: {target_column}')
    # print(f' Target missing rate: {target_missing_rate}')
    # print(f' Beta_1: {beta_1}')
    
    # print(f'\n Computed beta_0: {beta_0:.2f}')
    # print(f' Actual missing: {missingpr:.2f} %')
    # print('\n _____________________________________________ \n')
    
    # Plot missingness probability function
    # plt.figure(figsize=(10, 6))
    # plt.scatter(df_mnar[target_column], missing_prob, alpha=0.5)
    # sns.regplot(x=df_mnar[target_column], y=missing_prob, logistic=True, scatter=False, color='red')
    # plt.xlabel(f'{target_column}')  
    # plt.ylabel(f'Probability of missing {target_column}')
    # plt.title(f'MNAR (Logistic): Probability of missingness by {target_column}')
    # plt.ylim(0, 1)
    # plt.show()
    
    # Violin plot 
    # df_mnar['missing'] = df_mnar[target_column].isna()
    # plt.figure(figsize=(8, 6))
    # sns.violinplot(
    #     x='missing',
    #     y=target_column,
    #     hue='missing',  # Voeg deze regel toe
    #     data=pd.concat([df_mnar[['missing']], df[[target_column]]], axis=1),
    #     inner="quart",
    #     palette={False: 'C0', True: 'orange'},
    #     legend=False  # voorkomt dubbele legenda
    # )
    # plt.xticks([0, 1], ['Observed', 'Missing'])
    # plt.xlabel(f'{target_column} Missingness')
    # plt.ylabel(f'{target_column}')
    # plt.title('MNAR Violin plot')
    # plt.show()
    # df_mnar = df_mnar.drop(columns=['missing'])
    
    return df_mnar, missingpr