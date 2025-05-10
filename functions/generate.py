# -*- coding: utf-8 -*-
"""
Created on Tue Apr  8 15:54:18 2025

@author: Marcin
"""

import numpy as np
import pandas as pd
from scipy.special import expit
import seaborn as sns
import matplotlib.pyplot as plt

def generate_patient_data(nsamples=10000, seed=123):
    np.random.seed(seed)
    
    # Generate Age and Weight
    age = np.random.normal(loc=50, scale=10, size=nsamples)
    weight = np.random.normal(loc=70, scale=15, size=nsamples)
    
    # Disease Stage (Ordinal, affected by Age)
    stage_tags = ['I', 'II', 'III', 'IV']
    stage_intercepts = [2, 3, 4]
    stage_beta_age = -0.1
    
    stage_logodds = np.array(stage_intercepts).reshape(len(stage_intercepts),1) + np.vstack([stage_beta_age * age] * len(stage_intercepts))
    stage_cumprob = expit(stage_logodds)
    stage_probs = np.hstack([stage_cumprob[0].reshape(-1,1), 
                            stage_cumprob[1].reshape(-1,1) - stage_cumprob[0].reshape(-1,1),
                            stage_cumprob[2].reshape(-1,1) - stage_cumprob[1].reshape(-1,1),
                            1 - stage_cumprob[2].reshape(-1,1)])
    
    stage = np.array([np.random.choice(stage_tags, p=prob) for prob in stage_probs])
    
    # Therapy (Binary, Random)
    therapy = np.random.choice([False, True], size=nsamples, p=[0.5, 0.5])
    
    # Blood Pressure (BP)
    bp_intercept = 120
    bp_beta_stage = [0, 10, 20, 30]
    bp_beta_therapy = -20
    bp_beta_weight = 0.5
    bp_beta_age = 1
    
    stage_to_bp_beta = dict(zip(stage_tags, bp_beta_stage))
    bp_betas = bp_intercept + np.array([stage_to_bp_beta[s] for s in stage]) + bp_beta_therapy * therapy + bp_beta_weight * weight + bp_beta_age * age
    bp = np.random.normal(loc=bp_betas, scale=10, size=nsamples)
    
    # Hospital Death (New Binary Variable)
    death_intercept = -6
    death_beta_age = 0.04       # 1 year ≈ OR 1.04
    death_beta_stage = 0      # Per-stage increase
    death_beta_bp = 0.02        # mmHg increase
    death_beta_weight = 0  
    death_beta_therapy = 0   
    
    stage_num = np.array([stage_tags.index(s)+1 for s in stage])
    log_odds = (death_intercept +
                death_beta_age * age +
                death_beta_stage * stage_num +
                death_beta_bp * bp +
                death_beta_weight * weight +
                death_beta_therapy * therapy.astype(int))
    
    death_prob = expit(log_odds)
    hospitaldeath = np.random.binomial(1, death_prob)
    
    data = pd.DataFrame({
        'age': age,
        'weight': weight,
        'stage': stage,
        'therapy': therapy,
        'bp': bp,
        'hospitaldeath': hospitaldeath
    })
    
    return data

def plot_relationships(data):
    sns.set(style="whitegrid")
    
    # Original plots
    plt.figure(figsize=(6, 5))
    sns.violinplot(x='stage', y='age', data=data, order=['I', 'II', 'III', 'IV'])
    plt.title("Effect of Age on Disease Stage")
    plt.show()
    
    plt.figure(figsize=(6, 5))
    sns.regplot(x='weight', y='bp', data=data, lowess=True, scatter_kws={'alpha':0.5})
    plt.title("Effect of Weight on Blood Pressure")
    plt.show()
    
    plt.figure(figsize=(6, 5))
    sns.regplot(x='age', y='bp', data=data, lowess=True, scatter_kws={'alpha':0.5})
    plt.title("Effect of age on Blood Pressure")
    plt.show()
    
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    sns.barplot(x='stage', y='bp', data=data, ax=axes[0], order=['I', 'II', 'III', 'IV'])
    axes[0].set_title("Effect of Disease Stage on Blood Pressure")
    sns.barplot(x='therapy', y='bp', data=data, ax=axes[1])
    axes[1].set_title("Effect of Therapy on Blood Pressure")
    plt.show()
    
    
    # New hospital death plots - optimized layout
    plt.figure(figsize=(15, 18))  # Adjusted for better spacing
    
    # --- Plot 1: BP vs Death ---
    plt.subplot(3, 2, 1)  # Row 1, Col 1
    sns.regplot(x='bp', y='hospitaldeath', data=data, logistic=True, ci=None,
                scatter_kws={'alpha':0.2, 'color':'gray'}, line_kws={'color':'red'})
    plt.title("A) Death Probability by Blood Pressure", pad=20)
    
    # --- Plot 2: Age vs Death ---
    plt.subplot(3, 2, 2)  # Row 1, Col 2
    sns.regplot(x='age', y='hospitaldeath', data=data, logistic=True, ci=None,
                scatter_kws={'alpha':0.2, 'color':'gray'}, line_kws={'color':'red'})
    plt.title("B) Death Probability by Age", pad=20)
    
    # --- Plot 3: Stage vs Death ---
    plt.subplot(3, 2, 3)  # Row 2, Col 1 (span full width)
    plt.subplots_adjust(wspace=0.3, hspace=0.4)  # Add spacing
    sns.barplot(x='stage', y='hospitaldeath', data=data, order=['I', 'II', 'III', 'IV'])
    plt.title("C) Mortality Rate by Stage", pad=20)
    
    # --- Plot 4: Therapy vs Death ---
    plt.subplot(3, 2, 4)  # Row 2, Col 2
    sns.barplot(x='therapy', y='hospitaldeath', data=data)
    plt.title("D) Mortality Rate by Therapy", pad=20)
    
    # --- Plot 5: Weight vs Death ---
    plt.subplot(3, 1, 3)  # Row 3 (full width)
    sns.regplot(x='weight', y='hospitaldeath', data=data, logistic=True, ci=None,
                scatter_kws={'alpha':0.2, 'color':'gray'}, line_kws={'color':'red'})
    plt.title("E) Death Probability by Weight", pad=20)
    
    plt.tight_layout()
    plt.show()

# Generate and plot data
data = generate_patient_data(2000)
plot_relationships(data)