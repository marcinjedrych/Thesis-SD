import numpy as np
import pandas as pd
from scipy.special import expit
import seaborn as sns
import matplotlib.pyplot as plt

def sample_disease(nsamples=10000, seed=123):
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

# Example usage
data = sample_disease(20)
print(data.head())


def plot_relationships(data):
    sns.set(style="whitegrid")
    
    # Effect of Age on Disease Stage (Violin Plot)
    plt.figure(figsize=(6, 5))
    sns.violinplot(x='stage', y='age', data=data, order=['I', 'II', 'III', 'IV'])
    plt.title("Effect of Age on Disease Stage")
    plt.show()
    
    # Scatterplots with LOESS smooth line
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    sns.regplot(x='weight', y='bp', data=data, lowess=True, scatter_kws={'alpha':0.5}, ax=axes[0])
    axes[0].set_title("Effect of Weight on Blood Pressure")
    sns.regplot(x='age', y='bp', data=data, lowess=True, scatter_kws={'alpha':0.5}, ax=axes[1])
    axes[1].set_title("Effect of Age on Blood Pressure")
    plt.show()
    
    # Barplots for categorical variables
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    sns.barplot(x='stage', y='bp', data=data, ax=axes[0], order=['I', 'II', 'III', 'IV'])
    axes[0].set_title("Effect of Disease Stage on Blood Pressure")
    sns.barplot(x='therapy', y='bp', data=data, ax=axes[1])
    axes[1].set_title("Effect of Therapy on Blood Pressure")
    plt.show()

# Example usage
data = sample_disease(500)
print(data.head())
plot_relationships(data)
