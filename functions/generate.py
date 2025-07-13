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
from sklearn.model_selection import train_test_split

def generate_patient_data(nsamples=10000, seed=123):
    np.random.seed(seed)
    
    # Generate Stage independently
    stage_tags = ['I', 'II', 'III', 'IV']
    stage_probs = [0.25, 0.25, 0.25, 0.25]  # uniform distribution
    stage = np.random.choice(stage_tags, size=nsamples, p=stage_probs)
    stage_num = np.array([stage_tags.index(s) + 1 for s in stage])  # For modeling
    
    # Generate Therapy independently
    therapy = np.random.choice([False, True], size=nsamples, p=[0.5, 0.5])
    
    # Generate latent2 (previously Age)
    latent2 = np.random.normal(loc=60, scale=10, size=nsamples)
    #latent2 = np.random.gamma(shape=36, scale=1.6667, size=nsamples)
    
    # Generate BP independently
    bp = np.random.normal(loc=120, scale=10, size=nsamples)
    
    # Generate Weight
    # Weight based on BP (e.g., weight increases with BP)
    latent1_intercept = 30
    latent1_beta_bp = 0.4
    
    latent1 = (
        latent1_intercept +
        latent1_beta_bp * bp +
        np.random.normal(0, 15, nsamples)
    )

    # Define effects of independent predictors on hospitaldeath
    death_intercept = -36
    death_beta_stage = 0.27
    death_beta_bp = 0.3
    death_beta_therapy = -0.2
    death_beta_latent2 = 0.04
    death_beta_latent1 = -0.04
    
    log_odds = (
        death_intercept +
        death_beta_stage * stage_num +
        death_beta_bp * bp +
        death_beta_therapy * therapy.astype(int) +
        death_beta_latent2 * latent2 +
        death_beta_latent1 * latent1 +
        np.random.normal(0, 13, size=len(bp))  # random error
    )
    
    death_prob = expit(log_odds)
    hospitaldeath = np.random.binomial(1, death_prob)
    
    data = pd.DataFrame({
        'stage': stage,
        'therapy': therapy,
        'bp': bp,
        'hospitaldeath': hospitaldeath,
        'latent1' : latent1,
        'latent2': latent2
    })
    
    return data

def plot_relationships(data):
    
    sns.set(style="whitegrid")
    
    if 'hospitaldeath' in data.columns:
        #class balance
        data['hospitaldeath'].value_counts().plot(kind='bar')
        plt.xlabel('hospitaldeath')
        plt.ylabel('Frequency')
        plt.title('Class Balance of hospdeath')
        plt.xticks([0, 1], rotation=0)
        plt.show()
        
    # Original plots
    # plt.figure(figsize=(6, 5))
    # sns.violinplot(x='stage', y='age', data=data, order=['I', 'II', 'III', 'IV'], hue = 'stage')
    # plt.title("Effect of Age on Disease Stage")
    # plt.show()
    
    # plt.figure(figsize=(6, 5))
    # sns.regplot(x='bp', y='latent1', data=data, lowess=True, scatter_kws={'alpha':0.5})
    # plt.title("Effect of Blood Pressure on latent variable 1")
    # plt.show()
    
    # plt.figure(figsize=(6, 5))
    # sns.regplot(x='age', y='bp', data=data, lowess=True, scatter_kws={'alpha':0.5})
    # plt.title("Effect of age on Blood Pressure")
    # plt.show()
    
    # fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    # sns.barplot(x='stage', y='bp', data=data, ax=axes[0], order=['I', 'II', 'III', 'IV'], hue = 'stage')
    # axes[0].set_title("Effect of Disease Stage on Blood Pressure")
    # sns.barplot(x='therapy', y='bp', data=data, ax=axes[1], hue= 'therapy', legend=False)
    # axes[1].set_title("Effect of Therapy on Blood Pressure")
    # plt.show()
    
    
    if 'hospitaldeath' in data.columns:
        
        plt.figure(figsize=(15, 18))
    
        # --- Plot 1: BP vs Death ---
        plt.subplot(3, 2, 1)  # Row 1, Col 1
        sns.regplot(x='bp', y='hospitaldeath', data=data, logistic=True, ci=None,
                    scatter_kws={'alpha':0.2, 'color':'gray'}, line_kws={'color':'red'})
        plt.title("A) Death Probability by Blood Pressure", pad=20)
        
        # --- Plot 2: Age vs Death ---
        plt.subplot(3, 2, 2)  # Row 1, Col 2
        sns.regplot(x='latent2', y='hospitaldeath', data=data, logistic=True, ci=None,
                    scatter_kws={'alpha':0.2, 'color':'gray'}, line_kws={'color':'red'})
        plt.title("B) Death Probability by latent variable 2", pad=20)
        
        # --- Plot 3: Stage vs Death ---
        plt.subplot(3, 2, 3)  # Row 2, Col 1 (span full width)
        plt.subplots_adjust(wspace=0.3, hspace=0.4)  # Add spacing
        sns.barplot(x='stage', y='hospitaldeath', data=data, order=['I', 'II', 'III', 'IV'], hue='stage', legend = False)
        plt.title("C) Mortality Rate by Stage", pad=20)
        
        # --- Plot 4: Therapy vs Death ---
        plt.subplot(3, 2, 4)  # Row 2, Col 2
        sns.barplot(x='therapy', y='hospitaldeath', data=data, hue='therapy', legend = False)
        plt.title("D) Mortality Rate by Therapy", pad=20)
        
        # --- Plot 5: Weight vs Death ---
        plt.subplot(3, 1, 3)  # Row 3 (full width)
        sns.regplot(x='latent1', y='hospitaldeath', data=data, logistic=True, ci=None,
                    scatter_kws={'alpha':0.2, 'color':'gray'}, line_kws={'color':'red'})
        plt.title("E) Death Probability by latent variable 1", pad=20)
        
        plt.tight_layout()
        plt.show()

# Generate and plot data
data = generate_patient_data(30000, seed = 123)
plot_relationships(data)


# SEPERABILTY?

train_idx, test_idx = train_test_split(data.index, test_size=0.2, random_state=123)
train_data = data.loc[train_idx]
test_data = data.loc[test_idx]

stage_order = {'I': 1, 'II': 2, 'III': 3, 'IV': 4}
train_data['stage'] = pd.Categorical(train_data['stage'], categories=stage_order.keys(), ordered=True)
# If you need the numeric version (e.g., for multiplication), extract codes + 1
train_data['stage_num'] = train_data['stage'].cat.codes + 1

from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression

train_data = train_data.drop(columns=['stage'])

# Features en target
X = train_data.drop(columns=['hospitaldeath','latent1','latent2'])
y = train_data['hospitaldeath']

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Model fit
model = LogisticRegression(solver='liblinear')  # 'liblinear' is stabieler bij kleine datasets
model.fit(X_scaled, y)

# Coëfficiënts
print(pd.Series(model.coef_[0], index=X.columns).sort_values())

probs = model.predict_proba(X_scaled)[:, 1]

plt.figure(figsize=(8,4))
sns.histplot(probs, bins=20, kde=False)
plt.title("Histogram of hospdeath predictions")
plt.xlabel("Predicted probability")
plt.ylabel("# observations")
plt.show()

df = train_data.copy()
df['hospitaldeath'] = y
print(df.groupby('hospitaldeath').mean())

from sklearn.decomposition import PCA

pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_scaled)

plt.figure(figsize=(6,6))
sns.scatterplot(x=X_pca[:,0], y=X_pca[:,1], hue=y, palette='coolwarm', alpha=0.6)
plt.title("PCA van predictors gekleurd op hospdeath")
plt.xlabel("PC1")
plt.ylabel("PC2")
plt.legend(title="hospdeath")
plt.show()
