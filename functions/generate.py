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

plots = False
seperability_check = False

def generate_patient_data(nsamples=10000, seed=123):
    np.random.seed(seed)
    
    # Generate Stage independently
    stage_tags = ['I', 'II', 'III', 'IV']
    stage_probs = [0.25, 0.25, 0.25, 0.25]  # uniform distribution
    stage = np.random.choice(stage_tags, size=nsamples, p=stage_probs)
    stage_num = np.array([stage_tags.index(s) + 1 for s in stage])  # For modeling
    
    # Generate Therapy independently
    therapy = np.random.choice([False, True], size=nsamples, p=[0.5, 0.5])
    
    # Generate latent2 (previously age)
    latent2 = np.random.normal(loc=60, scale=10, size=nsamples)
    #latent2 = np.random.gamma(shape=36, scale=1.6667, size=nsamples) # try different distribution
    
    # Generate BP independently
    bp = np.random.normal(loc=120, scale=10, size=nsamples)
    
    # Generate latent (previously weight)
    latent1_intercept = 30
    latent1_beta_bp = 0.4
    
    latent1 = (
        latent1_intercept +
        latent1_beta_bp * bp +
        np.random.normal(0, 15, nsamples)
    )

    # Define effects of independent predictors on hospitaldeath
    death_intercept = -48 #-36
    death_beta_stage = 0.27
    death_beta_bp = 0.4 #0.3
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

# --- Loop ----------------------
n_runs = 250
base_seed = 122
dfs = []
for i in range(n_runs):
    df_i = generate_patient_data(nsamples=10000, seed=base_seed + i)
    df_i['_run'] = i
    dfs.append(df_i)
all_data = pd.concat(dfs, ignore_index=True)
  
def plot_relationships(data, aggregate=False, nbins=40, min_count=40):
    
    sns.set(style="whitegrid")
    
    if not aggregate or '_run' not in data.columns:
        
        # -------- ORIGINAL SINGLE-SEED PLOTS --------
        if 'hospitaldeath' in data.columns:
            #class balance
            data['hospitaldeath'].value_counts().plot(kind='bar')
            plt.xlabel('hospitaldeath')
            plt.ylabel('Frequency')
            plt.title('Class Balance of hospdeath')
            plt.xticks([0, 1], rotation=0)
            plt.show()
        
        # --- Plot 1: BP vs Death ---
        #plt.subplot(3, 2, 1)  # Row 1, Col 1
        sns.regplot(x='bp', y='hospitaldeath', data=data, logistic=True, ci=None,
                    scatter_kws={'alpha':0.2, 'color':'gray'}, line_kws={'color':'red'})
        plt.title("A) Death Probability by Blood Pressure", pad=20)
        plt.show()
        
        # # --- Plot 2: latent2 vs Death ---
        # plt.subplot(3, 2, 2)  # Row 1, Col 2
        # sns.regplot(x='latent2', y='hospitaldeath', data=data, logistic=True, ci=None,
        #             scatter_kws={'alpha':0.2, 'color':'gray'}, line_kws={'color':'red'})
        # plt.title("B) Death Probability by latent variable 2", pad=20)
        
        # --- Plot 3: Stage vs Death ---
        #plt.subplot(3, 2, 3)  # Row 2, Col 1 (span full width)
        plt.subplots_adjust(wspace=0.3, hspace=0.4)  # Add spacing
        sns.barplot(x='stage', y='hospitaldeath', data=data, order=['I', 'II', 'III', 'IV'], hue='stage', legend=False)
        plt.title("C) Mortality Rate by Stage", pad=20)
        plt.show()
        
        # --- Plot 4: Therapy vs Death ---
        #plt.subplot(3, 2, 4)  # Row 2, Col 2
        sns.barplot(x='therapy', y='hospitaldeath', data=data, hue='therapy', legend=False)
        plt.title("D) Mortality Rate by Therapy", pad=20)
        plt.show()
        return

    # ----------------- MULTI-SEED PLOTS + VARIABILITY -----------------

    # 1) Class balance (both 0 and 1) with 95% CI (across seeds) — counts to mirror your original
    run_counts = (
        data.groupby(['_run', 'hospitaldeath'])
            .size().rename('count').reset_index()
    )

    bal = (run_counts.groupby('hospitaldeath')['count']
                     .agg(mean='mean', sd='std', n='size')
                     .reset_index())
    
    plt.figure()
    ax = sns.barplot(x='hospitaldeath', y='mean', data=bal,
                     hue='hospitaldeath', dodge=False, legend=False)  # keep seaborn default colors
    ax.errorbar(x=np.arange(len(bal)), y=bal['mean'],
                yerr=bal['sd'].values,         # <-- mean ± 1 SD
                fmt='none', capsize=6, linewidth=1.5, color='black')
    ax.set_title("Class Counts (mean ± 1 SD across seeds)")
    ax.set_xlabel('hospitaldeath'); ax.set_ylabel('Count')
    plt.show()

    # 2) BP vs Death
    bins = pd.cut(data['bp'], bins=nbins, include_lowest=True)
    tmp = (
        data.assign(_bin=bins)
            .groupby(['_bin', '_run'])
            .agg(bp_center=('bp', 'mean'),
                 p=('hospitaldeath', 'mean'),
                 n=('hospitaldeath', 'size'))
            .reset_index()
    )
    tmp = tmp[tmp['n'] >= min_count]
    
    curve = (
        tmp.groupby('_bin')
           .agg(bp_center=('bp_center', 'mean'),
                p_mean=('p', 'mean'),
                p_lo=('p', lambda s: s.quantile(0.025)),
                p_hi=('p', lambda s: s.quantile(0.975)))
           .sort_values('bp_center')
    )
    
    plt.figure()
    plt.fill_between(curve['bp_center'], curve['p_lo'], curve['p_hi'], alpha=0.25, linewidth=0, color="red")
    plt.plot(curve['bp_center'], curve['p_mean'], lw=2, color="red")
    plt.title("Death Probability by Blood Pressure")
    plt.xlabel('bp'); plt.ylabel('P(hospdeath=1)'); plt.ylim(0, 1)
    plt.show()


    # 3) Stage vs Death 
    stage_order = ['I', 'II', 'III', 'IV']
    stage = (data.groupby(['_run', 'stage'])['hospitaldeath'].mean().reset_index())
    
    # mean and standard deviation across runs per stage
    stage_agg = (stage.groupby('stage')['hospitaldeath']
                      .agg(mean='mean', sd='std')
                      .reset_index())
    
    stage_agg['stage'] = pd.Categorical(stage_agg['stage'],
                                        categories=stage_order, ordered=True)
    stage_agg = stage_agg.sort_values('stage')
    
    plt.figure()
    ax = sns.barplot(x='stage', y='mean', data=stage_agg, order=stage_order,
                     hue='stage', legend=False)
    ax.errorbar(x=np.arange(len(stage_agg)), y=stage_agg['mean'],
                yerr=stage_agg['sd'].values,          # mean ± 1 SD
                fmt='none', capsize=6, linewidth=1.5, color='black')
    ax.set_ylim(0, 1)
    ax.set_title("Mortality Rate by Stage (mean ± 1 SD)")
    ax.set_xlabel('stage'); ax.set_ylabel('Rate')
    plt.show()

    # 4) Therapy vs Death 
    therapy = (data.groupby(['_run', 'therapy'])['hospitaldeath'].mean().reset_index())
    
    # mean and SD across runs per therapy group
    therapy_agg = (therapy.groupby('therapy')['hospitaldeath']
                        .agg(mean='mean', sd='std')
                        .reset_index())
    
    therapy_agg['therapy'] = pd.Categorical(therapy_agg['therapy'],
                                            categories=[False, True], ordered=True)
    therapy_agg = therapy_agg.sort_values('therapy')
    
    plt.figure()
    ax = sns.barplot(x='therapy', y='mean', data=therapy_agg,
                     order=[False, True], hue='therapy', dodge=False, legend=False)
    ax.errorbar(x=np.arange(len(therapy_agg)), y=therapy_agg['mean'],
                yerr=therapy_agg['sd'].values,     # mean ± 1 SD
                fmt='none', capsize=6, linewidth=1.5, color='black')
    ax.set_ylim(0, 1)
    ax.set_title("Mortality Rate by Therapy (mean ± 1 SD across seeds)")
    ax.set_xlabel('therapy'); ax.set_ylabel('Rate')
    plt.show()


data = generate_patient_data(10000, seed=122)
if plots is True:
    #plot_relationships(data, aggregate=False)  # 1 run
    plot_relationships(all_data, aggregate=True)   # shows error bars / bands


# SEPERABILTY?
if seperability_check is True:
    train_idx, test_idx = train_test_split(data.index, test_size=0.2, random_state=123)
    train_data = data.loc[train_idx]
    test_data = data.loc[test_idx]
    
    stage_order = {'I': 1, 'II': 2, 'III': 3, 'IV': 4}
    train_data['stage'] = pd.Categorical(train_data['stage'], categories=stage_order.keys(), ordered=True)
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
    model = LogisticRegression(max_iter=1000, random_state=123)  
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
