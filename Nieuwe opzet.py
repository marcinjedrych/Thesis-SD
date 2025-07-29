# -*- coding: utf-8 -*-
"""
Created on Sun Jul 13 15:09:06 2025

@author: Marcin
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from functions.generate import generate_patient_data
from sklearn.model_selection import train_test_split
from functions.missingness import mcar, mar, mnar
from functions.ctgan_syn import generate_synthetic_data
from functions.strategies import complete_cases, MI_impute, ensemble , missing_indicator
from functions.model import logistic_regression

import time
start_time = time.time()

n_iter = 100
exclude = ['latent1', 'latent2']

# 1 BASELINE

def run_baseline(n_iter, exclude=None, synthetic=False):
    results = []
    
    for i in range(n_iter):

        data = generate_patient_data(nsamples=250, seed=i)
        train_data, test_data = train_test_split(data, test_size=0.2, random_state=i)
        
        #test data check
        #print('baseline check: ', test_data[:3]) if i == 2 else None

        if synthetic:
            train_data = generate_synthetic_data(train_data)
        
        result = logistic_regression(train_data, test_data, 'hospitaldeath', exclude_vars=exclude)
        results.append(result)
    
    return results


# 2 Missing data

target_missing_rate=0.45
def run_models(n_iter, exclude=None, synthetic=False, m_type=None, strategy="CCA", m_amount = []):
    if m_type is None:
        m_type = mcar
    
    results = []
    
    for i in range(n_iter):
        
        # generate
        data = generate_patient_data(nsamples=250, seed=i)
        train_data, test_data = train_test_split(data, test_size=0.2, random_state=i)
        
        #test data check
        #print('non-baseline check: ', test_data[:3]) if i == 2 else None

        # missingness
        train_data, m_proportion = m_type(train_data, target_column="bp", target_missing_rate=target_missing_rate)
        m_amount.append(m_proportion)
        
        # strategy
        if strategy == "MI":
            dfs = MI_impute(train_data)
            
            if synthetic:
                imp_and_syn = []
                for df in dfs:
                    syn_df = generate_synthetic_data(df)
                    imp_and_syn.append(syn_df)
                _, _, results = ensemble(imp_and_syn, test_data, target="hospitaldeath")
            else:
                _, _, results = ensemble(dfs, test_data, target="hospitaldeath")
        
        else:
            if strategy == "CCA":
                train_data = complete_cases(train_data)
            else:
                train_data = missing_indicator(train_data, 'bp')
            
            if synthetic:
                train_data = generate_synthetic_data(train_data)
            
            result = logistic_regression(train_data, test_data, 'hospitaldeath', exclude_vars=exclude)
            results.append(result)
    
    return results, m_amount


### 1. BASELINE
baseline_original = run_baseline(n_iter, exclude)
baseline_synthetic = run_baseline(n_iter, exclude, synthetic = True)

### 2. MISSINGNESS (ORIGINAL DATA)
## 2.1 MCAR   
original_mcar_cca, M_A = run_models(n_iter, exclude)
original_mcar_mi, M_A = run_models(n_iter, exclude, strategy = "MI", m_amount = M_A)
original_mcar_ind, M_A = run_models(n_iter, exclude, strategy = "IND", m_amount = M_A)

print(f"original MCAR \n missing proportion: {np.mean(M_A)}%")

## 2.2 MAR
original_mar_cca, M_A = run_models(n_iter, exclude, m_type=mar)
original_mar_mi, M_A = run_models(n_iter, exclude, strategy = "MI", m_type = mar, m_amount = M_A)
original_mar_ind, M_A = run_models(n_iter, exclude, strategy = "IND", m_type=mar, m_amount = M_A)

print(f"original MAR \n missing proportion: {np.mean(M_A)}%")

### 2.3 MNAR
original_mnar_cca, M_A = run_models(n_iter, exclude, m_type=mnar)
original_mnar_mi, M_A = run_models(n_iter, exclude, strategy = "MI", m_type = mnar, m_amount = M_A)
original_mnar_ind, M_A = run_models(n_iter, exclude, strategy = "IND", m_type=mnar, m_amount = M_A)

print(f"original MNAR \n missing proportion: {np.mean(M_A)}%")

### 3. MISSINGNESS (SYNTHETIC DATA)
## 3.1 MCAR   
syn_mcar_cca, M_A = run_models(n_iter, exclude, synthetic=True)
syn_mcar_mi, M_A = run_models(n_iter, exclude,strategy = "MI", synthetic=True, m_amount = M_A)
syn_mcar_ind, M_A = run_models(n_iter, exclude, strategy = "IND", synthetic=True, m_amount = M_A)

print(f"synthetic MCAR \n missing proportion: {np.mean(M_A)}%")

## 3.2 MAR
syn_mar_cca, M_A = run_models(n_iter, exclude, m_type=mar, synthetic=True)
syn_mar_mi, M_A = run_models(n_iter, exclude, strategy = "MI", m_type = mar, synthetic=True, m_amount = M_A)
syn_mar_ind, M_A = run_models(n_iter, exclude, strategy = "IND", m_type=mar, synthetic=True, m_amount = M_A)

print(f"synthetic MAR \n missing proportion: {np.mean(M_A)}%")

### 3.3 MNAR
syn_mnar_cca, M_A = run_models(n_iter, exclude, m_type=mnar, synthetic=True)
syn_mnar_mi, M_A = run_models(n_iter, exclude, strategy = "MI", m_type = mnar, synthetic=True, m_amount = M_A)
syn_mnar_ind, M_A = run_models(n_iter, exclude, strategy = "IND", m_type=mnar, synthetic=True, m_amount = M_A)

print(f"synthetic MNAR \n missing proportion: {np.mean(M_A)}%")

##part2

# helper function
def aggregate_results(results_list):
    df = pd.DataFrame(results_list)
    return df.mean(), df.std()

# store AUC means and SDs
summary = {
    'missingness': [],
    'data_type': [],
    'strategy': [],
    'auc_mean': [],
    'auc_sd': []
}

def collect_metrics(results, miss_type, data_type, strategy):
    mean_vals, std_vals = aggregate_results(results)
    summary['missingness'].append(miss_type)
    summary['data_type'].append(data_type)
    summary['strategy'].append(strategy)
    summary['auc_mean'].append(mean_vals['AUC'])
    summary['auc_sd'].append(std_vals['AUC'])

# original data
collect_metrics(original_mcar_cca, 'MCAR', 'original', 'CCA')
collect_metrics(original_mcar_mi,  'MCAR', 'original', 'MI')
collect_metrics(original_mcar_ind, 'MCAR', 'original', 'IND')

collect_metrics(original_mar_cca, 'MAR', 'original', 'CCA')
collect_metrics(original_mar_mi,  'MAR', 'original', 'MI')
collect_metrics(original_mar_ind, 'MAR', 'original', 'IND')

collect_metrics(original_mnar_cca, 'MNAR', 'original', 'CCA')
collect_metrics(original_mnar_mi,  'MNAR', 'original', 'MI')
collect_metrics(original_mnar_ind, 'MNAR', 'original', 'IND')

# synthetic data
collect_metrics(syn_mcar_cca, 'MCAR', 'synthetic', 'CCA')
collect_metrics(syn_mcar_mi,  'MCAR', 'synthetic', 'MI')
collect_metrics(syn_mcar_ind, 'MCAR', 'synthetic', 'IND')

collect_metrics(syn_mar_cca, 'MAR', 'synthetic', 'CCA')
collect_metrics(syn_mar_mi,  'MAR', 'synthetic', 'MI')
collect_metrics(syn_mar_ind, 'MAR', 'synthetic', 'IND')

collect_metrics(syn_mnar_cca, 'MNAR', 'synthetic', 'CCA')
collect_metrics(syn_mnar_mi,  'MNAR', 'synthetic', 'MI')
collect_metrics(syn_mnar_ind, 'MNAR', 'synthetic', 'IND')


# Compute baseline AUC statistics
baseline_orig_mean, baseline_orig_std = aggregate_results(baseline_original)
baseline_syn_mean, baseline_syn_std = aggregate_results(baseline_synthetic)

def plot_baseline(mean_orig, std_orig, mean_syn, std_syn):
    labels = ['Original', 'Synthetic']
    auc_means = [mean_orig['AUC'], mean_syn['AUC']]
    #auc_stds = [std_orig['AUC'], std_syn['AUC']]

    plt.figure(figsize=(5, 5))
    bars = plt.bar(labels, auc_means, capsize=5, color=['cornflowerblue', 'lightcoral'])
    
    # Optional: add exact values on top of bars
    for bar, auc in zip(bars, auc_means):
        yval = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2, yval + 0.01, f'{auc:.3f}', ha='center', va='bottom')

    plt.title('Baseline AUC Comparison')
    plt.ylabel('AUC (mean)')
    plt.ylim(0.5, 1)  # Adjust depending on your results
    plt.tight_layout()
    plt.show()

plot_baseline(baseline_orig_mean, baseline_orig_std,
                         baseline_syn_mean, baseline_syn_std)

# build DataFrame
summary_df = pd.DataFrame(summary)

## plot AUC with error bars
original_df = summary_df[summary_df['data_type'] == 'original']
synthetic_df = summary_df[summary_df['data_type'] == 'synthetic']

def plot_summary(df, title, baseline_auc):
    plt.figure(figsize=(10, 5))
    ax = sns.barplot(
    data=df,
    x='missingness',
    y='auc_mean',
    hue='strategy',
    hue_order=['CCA', 'MI', 'IND'],  # expliciete volgorde
    ci=None,
    palette={'CCA': 'blue', 'MI': 'orange', 'IND': 'green'}  # kleur per strategie
)


    # Voeg correcte error bars toe per bar
    # We nemen de exacte posities uit de barplot
    for bars, strategy in zip(ax.containers, ['CCA', 'MI', 'IND']):
        for bar, (_, row) in zip(bars, df[df['strategy'] == strategy].iterrows()):
            height = bar.get_height()
            x = bar.get_x() + bar.get_width() / 2
            err = row['auc_sd']
            ax.errorbar(x, height, yerr=err, fmt='none', ecolor='black', capsize=3)
            
    ax.axhline(baseline_auc, linestyle='--', color='red', label='Baseline')

    ax.set_title(title)
    ax.set_ylabel('AUC (mean ± SD)')
    ax.set_xlabel('Missingness Type')
    ax.legend(title='Strategy', bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0.)
    plt.tight_layout()
    plt.show()

plot_summary(original_df, 'Original Data – AUC by Missingness and Strategy', baseline_orig_mean['AUC'])
plot_summary(synthetic_df, 'Synthetic Data – AUC by Missingness and Strategy', baseline_syn_mean['AUC'])

end_time = time.time()
print(f"Script runtime: {end_time - start_time:.2f} seconds")
