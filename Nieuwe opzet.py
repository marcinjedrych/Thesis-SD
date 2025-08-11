# -*- coding: utf-8 -*-
"""
Created on Sun Jul 13 15:09:06 2025

@author: Marcin
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import gc  # FIX: for manual garbage collection

from functions.generate import generate_patient_data
from sklearn.model_selection import train_test_split
from functions.missingness import mcar, mar, mnar
from functions.ctgan_syn import generate_synthetic_data
from functions.strategies import complete_cases, MI_impute, ensemble, missing_indicator
from functions.model import logistic_regression

import time
start_time = time.time()

n_iter = 200
exclude = ['latent1', 'latent2']

# 1 BASELINE
def run_baseline(n_iter, exclude=None, synthetic=False):
    results = []
    for i in range(n_iter):
        data = generate_patient_data(nsamples=500, seed=i)
        train_data, test_data = train_test_split(data, test_size=0.2, random_state=i)

        if synthetic:
            train_data = generate_synthetic_data(train_data)

        result = logistic_regression(train_data, test_data, 'hospitaldeath', exclude_vars=exclude)
        results.append(result)

        # FIX: free memory
        del data, train_data, test_data, result
        gc.collect()

    return results


# 2 Missing data
target_missing_rate = 0.45
def run_models(n_iter, exclude=None, synthetic=False, m_type=None, strategy="CCA"):
    if m_type is None:
        m_type = mcar

    results = []
    m_amount = []

    for i in range(n_iter):
        data = generate_patient_data(nsamples=500, seed=i)
        train_data, test_data = train_test_split(data, test_size=0.2, random_state=i)

        train_data, m_proportion = m_type(train_data, target_column="bp", target_missing_rate=target_missing_rate)
        m_amount.append(m_proportion)

        if strategy == "MI":
            
            dfs = MI_impute(train_data)

            if synthetic:
                imp_and_syn = []
                for df in dfs:
                    syn_df = generate_synthetic_data(df)
                    imp_and_syn.append(syn_df)
                _, _, mi_results = ensemble(imp_and_syn, test_data, target="hospitaldeath")
            else:
                _, _, mi_results = ensemble(dfs, test_data, target="hospitaldeath")

            results.extend(mi_results)  # FIX: extend instead of append list
            del dfs
            if synthetic:
                del imp_and_syn
        else:
            if strategy == "CCA":
                train_data = complete_cases(train_data)
            else:
                train_data = missing_indicator(train_data, 'bp')

            if synthetic:
                train_data = generate_synthetic_data(train_data)

            result = logistic_regression(train_data, test_data, 'hospitaldeath', exclude_vars=exclude)
            results.append(result)

        # FIX: free memory
        del data, train_data, test_data
        gc.collect()

    df = pd.DataFrame(results)
    mean_metrics = df.mean(numeric_only=True)

    print("\nResults:")
    for metric, value in mean_metrics.items():
        print(f"{metric}: {value:.4f}")
    print("\n")
    return results, m_amount


### 1. BASELINE
baseline_original = run_baseline(n_iter, exclude)
# baseline_synthetic = run_baseline(n_iter, exclude, synthetic=True)

### 2. MISSINGNESS (ORIGINAL DATA)
## 2.1 MCAR 
print("original MCAR")  
original_mcar_cca, _ = run_models(n_iter, exclude)
original_mcar_mi, _ = run_models(n_iter, exclude, strategy="MI")
original_mcar_ind, _ = run_models(n_iter, exclude, strategy="IND")

## 2.2 MAR
print("original MAR") 
original_mar_cca, _ = run_models(n_iter, exclude, m_type=mar)
original_mar_mi, _ = run_models(n_iter, exclude, strategy="MI", m_type=mar)
original_mar_ind, _ = run_models(n_iter, exclude, strategy="IND", m_type=mar)

### 2.3 MNAR
print("original MNAR") 
original_mnar_cca, _ = run_models(n_iter, exclude, m_type=mnar)
original_mnar_mi, _ = run_models(n_iter, exclude, strategy="MI", m_type=mnar)
original_mnar_ind, _ = run_models(n_iter, exclude, strategy="IND", m_type=mnar)

## 3. MISSINGNESS (SYNTHETIC DATA)
# Uncomment when needed
# print("synthetic MCAR") 
# syn_mcar_cca, _ = run_models(n_iter, exclude, synthetic=True)
# syn_mcar_mi, _ = run_models(n_iter, exclude, strategy="MI", synthetic=True)
# syn_mcar_ind, _ = run_models(n_iter, exclude, strategy="IND", synthetic=True)
# print("synthetic MAR") 
# syn_mar_cca, _ = run_models(n_iter, exclude, m_type=mar, synthetic=True)
# syn_mar_mi, _ = run_models(n_iter, exclude, strategy="MI", m_type=mar, synthetic=True)
# syn_mar_ind, _ = run_models(n_iter, exclude, strategy="IND", m_type=mar, synthetic=True)
# print("synthetic MNAR") 
# syn_mnar_cca, _ = run_models(n_iter, exclude, m_type=mnar, synthetic=True)
# syn_mnar_mi, _ = run_models(n_iter, exclude, strategy="MI", m_type=mnar, synthetic=True)
# syn_mnar_ind, _ = run_models(n_iter, exclude, strategy="IND", m_type=mnar, synthetic=True)

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
collect_metrics(original_mcar_mi, 'MCAR', 'original', 'MI')
collect_metrics(original_mcar_ind, 'MCAR', 'original', 'IND')

collect_metrics(original_mar_cca, 'MAR', 'original', 'CCA')
collect_metrics(original_mar_mi, 'MAR', 'original', 'MI')
collect_metrics(original_mar_ind, 'MAR', 'original', 'IND')

collect_metrics(original_mnar_cca, 'MNAR', 'original', 'CCA')
collect_metrics(original_mnar_mi, 'MNAR', 'original', 'MI')
collect_metrics(original_mnar_ind, 'MNAR', 'original', 'IND')

# baseline stats
baseline_orig_mean, baseline_orig_std = aggregate_results(baseline_original)
# baseline_syn_mean, baseline_syn_std = aggregate_results(baseline_synthetic)

def plot_baseline(mean_orig, std_orig, mean_syn, std_syn):
    labels = ['Original', 'Synthetic']
    auc_means = [mean_orig['AUC'], mean_syn['AUC']]

    plt.figure(figsize=(5, 5))
    bars = plt.bar(labels, auc_means, capsize=5, color=['cornflowerblue', 'lightcoral'])
    for bar, auc in zip(bars, auc_means):
        yval = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2, yval + 0.01, f'{auc:.3f}', ha='center', va='bottom')

    plt.title('Baseline AUC Comparison')
    plt.ylabel('AUC (mean)')
    plt.ylim(0.5, 1)
    plt.tight_layout()
    plt.show()

#plot_baseline(baseline_orig_mean, baseline_orig_std, baseline_syn_mean, baseline_syn_std)

# build DataFrame
summary_df = pd.DataFrame(summary)

def plot_summary(df, title, baseline_auc):
    plt.figure(figsize=(10, 5))
    ax = sns.barplot(
        data=df,
        x='missingness',
        y='auc_mean',
        hue='strategy',
        hue_order=['CCA', 'MI', 'IND'],
        ci=None,
        palette={'CCA': 'blue', 'MI': 'orange', 'IND': 'green'}
    )

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

original_df = summary_df[summary_df['data_type'] == 'original']
plot_summary(original_df, 'Original Data – AUC by Missingness and Strategy', baseline_orig_mean['AUC'])

end_time = time.time()
print(f"Script runtime: {end_time - start_time:.2f} seconds")
