# -*- coding: utf-8 -*-
"""
Created on Sun Jul 13 15:09:06 2025

@author: Marcin
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import gc

from functions.generate import generate_patient_data
from sklearn.model_selection import train_test_split
from functions.missingness import mcar, mar, mnar
from functions.ctgan_syn import generate_synthetic_data
from functions.strategies import complete_cases, MI_impute, ensemble, missing_indicator
from functions.model import logistic_regression

synt = True
baseline = False
nsamples = 200
n_iter = 100
exclude = ['latent1', 'latent2']
setting = "syn" if synt else "org"
filename = f"{n_iter}it-{setting}-{nsamples}rec"

import time
start_time = time.time()

# 1 BASELINE
def run_baseline(n_iter, exclude=None, synthetic=False):
    results = []
    for i in range(n_iter):
        data = generate_patient_data(nsamples=nsamples, seed=i)
        train_data, test_data = train_test_split(data, test_size=0.2, random_state=i)

        if synthetic:
            train_data = generate_synthetic_data(train_data)

        result = logistic_regression(train_data, test_data, 'hospitaldeath', exclude_vars=exclude)
        results.append(result)

        # free memory
        del data, train_data, test_data, result
        gc.collect()

    return results


# 2 Missing data
target_missing_rate = 0.44
def run_models(n_iter, exclude=None, synthetic=False, m_type='mcar', strategy="CCA"):

    results = []
    m_amount = []

    for i in range(n_iter):
        data = generate_patient_data(nsamples=nsamples, seed=i)
        train_data, test_data = train_test_split(data, test_size=0.2, random_state=i)
        
        if m_type == 'mcar':
            train_data, m_proportion = mcar(train_data, target_column="bp", target_missing_rate=target_missing_rate)
        elif m_type == 'mar':
            train_data, _ = mar(train_data, target_column="bp", target_missing_rate=target_missing_rate)
        else:
            train_data, _ = mnar(train_data, target_column="bp", target_missing_rate=target_missing_rate)
 
        #m_amount.append(m_proportion)  #to check proportion of missingness

        if strategy == "MI":
            
            dfs = MI_impute(train_data)

            if synthetic:
                imp_and_syn = []
                for df in dfs:
                    syn_df = generate_synthetic_data(df)
                    imp_and_syn.append(syn_df)
                print(f'synthesizer MI {i}')
                _, _, mi_results = ensemble(imp_and_syn, test_data, target="hospitaldeath")
            else:
                _, _, mi_results = ensemble(dfs, test_data, target="hospitaldeath")

            results.extend(mi_results)
            del dfs
            if synthetic:
                del imp_and_syn
        else:
            if strategy == "CCA":
                #print("Number of rows before CCA:", train_data.shape[0])
                train_data = complete_cases(train_data)
                #print("Number of rows after CCA:", train_data.shape[0])
            else:
                train_data = missing_indicator(train_data, 'bp')

            if synthetic:
                train_data = generate_synthetic_data(train_data)

            result = logistic_regression(train_data, test_data, 'hospitaldeath', exclude_vars=exclude)
            results.append(result)

        # free memory
        del data, train_data, test_data
        gc.collect()

    df = pd.DataFrame(results)
    mean_metrics = df.mean(numeric_only=True)

    print(f"\n{strategy} finished")
    print("\n")
    return results, m_amount

### 2. MISSINGNESS (ORIGINAL DATA)
## 2.1 MCAR 
if synt is False:
    
    ### 1.1 BASELINE
    baseline_original = run_baseline(n_iter, exclude)
    if baseline is True:
        baseline_synthetic = run_baseline(n_iter, exclude, synthetic=True)
        
    ##1.2 MISSINGNESS (ORIGINAL DATA)
    ## MCAR
    print("original MCAR")  
    original_mcar_cca, _ = run_models(n_iter, exclude)
    original_mcar_mi, _ = run_models(n_iter, exclude, strategy="MI")
    original_mcar_ind, _ = run_models(n_iter, exclude, strategy="IND")  
     ## MAR
    print("original MAR") 
    original_mar_cca, _ = run_models(n_iter, exclude, m_type='mar')
    original_mar_mi, _ = run_models(n_iter, exclude, strategy="MI", m_type='mar')
    original_mar_ind, _ = run_models(n_iter, exclude, strategy="IND", m_type='mar')
    ### MNAR
    print("original MNAR") 
    original_mnar_cca, _ = run_models(n_iter, exclude, m_type='mnar')
    original_mnar_mi, _ = run_models(n_iter, exclude, strategy="MI", m_type='mnar')  
    original_mnar_ind, _ = run_models(n_iter, exclude, strategy="IND", m_type='mnar')
    
else:
    ### 2.1. BASELINE
    baseline_synthetic = run_baseline(n_iter, exclude, synthetic=True)
    if baseline is True:
        baseline_original = run_baseline(n_iter, exclude)
        
    ## 2.2. MISSINGNESS (SYNTHETIC DATA)
    ## MCAR
    print("synthetic MCAR") 
    syn_mcar_cca, pr1  = run_models(n_iter, exclude, synthetic=True)
    syn_mcar_mi, pr2 = run_models(n_iter, exclude, strategy="MI", synthetic=True)
    syn_mcar_ind, pr3 = run_models(n_iter, exclude, strategy="IND", synthetic=True)
    ## MAR
    print("synthetic MAR") 
    syn_mar_cca, pr4 = run_models(n_iter, exclude, m_type=mar, synthetic=True)
    syn_mar_mi, pr5 = run_models(n_iter, exclude, strategy="MI", m_type=mar, synthetic=True)
    syn_mar_ind, pr6 = run_models(n_iter, exclude, strategy="IND", m_type=mar, synthetic=True)
    ## MNAR
    print("synthetic MNAR") 
    syn_mnar_cca, pr7 = run_models(n_iter, exclude, m_type=mnar, synthetic=True)
    syn_mnar_mi, pr8 = run_models(n_iter, exclude, strategy="MI", m_type=mnar, synthetic=True)
    syn_mnar_ind, pr9 = run_models(n_iter, exclude, strategy="IND", m_type=mnar, synthetic=True)

# helper function
def aggregate_results(results_list):
    df = pd.DataFrame(results_list)
    return df.mean(), df.std()

# store AUC means and SDs
summary = {
    'missingness': [],
    'data_type': [],
    'strategy': [],
    'AUC_mean': [],
    'AUC_sd': [],
    'Accuracy_mean': [],
    'Accuracy_sd': [],
    'Recall_mean': [],
    'Recall_sd': [],
    'Precision_mean': [],
    'Precision_sd': [],
    'Brier_mean':[],
    'Brier_sd':[]
}

def collect_metrics(results, miss_type, data_type, strategy):
    mean_vals, std_vals = aggregate_results(results)
    summary['missingness'].append(miss_type)
    summary['data_type'].append(data_type)
    summary['strategy'].append(strategy)
    summary['AUC_mean'].append(mean_vals['AUC'])
    summary['AUC_sd'].append(std_vals['AUC'])
    summary['Accuracy_mean'].append(mean_vals['Accuracy'])
    summary['Accuracy_sd'].append(std_vals['Accuracy'])
    summary['Recall_mean'].append(mean_vals['Recall'])
    summary['Recall_sd'].append(std_vals['Recall'])
    summary['Precision_mean'].append(mean_vals['Precision'])
    summary['Precision_sd'].append(std_vals['Precision'])
    summary['Brier_mean'].append(mean_vals['Brier Score'])
    summary['Brier_sd'].append(std_vals['Brier Score'])

if synt is False:
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
    if baseline is True:
        baseline_syn_mean, baseline_syn_std = aggregate_results(baseline_synthetic)

else:
    #synthetic
    collect_metrics(syn_mcar_cca, 'MCAR', 'synthetic', 'CCA')
    collect_metrics(syn_mcar_mi, 'MCAR', 'synthetic', 'MI')
    collect_metrics(syn_mcar_ind, 'MCAR', 'synthetic', 'IND')
    
    collect_metrics(syn_mar_cca, 'MAR', 'synthetic', 'CCA')
    collect_metrics(syn_mar_mi, 'MAR', 'synthetic', 'MI')
    collect_metrics(syn_mar_ind, 'MAR', 'synthetic', 'IND')
    
    collect_metrics(syn_mnar_cca, 'MNAR', 'synthetic', 'CCA')
    collect_metrics(syn_mnar_mi, 'MNAR', 'synthetic', 'MI')
    collect_metrics(syn_mnar_ind, 'MNAR', 'synthetic', 'IND')

    #baseline stats
    baseline_syn_mean, baseline_syn_std = aggregate_results(baseline_synthetic)
    if baseline is True:
        baseline_orig_mean, baseline_orig_std = aggregate_results(baseline_original)
        
def plot_baseline(mean_orig, std_orig, mean_syn, std_syn,
                           savepath="baseline_results.pdf"):
    """
    Create a 3-panel baseline figure comparing Original vs Synthetic on:
    Accuracy, AUC, and Brier Score. Includes SD error bars and value labels.
    """
    metrics = [
        ("Accuracy", "Accuracy"),
        ("AUC", "AUC"),
        ("Brier Score", "Brier score")
    ]
    labels = ["Original", "Synthetic"]
    colors = ["cornflowerblue", "lightcoral"] 

    fig, axes = plt.subplots(1, 3, figsize=(13, 4), sharey=False)
    for ax, (key, pretty_name) in zip(axes, metrics):
        means = [float(mean_orig[key]), float(mean_syn[key])]
        sds   = [float(std_orig[key]),  float(std_syn[key])]

        x = np.arange(len(labels))
        bars = ax.bar(x, means, yerr=sds, capsize=4, color = colors)

        # y-limits dynamic for Brier
        if key == "Brier Score":
            upper = max(0.05, max(means) + max(sds) + 0.02)
            ax.set_ylim(0.0, upper)
        else:
            upper = min(1.0, max(means) + max(sds) + 0.05)
            ax.set_ylim(0.0, max(0.6, upper))

        # grid + labels
        ax.grid(axis="y", alpha=0.3)
        ax.set_xticks(x)
        ax.set_xticklabels(labels)
        ax.set_title(pretty_name)
        ax.set_ylabel("Score")

        # value annotations above bars
        offset = 0.01 if max(sds) == 0 else max(sds) * 0.15
        for xi, yi in zip(x, means):
            ax.text(xi, yi + offset, f"{yi:.3f}", ha="center", va="bottom", fontsize=9)

    fig.suptitle("Baseline performance: Original vs Synthetic", y=1.02, fontsize=12)
    fig.tight_layout()
    fig.savefig(savepath, bbox_inches="tight")
    print(f"Saved figure to {savepath}")
    plt.show()


if baseline is True:
    plot_baseline(baseline_orig_mean, baseline_orig_std, baseline_syn_mean, baseline_syn_std)

# build DataFrame of results
summary_df = pd.DataFrame(summary)

def plot_summary(df, title, baseline, metric = 'AUC'):
    plt.figure(figsize=(10, 5))
    ax = sns.barplot(
        data=df,
        x='missingness',
        y= metric + '_mean',
        hue='strategy',
        hue_order=['CCA', 'MI', 'IND'],
        ci=None,
        palette={'CCA': 'blue', 'MI': 'orange', 'IND': 'green'}
    )

    for bars, strategy in zip(ax.containers, ['CCA', 'MI', 'IND']):
        for bar, (_, row) in zip(bars, df[df['strategy'] == strategy].iterrows()):
            height = bar.get_height()
            x = bar.get_x() + bar.get_width() / 2
            err = row[metric +'_sd']
            ax.errorbar(x, height, yerr=err, fmt='none', ecolor='black', capsize=3)

    ax.axhline(baseline, linestyle='--', color='red', label='Baseline')
    ax.set_title(title)
    ax.set_ylabel(f'{metric} (mean ± SD)')
    ax.set_xlabel('Missingness Type')
    ax.legend(title='Strategy', bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0.)
    plt.tight_layout()
    
    filename2 = filename + "-" + metric + ".pdf"
    if filename is not None:
        plt.gcf().savefig(filename2, bbox_inches="tight")
        print(f"Saved figure to {filename}")
    
    plt.show()

if synt is False:
    original_df = summary_df[summary_df['data_type'] == 'original']
    filename3 = filename + ".csv"
    original_df.to_csv(filename3, index=False) 
    plot_summary(original_df, 'Original Data – AUC by Missingness and Strategy', baseline_orig_mean['AUC'])
    plot_summary(original_df, 'Original Data – Accuracy by Missingness and Strategy', baseline_orig_mean['Accuracy'], metric = 'Accuracy')
    plot_summary(original_df, 'Original Data – Recall by Missingness and Strategy', baseline_orig_mean['Recall'], metric = 'Recall')
    plot_summary(original_df, 'Original Data – Precision by Missingness and Strategy', baseline_orig_mean['Precision'], metric = 'Precision')
    plot_summary(original_df, 'Original Data – Brier score by Missingness and Strategy', baseline_orig_mean['Brier Score'], metric = 'Brier')
else:
    synthetic_df = summary_df[summary_df['data_type'] == 'synthetic']
    filename3 = filename + ".csv"
    synthetic_df.to_csv(filename3, index=False) 
    plot_summary(synthetic_df, 'Synthetic Data – AUC by Missingness and Strategy', baseline_syn_mean['AUC'])
    plot_summary(synthetic_df, 'Synthetic Data – Accuracy by Missingness and Strategy', baseline_syn_mean['Accuracy'], metric = 'Accuracy')
    plot_summary(synthetic_df, 'Synthetic Data – Recall by Missingness and Strategy', baseline_syn_mean['Recall'], metric = 'Recall')
    plot_summary(synthetic_df, 'Synthetic Data – Precision by Missingness and Strategy', baseline_syn_mean['Precision'], metric = 'Precision')
    plot_summary(synthetic_df, 'Synthetic Data – Brier score by Missingness and Strategy', baseline_syn_mean['Brier Score'], metric = 'Brier')

end_time = time.time()
print(f"Script runtime: {end_time - start_time:.2f} seconds")
