# -*- coding: utf-8 -*-
"""
Created on Sat May 10 18:37:01 2025

@author: Marcin
"""

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import re
sns.set_style("whitegrid")

results_sd = pd.read_excel("metrics_sd.xlsx").dropna()
results = pd.read_excel("metrics_mean.xlsx").dropna()

# Filter original and synthetic models
original_models = results[results['Model'].str.contains('Original')]
synthetic_models = results[results['Model'].str.contains('Synthetic')]

# clean model names
def clean_model_name(name):
    return re.sub(r'\s*\(.*?\)|\bOriginal\b|\bSynthetic\b', '', name, flags=re.IGNORECASE).strip()

original_models['Model'] = original_models['Model'].apply(clean_model_name)
synthetic_models['Model'] = synthetic_models['Model'].apply(clean_model_name)

def plots(df_mean, df_sd, label, metric="AUC", ylim=None):
    baseline = df_mean.iloc[0][metric]  # Get baseline value from the first row

    fig, axes = plt.subplots(1, 3, figsize=(18, 6), sharey=True)
    ylim = (0.3, 0.6) if metric == "AUC" else None
    titles = ["MCAR", "MAR", "MNAR"]
    indices = [1, 4, 7]

    for ax, title, idx in zip(axes, titles, indices):
        data = df_mean[idx:idx+3].copy()
        errors = df_sd[idx:idx+3][metric].values
        labels = data["Model"].values
        colors = [sns.color_palette("tab10")[i] for i in range(len(labels))]

        ax.bar(labels, data[metric], yerr=errors, capsize=5, color=colors)
        ax.set_title(f"{label} {title}")
        ax.axhline(baseline, color="red", linestyle="--")
        if ylim:
            ax.set_ylim(*ylim)
        if ax == axes[0]:
            ax.set_ylabel(metric)
        else:
            ax.set_ylabel("")
        ax.set_xlabel("")

    lines = [plt.Line2D([0], [0], color="red", linestyle="--")]
    labels = [f"Baseline {label}"]
    fig.legend(lines, labels, loc='upper right', bbox_to_anchor=(0.98, 0.98))

    fig.suptitle(metric, fontsize=16)
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.show()

# Identify all metrics (exclude 'Model' column)
metric_columns = [col for col in results.columns if col not in ("Model","Precision","Recall")]

# Plot baseline metrics for the first two rows with standard deviation error bars
baseline_means = results.iloc[:2].copy()
baseline_sds = results_sd.iloc[:2].copy()

fig, axes = plt.subplots(1, len(metric_columns), figsize=(6 * len(metric_columns), 6))

for ax, metric in zip(axes, metric_columns):
    means = baseline_means[metric].values
    errors = baseline_sds[metric].values
    labels = baseline_means["Model"].values
    colors = [sns.color_palette("tab10")[i] for i in range(len(labels))]

    ax.bar(labels, means, yerr=errors, capsize=5, color=colors)
    ax.set_title(metric)
    ax.set_xlabel("")
    ax.set_ylabel(metric)

plt.tight_layout()
plt.show()

# Plot all metrics dynamically for Original and Synthetic models
for metric in metric_columns:
    plots(original_models, results_sd[results_sd['Model'].str.contains('Original')].copy(), "Original", metric=metric)
    plots(synthetic_models, results_sd[results_sd['Model'].str.contains('Synthetic')].copy(), "Synthetic", metric=metric)
