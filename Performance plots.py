# -*- coding: utf-8 -*-
"""
Created on Sat May 10 18:37:01 2025

@author: Marcin
"""
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import re


results = pd.read_excel("performance.xlsx")
#results = pd.read_excel("model_performance_summary.xlsx")
results = results.dropna()

## BASELINE
fig, axes = plt.subplots(1, 3, figsize=(18, 6))
sns.barplot(ax=axes[0], x="Model", y="AUC", data=results[:2])
axes[0].set_title("AUC")
axes[0].set_ylim(0.5, 0.7)
sns.barplot(ax=axes[1], x="Model", y="Accuracy", data=results[:2])
axes[1].set_title("Accuracy")
axes[1].set_ylim(0.4, 0.7)
sns.barplot(ax=axes[2], x="Model", y="Brier Score", data=results[:2])
axes[2].set_title("Brier Score")
axes[2].set_ylim(0.1, 0.3)
plt.tight_layout()
plt.show()

## MISSINGNESS MODELS

# Filter original and synthetic models
original_models = results[results['Model'].str.contains('Original')]
synthetic_models = results[results['Model'].str.contains('Synthetic')]

# clean model names
def clean_model_name(name):
    return re.sub(r'\s*\(.*?\)|\bOriginal\b|\bSynthetic\b', '', name, flags=re.IGNORECASE).strip()
original_models['Model'] = original_models['Model'].apply(clean_model_name)
synthetic_models['Model'] = synthetic_models['Model'].apply(clean_model_name)

def plots(df, label, metric="AUC", ylim=None):
    import matplotlib.pyplot as plt
    import seaborn as sns

    baseline = df.iloc[0][metric]  # Get baseline value from the first row

    fig, axes = plt.subplots(1, 3, figsize=(18, 6), sharey=True)

    titles = ["MCAR", "MAR", "MNAR"]
    indices = [1, 4, 7]

    for ax, title, idx in zip(axes, titles, indices):
        sns.barplot(ax=ax, x="Model", y=metric, data=df[idx:idx+3])
        ax.set_title(f"{label} {title}")
        ax.axhline(baseline, color="red", linestyle="--")
        if ylim:
            ax.set_ylim(*ylim)
        if ax == axes[0]:
            ax.set_ylabel(metric)
        else:
            ax.set_ylabel("")
        ax.set_xlabel("")

    # Add a single legend in the top right of the figure (not subplot)
    lines = [plt.Line2D([0], [0], color="red", linestyle="--")]
    labels = [f"Baseline {label}"]
    fig.legend(lines, labels, loc='upper right', bbox_to_anchor=(0.98, 0.98))

    fig.suptitle(metric, fontsize=16)
    plt.tight_layout(rect=[0, 0, 1, 0.95])  # leave space for the suptitle

    plt.show()


plots(original_models, "Original", metric="AUC", ylim=(0.5,0.7))
plots(synthetic_models, "Synthetic", metric="AUC", ylim=(0.5,0.7))

plots(original_models, "Original", metric="Accuracy",ylim=(0.4,0.7))
plots(synthetic_models, "Synthetic", metric="Accuracy",ylim=(0.4,0.7))

plots(original_models, "Original", metric="Brier Score",ylim=(0.1,0.3))
plots(synthetic_models, "Synthetic",metric="Brier Score",ylim=(0.1,0.3))
