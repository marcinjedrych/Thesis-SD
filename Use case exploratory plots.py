# -*- coding: utf-8 -*-
"""
Created on Wed Aug 20 11:54:20 2025

@author: Marcin
"""
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import statsmodels.formula.api as smf

sns.set_theme(style="whitegrid")

# ---- Load data ----
file_path = r"C:\Users\Marcin\.cache\kagglehub\datasets\fedesoriano\stroke-prediction-dataset\versions\1\healthcare-dataset-stroke-data.csv"
df = pd.read_csv(file_path)

# ---- Basic cleaning ----
df['smoking_status'] = df['smoking_status'].replace("Unknown", np.nan)
df = df[df['gender'] != 'Other']  # drop rare category
print(df["stroke"].value_counts())

# ---- Smoking order and colors ----
smoke_order = ["never smoked", "formerly smoked", "smokes"]
smoke_palette = {
    "never smoked": "#66c2a5",     # greenish
    "formerly smoked": "#fc8d62",  # orange
    "smokes": "#8da0cb"            # purple-blue
}

# ============================================================================
# 1) Class balance in stroke (0/1)  ---- string-keyed palette to avoid key mismatch
# ============================================================================
counts = df['stroke'].value_counts().sort_index()
props  = df['stroke'].value_counts(normalize=True).sort_index()
classbal = pd.DataFrame({
    "stroke": counts.index.astype(int),
    "count": counts.values,
    "prop": props.values
})
# Plot over a string column so palette keys match
classbal["stroke_str"] = classbal["stroke"].astype(str)

palette_stroke = {"0": "#1f77b4", "1": "#ff7f0e"}  # 0 blue, 1 orange

plt.figure(figsize=(5,4))
ax = sns.barplot(data=classbal, x="stroke_str", y="prop",
                 palette=palette_stroke)
ax.set_xlabel("Stroke (0 = No, 1 = Yes)")
ax.set_ylabel("Proportion")
ax.set_title("Class Balance of Stroke")
ax.set_ylim(0, 1)
plt.tight_layout()
plt.savefig("stroke_class_balance.pdf")
plt.show()

# ============================================================================
# 2) Probability of stroke by smoking status
# ============================================================================
probs = (
    df.groupby("smoking_status", dropna=True)["stroke"]
      .mean()
      .reindex(smoke_order)   # keep desired order, may insert NaN if missing
      .dropna()
      .reset_index()
)

plt.figure(figsize=(5,4))
ax = sns.barplot(
    data=probs, x="smoking_status", y="stroke",
    order=[c for c in smoke_order if c in probs["smoking_status"].tolist()],
    palette=smoke_palette
)
ax.set_ylabel("Probability of Stroke")
ax.set_xlabel("Smoking Status")
ax.set_title("Chance of Stroke by Smoking Status")
ax.set_ylim(0, 0.2)  # adjust if your rates differ
plt.tight_layout()
plt.savefig("smoking_stroke.pdf")
plt.show()


# ============================================================================
# 3) Stroke probability vs BMI (continuous)
# ============================================================================
plt.figure(figsize=(5,4))
sns.regplot(
    data=df, x="bmi", y="stroke",
    logistic=True, ci=None,
    scatter_kws={"alpha": 0.3}
)
plt.ylabel("Probability of Stroke")
plt.xlabel("BMI")
plt.title("Stroke Probability vs BMI")
plt.ylim(0, 1)
plt.tight_layout()
plt.savefig("bmi_stroke.pdf")
plt.show()
