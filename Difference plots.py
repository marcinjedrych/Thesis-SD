# -*- coding: utf-8 -*-

from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

rec = 200
csv1 = Path(f"org-{rec}rec.csv")  # original results
csv2 = Path("syn-{rec}rec.csv")  # synthetic results
metric = "AUC"

strategies = ["CCA", "MI", "IND"]

df1 = pd.read_csv(csv1, sep=";", decimal=",")
df2 = pd.read_csv(csv2, sep=";", decimal=",")

# ensure numeric
for df in (df1, df2):
    df[f"{metric}_mean"] = pd.to_numeric(df[f"{metric}_mean"], errors="coerce")

# difference original − synthetic
diff = df1[f"{metric}_mean"] - df2[f"{metric}_mean"]

df_new = df1.iloc[:, :3].copy()
df_new["diff"] = diff

# filter
key_cols = df_new.columns[:3].tolist()
miss_col, data_col, strat_col = key_cols
df_new = df_new[df_new[strat_col].isin(strategies)].copy()
df_new["diff_pp"] = df_new["diff"] 

# pivot table: rows = missingness, cols = strategies
order_idx = ["MCAR", "MAR", "MNAR"]
pivot = (
    df_new.pivot_table(
        index=miss_col,
        columns=strat_col,
        values="diff_pp",
        aggfunc="mean"
    )
    .reindex(order_idx, axis=0)    
    [strategies]                     
)

# Plot
fig, ax = plt.subplots(figsize=(9, 5.2))
x = np.arange(len(pivot.index))
width = 0.8 / len(strategies)

bars = []
for i, strat in enumerate(strategies):
    vals = pivot[strat].values
    bar = ax.bar(x + (i - (len(strategies)-1)/2)*width,
                 vals, width, label=strat)
    bars.append(bar)
ax.axhline(0, linewidth=0.8, color="black")

# labels & title
ax.set_xticks(x)
ax.set_xticklabels(pivot.index, rotation=0)
ax.set_ylabel(f"{metric} difference (original − synthetic), pp")
ax.set_xlabel(miss_col)
ax.set_title(f"Mean {metric} difference (original − synthetic)")

# annotate each bar
def annotate(bars):
    for bar in bars:
        h = bar.get_height()
        ax.annotate(f"{h:.3f}",
                    xy=(bar.get_x() + bar.get_width()/2, h),
                    xytext=(0, 3), textcoords="offset points",
                    ha="center", va="bottom", fontsize=9)

for b in bars:
    annotate(b)

ax.legend(frameon=False)
fig.tight_layout()

# save
plt.savefig(f"Difference_plot-{metric}-{rec}.pdf")
plt.show()
