import numpy as np
from sklearn.metrics import roc_auc_score
import pandas as pd
import matplotlib.pyplot as plt

# True labels
y_true = np.array([0, 0, 1, 1, 0, 1, 0, 0, 1, 0, 1, 1, 0, 0, 0 , 0 ])

base_pred = np.array([0.233, 0.383, 0.375, 0.600, 0.420, 0.730, 0.150, 0.250, 0.910, 0.300, 0.8, 0.5, 0.1, 0.3 ,0.6, 0.3])

# Generate 15 prediction sets with small random noise around base_pred
rng = np.random.default_rng(42)
preds = []
for i in range(150):
    noise = rng.normal(0,0.2 , size=base_pred.shape)  # Gaussian noise
    y_pred_i = np.clip(base_pred + noise, 0, 1)       # keep in [0,1]
    preds.append(y_pred_i)

# Compute Mean AUC, Pooled AUC, and their difference for M=2..
rows = []
for M in range(2, 150):
    subset = preds[:M]
    aucs = [roc_auc_score(y_true, p) for p in subset]
    mean_auc = np.mean(aucs)
    pooled_auc = roc_auc_score(y_true, np.mean(subset, axis=0))
    diff = abs(pooled_auc - mean_auc)
    rows.append({"M": M, "Mean AUC": mean_auc, "Pooled AUC": pooled_auc, "Abs Diff": diff})

df = pd.DataFrame(rows)
print(df.round(4))

# Plot absolute difference vs M
plt.figure(figsize=(6,4))
plt.plot(df["M"], df["Abs Diff"], marker="o")
plt.title("Absolute Difference: |Pooled prediction AUC - Mean AUC|")
plt.xlabel("Number of models (M)")
plt.ylabel("Absolute difference")
plt.grid(True, linestyle="--", alpha=0.7)
plt.tight_layout()
plt.show()
