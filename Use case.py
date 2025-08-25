
import pandas as pd
import numpy as np
from functions.strategies import complete_cases, MI_impute, ensemble, missing_indicator
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    roc_auc_score, brier_score_loss, accuracy_score, precision_score, recall_score
)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
import seaborn as sns
import matplotlib.pyplot as plt
from functions.ctgan_syn import generate_synthetic_data

# -----------------------
# Initialization
# -----------------------
synt = True  # whether to generate synthetic training data
file_path = r"C:\Users\Marcin\.cache\kagglehub\datasets\fedesoriano\stroke-prediction-dataset\versions\1\healthcare-dataset-stroke-data.csv"
target = "stroke"
n_iter = 100
strategies = ["CCA", "MI_ensemble", "Missing_Indicator"]
setting = "syn" if synt else "org"
filename = f"Usecase-{n_iter}it-{setting}"

# Metrics to compute and store
metrics = ["AUC", "Brier", "Accuracy", "Precision", "Recall"]


import time
start_time = time.time()

# -----------------------
# helper functions
# -----------------------
def has_two_classes(y) -> bool:
    """Return True if y contains at least two classes."""
    return pd.Series(y).nunique() >= 2

def encode_train_test(train_df: pd.DataFrame, test_df: pd.DataFrame):
    """One-hot encode categoricals (train fit, test transform), return X/y for both."""
    X_train = train_df.drop(columns=[target])
    X_test = test_df.drop(columns=[target])
    y_train = train_df[target]
    y_test = test_df[target]

    cat_cols = X_train.select_dtypes(include=['object']).columns
    num_cols = X_train.select_dtypes(exclude=['object']).columns

    if len(cat_cols) > 0:
        X_train = X_train.copy()
        X_test = X_test.copy()
        X_train[cat_cols] = X_train[cat_cols].astype(str)
        X_test[cat_cols] = X_test[cat_cols].astype(str)

        try:
            enc = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
        except TypeError:
            enc = OneHotEncoder(handle_unknown="ignore", sparse=False)

        X_train_cat = enc.fit_transform(X_train[cat_cols])
        X_test_cat = enc.transform(X_test[cat_cols])

        X_train_cat_df = pd.DataFrame(
            X_train_cat,
            columns=enc.get_feature_names_out(cat_cols),
            index=X_train.index
        )
        X_test_cat_df = pd.DataFrame(
            X_test_cat,
            columns=enc.get_feature_names_out(cat_cols),
            index=X_test.index
        )

        X_train_final = pd.concat([X_train[num_cols], X_train_cat_df], axis=1)
        X_test_final = pd.concat([X_test[num_cols], X_test_cat_df], axis=1)
    else:
        X_train_final = X_train
        X_test_final = X_test

    return X_train_final, y_train, X_test_final, y_test

def collect_metrics(results_dict, strategy, y_true, y_prob, threshold=0.5):
    """Compute AUC, Brier, Accuracy, Precision, Recall and append to results."""
    y_hat = (y_prob >= threshold).astype(int)
    results_dict[strategy]["AUC"].append(roc_auc_score(y_true, y_prob))
    results_dict[strategy]["Brier"].append(brier_score_loss(y_true, y_prob))
    results_dict[strategy]["Accuracy"].append(accuracy_score(y_true, y_hat))
    results_dict[strategy]["Precision"].append(precision_score(y_true, y_hat, zero_division=0))
    results_dict[strategy]["Recall"].append(recall_score(y_true, y_hat, zero_division=0))

# -----------------------
# Load & clean data
# -----------------------
df = pd.read_csv(file_path)
df['smoking_status'] = df['smoking_status'].replace("Unknown", np.nan)
df = df[df['gender'] != 'Other']  # drop rare category
print(df["stroke"].value_counts())

# Storage for metrics across iterations
results = {strategy: {m: [] for m in metrics} for strategy in strategies}

# -----------------------
# Main loop
# -----------------------
for seed in range(n_iter):
    print(f"iteration: {seed}")

    # Build test on complete cases
    completes = df.dropna()

    # every iteration different test set possible
    train_idx, test_idx = train_test_split(
        completes.index,
        test_size=0.2,
        stratify=completes[target],
        random_state=seed
    )
    test_df = df.loc[test_idx]  # test is complete cases (by construction)
    train_df = df.loc[df.index.difference(test_idx)]  # trainset is all the rest

    # -------------------
    # 1) Complete Case Analysis (CCA)
    # -------------------
    cca_train = complete_cases(train_df)
    if synt:
        cca_train = generate_synthetic_data(cca_train)

    # If synthetic generation or CCA collapsed to one class, skip
    if target not in cca_train or not has_two_classes(cca_train[target]):
        print(f"Seed {seed}: CCA train single class, skipping.")
        continue

    X_cca_train, y_cca_train, X_cca_test, y_cca_test = encode_train_test(cca_train, test_df)

    scaler_cca = StandardScaler()
    X_cca_train = scaler_cca.fit_transform(X_cca_train)
    X_cca_test = scaler_cca.transform(X_cca_test)

    logreg_cca = LogisticRegression(max_iter=1000, class_weight="balanced")
    logreg_cca.fit(X_cca_train, y_cca_train)
    y_prob_cca = logreg_cca.predict_proba(X_cca_test)[:, 1]
    collect_metrics(results, "CCA", y_cca_test, y_prob_cca)
    print('CCA finished')

    # -------------------
    # 2) Multiple Imputation + ensemble model
    # -------------------
    mi_train_list = MI_impute(train_df)
    if synt:
        mi_train_list = [generate_synthetic_data(d) for d in mi_train_list]
    # ensemble() returns soft predictions for test_df
    _, soft_preds_mi, _ = ensemble(mi_train_list, test_df, target, label="MI Ensemble")
    collect_metrics(results, "MI_ensemble", test_df[target], soft_preds_mi)
    print('MI finished')
    # -------------------
    # 3) Missing Indicator approach
    # -------------------
    ind_train = missing_indicator(train_df, 'bmi')
    ind_train = missing_indicator(ind_train, 'smoking_status')
    ind_test = missing_indicator(test_df, 'bmi')
    ind_test = missing_indicator(ind_test, 'smoking_status')

    if synt:
        ind_train = generate_synthetic_data(ind_train)
    X_ind_train, y_ind_train, X_ind_test, y_ind_test = encode_train_test(ind_train, ind_test)

    scaler_ind = StandardScaler()
    X_ind_train = scaler_ind.fit_transform(X_ind_train)
    X_ind_test = scaler_ind.transform(X_ind_test)

    logreg_ind = LogisticRegression(max_iter=1000, class_weight="balanced")
    logreg_ind.fit(X_ind_train, y_ind_train)
    y_prob_ind = logreg_ind.predict_proba(X_ind_test)[:, 1]
    collect_metrics(results, "Missing_Indicator", y_ind_test, y_prob_ind)
    print('IND finished')
# -----------------------
# Aggregate & plot
# -----------------------
# Tidy aggregation
agg_rows = []
for s in strategies:
    for m in metrics:
        vals = results[s][m]
        agg_rows.append({
            "Strategy": s,
            "Metric": m,
            "Mean": np.mean(vals) if len(vals) > 0 else np.nan,
            "Std":  np.std(vals)  if len(vals) > 0 else np.nan,
            "N": len(vals)
        })
agg_df = pd.DataFrame(agg_rows)

# Plot
palette = {
    "CCA": "blue",
    "MI_ensemble": "orange",
    "Missing_Indicator": "green"
}
plot_title = "Synthetic" if synt else "Original"

for metric in metrics:
    sub = agg_df[agg_df["Metric"] == metric].copy()

    # If no data for this metric, skip
    if sub["Mean"].isna().all():
        print(f"Skipping {metric}: no valid results.")
        continue

    plt.figure(figsize=(8, 6))
    ax = sns.barplot(data=sub, x="Strategy", y="Mean",
                     palette={"CCA":"blue","MI_ensemble":"orange","Missing_Indicator":"green"})

    # Error bars
    x_positions = range(len(sub))
    ax.errorbar(
        x=list(x_positions),
        y=sub["Mean"],
        yerr=sub["Std"],
        fmt='none',
        c='black',
        capsize=5
    )

    # Labels on bars
    for idx, row in sub.reset_index(drop=True).iterrows():
        if pd.notnull(row["Mean"]):
            ax.text(idx, row["Mean"] + 0.015, f"{row['Mean']:.3f} ± {row['Std']:.3f}",
                    ha='center', fontsize=10)

    ax.set_title(f"{metric} — Comparison of Strategies on {plot_title} Data")

    # adaptive y-limits
    if metric == "Brier":
        # lower is better , max 0.5
        ymin = 0.0
        ymax = min(1.0, max(0.05, np.nanmax(sub["Mean"]) + 0.1))
    else:
        # classification metrics in [0,1]
        ymin = 0.0
        ymax = 1.0
    ax.set_ylim(ymin, ymax)

    ylabel = f"Mean {metric} ± SD" + (" (lower is better)" if metric == "Brier" else "")
    ax.set_ylabel(ylabel)

    ax.grid(axis='y', linestyle='--', alpha=0.7)
    plt.tight_layout()
    
    filename2 = filename + '-' + metric + '.pdf'
    if filename is not None:
        plt.gcf().savefig(filename2, bbox_inches="tight")
        print(f"Saved figure to {filename}")
        
    plt.show()

end_time = time.time()
print(f"Script runtime: {end_time - start_time:.2f} seconds")
