# --- CLEAR, ROBUST VERSION ---

import pandas as pd
import numpy as np
from functions.strategies import complete_cases, MI_impute, ensemble, missing_indicator
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
import seaborn as sns
import matplotlib.pyplot as plt
from functions.ctgan_syn import generate_synthetic_data

# -----------------------
# Config
# -----------------------
synt = True  # whether to generate synthetic training data
file_path = r"C:\Users\Marcin\.cache\kagglehub\datasets\fedesoriano\stroke-prediction-dataset\versions\1\healthcare-dataset-stroke-data.csv"
target = "stroke"
n_iter = 100
strategies = ["CCA", "MI_ensemble", "Missing_Indicator"]

# -----------------------
# Helpers
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

    # Consistent string dtype for categoricals
    if len(cat_cols) > 0:
        X_train = X_train.copy()
        X_test = X_test.copy()
        X_train[cat_cols] = X_train[cat_cols].astype(str)
        X_test[cat_cols] = X_test[cat_cols].astype(str)

        # Compatibility with different sklearn versions:
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

# -----------------------
# Load & clean data
# -----------------------
df = pd.read_csv(file_path)
df['smoking_status'] = df['smoking_status'].replace("Unknown", np.nan)
df = df[df['gender'] != 'Other']  # drop rare category

# Storage for AUC scores across iterations
auc_results_all = {strategy: [] for strategy in strategies}

# -----------------------
# Main loop
# -----------------------
for seed in range(n_iter):
    
    print(f"iteration: {seed}")
    # Build test on complete cases, stratified by target to ensure both classes in test
    completes = df.dropna()

    # every iteration different test set possible
    train_idx, test_idx = train_test_split(
        completes.index,
        test_size=0.2,
        stratify=completes[target],
        random_state=seed
    )
    test_df = df.loc[test_idx]  # test is complete cases (by construction)
    train_df = df.loc[df.index.difference(test_idx)] # trainset is all the rest

    # -------------------
    # 1) Complete Case Analysis (CCA)
    # -------------------
    #print('CCA...\n')
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

    logreg_cca = LogisticRegression(max_iter=1000)
    logreg_cca.fit(X_cca_train, y_cca_train)
    y_pred_cca = logreg_cca.predict_proba(X_cca_test)[:, 1]
    auc_cca = roc_auc_score(y_cca_test, y_pred_cca)
    auc_results_all["CCA"].append(auc_cca)

    # -------------------
    # 2) Multiple Imputation + internal ensemble model
    # -------------------
    #print('MI...\n')
    mi_train_list = MI_impute(train_df)
    if synt:
        mi_train_list = [generate_synthetic_data(d) for d in mi_train_list]
    # ensemble() returns soft predictions for test_df
    _, soft_preds_mi, _ = ensemble(mi_train_list, test_df, target, label="MI Ensemble")
    auc_results_all["MI_ensemble"].append(roc_auc_score(test_df[target], soft_preds_mi))

    # -------------------
    # 3) Missing Indicator approach
    # -------------------
    #print('Learned NA...\n')
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

    logreg_ind = LogisticRegression(max_iter=1000)
    logreg_ind.fit(X_ind_train, y_ind_train)
    y_pred_ind = logreg_ind.predict_proba(X_ind_test)[:, 1]
    auc_ind = roc_auc_score(y_ind_test, y_pred_ind)
    auc_results_all["Missing_Indicator"].append(auc_ind)

# -----------------------
# Aggregate & plot
# -----------------------
auc_df = pd.DataFrame({
    "Strategy": strategies,
    "AUC_mean": [np.mean(auc_results_all[s]) if len(auc_results_all[s]) > 0 else np.nan for s in strategies],
    "AUC_std":  [np.std(auc_results_all[s])  if len(auc_results_all[s]) > 0 else np.nan for s in strategies]
})

plt.figure(figsize=(8, 6))
ax = sns.barplot(data=auc_df, x="Strategy", y="AUC_mean", palette={
    "CCA": "blue",
    "MI_ensemble": "orange",
    "Missing_Indicator": "green"
})

# Add error bars manually
x_positions = range(len(auc_df))
ax.errorbar(
    x=list(x_positions),
    y=auc_df["AUC_mean"],
    yerr=auc_df["AUC_std"],
    fmt='none',
    c='black',
    capsize=5
)

plot_title = "Synthetic" if synt else "Original"
plt.title(f"Comparison of Strategies on {plot_title} Data (Logistic Regression AUC)")
plt.ylim(0.5, 1.0)
plt.ylabel("Mean AUC Score ± SD")
plt.grid(axis='y', linestyle='--', alpha=0.7)

# Add mean ± sd on bars
for idx, row in auc_df.iterrows():
    if pd.notnull(row.AUC_mean):
        plt.text(idx, row.AUC_mean + 0.015, f"{row.AUC_mean:.3f} ± {row.AUC_std:.3f}",
                 ha='center', fontsize=10)

plt.tight_layout()
plt.show()

# Optional: print how many successful runs per strategy
print("\nSuccessful runs per strategy:")
for s in strategies:
    print(f"{s}: {len(auc_results_all[s])}/{n_iter}")
