import pandas as pd
import numpy as np
from functions.strategies import complete_cases, MI_impute, ensemble, missing_indicator
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
import seaborn as sns
import matplotlib.pyplot as plt

# Load dataset and clean
file_path = r"C:\Users\Marcin\.cache\kagglehub\datasets\fedesoriano\stroke-prediction-dataset\versions\1\healthcare-dataset-stroke-data.csv"
df = pd.read_csv(file_path)
df['smoking_status'] = df['smoking_status'].replace("Unknown", np.nan)
df = df[df['gender'] != 'Other']  # filter out rare category

target = "stroke"
auc_results = {}

def encode_train_test(train_df, test_df):
    X_train = train_df.drop(columns=[target])
    X_test = test_df.drop(columns=[target])
    y_train = train_df[target]
    y_test = test_df[target]

    cat_cols = X_train.select_dtypes(include=['object']).columns
    num_cols = X_train.select_dtypes(exclude=['object']).columns

    # Convert categorical columns to string to avoid mixed types
    X_train[cat_cols] = X_train[cat_cols].astype(str)  # Fix: Convert to string
    X_test[cat_cols] = X_test[cat_cols].astype(str)    # Fix: Convert to string

    if len(cat_cols) > 0:
        enc = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
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
        # Combine numerical and encoded categorical features
        X_train_final = pd.concat([X_train[num_cols], X_train_cat_df], axis=1)
        X_test_final = pd.concat([X_test[num_cols], X_test_cat_df], axis=1)
    else:
        # No categorical columns
        X_train_final = X_train
        X_test_final = X_test

    return X_train_final, y_train, X_test_final, y_test

import numpy as np


# Number of iterations
n_iter = 250

test_size = int(0.2 * len(df))
strategies = ["CCA", "MI_ensemble", "Missing_Indicator"]

# Storage for AUC scores across iterations
auc_results_all = {strategy: [] for strategy in strategies}

from sklearn.preprocessing import StandardScaler
for seed in range(n_iter):
    # Sample complete cases for test set
    completes = df.dropna()
    test_df = completes.sample(n=test_size, random_state=seed)

    # Training data is everything else
    train_df = df.loc[~df.index.isin(test_df.index)]
    
    # === Model 1: Complete Case Analysis ===
    cca_train = complete_cases(train_df)
    X_cca_train, y_cca_train, X_cca_test, y_cca_test = encode_train_test(cca_train, test_df)
    
    scaler = StandardScaler()
    X_cca_train = scaler.fit_transform(X_cca_train)
    X_cca_test = scaler.transform(X_cca_test)
    
    logreg = LogisticRegression(max_iter=1000)
    logreg.fit(X_cca_train, y_cca_train)
    y_pred_cca = logreg.predict_proba(X_cca_test)[:, 1]
    auc_cca = roc_auc_score(y_cca_test, y_pred_cca)
    auc_results_all["CCA"].append(auc_cca)
    
    # # 2. Multiple Imputation with ensemble modeling internally (no encoding needed here)
    mi_train_list = MI_impute(train_df)
    _, soft_preds_mi, _ = ensemble(mi_train_list, test_df, target, label="MI Ensemble")
    auc_results_all["MI_ensemble"].append(roc_auc_score(test_df[target], soft_preds_mi))
    #auc_results["MI_ensemble"] = roc_auc_score(test_df[target], soft_preds_mi)

    
    # === Model 3: Missing Indicator ===
    ind_train = missing_indicator(train_df, 'bmi')
    ind_train = missing_indicator(ind_train, 'smoking_status')
    ind_test = missing_indicator(test_df, 'bmi')
    ind_test = missing_indicator(ind_test, 'smoking_status')
    X_ind_train, y_ind_train, X_ind_test, y_ind_test = encode_train_test(ind_train, ind_test)
    
    X_ind_train = scaler.fit_transform(X_ind_train)
    X_ind_test = scaler.transform(X_ind_test)
    
    logreg.fit(X_ind_train, y_ind_train)
    y_pred_ind = logreg.predict_proba(X_ind_test)[:, 1]
    auc_ind = roc_auc_score(y_ind_test, y_pred_ind)
    auc_results_all["Missing_Indicator"].append(auc_ind)


# Convert results to DataFrame for plotting
auc_df = pd.DataFrame({
    "Strategy": [],
    "AUC_mean": [],
    "AUC_std": []
})

for strategy in strategies:
    mean_auc = np.mean(auc_results_all[strategy])
    std_auc = np.std(auc_results_all[strategy])
    auc_df = pd.concat([auc_df, pd.DataFrame({
        "Strategy": [strategy],
        "AUC_mean": [mean_auc],
        "AUC_std": [std_auc]
    })], ignore_index=True)

plt.figure(figsize=(8, 6))

# Draw barplot without yerr
ax = sns.barplot(data=auc_df, x="Strategy", y="AUC_mean", palette={
    "CCA": "blue",
    "MI_ensemble": "orange",
    "Missing_Indicator": "green"
})

# Add error bars manually using plt.errorbar
x_positions = range(len(auc_df))
ax.errorbar(x=x_positions, 
            y=auc_df["AUC_mean"], 
            yerr=auc_df["AUC_std"], 
            fmt='none',  # no marker
            c='black',   # error bar color
            capsize=5)   # size of error bar caps

plt.title("Comparison of Strategies on Original Data (Logistic Regression AUC)")
plt.ylim(0.5, 1.0)
plt.ylabel("Mean AUC Score ± SD")
plt.grid(axis='y', linestyle='--', alpha=0.7)

# Add mean AUC value on top of bars
for idx, row in auc_df.iterrows():
    plt.text(idx, row.AUC_mean + 0.015, f"{row.AUC_mean:.3f} ± {row.AUC_std:.3f}", 
             ha='center', fontsize=10)

plt.tight_layout()
plt.show()

# # Calculate 20% of total data as test size (number of rows)
# test_size = int(0.2 * len(df))

# # Filter complete cases (no NA) across all columns or just variables of interest?
# # Assuming we want no missing in the whole row for the test set:
# completes = df.dropna()

# # Sample test_size rows from complete cases without replacement
# test_df = completes.sample(n=test_size, random_state=10)

# # Training data is all data except the test rows (including those with missing values)
# train_df = df.loc[~df.index.isin(test_df.index)]

# # Print sizes
# print(f"Total rows: {len(df)}")
# print(f"Test set size (complete cases sample): {len(test_df)}")
# print(f"Training set size (remaining data): {len(train_df)}")

# # Calculate and print missingness proportion for smoking_status and bmi in training data
# missing_smoking = train_df['smoking_status'].isna().mean()
# missing_bmi = train_df['bmi'].isna().mean()

# print(f"Missing proportion in training data - smoking_status: {missing_smoking:.3f}")
# print(f"Missing proportion in training data - bmi: {missing_bmi:.3f}")

# ### ORIGINAL DATA MODELS

# # 1. Complete Case Analysis
# cca_train = complete_cases(train_df)
# X_cca_train, y_cca_train, X_cca_test, y_cca_test = encode_train_test(cca_train, test_df)

# logreg = LogisticRegression(max_iter=1500)
# logreg.fit(X_cca_train, y_cca_train)
# y_pred_cca = logreg.predict_proba(X_cca_test)[:, 1]
# auc_results["CCA"] = roc_auc_score(y_cca_test, y_pred_cca)

# # 2. Multiple Imputation with ensemble modeling internally (no encoding needed here)
# mi_train_list = MI_impute(train_df)
# _, soft_preds_mi, _ = ensemble(mi_train_list, test_df, target, label="MI Ensemble")
# auc_results["MI_ensemble"] = roc_auc_score(test_df[target], soft_preds_mi)

# # 3. Missing Indicator Method
# ind_train = missing_indicator(train_df, 'bmi')
# ind_train = missing_indicator(ind_train, 'smoking_status')
# ind_test = missing_indicator(test_df, 'bmi')
# ind_test = missing_indicator(ind_test, 'smoking_status')
# X_ind_train, y_ind_train, X_ind_test, y_ind_test = encode_train_test(ind_train, ind_test)

# logreg.fit(X_ind_train, y_ind_train)
# y_pred_ind = logreg.predict_proba(X_ind_test)[:, 1]
# auc_results["Missing_Indicator"] = roc_auc_score(y_ind_test, y_pred_ind)

# # Plot AUC comparison with custom colors
# auc_df = pd.DataFrame(list(auc_results.items()), columns=["Strategy", "AUC"])

# # Create a color palette mapping each strategy to a specific color
# palette = {
#     "CCA": "blue",
#     "MI_ensemble": "orange",
#     "Missing_Indicator": "green"
# }

# # Create the bar plot with custom colors
# plt.figure(figsize=(8, 6))
# sns.barplot(
#     data=auc_df,
#     x="Strategy",
#     y="AUC",
#     palette=palette
# )
# plt.title("Comparison of Strategies on Origina data (Logistic Regression AUC)")
# plt.ylim(0.5, 1.0)
# plt.ylabel("AUC Score")
# plt.grid(axis='y', linestyle='--', alpha=0.7)

# # Add value labels on top of bars
# for index, row in auc_df.iterrows():
#     plt.text(index, row.AUC + 0.01, f"{row.AUC:.3f}", 
#              ha='center', fontsize=10)

# plt.tight_layout()
# plt.show()


### SYNTHETIC DATA MODELS

