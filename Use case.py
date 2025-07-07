# -*- coding: utf-8 -*-
"""
Created on Sun Jul  6 18:29:32 2025

@author: Marcin
"""

from ucimlrepo import fetch_ucirepo
import pandas as pd
import numpy as np
from functions.strategies import complete_cases, multiple_imputation, missing_indicator
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import OneHotEncoder
import seaborn as sns
import matplotlib.pyplot as plt

# ------------------ Load Dataset 1 Income ------------------ #
adult = fetch_ucirepo(id=2) 
features = adult.data.features 
target = adult.data.targets 
df1 = pd.concat([features, target], axis=1)
df1.replace("?", np.nan, inplace=True)
df1['income'] = df1['income'].apply(lambda x: 1 if x.strip() == '>50K' else 0)

# ------------------ Load Dataset 2 Stroke prediction (Kaggle) ------------------ #
file_path = r"C:\Users\Marcin\.cache\kagglehub\datasets\fedesoriano\stroke-prediction-dataset\versions\1\healthcare-dataset-stroke-data.csv"
df2 = pd.read_csv(file_path)
df2['smoking_status'] = df2['smoking_status'].replace("Unknown", np.nan)

# ------------------ Load Dataset 3 Pima Indians Diabetes Database(Kaggle) ------------------ #
file_path = r"C:\Users\Marcin\.cache\kagglehub\datasets\uciml\pima-indians-diabetes-database\versions\1\diabetes.csv"
df3 = pd.read_csv(file_path)
df3["BloodPressure"] = df3["BloodPressure"].replace(0, np.nan)
df3["SkinThickness"] = df3["SkinThickness"].replace(0, np.nan)
df3["Insulin"] = df3["Insulin"].replace(0, np.nan)


def usecase(df, missing_predictors, target, selected_predictors = None):
    complete_df = df.dropna()
    test_df = complete_df.sample(frac=0.2, random_state=42)
    train_df = df.drop(index=test_df.index)  # contains missingness

    results = []
    n_iterations = 40
    subset_size = 200
    random_state = 42
    
    if selected_predictors is None:
        selected_predictors = [col for col in df.columns if col != target]

    from sklearn.preprocessing import StandardScaler

    def preprocess_and_model(strategy_fn, train_data, test_data, strategy_label, seed):
        sample = train_data.sample(n=subset_size, random_state=seed)

        sample = sample[selected_predictors + [target]]
        test = test_data[selected_predictors + [target]]

        if strategy_label == "Missing Indicator":
            for predictor in missing_predictors:
                if predictor in sample.columns and sample[predictor].isnull().any():
                    sample = missing_indicator(sample, predictor)
        elif strategy_fn is not None:
            sample = strategy_fn(sample)

        # One-hot encoding
        sample = pd.get_dummies(sample, drop_first=True)
        test = pd.get_dummies(test, drop_first=True)
        sample, test = sample.align(test, join='left', axis=1, fill_value=0)

        X_train = sample.drop(columns=[target])
        y_train = sample[target]
        X_test = test.drop(columns=[target])
        y_test = test[target]

        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)

        model = LogisticRegression(max_iter=1000, random_state=123)
        model.fit(X_train_scaled, y_train)
        y_prob = model.predict_proba(X_test_scaled)[:, 1]
        auc = roc_auc_score(y_test, y_prob)

        return auc

    for i in range(n_iterations):
        seed = random_state + i
        auc_cca = preprocess_and_model(complete_cases, train_df, test_df, "Complete Cases", seed)
        auc_mi = preprocess_and_model(multiple_imputation, train_df, test_df, "Multiple Imputation", seed)
        auc_ind = preprocess_and_model(None, train_df, test_df, "Missing Indicator", seed)

        results.extend([
            {'Model': 'Complete Cases', 'AUC': auc_cca},
            {'Model': 'Multiple Imputation', 'AUC': auc_mi},
            {'Model': 'Missing Indicator', 'AUC': auc_ind}
        ])

    # ------------------ Plot AUCs with Error Bars (±1 SD) ------------------ #
    results_df = pd.DataFrame(results)
    summary_df = results_df.groupby('Model').agg(
        mean_auc=('AUC', 'mean'),
        std_auc=('AUC', 'std')
    ).reset_index()

    model_order = ['Complete Cases', 'Multiple Imputation', 'Missing Indicator']
    colors = ['blue', 'orange', 'green']
    summary_df['Model'] = pd.Categorical(summary_df['Model'], categories=model_order, ordered=True)
    summary_df = summary_df.sort_values('Model')

    plt.figure(figsize=(8, 6))
    x = summary_df['Model']
    y = summary_df['mean_auc']
    yerr = summary_df['std_auc']

    plt.bar(x, y, yerr=yerr, capsize=8, color=colors)
    plt.title('AUC (Mean ± SD) Across Missingness Strategies\n(30x Samples of 200rec)')
    plt.ylabel('Mean AUC')
    plt.ylim(0.5, 1.0)
    plt.grid(axis='y')
    plt.tight_layout()
    plt.show()


def overview(df, target_var):
    if target_var not in df.columns:
        raise ValueError(f"Target variable '{target_var}' not found in DataFrame.")
    
    total_rows = len(df)
    
    # Calculate missing values
    missing_count = df.isnull().sum()
    missing_percentage = (missing_count / total_rows) * 100

    # Calculate correlation with the target
    correlations = {}
    for col in df.columns:
        if col == target_var:
            correlations[col] = 1.0
        else:
            try:
                correlations[col] = df[[col, target_var]].dropna().corr().iloc[0, 1]
            except:
                correlations[col] = None  # in case of non-numeric column or failure
    
    # Combine into a DataFrame
    overview_df = pd.DataFrame({
        'Missing Count': missing_count,
        'Missing Percentage': missing_percentage.round(2),
        'Cor with Target': pd.Series(correlations)
    })

    overview_df = overview_df.sort_values(by='Missing Percentage', ascending=False)

    print("\nOverview of Variables:\n")
    print(overview_df)
    print('\n')
    
    return overview_df


print('Case 1: Binary variable income (<=50k or >50k)')
overview(df1, 'income')
print('Case 2: Binary variable stroke (0 or 1)')
overview(df2, 'stroke')
print('Case 3: Binary variable diabetes (0 or 1)')
overview(df3, 'Outcome')

usecase(df1, ["workclass", "occupation","native-country"], "income")
usecase(df2, ["age", "bmi","smoking_status"], "stroke")
#usecase(df2, ["age", "bmi","smoking_status"], "stroke", ["age", "bmi","smoking_status"])
usecase(df3, ["BloodPressure", "SkinThickness", "Insulin"], "Outcome" )
#usecase(df3, ["BloodPressure", "SkinThickness", "Insulin"], "Outcome" , ["Age", "SkinThickness", "Insulin"])  

#0missingvariables apart?  