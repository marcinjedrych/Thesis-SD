# -*- coding: utf-8 -*-
"""
Created on Sat Apr 26 16:16:50 2025

MCAR models

@author: Marcin
"""


import matplotlib.pyplot as plt
import seaborn as sns
import os
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, recall_score, precision_score, confusion_matrix
from functions.other import results_to_excel, format_and_sample
from sklearn.calibration import calibration_curve
from sklearn.metrics import brier_score_loss
from sklearn.metrics import roc_auc_score
from sklearn.metrics import roc_curve
from functions.strategies import missing_indicator, ensemble

plots = False
data = 'Data'

# Initialize an empty list to store results
results = []

def logistic_regression(train, test, target, label, NA = False):
    
    #if NA in dataset use missing indicator (for Learned NA)
    if NA is True:
        train = missing_indicator(train, 'bp')
        test = missing_indicator(test, 'bp')
    
    # Handle categorical variables
    train = pd.get_dummies(train, columns=['stage'])
    test = pd.get_dummies(test, columns=['stage'])
    
    # Separate predictors (X) and outcome (y)
    X_train = train.drop(columns=[target])
    X_train = X_train.drop(columns=['Index','latent1','latent2'])
    print(X_train.columns)
    X_test = test.drop(columns=[target])
    X_test = X_test.drop(columns=['Index','latent1','latent2'])
    
    y_train = train[target]
    y_test = test[target]

    # Train logistic regression model
    model = LogisticRegression(max_iter=1000, random_state=123)
    model.fit(X_train, y_train)

    # Predictions
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]  # probability of class 1

    # discrimination metrics 
    acc = accuracy_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    precision = precision_score(y_test,y_pred)
    auc = roc_auc_score(y_test, y_prob)  
    
    fpr, tpr, _ = roc_curve(y_test, y_prob)  # ROC curve values
    conf_matrix = confusion_matrix(y_test, y_pred)
    
    # Calibration metrics
    brier = brier_score_loss(y_test, y_prob) # mean squared difference between predicted probabilities and actual binary outcomes.
    prob_true, prob_pred = calibration_curve(y_test, y_prob, n_bins=10)
    
    # Save results
    results.append({
        'Model': label,
        'Accuracy': acc,
        'Recall': recall,
        'Precision': precision,
        'AUC': auc,
        'Brier Score': brier
    })
    
    # Output
    print(f"\nPerformance for {label}:")
    print(f"  - Accuracy: {acc:.4f}")
    print(f"  - Recall: {recall:.4f}")
    print(f"  - Precision: {precision:.4f}")
    print(f"  - AUC: {auc:.4f}")
    print(f"  - Brier Score: {brier:.4f}")
    print("  - Confusion Matrix:")
      
    if plots is True:
        # Plot confusion matrix
        plt.figure(figsize=(6,5))
        sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues')
        plt.xlabel('Predicted')
        plt.ylabel('Actual')
        plt.title(f'Confusion Matrix for {label}')
        plt.show()
        
        # ROC Curve
        plt.figure(figsize=(6,5))
        plt.plot(fpr, tpr, label=f'AUC = {auc:.2f}')
        plt.plot([0, 1], [0, 1], linestyle='--', color='gray')
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title(f'ROC Curve for {label}')
        plt.legend(loc='lower right')
        plt.grid(True)
        plt.show()
    
        
        # Calibration plot: predicted probabilities vs. observed frequencies, perfectly calibrated model lies on the diagonal (y = x).
        plt.figure(figsize=(6,5))
        plt.plot(prob_pred, prob_true, marker='o', label='Model')
        plt.plot([0, 1], [0, 1], linestyle='--', color='gray', label='Perfectly Calibrated')
        plt.xlabel('Mean Predicted Probability')
        plt.ylabel('Actual hospitaldeath = 1')
        plt.title(f'Calibration Curve for {label}')
        plt.legend()
        plt.grid(True)
        plt.show()

    return results
    
# Test data
test_data = pd.read_excel(f"{data}/test_data.xlsx")
test_data = test_data.rename(columns={'Unnamed: 0': 'Index'})

# Define your target variable
target_variable = 'hospitaldeath'
n_iter = 50
subset_size = 200
cca_subset_size = round(0.45*200)
mi_path = f"{data}/Original/Multiple Imputation/"
mi_path2 = f"{data}/Original/Multiple Imputation/"

# MCAR - Original
original_mcar_cca = pd.read_excel(f"{data}/Original/Complete Case Analysis/bp_mcar_cca.xlsx")

excel_files = [f for f in os.listdir(mi_path) if f.lower().endswith(('.xlsx', '.xls')) and 'mcar' in f.lower()]
mcar_dataframes = [pd.read_excel(os.path.join(mi_path, file)) for file in excel_files]
#original_mcar_mi = pd.read_excel(f"{data}/Original/Multiple Imputation/bp_mcar_mi.xlsx")

original_mcar_na = pd.read_excel(f"{data}/Original/Learned NA/bp_mcar.xlsx")

# MCAR- Synthetic
syn_mcar_cca = pd.read_excel(f"{data}/Synthetic/Complete Case Analysis/bp_mcar_cca.xlsx")

excel_files = [f for f in os.listdir(mi_path2) if f.lower().endswith(('.xlsx', '.xls')) and 'mcar' in f.lower()]
mcar_syn_dataframes = [pd.read_excel(os.path.join(mi_path2, file)) for file in excel_files]
#syn_mcar_mi = pd.read_excel(f"{data}/Synthetic/Multiple Imputation/bp_mcar_mi.xlsx")

syn_mcar_na = pd.read_excel(f"{data}/Synthetic/Learned NA/bp_mcar.xlsx")

# MAR - Original
original_mar_cca = pd.read_excel(f"{data}/Original/Complete Case Analysis/bp_mar_cca.xlsx")

excel_files = [f for f in os.listdir(mi_path) if f.lower().endswith(('.xlsx', '.xls')) and 'mar' in f.lower()]
mar_dataframes = [pd.read_excel(os.path.join(mi_path, file)) for file in excel_files]
#original_mar_mi = pd.read_excel(f"{data}/Original/Multiple Imputation/bp_mar_mi.xlsx")

original_mar_na = pd.read_excel(f"{data}/Original/Learned NA/bp_mar.xlsx")

# MAR - Synthetic
syn_mar_cca = pd.read_excel(f"{data}/Synthetic/Complete Case Analysis/bp_mar_cca.xlsx")

excel_files = [f for f in os.listdir(mi_path2) if f.lower().endswith(('.xlsx', '.xls')) and 'mar' in f.lower()]
mar_syn_dataframes = [pd.read_excel(os.path.join(mi_path2, file)) for file in excel_files]
#syn_mar_mi = pd.read_excel(f"{data}/Synthetic/Multiple Imputation/bp_mar_mi.xlsx")

syn_mar_na = pd.read_excel(f"{data}/Synthetic/Learned NA/bp_mar.xlsx")

# MNAR - Original
original_mnar_cca = pd.read_excel(f"{data}/Original/Complete Case Analysis/bp_mnar_cca.xlsx")

excel_files = [f for f in os.listdir(mi_path) if f.lower().endswith(('.xlsx', '.xls')) and 'mnar' in f.lower()]
mnar_dataframes = [pd.read_excel(os.path.join(mi_path, file)) for file in excel_files]
#original_mnar_mi = pd.read_excel(f"{data}/Original/Multiple Imputation/bp_mnar_mi.xlsx")

original_mnar_na = pd.read_excel(f"{data}/Original/Learned NA/bp_mnar.xlsx")

# MNAR - Synthetic
syn_mnar_cca = pd.read_excel(f"{data}/Synthetic/Complete Case Analysis/bp_mnar_cca.xlsx")

excel_files = [f for f in os.listdir(mi_path2) if f.lower().endswith(('.xlsx', '.xls')) and 'mnar' in f.lower()]
mnar_syn_dataframes = [pd.read_excel(os.path.join(mi_path2, file)) for file in excel_files]
#syn_mnar_mi = pd.read_excel(f"{data}/Synthetic/Multiple Imputation/bp_mnar_mi.xlsx")

syn_mnar_na = pd.read_excel(f"{data}/Synthetic/Learned NA/bp_mnar.xlsx")

baseline_len = len(pd.read_excel(f"{data}/Original/no_missing.xlsx"))
sum_mis = len(original_mar_cca) + len(original_mcar_cca) + len(original_mnar_cca)
avg_n = sum_mis // 3
subset_size = 200
avg_mis = 1 - (avg_n / baseline_len)
cca_subset_size = round(avg_mis*subset_size)

for i in range(n_iter):
    
    ### CCA HAS TO HAVE SMALLER SUBSET!
    
    # MCAR - original
    original_mcar_cca_sample = format_and_sample(original_mcar_cca, data = data, nsubset = cca_subset_size, random_state=i)
    original_mcar_mi = []
    for df in mcar_dataframes:
        sample = format_and_sample(df, data = data, nsubset = subset_size, random_state=i)
        original_mcar_mi.append(sample)
    #original_mcar_mi_sample = format_and_sample(original_mcar_mi, data = data, nsubset = subset_size, random_state=i)
    original_mcar_na_sample = format_and_sample(original_mcar_na, data = data, nsubset = subset_size, random_state=i)
    
    # MCAR - Synthetic
    syn_mcar_cca_sample = format_and_sample(syn_mcar_cca, data = data, nsubset = cca_subset_size, random_state=i)
    syn_mcar_mi = []
    for df in mcar_syn_dataframes:
        sample = format_and_sample(df, data = data, nsubset = subset_size, random_state=i)
        syn_mcar_mi.append(sample)
    #syn_mcar_mi_sample = format_and_sample(syn_mcar_mi, data = data, nsubset = subset_size, random_state=i)
    syn_mcar_na_sample = format_and_sample(syn_mcar_na, data = data, nsubset = subset_size, random_state=i)
    
    # MAR - Original
    original_mar_cca_sample = format_and_sample(original_mar_cca, data = data, nsubset = cca_subset_size, random_state=i)
    original_mar_mi = []
    for df in mar_dataframes:
        sample = format_and_sample(df, data = data, nsubset = subset_size, random_state=i)
        original_mar_mi.append(sample)
    #original_mar_mi_sample = format_and_sample(original_mar_mi, data = data, nsubset = subset_size, random_state=i)
    original_mar_na_sample = format_and_sample(original_mar_na, data = data, nsubset = subset_size, random_state=i)
    
    # MAR - Synthetic
    syn_mar_cca_sample = format_and_sample(syn_mar_cca, data = data, nsubset = cca_subset_size, random_state=i)
    syn_mar_mi = []
    for df in mar_syn_dataframes:
        sample = format_and_sample(df, data = data, nsubset = subset_size, random_state=i)
        syn_mar_mi.append(sample)
    #syn_mar_mi_sample = format_and_sample(syn_mar_mi, data = data, nsubset = subset_size, random_state=i)
    syn_mar_na_sample = format_and_sample(syn_mar_na, data = data, nsubset = subset_size, random_state=i)
    
    # MNAR - Original
    original_mnar_cca_sample = format_and_sample(original_mnar_cca, data = data, nsubset = cca_subset_size, random_state=i)
    original_mnar_mi = []
    for df in mnar_dataframes:
        sample = format_and_sample(df, data = data, nsubset = subset_size, random_state=i)
        original_mnar_mi.append(sample)
    #original_mnar_mi_sample = format_and_sample(original_mnar_mi, data = data, nsubset = subset_size, random_state=i)
    original_mnar_na_sample = format_and_sample(original_mnar_na, data = data, nsubset = subset_size, random_state=i)
    
    # MNAR - Synthetic
    syn_mnar_cca_sample = format_and_sample(syn_mnar_cca, data = data, nsubset = cca_subset_size, random_state=i)
    syn_mnar_mi = []
    for df in mnar_syn_dataframes:
        sample = format_and_sample(df, data = data, nsubset = subset_size, random_state=i)
        syn_mnar_mi.append(sample)
    #syn_mnar_mi_sample = format_and_sample(syn_mnar_mi, data = data, nsubset = subset_size, random_state=i)
    syn_mnar_na_sample = format_and_sample(syn_mnar_na, data = data, nsubset = subset_size, random_state=i)

    # Run logistic regression for each 
    # MCAR
    logistic_regression(original_mcar_cca_sample, test_data, target=target_variable, label="Original CCA (MCAR)")
    ensemble(original_mcar_mi, test_data, target=target_variable)
    #logistic_regression(original_mcar_mi_sample, test_data, target=target_variable, label="Original MI (MCAR)")
    logistic_regression(original_mcar_na_sample, test_data, target=target_variable, label="Original Learned NA (MCAR)", NA = True)
    
    logistic_regression(syn_mcar_cca_sample, test_data, target=target_variable, label="Synthetic CCA (MCAR)")
    #logistic_regression(syn_mcar_mi_sample, test_data, target=target_variable, label="Synthetic MI (MCAR)")
    ensemble(syn_mcar_mi, test_data, target=target_variable)
    logistic_regression(syn_mcar_na_sample, test_data, target=target_variable, label="Synthetic Learned NA (MCAR)", NA = True)
    
    # MAR
    logistic_regression(original_mar_cca_sample, test_data, target=target_variable, label="Original CCA (MAR)")
    #logistic_regression(original_mar_mi_sample, test_data, target=target_variable, label="Original MI (MAR)")
    ensemble(original_mar_mi, test_data, target=target_variable)
    logistic_regression(original_mar_na_sample, test_data, target=target_variable, label="Original Learned NA (MAR)", NA = True)
    
    logistic_regression(syn_mar_cca_sample, test_data, target=target_variable, label="Synthetic CCA (MAR)")
    #logistic_regression(syn_mar_mi_sample, test_data, target=target_variable, label="Synthetic MI (MAR)")
    ensemble(syn_mar_mi, test_data, target=target_variable)
    logistic_regression(syn_mar_na_sample, test_data, target=target_variable, label="Synthetic Learned NA (MAR)", NA = True)
    
    # MNAR
    logistic_regression(original_mnar_cca_sample, test_data, target=target_variable, label="Original CCA (MNAR)")
    #logistic_regression(original_mnar_mi_sample, test_data, target=target_variable, label="Original MI (MNAR)")
    ensemble(original_mnar_mi, test_data, target=target_variable)
    logistic_regression(original_mnar_na_sample, test_data, target=target_variable, label="Original Learned NA (MNAR)", NA = True)
    
    logistic_regression(syn_mnar_cca_sample, test_data, target=target_variable, label="Synthetic CCA (MNAR)")
    #logistic_regression(syn_mnar_mi_sample, test_data, target=target_variable, label="Synthetic MI (MNAR)")
    ensemble(syn_mnar_mi, test_data, target=target_variable)
    logistic_regression(syn_mnar_na_sample, test_data, target=target_variable, label="Synthetic Learned NA (MNAR)", NA = True)


metrics_df = pd.DataFrame(results)

# Specify only the numeric columns you want to summarize
metric_cols = ['Accuracy', 'Recall', 'Precision', 'AUC', 'Brier Score']

# Compute mean and std, grouped by Model
grouped_mean = metrics_df.groupby('Model', sort=False)[metric_cols].mean().round(3).reset_index()
grouped_sd = metrics_df.groupby('Model', sort=False)[metric_cols].std().round(3).reset_index()

try:
    old_metrics = pd.read_excel('metrics_mean.xlsx')
    old_metrics_sd = pd.read_excel('metrics_sd.xlsx')
    grouped_mean = pd.concat([old_metrics, grouped_mean]).drop_duplicates()
    grouped_sd = pd.concat([old_metrics_sd, grouped_sd]).drop_duplicates()
except FileNotFoundError:
    pass

grouped_mean.to_excel('metrics_mean.xlsx', index=False)
grouped_sd.to_excel('metrics_sd.xlsx', index=False)


 

 

