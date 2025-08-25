# -*- coding: utf-8 -*-
"""
Created on Sun Jul 13 11:59:19 2025

@author: Marcin
"""

from functions.strategies import MI_impute, ensemble
from functions.missingness import mcar, mar, mnar
from functions.generate import generate_patient_data
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score


root = 'Data' # 'Data'

# Generate patient data and split
data = generate_patient_data(nsamples=200, seed=125)
train_idx, test_idx = train_test_split(data.index, test_size=0.2, random_state=123)
train_data = data.loc[train_idx]
test_data = data.loc[test_idx]
test_data.to_excel(f"{root}/test_data.xlsx", index = True)  #export test data

# Missingness in continuous predictor bp
bp_mcar,_ = mcar(train_data, 'bp', target_missing_rate=0.45)
bp_mar,_ = mar(train_data, 'bp', target_missing_rate=0.45, beta_1=0.5)
bp_mnar,_ = mnar(train_data, 'bp', target_missing_rate=0.45)

### Multiple Imputation
for i in range(2,20):
    imputed_datasets = MI_impute(bp_mar, n_imputations=i) #on MCAR data here
    hard_preds, soft_preds, _ = ensemble(imputed_datasets, test_data, target='hospitaldeath')
    
    y_true = test_data['hospitaldeath'].values
    auc = roc_auc_score(y_true, soft_preds)
    print(f"n_imputations: {i}, AUC: {auc}")

# Evaluate
from sklearn.metrics import roc_auc_score, accuracy_score
true_labels = test_data['hospitaldeath']
print("AUC:", roc_auc_score(true_labels, soft_preds))
print("Accuracy:", accuracy_score(true_labels, hard_preds))
