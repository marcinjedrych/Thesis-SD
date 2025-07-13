# -*- coding: utf-8 -*-
"""
Created on Sun Jul 13 11:59:19 2025

@author: Marcin
"""

from functions.strategies import MI_impute, ensemble
from functions.missingness import mcar, mar, mnar
from functions.generate import generate_patient_data
from sklearn.model_selection import train_test_split

root = 'Data' # 'Data'

# Generate patient data and split
data = generate_patient_data(nsamples=10000, seed=125)
train_idx, test_idx = train_test_split(data.index, test_size=0.2, random_state=123)
train_data = data.loc[train_idx]
test_data = data.loc[test_idx]
test_data.to_excel(f"{root}/test_data.xlsx", index = True)  #export test data

# Missingness in continuous predictor bp
bp_mcar = mcar(train_data, 'bp', missing_rate=0.45)
bp_mar = mar(train_data, 'bp', 'stage', target_missing_rate=0.45, beta_1=0.5)
bp_mnar = mnar(train_data, 'bp', target_missing_rate=0.45)

### Multiple Imputation
imputed_datasets = MI_impute(bp_mcar, n_imputations=10) #on MCAR data here
hard_preds, soft_preds = ensemble(imputed_datasets, test_data, target='hospitaldeath')

# Evaluate
from sklearn.metrics import roc_auc_score, accuracy_score
true_labels = test_data['hospitaldeath']
print("AUC:", roc_auc_score(true_labels, soft_preds))
print("Accuracy:", accuracy_score(true_labels, hard_preds))
