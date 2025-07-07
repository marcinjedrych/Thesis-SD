# -*- coding: utf-8 -*-
"""
Created on Sun Mar  2 14:32:57 2025

Missingness scenarios

@author: Marcin
"""

from sklearn.model_selection import train_test_split
from functions.generate import generate_patient_data, plot_relationships
from functions.missingness import plot_missingness, mcar, mar, mnar
from functions.strategies import multiple_imputation, complete_cases
import pandas as pd

#BINARY OUTCOME

#data = generate_patient_data(2000)
#plot_relationships(data)

# Generate patient data and split
data = generate_patient_data(nsamples=10000, seed=123)
train_idx, test_idx = train_test_split(data.index, test_size=0.2, random_state=123)
train_data = data.loc[train_idx]
test_data = data.loc[test_idx]
test_data.to_excel("Data/test_data.xlsx", index = True)  #export test data

# Define mapping and convert to ordered categorical
stage_order = {'I': 1, 'II': 2, 'III': 3, 'IV': 4}
train_data['stage'] = pd.Categorical(train_data['stage'], categories=stage_order.keys(), ordered=True)
# If you need the numeric version (e.g., for multiplication), extract codes + 1
train_data['stage_num'] = train_data['stage'].cat.codes + 1

#missingness in continuous predictor bp
bp_mcar = mcar(train_data, 'bp', missing_rate=0.45)
bp_mar = mar(train_data, 'bp', 'stage', target_missing_rate=0.45)
bp_mnar = mnar(train_data, 'bp', target_missing_rate=0.45)

#plot_missingness(data, "... Missingness Pattern")

# Export to datafolder
train_data.to_excel("Data/Original/no_missing.xlsx", index = True) #baseline (train)

# Handle missingness (3 ways)

# 1. CCA
bp_mcar_cca = complete_cases(bp_mcar)
bp_mcar_cca = bp_mcar_cca.drop(columns='stage_num', errors='ignore')
bp_mar_cca = complete_cases(bp_mar)
bp_mar_cca = bp_mar_cca.drop(columns='stage_num', errors='ignore')
bp_mnar_cca = complete_cases(bp_mnar)
bp_mnar_cca = bp_mnar_cca.drop(columns='stage_num', errors='ignore')
bp_mcar_cca.to_excel("Data/Original/Complete Case Analysis/bp_mcar_cca.xlsx", index= True)
bp_mar_cca.to_excel("Data/Original/Complete Case Analysis/bp_mar_cca.xlsx", index= True)
bp_mnar_cca.to_excel("Data/Original/Complete Case Analysis/bp_mnar_cca.xlsx", index= True)

# 3. Multiple Imputation (MI)
bp_mcar_mi = multiple_imputation(bp_mcar)
bp_mcar_mi = bp_mcar_mi.drop(columns='stage_num', errors='ignore')
bp_mar_mi = multiple_imputation(bp_mar)
bp_mar_mi = bp_mar_mi.drop(columns='stage_num', errors='ignore')
bp_mnar_mi = multiple_imputation(bp_mnar)
bp_mnar_mi = bp_mnar_mi.drop(columns='stage_num', errors='ignore')
bp_mcar_mi.to_excel("Data/Original/Multiple Imputation/bp_mcar_mi.xlsx", index= True)
bp_mar_mi.to_excel("Data/Original/Multiple Imputation/bp_mar_mi.xlsx", index= True)
bp_mnar_mi.to_excel("Data/Original/Multiple Imputation/bp_mnar_mi.xlsx", index= True)

# 3. keep NA's
bp_mcar = bp_mcar.drop(columns='stage_num', errors='ignore')
bp_mcar.to_excel("Data/Original/Learned NA/bp_mcar.xlsx", index= True)
bp_mar.to_excel("Data/Original/Learned NA/bp_mar.xlsx", index= True)
bp_mnar.to_excel("Data/Original/Learned NA/bp_mnar.xlsx", index= True)
