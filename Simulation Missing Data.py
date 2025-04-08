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

data = generate_patient_data(2000)
plot_relationships(data)

# Generate patient data and split
data = generate_patient_data(nsamples=10000, seed=123)
train_idx, test_idx = train_test_split(data.index, test_size=0.2, random_state=123)
train_data = data.loc[train_idx]
test_data = data.loc[test_idx]
test_data.to_excel("Data/test_data.xlsx", index = True)  #export test data

#missingness in continuous predictor bp
bp_mcar = mcar(train_data, 'bp')
bp_mar = mar(train_data, 'bp', 'age')
bp_mnar = mnar(train_data, 'bp')

#plot_missingness(data, "... Missingness Pattern")

# Export to datafolder
train_data.to_excel("Data/Original/no_missing.xlsx", index = True) #baseline (train)

# Handle missingness (3 ways)

# 1. keep NA's
bp_mcar.to_excel("Data/Original/Learned NA/bp_mcar.xlsx", index= True)
bp_mar.to_excel("Data/Original/Learned NA/bp_mar.xlsx", index= True)
bp_mnar.to_excel("Data/Original/Learned NA/bp_mnar.xlsx", index= True)

# 2. CCA
bp_mcar_cca = complete_cases(bp_mcar)
bp_mar_cca = complete_cases(bp_mar)
bp_mnar_cca = complete_cases(bp_mnar)
bp_mcar_cca.to_excel("Data/Original/Complete Case Analysis/bp_mcar_cca.xlsx", index= True)
bp_mar_cca.to_excel("Data/Original/Complete Case Analysis/bp_mar_cca.xlsx", index= True)
bp_mnar_cca.to_excel("Data/Original/Complete Case Analysis/bp_mnar_cca.xlsx", index= True)

# 3. Multiple Imputation (MI)
bp_mcar_mi = multiple_imputation(bp_mcar)
bp_mar_mi = multiple_imputation(bp_mar)
bp_mnar_mi = multiple_imputation(bp_mnar)
bp_mcar_mi.to_excel("Data/Original/Multiple Imputation/bp_mcar_mi.xlsx", index= True)
bp_mar_mi.to_excel("Data/Original/Multiple Imputation/bp_mar_mi.xlsx", index= True)
bp_mnar_mi.to_excel("Data/Original/Multiple Imputation/bp_mnar_mi.xlsx", index= True)