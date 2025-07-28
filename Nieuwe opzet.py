# -*- coding: utf-8 -*-
"""
Created on Sun Jul 13 15:09:06 2025

@author: Marcin
"""

from functions.generate import generate_patient_data
from sklearn.model_selection import train_test_split
from functions.missingness import mcar, mar, mnar
from functions.ctgan_syn import generate_synthetic_data
from functions.strategies import complete_cases, MI_impute, ensemble , missing_indicator
from functions.model import logistic_regression


root = 'Data' # 'Data'

# # Generate patient data and split
# data = generate_patient_data(nsamples=10000, seed=125)
# train_idx, test_idx = train_test_split(data.index, test_size=0.2, random_state=123)
# train_data = data.loc[train_idx]
# test_data = data.loc[test_idx]

n_iter = 3
exclude = ['latent1', 'latent2']

# 1 BASELINE

def run_baseline(n_iter, exclude=None, synthetic=False):
    results = []
    
    for i in range(n_iter):

        data = generate_patient_data(nsamples=250, seed=i)
        train_data, test_data = train_test_split(data, test_size=0.2, random_state=i)
        
        if synthetic:
            train_data = generate_synthetic_data(train_data)
        
        result = logistic_regression(train_data, test_data, 'hospitaldeath', exclude_vars=exclude)
        results.append(result)
    
    return results


# 2 Missing data

target_missing_rate=0.45
def run_models(n_iter, exclude=None, synthetic=False, m_type=None, strategy="CCA"):
    if m_type is None:
        m_type = mcar
    
    results = []
    
    for i in range(n_iter):
        # Genereer nieuwe dataset van 250
        data = generate_patient_data(nsamples=250, seed=i)
        train_data, test_data = train_test_split(data, test_size=0.2, random_state=i)
        
        # Induceer missingness
        train_data = m_type(train_data, target_column="bp", target_missing_rate=target_missing_rate)
        
        if strategy == "MI":
            dfs = MI_impute(train_data)
            
            if synthetic:
                imp_and_syn = []
                for df in dfs:
                    syn_df = generate_synthetic_data(df)
                    imp_and_syn.append(syn_df)
                _, _, results = ensemble(imp_and_syn, test_data, target="hospitaldeath")
            else:
                _, _, results = ensemble(dfs, test_data, target="hospitaldeath")
        
        else:
            if strategy == "CCA":
                train_data = complete_cases(train_data)
            else:
                train_data = missing_indicator(train_data, 'bp')
            
            if synthetic:
                train_data = generate_synthetic_data(train_data)
            
            result = logistic_regression(train_data, test_data, 'hospitaldeath', exclude_vars=exclude)
            results.append(result)
    
    return results


### 1. BASELINE
baseline_original = run_baseline(n_iter, exclude)
baseline_synthetic = run_baseline(n_iter, exclude, synthetic = True)

### 2. MISSINGNESS (ORIGINAL DATA)
## 2.1 MCAR   
original_mcar_cca = run_models(n_iter, exclude)
original_mcar_mi = run_models(n_iter, exclude, strategy = "MI")
original_mcar_ind = run_models(n_iter, exclude, strategy = "IND")

## 2.2 MAR
original_mar_cca = run_models(n_iter, exclude, m_type=mar)
original_mar_mi = run_models(n_iter, exclude, strategy = "MI", m_type = mar)
original_mar_ind = run_models(n_iter, exclude, strategy = "IND", m_type=mar)

### 2.3 MNAR
original_mnar_cca = run_models(n_iter, exclude, m_type=mnar)
original_mnar_mi = run_models(n_iter, exclude, strategy = "MI", m_type = mnar)
original_mnar_ind = run_models(n_iter, exclude, strategy = "IND", m_type=mnar)

### 3. MISSINGNESS (SYNTHETIC DATA)
## 3.1 MCAR   
syn_mcar_cca = run_models(n_iter, exclude, synthetic=True)
syn_mcar_mi = run_models(n_iter, exclude,strategy = "MI", synthetic=True)
syn_mcar_ind = run_models(n_iter, exclude, strategy = "IND", synthetic=True)

## 3.2 MAR
syn_mar_cca = run_models(n_iter, exclude, m_type=mar, synthetic=True)
syn_mar_mi = run_models(n_iter, exclude, strategy = "MI", m_type = mar, synthetic=True)
syn_mar_ind = run_models(n_iter, exclude, strategy = "IND", m_type=mar, synthetic=True)

### 3.3 MNAR
syn_mnar_cca = run_models(n_iter, exclude, m_type=mnar, synthetic=True)
syn_mnar_mi = run_models(n_iter, exclude, strategy = "MI", m_type = mnar, synthetic=True)
syn_mnar_ind = run_models(n_iter, exclude, strategy = "IND", m_type=mnar, synthetic=True)

