# -*- coding: utf-8 -*-
"""
Created on Sun Jul 13 15:09:06 2025

@author: Marcin
"""

from functions.generate import generate_patient_data
from functions.missingness import mcar, mar, mnar
from functions.ctgan_syn import generate_synthetic_data
from functions.strategies import complete_cases, MI_impute, ensemble, missing_indicator
from functions.model import logistic_regression
from sklearn.model_selection import train_test_split

root = 'Data' # 'Data'

# Generate patient data and split
data = generate_patient_data(nsamples=10000, seed=125)
train_idx, test_idx = train_test_split(data.index, test_size=0.2, random_state=123)
train_data = data.loc[train_idx]
test_data = data.loc[test_idx]

n_iter = 50
n_subset = 200
exclude = ['latent1', 'latent2']

# 1 BASELINE

def run_baseline(train_data, test_data, n_iter, n_subset, exclude = None, synthetic = False):
    
    results = []
    for i in range(n_iter):
    
        sample = train_data.sample(n=n_subset, random_state=i)
        
        if synthetic is True:
            sample = generate_synthetic_data(sample)

        result = logistic_regression(sample, test_data, 'hospitaldeath', exclude_vars=exclude)
        
        results.append(result)

    return results
    

baseline_original = run_baseline(train_data, test_data, n_iter, n_subset, exclude)
baseline_synthetic = run_baseline(train_data, test_data, n_iter, n_subset, exclude, synthetic = True)

# 2 Missing data

target_missing_rate=0.45

def run_models(train_data, test_data, n_iter, n_subset, exclude = None, synthetic = False, m_type = None, strategy = None):
    
    if m_type is None:
        m_type = mcar()
    if strategy is None:
        strategy = complete_cases()
        
    results = []
    for i in range(n_iter):
    
        sample = train_data.sample(n=n_subset, random_state=i)
        
        sample = m_type(sample, target_column = "hospitaldeath", target_missing_rate = target_missing_rate)
        
        if synthetic is True:
            sample = generate_synthetic_data(sample)

        result = logistic_regression(sample, test_data, 'hospitaldeath', exclude_vars=exclude)
        
        results.append(result)

    return results
    
original_mcar_cca = run_models(train_data, test_data, n_iter, n_subset, exclude)

# ! special case
#original_mcar_mi = run_models(train_data, test_data, n_iter, n_subset, exclude)

original_mcar_ind = run_models(train_data, test_data, n_iter, n_subset, exclude, strategy = missing_indicator())


# ..... Na Egypte
