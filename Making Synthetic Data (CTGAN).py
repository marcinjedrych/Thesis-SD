# -*- coding: utf-8 -*-
"""
Created on Tue Mar 25 20:50:38 2025

Code to make synthetic versions

@author: Marcin
"""

import pandas as pd
from functions.ctgan_syn import generate_synthetic_data

import os

# 1. Process 'no_missing.xlsx' and save synthetic data
df = pd.read_excel(r"Data/Original/no_missing.xlsx")
synthetic_data = generate_synthetic_data(df)
synthetic_data.to_excel(r"Data/Synthetic/synthetic_no_missing.xlsx", index=False)

# 2. Complete Case Analysis (CCA)
for filename in os.listdir(r"Data/Original/Complete Case Analysis"):
    if filename.endswith(".xlsx"):
        df = pd.read_excel(rf"Data/Original/Complete Case Analysis/{filename}")
        synthetic_data = generate_synthetic_data(df)
        synthetic_data.to_excel(rf"Data/Synthetic/Complete Case Analysis/{filename}", index=False)

# 3. Multiple Imputation
for filename in os.listdir(r"Data/Original/Multiple Imputation"):
    if filename.endswith(".xlsx"):
        df = pd.read_excel(rf"Data/Original/Multiple Imputation/{filename}")
        synthetic_data = generate_synthetic_data(df)
        synthetic_data.to_excel(rf"Data/Synthetic/Multiple imputation/{filename}", index=False)

# 4. Learned NA
for filename in os.listdir(r"Data/Original/Learned NA"):
    if filename.endswith(".xlsx"):
        df = pd.read_excel(rf"Data/Original/Learned NA/{filename}")
        synthetic_data = generate_synthetic_data(df)
        synthetic_data.to_excel(rf"Data/Synthetic/Learned NA/{filename}", index=False)