# -*- coding: utf-8 -*-
"""
Created on Tue Apr  8 17:50:18 2025

@author: Marcin
"""
import pandas as pd
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
from sklearn.preprocessing import OrdinalEncoder

def multiple_imputation(df, n_imputations=5):
    
    """ (from stefvanbuuren.name)
        - When covariates have missingness & MAR: estimated statistics and regression coefficients biased with complete case analysis
        - When covariates have missingness & MCAR: reduction in sample size will still reduce precision of estimated coefficients in CCA"""
        
    df_copy = df.copy()
    
    # Identify categorical columns
    categorical_cols = df_copy.select_dtypes(include=["object", "category"]).columns
    
    # Encode categorical columns
    encoder = OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1)
    df_copy[categorical_cols] = encoder.fit_transform(df_copy[categorical_cols])
    
    # Initialize the imputer
    imputer = IterativeImputer(max_iter=10, sample_posterior=True, random_state=0)  
    
    # Generate multiple imputed datasets
    imputed_datasets = []
    for _ in range(n_imputations):
        imputed_data = imputer.fit_transform(df_copy)
        imputed_df = pd.DataFrame(imputed_data, columns=df.columns)
        imputed_datasets.append(imputed_df)
    
    # Combine results: Take mean for numerical, mode for categorical
    combined_imputed_df = pd.concat(imputed_datasets).groupby(level=0).mean()
    
    for col in categorical_cols:
        combined_imputed_df[col] = pd.concat(imputed_datasets)[col].groupby(level=0).agg(lambda x: x.mode()[0])
        combined_imputed_df[col] = combined_imputed_df[col].round().astype(int)
    
    # Decode categorical columns back
    combined_imputed_df[categorical_cols] = encoder.inverse_transform(combined_imputed_df[categorical_cols])
    
    return combined_imputed_df

def complete_cases(df):
    
    """from paper multiple imputation & stefvanbuuren.name: 
        - When only outcome variable has missingness this approach is valid under MCAR, MAR. (best choice)
        - Also when missing data probability does not depend on the outcome variable Y (even under MNAR)
        - Or when Logistic Regression with missing data only in Y or X (but not both)"""

    return df.dropna()

# needed for learned NA but not before training (NOT before making data synthetic!)
def missing_indicator(df, column):

    df = df.copy()
    indicator_col = f"{column}_missing"
    
    # Create missing indicator
    df[indicator_col] = df[column].isna().astype(int)
    
    # Impute missing values with 0
    df[column] = df[column].fillna(0)
    
    return df