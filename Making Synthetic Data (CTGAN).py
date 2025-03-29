# -*- coding: utf-8 -*-
"""
Created on Tue Mar 25 20:50:38 2025

Code to convert dataframe in synthetic one

@author: Marcin
"""


import os
import pandas as pd
from sdv.single_table import CTGANSynthesizer
from sdv.metadata import SingleTableMetadata
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
from sklearn.preprocessing import OrdinalEncoder


def multiple_imputation(df, n_imputations=5):
    
    print('Multiple Imputation...')
    df_copy = df.copy()
    
    # Identify categorical columns
    categorical_cols = df_copy.select_dtypes(include=["object", "category"]).columns
    
    # Encode categorical columns
    encoder = OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1)
    df_copy[categorical_cols] = encoder.fit_transform(df_copy[categorical_cols])
    
    # Initialize the imputer
    imputer = IterativeImputer(max_iter=10, sample_posterior=True, random_state=123)  
    
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
    
    print('CCA ...')
    return df.dropna()

metadata = {
    "primary_key": None,  # No explicit primary key in the dataset
    "columns": {
        "age": {"sdtype": "numerical", "computer_representation": "Float"},
        "weight": {"sdtype": "numerical", "computer_representation": "Float"},
        "stage": {"sdtype": "categorical"},
        "therapy": {"sdtype": "categorical"},
        "bp": {"sdtype": "numerical", "computer_representation": "Float"},
    }
}

def generate_synthetic_data(df, index, output_folder, filename):
    metadata_obj = SingleTableMetadata()
    metadata_obj.detect_from_dataframe(df)
    synthesizer = CTGANSynthesizer(metadata_obj)
    synthesizer.fit(df)
    synthetic_data = synthesizer.sample(num_rows=len(df))
    synthetic_data.insert(0, 'Index', index)
    os.makedirs(output_folder, exist_ok=True)
    synthetic_data.to_excel(os.path.join(output_folder, f"synthetic_{filename}"), index=False)

data_folder = "data"
synthetic_base_folder = os.path.join(data_folder, "Synthetic")
subfolders = {"Complete Case Analysis": complete_cases, "Multiple Imputation": multiple_imputation, "Learned NA": lambda x: x}

xlsx_files = [f for f in os.listdir(data_folder) if f.endswith(".xlsx")]

for file in xlsx_files:
    file_path = os.path.join(data_folder, file)
    print(f"Processing {file}...")
    df = pd.read_excel(file_path)
    df = df.rename(columns={'Unnamed: 0': 'Index'})
    index = df['Index']
    df = df.drop(columns=['Index'])
    
    if df.empty:
        print(f"Skipping {file} (empty dataset).")
        continue
    
    for subfolder, preprocess_func in subfolders.items():
        processed_df = preprocess_func(df)
        output_folder = os.path.join(synthetic_base_folder, subfolder)
        generate_synthetic_data(processed_df, index, output_folder, file)
        print(f"Synthetic data saved for {subfolder}: {file}")

print("Processing complete.")
