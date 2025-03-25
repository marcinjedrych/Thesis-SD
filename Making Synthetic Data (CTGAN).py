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

# Define paths
data_folder = "data"
synthetic_folder = os.path.join(data_folder, "Synthetic")

# Create the Synthetic folder if it doesn't exist
os.makedirs(synthetic_folder, exist_ok=True)

# Get all .xlsx files in the data folder
xlsx_files = [f for f in os.listdir(data_folder) if f.endswith(".xlsx")]

for file in xlsx_files:
    file_path = os.path.join(data_folder, file)
    print(f"Processing {file}...")
    
    # Read the Excel file into a dataframe
    df = pd.read_excel(file_path)
    df = df.rename(columns={'Unnamed: 0': 'Index'})
    index = df['Index']
    df = df.drop(columns=['Index'])
    
    if df.empty:
        print(f"Skipping {file} (empty dataset).")
        continue
    
    # Infer metadata automatically
    metadata_obj = SingleTableMetadata()
    metadata_obj.detect_from_dataframe(df)
    
    # Initialize the synthesizer
    synthesizer = CTGANSynthesizer(metadata_obj)
    
    # Fit the synthesizer
    print(f"Fitting synthesizer for {file}...")
    synthesizer.fit(df)
    
    # Generate synthetic data
    print(f"Generating synthetic data for {file}...")
    synthetic_data = synthesizer.sample(num_rows=len(df))
    
    synthetic_data.insert(0, 'Index', index)
                                
    # Save synthetic data to an Excel file
    synthetic_file_path = os.path.join(synthetic_folder, f"synthetic_{file}")
    synthetic_data.to_excel(synthetic_file_path, index=False)
    
    print(f"Synthetic data saved: {synthetic_file_path}\n")

print("Processing complete.")
