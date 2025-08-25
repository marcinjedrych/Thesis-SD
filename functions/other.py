# -*- coding: utf-8 -*-
"""
Created on Sat Apr 26 16:13:34 2025
 
handy functions

@author: Marcin
"""

import os
import pandas as pd

def results_to_excel(results, output_file='model_performance_summary.xlsx'):

    # Convert the list of results to a DataFrame
    results_df = pd.DataFrame(results)

    # Check if the output file already exists
    if os.path.exists(output_file):
        # Load the existing file
        existing_df = pd.read_excel(output_file)
        
        # Combine old and new results
        combined_df = pd.concat([existing_df, results_df], ignore_index=True)
        
        # Drop duplicates based on 'Model', keeping the most recent
        combined_df = combined_df.drop_duplicates(subset=['Model'], keep='last')
    else:
        # If no file exists, use the new results
        combined_df = results_df

    # Save to Excel
    combined_df.to_excel(output_file, index=False)


def format_and_sample(input_df, data= "Data", reference_path="/test_data.xlsx", nsubset = 200, random_state=123):
    
    path = data + reference_path
    
    # Load and format reference data
    reference_df = pd.read_excel(path)
    reference_df = reference_df.rename(columns={'Unnamed: 0': 'Index'})
    
    expected_columns = reference_df.columns.tolist()
    
    # If 'Index' is missing, add it as a regular index
    if 'Index' not in input_df.columns:
        input_df = input_df.reset_index().rename(columns={'index': 'Index'})
    
    # Re-check missing columns
    missing_cols = [col for col in expected_columns if col not in input_df.columns]
    if missing_cols:
        raise ValueError(f"The input dataframe is missing the following required columns: {missing_cols}")
    
    # Drop extra columns and reorder
    cleaned_df = input_df[expected_columns].copy()
    sample = cleaned_df.sample(n=nsubset, random_state = random_state)
    
    return sample

