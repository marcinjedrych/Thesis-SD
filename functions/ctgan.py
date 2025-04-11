# -*- coding: utf-8 -*-
"""
Created on Fri Apr 11 15:49:58 2025

@author: Marcin
"""

from sdv.single_table import CTGANSynthesizer
from sdv.metadata import SingleTableMetadata

def generate_synthetic_data(df):
    
    if 'Unnamed: 0' in df.columns:
        df = df.rename(columns={'Unnamed: 0': 'Index'})
    index = df['Index']
    df = df.drop(columns=['Index'])

    metadata_obj = SingleTableMetadata()
    metadata_obj.detect_from_dataframe(df)
    synthesizer = CTGANSynthesizer(metadata_obj)
    synthesizer.fit(df)
    synthetic_data = synthesizer.sample(num_rows=len(df))

    synthetic_data.insert(0, 'Index', index)
    
    return synthetic_data
