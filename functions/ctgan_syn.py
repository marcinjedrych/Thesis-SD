# -*- coding: utf-8 -*-
"""
Created on Fri Apr 11 15:49:58 2025

@author: Marcin
"""

import warnings

warnings.filterwarnings(
    "ignore",
    message="We strongly recommend saving the metadata using 'save_to_json' for replicability in future SDV versions."
)


from sdv.metadata import Metadata
from sdv.single_table import CTGANSynthesizer

def generate_synthetic_data(df):
    metadata = Metadata.detect_from_dataframe(
        data=df,
        table_name='my_table'
    )

    synthesizer = CTGANSynthesizer(metadata=metadata, epochs=100)
    synthesizer.fit(data=df)
    return synthesizer.sample(num_rows=len(df))

# Old function (deprecated)
# from sdv.single_table import CTGANSynthesizer
# from sdv.metadata import SingleTableMetadata

# def generate_synthetic_data(df):
#     df = df.copy()

#     if 'Index' in df.columns:
#         df = df.drop(columns=['Index'])
#     if 'Unnamed: 0' in df.columns:
#         df = df.drop(columns=['Unnamed: 0'])

#     metadata_obj = SingleTableMetadata()
#     metadata_obj.detect_from_dataframe(df)
#     synthesizer = CTGANSynthesizer(metadata_obj, epochs=100)
#     synthesizer.fit(df)
#     synthetic_data = synthesizer.sample(num_rows=len(df))

#     return synthetic_data