# -*- coding: utf-8 -*-
"""
Created on Fri Apr 11 15:49:58 2025

@author: Marcin
"""

import warnings
import matplotlib.pyplot as plt
import seaborn as sns
from functions.generate import generate_patient_data

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

    synthesizer = CTGANSynthesizer(metadata=metadata, epochs=400)
    synthesizer.fit(data=df)

    # Get and plot losses
    # loss_df = synthesizer.get_loss_values()
    # plt.figure(figsize=(8, 5))
    # sns.lineplot(data=loss_df[['Generator Loss', 'Discriminator Loss']])
    # plt.title("CTGAN Loss over Epochs")
    # plt.xlabel("Epoch")
    # plt.ylabel("Loss")
    # plt.grid(True)
    # plt.tight_layout()
    # plt.show()

    return synthesizer.sample(num_rows=len(df))


# data = generate_patient_data(nsamples=1000, seed = 123)
# syn = generate_synthetic_data(data)


    