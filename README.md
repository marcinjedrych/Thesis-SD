# Thesis: Handling Missing Data in Synthetic Data Generation

This repository contains the code and resources for my Master's thesis in Statistical Data Analysis. The project investigates strategies for handling missing data in the context of synthetic data generation and their impact on predictive model performance.

## Overview

The goal of this thesis is to evaluate the effectiveness of various missing data handling techniques: Complete Case Analysis (CCA), Multiple Imputation (MI), and the Indicator Method. This work aims to invistigate this particulary in the context of synthetic data generation. Synthetic datasets are created using CTGAN, and the performance of models trained on synthetic data is compared against those trained on original datasets.

## Repository Structure

### Main Scripts

**1. Simulation.py** 

   - Generation of datasets (`generate.py`)   
   - Inducing of missing data under MCAR, MAR, and MNAR mechanisms (`missingness.py`)  
   - Application of missing data handling strategies: CCA, MI, and Indicator (`strategies.py`)
   - Training of CTGAN models to generate synthetic datasets (`ctgan_syn.py`) 
   - Training of logistic regression models for baseline, original missing, and synthetic missing data (`model.py`)
   - Evaluation of model performance and visualisation 

**2. Use case.py**  
   - Applies the full pipeline on a real-world use case.

### functions/ Folder

This folder contains modular helper scripts used across the pipeline:

- **ctgan_syn.py** – Functions to train CTGAN and generate synthetic datasets  
- **generate.py** – Functions to simulate the initial synthetic population  
- **missingness.py** – Implements MCAR, MAR, and MNAR missing data mechanisms  
- **strategies.py** – Missing data handling strategies (CCA, MI, Indicator Method)  
- **other.py** – Utility functions for formatting, evaluation, plotting

## Contact

For questions or collaboration, feel free to reach out via [LinkedIn](https://www.linkedin.com/in/marcin-j%C4%99drych-290435165/) or [email](marcin.jedrych@ugent.be).)
