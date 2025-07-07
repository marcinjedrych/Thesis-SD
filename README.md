# Thesis: Handling Missing Data in Synthetic Data Generation

This repository contains the code and resources for my Master's thesis in Statistical Data Analysis. The project investigates strategies for handling missing data in the context of synthetic data generation and their impact on predictive model performance.

## Overview

The goal of this thesis is to evaluate the effectiveness of various missing data handling techniques: Complete Case Analysis (CCA), Multiple Imputation (MI), and the Indicator Method. This work aims to invistigate this particulary in the context of synthetic data generation. Synthetic datasets are created using CTGAN, and the performance of models trained on synthetic data is compared against those trained on original datasets.

## Repository Structure

### Main Scripts

**1. Simulation Missing Data.py**  
   - Simulates a dataset using predefined functions (`generate.py`)  
   - Splits data into training and test sets without missingness  
   - Introduces missing data under MCAR, MAR, and MNAR mechanisms (`missingness.py`)  
   - Applies missing data handling strategies: CCA, MI, and Indicator (`strategies.py`)  
   - Saves the generated datasets in `Data/Original/`

**2. Making Synthetic Data (CTGAN).py**  
   - Trains CTGAN models to generate synthetic datasets based on the incomplete training sets (`ctgan_syn.py`)  
   - Saves synthetic datasets in `Data/Synthetic/`

**3. Baseline model.py**  
   - Trains logistic regression models on complete original and synthetic datasets  
   - Compares model performance between synthetic and original data

**4. Missing data models.py**  
   - Trains logistic regression models on datasets with missing data (handled by CCA, MI, or Indicator)  
   - Evaluates model performance on both original and synthetic datasets

**5. Performance plots.py**  
   - Visualizes model performance metrics (e.g., AUC, accuracy)  
   - Compares results across different missingness types and handling strategies

**6. Use case.py**  
   - Applies the full pipeline on a real-world use case (if applicable)

### functions/ Folder

This folder contains modular helper scripts used across the pipeline:

- **ctgan_syn.py** – Functions to train CTGAN and generate synthetic datasets  
- **generate.py** – Functions to simulate the initial synthetic population  
- **missingness.py** – Implements MCAR, MAR, and MNAR missing data mechanisms  
- **strategies.py** – Missing data handling strategies (CCA, MI, Indicator Method)  
- **other.py** – Utility functions for formatting, evaluation, plotting

## Contact

For questions or collaboration, feel free to reach out via [LinkedIn](https://www.linkedin.com/in/marcin-j%C4%99drych-290435165/) or [email](marcin.jedrych@ugent.be).)