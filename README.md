# Thesis-SD
This is a repository for my thesis in statistical data-analysis. I will investigate strategies to handle missing data in synthetic data generation

1. Simulation Missing Data.py
   Generates data set (func.generate)
   Creates test set and training set without missingness
   Creates training sets with MCAR, MAR, MNAR missingness (func.missingness)
   Applies CCA/MI/Learned NA strategies (func.strategies)
   Save datasets in Data/Original

2. Making Synthetic Data (CTGAN).py
   Makes synthetic versions of traingsets (func.ctgan)
   Save datasets in Data/Synthetic

3. Baseline model.py
   Logistic regression model to predict hospital death
   Synthetic vs. Original on complete datasets.
   
5. Simulation Missing Data.py
  Logistic regression model to predict hospital death
  Synthetic vs. Original on complete datasets.

6.
