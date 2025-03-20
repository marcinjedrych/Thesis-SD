# -*- coding: utf-8 -*-
"""
Created on Thu Mar 20 15:12:58 2025

Code to predict bloodpressure and compare
missingness in outcome variable (bloodpressure) vs. missingness in predictor variable (weight)

@author: Marcin
"""
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
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

def train_and_evaluate(df, target, label, strategy = "CCA"):
    
    # handling missingness
    if strategy == "CCA":
        df = complete_cases(df)
    else:
        df = multiple_imputation(df)
    
    # Handle categorical variables
    df = pd.get_dummies(df, columns=['stage'])
    
    # Separate predictors (X) and outcome (y)
    X = df.drop(columns=[target])  # Use all available predictors
    y = df[target]

    # Split data into training and test sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Train the regression model
    model = LinearRegression()
    model.fit(X_train, y_train)

    # Make predictions
    y_pred = model.predict(X_test)

    # Evaluate performance
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    #print(f"Performance for {label}:")
    #print(f"  - Mean Squared Error: {mse:.2f}")
    #print(f"  - R² Score: {r2:.2f}\n")

    return model, mse, r2

# Load data into a dictionary
datasets = {
    "MCAR": (pd.read_excel("data/weight_mcar.xlsx"), pd.read_excel("data/bp_mcar.xlsx")),
    "MAR": (pd.read_excel("data/weight_mar.xlsx"), pd.read_excel("data/bp_mar.xlsx")),
    "MNAR": (pd.read_excel("data/weight_mnar.xlsx"), pd.read_excel("data/bp_mnar.xlsx"))
}

# Train models and store results (CCA)
results = {}
print('\n ---- Complete case analysis (CCA) ---- \n')

for label, (missing_weight, missing_bp) in datasets.items():
    model_w, mse_w, r2_w = train_and_evaluate(missing_weight, 'bp', label)
    model_bp, mse_bp, r2_bp = train_and_evaluate(missing_bp, 'bp', label)
    results[label] = {"MSE m_weight": mse_w, "R² Score m_weight": r2_w, "MSE m_Bp": mse_bp, "R² Score m_Bp" : r2_bp }

comparison_cca = pd.DataFrame.from_dict(results, orient="index").reset_index().rename(columns={"index": "Missingness Type"})

print(comparison_cca)


# Train models and store results (MI)
results = {}
print('\n ---- Multiple Imputation (MI) ---- \n')

for label, (missing_weight, missing_bp) in datasets.items():
    model_w, mse_w, r2_w = train_and_evaluate(missing_weight, 'bp', label, strategy="MI")
    model_bp, mse_bp, r2_bp = train_and_evaluate(missing_bp, 'bp', label, strategy = "MI")
    results[label] = {"MSE m_weight": mse_w, "R² Score m_weight": r2_w, "MSE m_Bp": mse_bp, "R² Score m_Bp" : r2_bp }

comparison_mi = pd.DataFrame.from_dict(results, orient="index").reset_index().rename(columns={"index": "Missingness Type"})

print(comparison_mi)

