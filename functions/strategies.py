# -*- coding: utf-8 -*-
"""
Created on Tue Apr  8 17:50:18 2025

@author: Marcin
"""
import pandas as pd
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
from sklearn.preprocessing import OrdinalEncoder
import numpy as np


from autoimpute.imputations import MultipleImputer


def multiple_imputation(df, n_imputations=5):
    
    """ (from stefvanbuuren.name)
        - When covariates have missingness & MAR: estimated statistics and regression coefficients biased with complete case analysis
        - When covariates have missingness & MCAR: reduction in sample size will still reduce precision of estimated coefficients in CCA"""
        
    df_copy = df.copy()

    # Identify categorical columns
    categorical_cols = df_copy.select_dtypes(include=["object", "category"]).columns.tolist()

    # Encode categorical columns (if any)
    if categorical_cols:
        encoder = OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1)
        df_copy[categorical_cols] = encoder.fit_transform(df_copy[categorical_cols])
    else:
        encoder = None  # No categorical columns to encode

    # Initialize imputer
    imputer = IterativeImputer(max_iter=10, sample_posterior=True, random_state=None)

    # Perform multiple imputations
    imputed_datasets = []
    for _ in range(n_imputations):
        imputed_array = imputer.fit_transform(df_copy)
        imputed_df = pd.DataFrame(imputed_array, columns=df.columns)
        imputed_datasets.append(imputed_df)

    # Combine results
    combined_imputed_df = pd.concat(imputed_datasets).groupby(level=0).mean()

    # Handle categorical columns
    if categorical_cols:
        for i, col in enumerate(categorical_cols):
            try:
                # Get mode value across imputations
                mode_series = pd.concat(imputed_datasets)[col].groupby(level=0).agg(lambda x: x.mode().iloc[0] if not x.mode().empty else np.nan)

                # Round and convert to int
                mode_series = mode_series.round().astype("Int64")

                # Clamp to valid index
                n_categories = len(encoder.categories_[i])
                mode_series = mode_series.clip(0, n_categories - 1)

                combined_imputed_df[col] = mode_series

            except Exception as e:
                print(f"Warning: failed to process column '{col}': {e}")
                combined_imputed_df[col] = np.nan

        # Decode back to original categories
        try:
            combined_imputed_df[categorical_cols] = encoder.inverse_transform(combined_imputed_df[categorical_cols])
        except Exception as e:
            print(f"Decoder error: {e}")
            # fallback: leave as integer codes

    return combined_imputed_df


# import numpy as np
# from sklearn.linear_model import LinearRegression
# from sklearn.metrics import mean_squared_error
# from sklearn.model_selection import train_test_split

# def multiple_imputation(df, target_col, n_imputations=10):
#     df_copy = df.copy()
    
#     # Identify categorical columns
#     categorical_cols = df_copy.select_dtypes(include=["object", "category"]).columns.tolist()
#     if target_col in categorical_cols:
#         categorical_cols.remove(target_col)
    
#     # Encode categorical columns
#     encoder = OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1)
#     df_copy[categorical_cols] = encoder.fit_transform(df_copy[categorical_cols])
    
#     # Split into train/test before imputation to avoid data leakage
#     train_df, test_df = train_test_split(df_copy, test_size=0.2, random_state=42)

#     # Store MSEs from each imputation
#     imputed_mse_list = []

#     for _ in range(n_imputations):
#         # Initialize and fit imputer on training data
#         imputer = IterativeImputer(max_iter=10, sample_posterior=True, random_state=None)
#         train_imputed = imputer.fit_transform(train_df)
#         test_imputed = imputer.transform(test_df)

#         # Convert back to DataFrame
#         train_imputed_df = pd.DataFrame(train_imputed, columns=train_df.columns)
#         test_imputed_df = pd.DataFrame(test_imputed, columns=test_df.columns)

#         # Fit model
#         X_train = train_imputed_df.drop(columns=[target_col])
#         y_train = train_imputed_df[target_col]
#         X_test = test_imputed_df.drop(columns=[target_col])
#         y_test = test_imputed_df[target_col]

#         model = LinearRegression()
#         model.fit(X_train, y_train)
#         predictions = model.predict(X_test)

#         # Calculate performance metric
#         mse = mean_squared_error(y_test, predictions)
#         imputed_mse_list.append(mse)

#     # Apply Rubin's Rules (simplified for performance metrics like MSE)
#     Q_bar = np.mean(imputed_mse_list)  # average MSE (within-imputation)
#     B = np.var(imputed_mse_list, ddof=1)  # between-imputation variance
#     T = Q_bar + (1 + 1/n_imputations) * B  # total variance
    
#     return {
#         "MSE_mean": Q_bar,
#         "MSE_total_variance": T,
#         "MSE_std_error": np.sqrt(T),
#         "all_mse": imputed_mse_list
#     }


def complete_cases(df):
    
    """from paper multiple imputation & stefvanbuuren.name: 
        - When only outcome variable has missingness this approach is valid under MCAR, MAR. (best choice)
        - Also when missing data probability does not depend on the outcome variable Y (even under MNAR)
        - Or when Logistic Regression with missing data only in Y or X (but not both)"""

    return df.dropna()

# needed for learned NA but not before training (NOT before making data synthetic!)
def missing_indicator(df, column):

    df = df.copy()
    indicator_col = f"{column}_missing"
    
    # Create missing indicator
    df[indicator_col] = df[column].isna().astype(int)
    
    # Impute missing values with 0
    df[column] = df[column].fillna(0)
    
    return df