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


def multiple_imputation_old(df, n_imputations=5):
    
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

def MI_impute(df, n_imputations=10, random_state=123):
    df_copy = df.copy()
    categorical_cols = df_copy.select_dtypes(include=["object", "category"]).columns.tolist()

    # Encode categorical columns if needed
    if categorical_cols:
        encoder = OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1)
        df_copy[categorical_cols] = encoder.fit_transform(df_copy[categorical_cols])
    else:
        encoder = None

    imputed_datasets = []
    for i in range(n_imputations):
        # New random state per imputation for variation
        imputer = IterativeImputer(max_iter=10, sample_posterior=True, random_state=random_state + i)
        imputed_array = imputer.fit_transform(df_copy)
        imputed_df = pd.DataFrame(imputed_array, columns=df.columns)

        # Optional: decode back to original categories
        if encoder:
            try:
                imputed_df[categorical_cols] = encoder.inverse_transform(imputed_df[categorical_cols])
            except Exception as e:
                print(f"Decoder error: {e}")
        
        imputed_datasets.append(imputed_df)

    return imputed_datasets

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, recall_score, precision_score, roc_auc_score, brier_score_loss

results =[]
def ensemble(imputed_datasets, test_data, target, threshold=0.5, label='Ensemble'):
    """
    Trains a logistic regression model on each imputed dataset,
    makes predictions on the test data, returns predictions and performance metrics.

    Parameters:
    - imputed_datasets: list of imputed training datasets (each a pd.DataFrame)
    - test_data: pd.DataFrame with missing values already handled
    - target_col: str, name of the target column
    - threshold: float, threshold to convert probabilities to class labels
    - label: str, label for the model in the results

    Returns:
    - hard_preds: np.array of 0s and 1s (final binary predictions)
    - soft_preds: np.array of averaged predicted probabilities
    - results: list containing a dict of evaluation metrics
    """
    predictions = []

    # Identify categorical columns
    categorical_cols = test_data.select_dtypes(include=["object", "category"]).columns.tolist()
    encoder = None

    if categorical_cols:
        encoder = OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1)
        combined_data = pd.concat([df.drop(columns=[target]) for df in imputed_datasets] + 
                                  [test_data.drop(columns=[target])])
        encoder.fit(combined_data[categorical_cols])

    for df in imputed_datasets:
        X_train = df.drop(columns=[target])
        y_train = df[target]
        X_test = test_data.drop(columns=[target])

        if encoder:
            X_train[categorical_cols] = encoder.transform(X_train[categorical_cols])
            X_test[categorical_cols] = encoder.transform(X_test[categorical_cols])

        model = LogisticRegression(max_iter=1000)
        model.fit(X_train, y_train)

        preds = model.predict_proba(X_test)[:, 1]
        predictions.append(preds)

    predictions_array = np.array(predictions)
    soft_preds = predictions_array.mean(axis=0)
    hard_preds = (soft_preds >= threshold).astype(int)

    # Calculate metrics
    y_true = test_data[target].values
    acc = accuracy_score(y_true, hard_preds)
    recall = recall_score(y_true, hard_preds)
    precision = precision_score(y_true, hard_preds)
    auc = roc_auc_score(y_true, soft_preds)
    brier = brier_score_loss(y_true, soft_preds)

    results.append({
        'Model': label,
        'Accuracy': acc,
        'Recall': recall,
        'Precision': precision,
        'AUC': auc,
        'Brier Score': brier
    })

    return hard_preds, soft_preds, results




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