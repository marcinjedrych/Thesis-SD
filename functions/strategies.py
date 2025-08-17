# -*- coding: utf-8 -*-
"""
Created on Tue Apr  8 17:50:18 2025

@author: Marcin
"""

from sklearn.experimental import enable_iterative_imputer
import pandas as pd
from sklearn.impute import IterativeImputer
from sklearn.preprocessing import OrdinalEncoder
import numpy as np
from sklearn.preprocessing import StandardScaler

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
        
        imputer = IterativeImputer(max_iter=10, sample_posterior=True, random_state=random_state + i)
        imputed_array = imputer.fit_transform(df_copy)
        imputed_df = pd.DataFrame(imputed_array, columns=df.columns)

        # decode back to original categories
        if encoder:
            try:
                imputed_df[categorical_cols] = encoder.inverse_transform(imputed_df[categorical_cols])
            except Exception as e:
                print(f"Decoder error: {e}")
        
        imputed_datasets.append(imputed_df)

    return imputed_datasets

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, recall_score, precision_score, roc_auc_score, brier_score_loss

result_row =[]

def ensemble(imputed_datasets, test_data, target, threshold=0.5, label='Ensemble'):
    """
    Trains a logistic regression model on each imputed dataset,
    makes predictions on the test data, aggregates predictions and returns performance metrics.
    """

    predictions = []

    # Identify categorical columns
    categorical_cols = test_data.select_dtypes(include=["object", "category"]).columns.tolist()
    encoder = None
    
    if categorical_cols:
        # Identify categorical columns
        categorical_cols = test_data.select_dtypes(include=["object", "category"]).columns.tolist()
        if categorical_cols:
            from sklearn.preprocessing import OrdinalEncoder
            encoder = OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1)

            # Sample data for encoder fitting to avoid huge concat
            sampled_frames = []
            for df in imputed_datasets:
                sampled_frames.append(df[categorical_cols].astype(str))
            sampled_frames.append(test_data[categorical_cols].astype(str))
            combined_data = pd.concat(sampled_frames, ignore_index=True)

            encoder.fit(combined_data[categorical_cols])

    for df in imputed_datasets:
        X_train = df.drop(columns=[target])
        y_train = df[target]
        X_test = test_data.drop(columns=[target])

        if encoder:
            # Convert categorical columns to string before transform
            X_train[categorical_cols] = X_train[categorical_cols].astype(str)
            X_test[categorical_cols] = X_test[categorical_cols].astype(str)

            X_train[categorical_cols] = encoder.transform(X_train[categorical_cols])
            X_test[categorical_cols] = encoder.transform(X_test[categorical_cols])

        # Ensure test has same columns as train
        X_test = X_test.reindex(columns=X_train.columns, fill_value=0)

        # ===== Add scaling here =====
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_test = scaler.transform(X_test)
        # ============================

        model = LogisticRegression(max_iter=1000, random_state=123)
        model.fit(X_train, y_train)

        preds = model.predict_proba(X_test)[:, 1]
        predictions.append(preds)

    predictions_array = np.array(predictions)
    soft_preds = predictions_array.mean(axis=0)
    hard_preds = (soft_preds >= threshold).astype(int)

    # Calculate metrics
    y_true = test_data[target].values
    acc = accuracy_score(y_true, hard_preds)
    recall = recall_score(y_true, hard_preds, zero_division=0)
    precision = precision_score(y_true, hard_preds, zero_division=0)
    auc = roc_auc_score(y_true, soft_preds)
    brier = brier_score_loss(y_true, soft_preds)

    result_row.append({
        #'Model': label,
        'Accuracy': acc,
        'Recall': recall,
        'Precision': precision,
        'AUC': auc,
        'Brier Score': brier
    })

    return hard_preds, soft_preds, result_row


def complete_cases(df):
    
    """from paper multiple imputation & stefvanbuuren.name: 
        - When only outcome variable has missingness this approach is valid under MCAR, MAR. (best choice)
        - Also when missing data probability does not depend on the outcome variable Y (even under MNAR)
        - Or when Logistic Regression with missing data only in Y or X (but not both)"""

    return df.dropna()

# needed for learned NA, (training data cannot contain missingness with logistic regressio)
def missing_indicator(df, column):

    df = df.copy()
    indicator_col = f"{column}_missing"
    
    # Create missing indicator
    df[indicator_col] = df[column].isna().astype(int)
    
    # Impute missing values with 0
    df[column] = df[column].fillna(0)
    
    return df