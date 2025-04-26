# -*- coding: utf-8 -*-
"""
Created on Sat Apr 26 17:02:39 2025

@author: Marcin
"""

import pandas as pd
from sklearn.ensemble import StackingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import accuracy_score

# --- 1. Load your data
no_missing = pd.read_excel("data/original/no_missing.xlsx")
no_missing = no_missing.rename(columns={'Unnamed: 0': 'Index'})

test_data = pd.read_excel("data/test_data.xlsx")
test_data = test_data.rename(columns={'Unnamed: 0': 'Index'})

# --- 2. Separate predictors and target
target = 'hospitaldeath'

X_train = no_missing.drop(columns=[target])
y_train = no_missing[target]

X_test = test_data.drop(columns=[target])
y_test = test_data[target]

# --- 3. Identify categorical and numerical columns
categorical_cols = X_train.select_dtypes(include=['object', 'category']).columns.tolist()
numerical_cols = X_train.select_dtypes(include=['number']).columns.tolist()

# --- 4. Preprocessing
preprocessor = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), numerical_cols),
        ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_cols)
    ]
)

# --- 5. Define base learners
base_learners = [
    ('decision_tree', DecisionTreeClassifier(max_depth=5, random_state=42)),
    ('random_forest', RandomForestClassifier(n_estimators=100, random_state=42)),
    ('svc', SVC(probability=True, random_state=42))
]

# --- 6. Define meta-learner
meta_learner = LogisticRegression(max_iter=1000, random_state=42)

# --- 7. Create the SuperLearner pipeline
model = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('stacking', StackingClassifier(
        estimators=base_learners,
        final_estimator=meta_learner,
        cv=5
    ))
])

# --- 8. Fit model
model.fit(X_train, y_train)

# --- 9. Predict and evaluate
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"SuperLearner Accuracy: {accuracy:.4f}")

