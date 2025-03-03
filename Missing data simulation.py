# -*- coding: utf-8 -*-
"""
Created on Sun Mar  2 14:32:57 2025

Missingness scenarios

@author: Marcin
"""
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

def generate_patient_data(n=1000, seed=42):
    np.random.seed(seed)
    age = np.random.normal(50, 15, n).astype(int)  # Age in years
    blood_pressure = np.random.normal(120, 15, n)  # Systolic blood pressure
    cholesterol = np.random.normal(200, 30, n)  # Cholesterol level
    weight = np.random.normal(70, 10, n)  # Weight in kg
    
    df = pd.DataFrame({
        'Age': age,
        'BloodPressure': blood_pressure,
        'Cholesterol': cholesterol,
        'Weight': weight
    })
    return df

def mcar(df, missing_rate=0.1, seed=42):
    np.random.seed(seed)
    df_mcar = df.copy()
    mask = np.random.rand(*df.shape) < missing_rate
    df_mcar[mask] = np.nan
    return df_mcar

def mar(df, base_missing_rate=0.05, max_missing_rate=0.3, noise_level=0.05, seed=42):
    np.random.seed(seed)
    df_mar = df.copy()
    
    # Normalize age between 0 and 1
    min_age, max_age = df['Age'].min(), df['Age'].max()
    normalized_age = (df['Age'] - min_age) / (max_age - min_age)
    
    # Define missing probability with added noise
    missing_prob = base_missing_rate + (max_missing_rate - base_missing_rate) * normalized_age
    missing_prob += np.random.normal(0, noise_level, len(df))  # Adding Gaussian noise
    missing_prob = np.clip(missing_prob, 0, 1)  # Ensure probabilities stay within [0,1]
    
    # Apply missingness
    mask = np.random.rand(len(df)) < missing_prob
    df_mar.loc[mask, 'Cholesterol'] = np.nan
    
    return df_mar

def mnar(df, base_missing_rate=0.05, max_missing_rate=0.3, noise_level=0.05, seed=42):
    np.random.seed(seed)
    df_mnar = df.copy()
    
    # Normalize blood pressure between 0 and 1
    min_bp, max_bp = df['BloodPressure'].min(), df['BloodPressure'].max()
    normalized_bp = (df['BloodPressure'] - min_bp) / (max_bp - min_bp)
    
    # Define missing probability as a smooth function of blood pressure
    missing_prob = base_missing_rate + (max_missing_rate - base_missing_rate) * normalized_bp
    missing_prob += np.random.normal(0, noise_level, len(df))  # Adding Gaussian noise
    missing_prob = np.clip(missing_prob, 0, 1)  # Ensure probabilities stay within [0,1]
    
    # Apply missingness based on the calculated probabilities
    mask = np.random.rand(len(df)) < missing_prob
    df_mnar.loc[mask, 'BloodPressure'] = np.nan
    
    return missing_prob, df_mnar


def plot_missingness(df, title):
    plt.figure(figsize=(8, 6))
    sns.heatmap(df.isnull(), cmap='viridis', cbar=False, yticklabels=False)
    plt.title(title)
    plt.show()

# Generate patient data
data = generate_patient_data()

# Introduce different missingness mechanisms
data_mcar = mcar(data)
data_mar = mar(data)
probs, data_mnar = mnar(data)


# Plot missingness patterns
plot_missingness(data_mcar, "MCAR Missingness Pattern")
plot_missingness(data_mar, "MAR Missingness Pattern")
plot_missingness(data_mnar, "MNAR Missingness Pattern")


def plot_mar(df, base_missing_rate=0.05, max_missing_rate=0.3, noise_level=0.05):
    min_age, max_age = df['Age'].min(), df['Age'].max()
    ages = np.linspace(min_age, max_age, 100)
    normalized_age = (ages - min_age) / (max_age - min_age)
    missing_prob = base_missing_rate + (max_missing_rate - base_missing_rate) * normalized_age
    
    plt.figure(figsize=(8, 5))
    plt.plot(ages, missing_prob, label="Expected Missingness Probability", color='b')

    # Overlay actual missing data points
    missing_ages = df.loc[df['Cholesterol'].isnull(), 'Age']
    missing_probs = base_missing_rate + (max_missing_rate - base_missing_rate) * \
                    ((missing_ages - min_age) / (max_age - min_age))
    missing_probs += np.random.normal(0, noise_level, len(missing_probs))  # Add noise
    plt.scatter(missing_ages, missing_probs, color='red', alpha=0.5, label="Observed Missingness")
    
    plt.xlabel("Age")
    plt.ylabel("Probability of Missing Cholesterol")
    plt.title("Missingness Probability vs Age")
    plt.legend()
    plt.grid(True)
    plt.show()


plot_mar(data_mar)


def plot_mnar(df, probs, base_missing_rate=0.05, max_missing_rate=0.3, noise_level=0.05):
    
    plt.figure(figsize=(8, 5))

    # Scatter plot of the observed data
    plt.scatter(df['BloodPressure'], probs, color='red', alpha=0.5, label="Observed Missingness")
    
    # Fit a polynomial curve to the data (e.g., degree 2 polynomial)
    poly_degree = 2
    coeffs = np.polyfit(df['BloodPressure'], probs, poly_degree)
    poly = np.poly1d(coeffs)
    
    # Generate a smooth curve based on the fitted polynomial
    blood_pressure_range = np.linspace(df['BloodPressure'].min(), df['BloodPressure'].max(), 500)
    prob_range = poly(blood_pressure_range)
    
    # Plot the smooth curve
    plt.plot(blood_pressure_range, prob_range, color='blue', label="Smoothed Curve", linewidth=2)
    
    # Labels and title
    plt.xlabel("Blood Pressure (original data)")
    plt.ylabel("Probability of Missing Blood Pressure (MNAR)")
    plt.title("MNAR scenario")
    plt.legend()
    plt.grid(True)
    
    info_text = "This plot shows the bloodpressure values from the original data (without missingness) against the probaility of missingness in the MNAR case.\n This demonstrates that the missing pattern in data_mnar for bloodpressure is dependent on the value of bloodpressure itself."
    plt.figtext(0.5, -0.05, info_text, ha="center", va="top", fontsize=10, wrap=True)

    plt.show()


'''needs original data (before missingness'''
plot_mnar(data, probs)