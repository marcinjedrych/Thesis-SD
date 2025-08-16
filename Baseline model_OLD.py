

import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, recall_score, precision_score, confusion_matrix
from functions.other import results_to_excel
from sklearn.calibration import calibration_curve
from sklearn.metrics import brier_score_loss, roc_auc_score, roc_curve

data = 'Data'

# Load data
no_missing = pd.read_excel(f"{data}/original/no_missing.xlsx")
no_missing = no_missing.rename(columns={'Unnamed: 0': 'Index'})

test_data = pd.read_excel(f"{data}/test_data.xlsx")
test_data = test_data.rename(columns={'Unnamed: 0': 'Index'})

# Lists to store all results and the subset for Excel
results4plot = []

def logistic_regression(train, test, target, label=None, exclude_vars=None):
    if exclude_vars is None:
        exclude_vars = []

    # Handle categorical variables
    train = pd.get_dummies(train, columns=['stage'])
    test = pd.get_dummies(test, columns=['stage'])

    # Align columns
    train, test = train.align(test, join='left', axis=1, fill_value=0)

    # Define predictors and target
    X_train = train.drop(columns=[target, 'Index'] + exclude_vars)
    X_test = test.drop(columns=[target, 'Index'] + exclude_vars)
    y_train = train[target]
    y_test = test[target]

    # Train logistic regression model
    model = LogisticRegression(max_iter=1000, random_state=123)
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]

    # Metrics
    acc = accuracy_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred, zero_division=0)
    precision = precision_score(y_test, y_pred, zero_division=0)
    auc = roc_auc_score(y_test, y_prob)
    brier = brier_score_loss(y_test, y_prob)

    result_row = {
        'Model': label,
        'Accuracy': acc,
        'Recall': recall,
        'Precision': precision,
        'AUC': auc,
        'Brier Score': brier
    }

    results4plot.append(result_row)
    return result_row

"""1.) BP vs. no BP plot """
# Run both with and without 'bp'
conditions = [('with bp', ['latent1', 'latent2']), ('without bp', ['latent1', 'latent2','bp'])]
synthetic_no_missing = pd.read_excel(f"{data}/Synthetic/synthetic_no_missing.xlsx")

for condition_name, exclude in conditions:
    # Run original data
    result_orig = logistic_regression(no_missing, test_data, 'hospitaldeath',
                                      label=f'Original ({condition_name})',
                                      exclude_vars=exclude)
    # Run synthetic data
    result_synth = logistic_regression(synthetic_no_missing, test_data, 'hospitaldeath',
                                        label=f'Synthetic ({condition_name})',
                                        exclude_vars=exclude)


# Create plot DataFrame from all results
results_df = pd.DataFrame(results4plot)

# Melt for plotting
melted = pd.melt(results_df, id_vars='Model',
                 value_vars=['Accuracy', 'AUC', 'Recall', 'Precision', 'Brier Score'],
                 var_name='Metric', value_name='Score')

# Extract metadata from Model string
melted['Data'] = melted['Model'].str.extract(r'^(Original|Synthetic)')
melted['Condition'] = melted['Model'].str.extract(r'\((.*?)\)')

# Plot
sns.set(style="whitegrid")
for data_type in ['Original', 'Synthetic']:
    plt.figure(figsize=(8, 5))
    subset = melted[melted['Data'] == data_type]
    sns.lineplot(data=subset, x='Condition', y='Score', hue='Metric', marker='o')
    plt.title(f'Model Performance ({data_type})')
    plt.ylabel("Score")
    plt.xlabel("Condition")
    plt.ylim(0, 1)
    plt.grid(True)
    plt.tight_layout()
    plt.show()


"""BASELINE RESULTS"""

results, results_syn = [], []
synthetic_no_missing = pd.read_excel(f"{data}/Synthetic/synthetic_no_missing.xlsx")
n_iter = 50
subset_size = 200
exclude = ['latent1', 'latent2']

for i in range(n_iter):
    
    train_sample = no_missing.sample(n=subset_size, random_state=i)
    train_sample2 = synthetic_no_missing.sample(n=subset_size, random_state=i)
    
    # Run original data 
    result_orig = logistic_regression(train_sample, test_data, 'hospitaldeath',
                                      exclude_vars=exclude)
    # Run synthetic data
    result_synth = logistic_regression(train_sample2, test_data, 'hospitaldeath',
                                        exclude_vars=exclude)
    
    # Store all
    results.append(result_orig)
    results_syn.append(result_synth)

selected_metrics = ['Model','Accuracy', 'Recall', 'Precision', 'AUC', 'Brier Score']

metrics_df = pd.DataFrame(results)
metrics_df_syn = pd.DataFrame(results_syn)

mean_metrics = metrics_df[selected_metrics].mean().to_frame().T
sd_metrics = metrics_df[selected_metrics].std().to_frame().T
mean_metrics_syn = metrics_df_syn[selected_metrics].mean().to_frame().T
sd_metrics_syn = metrics_df_syn[selected_metrics].std().to_frame().T


mean_metrics['Model'] = 'Original Baseline'
sd_metrics['Model'] = 'Original Baseline'
mean_metrics_syn['Model'] = 'Synthetic Baseline'
sd_metrics_syn['Model'] = 'Synthetic Baseline'

mean_metrics = mean_metrics.round(3)
sd_metrics = sd_metrics.round(3)
mean_metrics_syn = mean_metrics_syn.round(3)
sd_metrics_syn = sd_metrics_syn.round(3)


results_to_excel(pd.DataFrame(mean_metrics), output_file='metrics_mean.xlsx')
results_to_excel(pd.DataFrame(sd_metrics), output_file='metrics_SD.xlsx')
results_to_excel(pd.DataFrame(mean_metrics_syn), output_file='metrics_mean.xlsx')
results_to_excel(pd.DataFrame(sd_metrics_syn), output_file='metrics_SD.xlsx')