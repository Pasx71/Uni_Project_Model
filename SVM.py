import pandas as pd
import numpy as np
from datasets import load_dataset
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, f1_score
from skopt import BayesSearchCV
from tqdm import tqdm

# Load the dataset
dataset = load_dataset("imodels/diabetes-readmission")

# Access the training and test sets directly
df_train = pd.DataFrame(dataset['train'])
df_test = pd.DataFrame(dataset['test'])

# Select the medication-related columns
medication_columns = [
    'metformin:Down', 'metformin:No', 'metformin:Steady', 'metformin:Up',
    'repaglinide:Down', 'repaglinide:No', 'repaglinide:Steady', 'repaglinide:Up',
    'nateglinide:Down', 'nateglinide:No', 'nateglinide:Steady', 'nateglinide:Up',
    'chlorpropamide:Down', 'chlorpropamide:No', 'chlorpropamide:Steady', 'chlorpropamide:Up',
    'glimepiride:Down', 'glimepiride:No', 'glimepiride:Steady', 'glimepiride:Up',
    'glipizide:Down', 'glipizide:No', 'glipizide:Steady', 'glipizide:Up',
    'glyburide:Down', 'glyburide:No', 'glyburide:Steady', 'glyburide:Up',
    'pioglitazone:Down', 'pioglitazone:No', 'pioglitazone:Steady', 'pioglitazone:Up',
    'rosiglitazone:Down', 'rosiglitazone:No', 'rosiglitazone:Steady', 'rosiglitazone:Up',
    'acarbose:Down', 'acarbose:No', 'acarbose:Steady', 'acarbose:Up',
    'miglitol:Down', 'miglitol:No', 'miglitol:Steady', 'miglitol:Up',
    'tolazamide:No', 'tolazamide:Steady', 'tolazamide:Up',
    'insulin:Down', 'insulin:No', 'insulin:Steady', 'insulin:Up',
    'glyburide-metformin:Down', 'glyburide-metformin:No', 'glyburide-metformin:Steady', 'glyburide-metformin:Up'
]

# Extract features and target variable for training and test data
X_train = df_train[medication_columns]
y_train = df_train['readmitted'].values

X_test = df_test[medication_columns]
y_test = df_test['readmitted'].values

# Standardize the data for SVM
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Set up the parameter search space for Bayesian optimization
search_space = {
    'C': (0.01, 1000.0, 'log-uniform'),     # Regularization parameter
    'gamma': (0.0001, 1.0, 'log-uniform'),  # Kernel coefficient for 'rbf'
    'kernel': ['rbf', 'linear', 'poly'],    # Kernel options
    'degree': (2, 5)                        # Degree of polynomial kernel
}

# Define the SVM model
svm = SVC(random_state=42)

# Perform Bayesian optimization using BayesSearchCV with a progress bar
opt = BayesSearchCV(svm, search_space, n_iter=50, cv=3, n_jobs=-1, verbose=1, scoring='f1')

# Loop with tqdm progress bar to simulate the optimization process
with tqdm(total=50) as pbar:
    for _ in range(50):  # simulate optimization steps
        opt.fit(X_train_scaled, y_train)
        pbar.update(1)  # Update the progress bar with each step

# Print the best score and best parameters found
print("Best F1-score: ", opt.best_score_)
print("Best hyperparameters: ", opt.best_params_)

# Evaluate the final model on the test set
y_pred = opt.predict(X_test_scaled)
f1 = f1_score(y_test, y_pred)

# Print classification report
print("Test F1 Score:", f1)
print(classification_report(y_test, y_pred))
