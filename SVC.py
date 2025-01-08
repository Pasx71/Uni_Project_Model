import pandas as pd
import numpy as np
from datasets import load_dataset
from sklearn.svm import LinearSVC
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

# Extract features and target variable
X_train = df_train[medication_columns]
y_train = df_train['readmitted'].values

X_test = df_test[medication_columns]
y_test = df_test['readmitted'].values

# Standardize the data
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Set up Bayesian search space
search_space = {
    'C': (0.01, 1000.0, 'log-uniform'),  # Regularization parameter
    'loss': ['squared_hinge'],           # Only squared_hinge supports dual=False
    'penalty': ['l2'],                   # Default penalty (compatible with LinearSVC)
    'dual': [False]                      # Set to False for better performance
}

# Define the LinearSVC model
model = LinearSVC(max_iter=10000, random_state=42)

# Perform Bayesian optimization
opt = BayesSearchCV(model, search_space, n_iter=30, cv=3, n_jobs=-1, verbose=1, scoring='f1')

# Fit the model with progress bar
with tqdm(total=30) as pbar:
    for _ in range(30):
        opt.fit(X_train_scaled, y_train)
        pbar.update(1)

# Best parameters and score
print("Best F1-score: ", opt.best_score_)
print("Best hyperparameters: ", opt.best_params_)

# Evaluate the final model on the test set
y_pred = opt.predict(X_test_scaled)
f1 = f1_score(y_test, y_pred)

# Print classification report
print("Test F1 Score:", f1)
print(classification_report(y_test, y_pred))
