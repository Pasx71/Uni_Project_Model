import pandas as pd
import numpy as np
from datasets import load_dataset
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import f1_score
from skopt import BayesSearchCV
from tqdm import tqdm

# Load the dataset
dataset = load_dataset("imodels/diabetes-readmission")
df = pd.DataFrame(dataset['train'])

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
X = df[medication_columns]
y = df['readmitted'].values

# Train-test split (adjust as needed)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Set up the model and the parameter search space for Bayesian optimization
search_space = {
    'n_estimators': (10, 1000),         # Range for n_estimators
    'max_depth': (5, 50),               # Range for max_depth
    'min_samples_split': (2, 20),       # Range for min_samples_split
    'max_features': [None, 'sqrt', 'log2'],  # Choices for max_features
    'min_samples_leaf': (1, 20),        # Range for min_samples_leaf
    'bootstrap': [True, False]          # Choices for bootstrap
}

# Define the Random Forest Classifier
rf = RandomForestClassifier(random_state=42)

# Perform Bayesian optimization using BayesSearchCV with a progress bar
opt = BayesSearchCV(rf, search_space, n_iter=50, cv=3, n_jobs=-1, verbose=1, scoring='f1')

# Loop with tqdm progress bar to simulate the optimization process
with tqdm(total=50) as pbar:
    for _ in range(50):  # simulate optimization steps
        opt.fit(X_train, y_train)
        pbar.update(1)  # Update the progress bar with each step

# Print the best score and best parameters found
print("Best F1-score: ", opt.best_score_)
print("Best hyperparameters: ", opt.best_params_)

# Evaluate the final model on the test set
y_pred = opt.predict(X_test)
f1 = f1_score(y_test, y_pred)

# Print classification report
print("Test F1 Score:", f1)
print(classification_report(y_test, y_pred))
