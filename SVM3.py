import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, f1_score
import pickle
from skopt import BayesSearchCV

# BayesSearchCV 

# Load the dataset
df = pd.read_csv('UNSW_NB15_training-set.csv')

# Drop the 'id' column
df = df.drop(columns=['id'])

# Define the features (X) and target (y)
X = df.drop(columns=['label'])
y = df['label']

# Identify categorical columns for one-hot encoding
categorical_cols = ['proto', 'service', 'state', 'attack_cat']

# Preprocessing: One-hot encode categorical columns
preprocessor = ColumnTransformer(
    transformers=[
        ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_cols)
    ],
    remainder='passthrough'  # Pass through the remaining columns without changes
)

# Define the SVM model
svm = SVC(kernel='rbf', random_state=42)

# Create a pipeline that preprocesses the data and then fits the SVM
pipeline = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('classifier', svm)
])

# Define the parameter space for BayesSearchCV
param_space = {
    'classifier__C': (1e-6, 1e+6, 'log-uniform'),
    'classifier__gamma': (1e-6, 1e+1, 'log-uniform')
}

# Create the BayesSearchCV object
opt = BayesSearchCV(
    estimator=pipeline,
    search_spaces=param_space,
    n_iter=32,
    cv=5,
    n_jobs=-1,
    verbose=2,
    random_state=42
)

# Split the dataset into the Training set and Test set
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Fit the BayesSearchCV to the training set
opt.fit(X_train, y_train)

# Get the best estimator
best_model = opt.best_estimator_

# Predict on the test set
y_pred = best_model.predict(X_test)

# Calculate accuracy and F1 score
accuracy = accuracy_score(y_test, y_pred)
f1_macro = f1_score(y_test, y_pred, average='macro')
print(f"Test set accuracy: {accuracy:.4f}")
print(f"Test set F1 Score (Macro): {f1_macro:.4f}")

# Save the best model to disk
filename = 'finalized_model_optimised.sav'
pickle.dump(best_model, open(filename, 'wb'))
