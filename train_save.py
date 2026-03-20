import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from xgboost import XGBClassifier
from imblearn.over_sampling import SMOTE
import joblib
import os

# Load data
df = pd.read_csv("Loan_default.csv")
print("Dataset loaded:", df.shape)

# Preprocessing (matching loandefault.ipynb)
print("Filling NaNs...")
df.fillna(df.mean(numeric_only=True), inplace=True)

le = LabelEncoder()
categorical_columns = ['Education','EmploymentType','MaritalStatus','LoanPurpose','HasMortgage','HasDependents','HasCoSigner']
for col in categorical_columns:
    if col in df.columns:
        df[col] = le.fit_transform(df[col].astype(str))

df = df.drop('LoanID', axis=1)

X = df.drop("Default", axis=1)
y = df["Default"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

sm = SMOTE(random_state=42)
X_train, y_train = sm.fit_resample(X_train, y_train)

# Train model
print("Training XGBoost...")
model = XGBClassifier(
    n_estimators=800,
    learning_rate=0.02,
    max_depth=6,
    subsample=0.9,
    colsample_bytree=0.9,
    gamma=0.1,
    scale_pos_weight=4,
    random_state=42
)
model.fit(X_train, y_train)

# Test
y_pred = model.predict(X_test)
accuracy = (y_pred == y_test).mean()
print(f"Test Accuracy: {accuracy:.4f}")

# Save
os.makedirs('models', exist_ok=True)
joblib.dump(model, 'models/model.joblib')
joblib.dump(scaler, 'models/scaler.joblib')
joblib.dump(X.columns.tolist(), 'models/features.joblib')
print("Saved model.joblib, scaler.joblib, features.joblib in /models/")

