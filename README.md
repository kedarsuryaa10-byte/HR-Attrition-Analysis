
This project provides a comprehensive analysis of employee attrition using HR data. It explores workforce demographics, attrition trends, job satisfaction, compensation, tenure, and department-wise turnover. The dashboard helps HR teams identify factors contributing to employee exits and develop effective retention strategies.
pip install pandas numpy matplotlib seaborn scikit-learn
# hr_retention_analysis.py

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix
)

# -----------------------------
# Load Dataset
# -----------------------------
df = pd.read_csv("HR_Employee_Attrition.csv")

print("Dataset Shape:", df.shape)
print(df.head())

# -----------------------------
# Data Cleaning
# -----------------------------
print("\nMissing Values:")
print(df.isnull().sum())

# Remove employee identifiers
drop_cols = ['EmployeeNumber', 'EmployeeCount', 'Over18', 'StandardHours']
df = df.drop(columns=[c for c in drop_cols if c in df.columns])

# -----------------------------
# Encode Categorical Variables
# -----------------------------
le = LabelEncoder()

for col in df.columns:
    if df[col].dtype == 'object':
        df[col] = le.fit_transform(df[col])

# -----------------------------
# Correlation Analysis
# -----------------------------
plt.figure(figsize=(12,8))
sns.heatmap(df.corr(), cmap="coolwarm")
plt.title("Correlation Matrix")
plt.show()

# -----------------------------
# Attrition Distribution
# -----------------------------
plt.figure(figsize=(6,4))
sns.countplot(x='Attrition', data=df)
plt.title("Employee Attrition")
plt.show()

# -----------------------------
# Feature Selection
# -----------------------------
X = df.drop('Attrition', axis=1)
y = df['Attrition']

# -----------------------------
# Train-Test Split
# -----------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

# -----------------------------
# Model Training
# -----------------------------
model = RandomForestClassifier(
    n_estimators=200,
    random_state=42
)

model.fit(X_train, y_train)

# -----------------------------
# Predictions
# -----------------------------
y_pred = model.predict(X_test)

# -----------------------------
# Evaluation
# -----------------------------
accuracy = accuracy_score(y_test, y_pred)

print("\nAccuracy:", round(accuracy * 100, 2), "%")

print("\nClassification Report:")
print(classification_report(y_test, y_pred))

# Confusion Matrix
cm = confusion_matrix(y_test, y_pred)

plt.figure(figsize=(6,4))
sns.heatmap(
    cm,
    annot=True,
    fmt='d',
    cmap='Blues'
)
plt.title("Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.show()

# -----------------------------
# Feature Importance
# -----------------------------
importance = pd.DataFrame({
    'Feature': X.columns,
    'Importance': model.feature_importances_
})

importance = importance.sort_values(
    by='Importance',
    ascending=False
)

print("\nTop Retention Factors:")
print(importance.head(10))

plt.figure(figsize=(10,6))
sns.barplot(
    data=importance.head(10),
    x='Importance',
    y='Feature'
)

plt.title("Top Factors Affecting Attrition")
plt.show()
