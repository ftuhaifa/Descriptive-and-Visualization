# -*- coding: utf-8 -*-
# -*- coding: utf-8 -*-

import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import BaggingClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import (
    accuracy_score,
    roc_auc_score,
    roc_curve,
    confusion_matrix,
    classification_report,
    f1_score,
    precision_score,
    recall_score,
)
import matplotlib.pyplot as plt
import seaborn as sns

# Load data
data = pd.read_csv("LungCancer25.csv")

# Remove rows with Surgery status equal to 3
data = data[data["Surgery status"] != 3]

# Encode the target variable into binary values
data["Sex_Encoded"] = pd.get_dummies(data["Sex"], drop_first=True)

# Select the relevant columns for prediction
X = data[["Survival years", "Surgery status", "Age", "Race"]]
y = data["Sex_Encoded"]

# Normalize features using min-max normalization
scaler = MinMaxScaler()
X = scaler.fit_transform(X)
from scipy.stats import zscore

from scipy.stats import zscore

# Convert X back to a DataFrame for easier processing
X_df = pd.DataFrame(X, columns=["Survival years", "Surgery status", "Age", "Race"])

# Reset the index of y to match X_df before filtering
y = y.reset_index(drop=True)

# Compute Z-scores for all numerical features
z_scores = np.abs(zscore(X_df))

# Set threshold (3 is common for Z-score outlier detection)
threshold = 3

# Identify rows where any feature has a Z-score > threshold
outlier_mask = (z_scores > threshold).any(axis=1)

# Print number of outliers
print(f"Number of detected outliers: {outlier_mask.sum()}")

# Remove outliers and reset index to align correctly
X_cleaned = X_df[~outlier_mask].reset_index(drop=True)
y_cleaned = y[~outlier_mask].reset_index(drop=True)

# Display shape before and after outlier removal
print(f"Original dataset size: {X_df.shape[0]}")
print(f"Dataset size after outlier removal: {X_cleaned.shape[0]}")

# Boxplot after outlier removal
plt.figure(figsize=(12, 6))
sns.boxplot(data=X_cleaned)
plt.title("Boxplot After Removing Outliers (Z-score method)")
plt.xticks(rotation=45)
plt.show()
