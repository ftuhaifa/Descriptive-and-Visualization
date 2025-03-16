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





# Visualizing Outliers

# Boxplot for numerical features
plt.figure(figsize=(12, 6))
sns.boxplot(data=pd.DataFrame(X, columns=["Survival years", "Surgery status", "Age", "Race"]))
plt.title("Boxplot of Features to Detect Outliers")
plt.xticks(rotation=45)
plt.show()

# Distribution plots for each feature
fig, axes = plt.subplots(2, 2, figsize=(12, 8))
features = ["Survival years", "Surgery status", "Age", "Race"]

for i, ax in enumerate(axes.flat):
    sns.histplot(pd.DataFrame(X, columns=features)[features[i]], bins=30, kde=True, ax=ax)
    ax.set_title(f"Distribution of {features[i]}")

plt.tight_layout()
plt.show()
