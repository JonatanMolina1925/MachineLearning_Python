"""
Principal Component Analysis (PCA) for Wine Quality Classification
==================================================================

This example demonstrates how Principal Component Analysis (PCA) can be
used for dimensionality reduction before classification.

The Wine Quality dataset is standardized, transformed using PCA, and then
classified using a Gaussian Naive Bayes classifier. The classification
performance is evaluated for different numbers of principal components.

Author: Jonatan Ali Medina Molina
"""

import matplotlib.pyplot as plt
import pandas as pd

from sklearn import preprocessing
from sklearn.decomposition import PCA
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.naive_bayes import GaussianNB
from pathlib import Path

# ---------------------------------------------------------------------
# Load dataset
# ---------------------------------------------------------------------
# Build the path to the dataset.
data_path = Path(__file__).resolve().parent.parent / "data" / "winequality-red.csv"

# Load the dataset.
data = pd.read_csv(data_path, sep=";")

# ---------------------------------------------------------------------
# Select input features and target variable
# ---------------------------------------------------------------------
X = data[
    [
        "fixed acidity",
        "volatile acidity",
        "citric acid",
        "residual sugar",
        "chlorides",
        "free sulfur dioxide",
        "total sulfur dioxide",
        "density",
        "pH",
        "sulphates",
        "alcohol",
    ]
]

# Wine quality score (target variable).
y = data["quality"]


# ---------------------------------------------------------------------
# Feature standardization
# ---------------------------------------------------------------------
# Standardize all features before applying PCA.
X = preprocessing.StandardScaler().fit_transform(X)


# ---------------------------------------------------------------------
# Apply PCA using all components
# ---------------------------------------------------------------------
pca = PCA()

pca.fit(X)

Z = pca.transform(X)

# Principal component loading matrix.
loadings = pd.DataFrame(
    pca.components_,
    columns=[
        "fixed acidity",
        "volatile acidity",
        "citric acid",
        "residual sugar",
        "chlorides",
        "free sulfur dioxide",
        "total sulfur dioxide",
        "density",
        "pH",
        "sulphates",
        "alcohol",
    ],
)

# Uncomment to inspect component loadings.
# print(loadings)


# ---------------------------------------------------------------------
# Baseline classification without PCA
# ---------------------------------------------------------------------
naive_bayes = GaussianNB()

naive_bayes.fit(X, y)

baseline_predictions = naive_bayes.predict(X)

print("========================================")
print("Without PCA")
print("========================================")
print(f"Accuracy: {accuracy_score(y, baseline_predictions):.4f}")
print("\nConfusion Matrix:")
print(confusion_matrix(y, baseline_predictions))


# ---------------------------------------------------------------------
# Evaluate PCA with different numbers of components
# ---------------------------------------------------------------------
accuracies = []

print("\nAccuracy using PCA")

for n_components in range(1, 10):

    pca = PCA(n_components=n_components)

    X_pca = pca.fit_transform(X)

    naive_bayes.fit(X_pca, y)

    predictions = naive_bayes.predict(X_pca)

    accuracy = accuracy_score(y, predictions)

    accuracies.append(accuracy)

    print(
        f"Components: {n_components:2d} | "
        f"Accuracy: {accuracy:.4f}"
    )


# ---------------------------------------------------------------------
# Plot accuracy vs. number of principal components
# ---------------------------------------------------------------------
plt.figure(figsize=(8, 5))

plt.plot(
    range(1, 10),
    accuracies,
    marker="o"
)

plt.xlabel("Number of Principal Components")
plt.ylabel("Classification Accuracy")
plt.title("Effect of PCA on Classification Accuracy")

plt.grid(True)

plt.show()