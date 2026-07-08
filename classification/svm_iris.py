"""
Support Vector Machine (SVM) Classification on the Iris Dataset
===============================================================

This example demonstrates three different approaches for binary
classification using Support Vector Machines (SVM) with scikit-learn.

Models included:
1. Linear SVM
2. Linear SVM with polynomial feature expansion
3. SVM with a polynomial kernel

The objective is to classify whether an Iris flower belongs to the
Iris-Virginica class using petal length and petal width as features.

Author: Jonatan Ali Medina Molina
"""

import numpy as np
from sklearn import datasets
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import PolynomialFeatures
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC, SVC


# ---------------------------------------------------------------------
# Load dataset
# ---------------------------------------------------------------------
# Load the Iris dataset.
iris = datasets.load_iris()

# Select only petal length and petal width.
X = iris["data"][:, (2, 3)]

# Binary classification:
# 1 -> Iris Virginica
# 0 -> Any other species
y = (iris["target"] == 2).astype(np.float64)


# ---------------------------------------------------------------------
# Linear SVM
# ---------------------------------------------------------------------
linear_svm = Pipeline([
    ("scaler", StandardScaler()),
    ("classifier", LinearSVC(
        C=1,
        loss="hinge",
        dual="auto"
    ))
])

linear_svm.fit(X, y)

prediction = linear_svm.predict([[5.5, 1.7]])

print("Linear SVM prediction:")
print(prediction)


# ---------------------------------------------------------------------
# Linear SVM with Polynomial Features
# ---------------------------------------------------------------------
polynomial_svm = Pipeline([
    ("poly_features", PolynomialFeatures(degree=3)),
    ("scaler", StandardScaler()),
    ("classifier", LinearSVC(
        C=10,
        loss="hinge",
        dual="auto"
    ))
])

polynomial_svm.fit(X, y)

prediction = polynomial_svm.predict([[5.5, 1.7]])

print("\nPolynomial Features + Linear SVM prediction:")
print(prediction)


# ---------------------------------------------------------------------
# Polynomial Kernel SVM
# ---------------------------------------------------------------------
kernel_svm = Pipeline([
    ("scaler", StandardScaler()),
    ("classifier", SVC(
        kernel="poly",
        degree=3,
        C=5,
        coef0=1
    ))
])

kernel_svm.fit(X, y)

prediction = kernel_svm.predict([[5.5, 1.7]])

print("\nPolynomial Kernel SVM prediction:")
print(prediction)