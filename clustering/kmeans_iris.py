"""
K-Means Clustering on the Iris Dataset
======================================

This example demonstrates how to use the K-Means clustering algorithm
provided by scikit-learn on the classic Iris dataset.

The script:
1. Loads the Iris dataset.
2. Performs K-Means clustering with three clusters.
3. Assigns a cluster label to each sample.
4. Compares the obtained clusters with the original class labels.
5. Displays the clustered samples in a 3D scatter plot.

Author: Jonatan Ali Medina Molina
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn import cluster
from sklearn import datasets


# ---------------------------------------------------------------------
# Load dataset
# ---------------------------------------------------------------------
# Load the Iris dataset included in scikit-learn.
data = datasets.load_iris()

# Convert the feature matrix into a DataFrame for easier manipulation.
X = pd.DataFrame(
    data.data,
    columns=data.feature_names
)

print("First five samples:")
print(X.head())


# ---------------------------------------------------------------------
# K-Means model
# ---------------------------------------------------------------------
# Create a K-Means model with three clusters.
model = cluster.KMeans(
    n_clusters=3,
    random_state=5
)

# Train the model.
results = model.fit(X)

# Predict the cluster assigned to each sample.
X["cluster"] = results.predict(X)

# Store the true class labels for comparison.
X["target"] = data.target

# Additional column used only to demonstrate DataFrame operations.
X["example"] = "LookatmeImimportant"

print("\nCluster assignments:")
print(X.head())


# ---------------------------------------------------------------------
# Compare clusters against true labels
# ---------------------------------------------------------------------
classification_result = (
    X[["cluster", "target", "example"]]
    .groupby(["cluster", "target"])
    .count()
)

print("\nCluster vs Target:")
print(classification_result)


# ---------------------------------------------------------------------
# Prepare data for visualization
# ---------------------------------------------------------------------
plot_data = X[
    [
        "sepal length (cm)",
        "sepal width (cm)",
        "petal length (cm)",
        "cluster",
    ]
].to_numpy()


# ---------------------------------------------------------------------
# 3D Visualization
# ---------------------------------------------------------------------
fig = plt.figure(figsize=(8, 6))
ax = fig.add_subplot(projection="3d")

for sample in plot_data:

    if sample[3] == 0:
        color = "red"
        marker = "x"

    elif sample[3] == 1:
        color = "blue"
        marker = "o"

    else:
        color = "green"
        marker = "+"

    ax.scatter(
        sample[0],
        sample[1],
        sample[2],
        color=color,
        marker=marker,
    )

ax.set_xlabel("Sepal Length (cm)")
ax.set_ylabel("Sepal Width (cm)")
ax.set_zlabel("Petal Length (cm)")
ax.set_title("K-Means Clustering on the Iris Dataset")

plt.show()