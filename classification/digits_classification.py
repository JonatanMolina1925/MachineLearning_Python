"""
Handwritten Digits Classification
=================================

This example demonstrates handwritten digit classification using the
Digits dataset available in scikit-learn.

Two different machine learning models are compared:

1. Gaussian Naive Bayes
2. Support Vector Machine (Polynomial Kernel)

The performance of each classifier is evaluated using a confusion matrix.

Author: Jonatan Ali Medina Molina
"""

import matplotlib.pyplot as plt
from sklearn import svm
from sklearn.datasets import load_digits
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score


# ---------------------------------------------------------------------
# Load dataset
# ---------------------------------------------------------------------
# Load the handwritten digits dataset.
digits = load_digits()

# Labels (digit classes from 0 to 9).
y = digits.target

# Flatten each 8x8 image into a one-dimensional feature vector.
n_samples = len(digits.images)
X = digits.images.reshape(n_samples, -1)


# ---------------------------------------------------------------------
# (Optional) Display a sample image
# ---------------------------------------------------------------------
# plt.figure()
# plt.gray()
# plt.imshow(digits.images[0], cmap=plt.cm.gray_r)
# plt.title(f"Digit: {digits.target[0]}")
# plt.show()


# ---------------------------------------------------------------------
# Split dataset
# ---------------------------------------------------------------------
# Divide the dataset into training and testing subsets.
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    random_state=0
)


# ---------------------------------------------------------------------
# Gaussian Naive Bayes
# ---------------------------------------------------------------------
naive_bayes = GaussianNB()

naive_bayes.fit(X_train, y_train)

nb_predictions = naive_bayes.predict(X_test)

print("========================================")
print("Gaussian Naive Bayes")
print("========================================")
print(f"Accuracy: {accuracy_score(y_test, nb_predictions):.4f}")
print("\nConfusion Matrix:")
print(confusion_matrix(y_test, nb_predictions))


# ---------------------------------------------------------------------
# Support Vector Machine
# ---------------------------------------------------------------------
svm_classifier = svm.SVC(
    kernel="poly",
    C=2,
    gamma="auto",
    max_iter=100000000
)

svm_classifier.fit(X_train, y_train)

svm_predictions = svm_classifier.predict(X_test)

print("\n========================================")
print("Support Vector Machine")
print("========================================")
print(f"Accuracy: {accuracy_score(y_test, svm_predictions):.4f}")
print("\nConfusion Matrix:")
print(confusion_matrix(y_test, svm_predictions))


# ---------------------------------------------------------------------
# (Optional) Display predictions
# ---------------------------------------------------------------------
predicted_images = list(zip(
    digits.images,
    svm_classifier.predict(X)
))

plt.figure(figsize=(8, 8))

for index, (image, prediction) in enumerate(predicted_images[:9]):
    plt.subplot(3, 3, index + 1)
    plt.axis("off")
    plt.imshow(
        image,
        cmap=plt.cm.gray_r,
        interpolation="nearest"
    )
    plt.title(f"Prediction: {prediction}")

plt.tight_layout()
plt.show()