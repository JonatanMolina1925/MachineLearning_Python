"""
Perceptron from Scratch
=======================

This example implements the Perceptron learning algorithm from scratch
using only NumPy.

The script demonstrates:
1. Weight initialization.
2. Perceptron learning rule.
3. Training over multiple epochs.
4. Binary classification of new samples.

Author: Jonatan Ali Medina Molina
"""

import numpy as np


class Perceptron:
    """
    Simple implementation of the Perceptron algorithm.
    """

    def __init__(
        self,
        X,
        y,
        threshold=0.5,
        learning_rate=0.1,
        max_epochs=10,
    ):
        """
        Initialize the perceptron.

        Parameters
        ----------
        X : array-like
            Training samples.
        y : array-like
            Target labels.
        threshold : float
            Decision threshold.
        learning_rate : float
            Weight update step.
        max_epochs : int
            Maximum number of training epochs.
        """
        self.X = X
        self.y = y
        self.threshold = threshold
        self.learning_rate = learning_rate
        self.max_epochs = max_epochs

    # -----------------------------------------------------------------
    # Weight initialization
    # -----------------------------------------------------------------
    def initialize(self, init_type="zeros"):
        """
        Initialize the model weights.

        Parameters
        ----------
        init_type : str
            'zeros' or 'random'.
        """

        if init_type == "random":
            self.weights = np.random.rand(len(self.X[0])) * 0.05

        elif init_type == "zeros":
            self.weights = np.zeros(len(self.X[0]))

        else:
            raise ValueError(
                "init_type must be 'zeros' or 'random'."
            )

    # -----------------------------------------------------------------
    # Training
    # -----------------------------------------------------------------
    def train(self):
        """
        Train the perceptron using the Perceptron learning rule.
        """

        epoch = 0

        while True:

            error_count = 0
            epoch += 1

            for sample, target in zip(self.X, self.y):
                error_count += self.train_observation(
                    sample,
                    target
                )

            print(
                f"Epoch {epoch:2d} | Errors: {error_count}"
            )

            if error_count == 0:
                print("\nTraining completed successfully.")
                break

            if epoch >= self.max_epochs:
                print(
                    "\nMaximum number of epochs reached."
                )
                break

    # -----------------------------------------------------------------
    # Update weights for one observation
    # -----------------------------------------------------------------
    def train_observation(self, sample, target):
        """
        Update the weights using one training sample.
        """

        prediction = (
            np.dot(sample, self.weights)
            > self.threshold
        )

        error = target - prediction

        if error != 0:

            for index, value in enumerate(sample):
                self.weights[index] += (
                    self.learning_rate
                    * error
                    * value
                )

            return 1

        return 0

    # -----------------------------------------------------------------
    # Prediction
    # -----------------------------------------------------------------
    def predict(self, sample):
        """
        Predict the class of a new sample.
        """

        return int(
            np.dot(sample, self.weights)
            > self.threshold
        )


# ---------------------------------------------------------------------
# Training dataset
# ---------------------------------------------------------------------
X = [
    (1, 0, 0),
    (1, 1, 0),
    (1, 1, 1),
    (1, 1, 1),
    (1, 0, 1),
    (1, 0, 1),
]

y = [1, 1, 0, 0, 1, 1]


# ---------------------------------------------------------------------
# Create and train the perceptron
# ---------------------------------------------------------------------
perceptron = Perceptron(X, y)

perceptron.initialize(init_type="zeros")

perceptron.train()


# ---------------------------------------------------------------------
# Test predictions
# ---------------------------------------------------------------------
print("\nPredictions")
print("---------------------")
print("(1, 1, 1) ->", perceptron.predict((1, 1, 1)))
print("(1, 0, 1) ->", perceptron.predict((1, 0, 1)))
print("\nLearned weights:")
print(perceptron.weights)