#!/usr/bin/env python

# Deep Learning Homework 1

import argparse

import numpy as np
import matplotlib.pyplot as plt

import utils


def softmax(x, axis=0, keepdims=True):
    max = np.max(x, axis=axis, keepdims=keepdims)
    e_x = np.exp(x - max)
    return e_x / e_x.sum(axis=axis, keepdims=keepdims)


class LinearModel(object):
    def __init__(self, n_classes, n_features, **kwargs):
        self.W = np.zeros((n_classes, n_features))

    def update_weight(self, x_i, y_i, **kwargs):
        raise NotImplementedError

    def train_epoch(self, X, y, **kwargs):
        for x_i, y_i in zip(X, y):
            self.update_weight(x_i, y_i, **kwargs)

    def predict(self, X):
        """X (n_examples x n_features)"""
        scores = np.dot(self.W, X.T)
        return scores.argmax(axis=0)

    def evaluate(self, X, y):
        """
        X (n_examples x n_features):
        y (n_examples): gold labels
        """
        y_hat = self.predict(X)
        n_correct = (y == y_hat).sum()
        n_possible = y.shape[0]
        return n_correct / n_possible


class Perceptron(LinearModel):
    def update_weight(self, x_i, y_i, **kwargs):
        """
        x_i (n_features): a single training example
        y_i (scalar): the gold label for that example
        other arguments are ignored
        """
        # Q1.1a
        scores = np.dot(self.W, x_i.T)
        y_hat = scores.argmax(axis=0)
        if y_hat != y_i:
            self.W[y_i] += x_i
            self.W[y_hat] -= x_i


class LogisticRegression(LinearModel):
    def update_weight(self, x_i, y_i, learning_rate=0.001):
        """
        x_i (n_features): a single training example
        y_i: the gold label for that example
        learning_rate (float): keep it at the default value for your plots
        """
        # Q1.1b
        y_hat = softmax(np.dot(self.W, x_i.T), axis=0).reshape((self.W.shape[0], 1))
        ey = np.zeros((self.W.shape[0], 1))
        ey[y_i] = 1
        x_i_T = x_i.reshape((1, x_i.shape[0]))
        grad = np.dot((y_hat - ey), x_i_T)
        self.W -= learning_rate * grad


class MLP(object):
    # Q3.2b. This MLP skeleton code allows the MLP to be used in place of the
    # linear models with no changes to the training loop or evaluation code
    # in main().
    def __init__(self, n_classes, n_features, hidden_size):
        # Initialize an MLP with a single hidden layer.
        # Use 200 hidden units, a relu activation function for the hidden layers, and a multinomial logistic loss (also
        # called cross-entropy) in the output layer (eventhough we are dealing with a binary
        # classification this will allow you to use the same code for a multi-class problem
        self.W1 = np.random.normal(loc=0.1, scale=0.1, size=(hidden_size, n_features))
        self.b1 = np.zeros((hidden_size, 1))
        self.W2 = np.random.normal(loc=0.1, scale=0.1, size=(n_classes, hidden_size))
        self.b2 = np.zeros((n_classes, 1))

    def predict(self, X):
        # Compute the forward pass of the network. At prediction time, there is
        # no need to save the values of hidden nodes, whereas this is required
        # at training time.
        z1 = np.dot(self.W1, X.T) + self.b1
        h1 = np.maximum(0, z1)
        z2 = np.dot(self.W2, h1) + self.b2
        h2 = softmax(z2, axis=0)
        return h2.argmax(axis=0)

    def evaluate(self, X, y):
        """
        X (n_examples x n_features)
        y (n_examples): gold labels
        """
        # Identical to LinearModel.evaluate()
        y_hat = self.predict(X)
        n_correct = (y == y_hat).sum()
        n_possible = y.shape[0]
        return n_correct / n_possible

    def train_epoch(self, X, y, learning_rate=0.001):
        """
        Dont forget to return the loss of the epoch.
        """
        for x_i, y_i in zip(X, y):
            y_i = y_i.reshape((1,))
            x_i = x_i.reshape((1, x_i.shape[0]))
            # Forward
            z1 = np.dot(self.W1, x_i.T) + self.b1
            h1 = np.maximum(0, z1)
            z2 = np.dot(self.W2, h1) + self.b2
            h2 = softmax(z2, axis=0)
            # Backward
            grad_z2 = h2 - np.eye(4)[y_i].T
            grad_W2 = np.dot(grad_z2, h1.T)
            grad_h1 = np.dot(self.W2.T, grad_z2)
            grad_z1 = grad_h1 * (z1 > 0)
            grad_W1 = np.dot(grad_z1, x_i)
            # Update weights
            self.W1 -= learning_rate * grad_W1
            self.W2 -= learning_rate * grad_W2
            self.b1 -= learning_rate * np.sum(grad_z1, axis=1, keepdims=True)
            self.b2 -= learning_rate * np.sum(grad_z2, axis=1, keepdims=True)
        # Forward
        z1 = np.dot(self.W1, X.T) + self.b1
        h1 = np.maximum(0, z1)
        z2 = np.dot(self.W2, h1) + self.b2
        h2 = softmax(z2, axis=0)
        return -np.sum(np.log(h2[y, np.arange(y.shape[0])]) / y.shape[0])


def plot(epochs, train_accs, val_accs):
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.plot(epochs, train_accs, label="train")
    plt.plot(epochs, val_accs, label="validation")
    plt.legend()
    plt.show()


def plot_loss(epochs, loss):
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.plot(epochs, loss, label="train")
    plt.legend()
    plt.show()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "model",
        choices=["perceptron", "logistic_regression", "mlp"],
        help="Which model should the script run?",
    )
    parser.add_argument(
        "-epochs",
        default=20,
        type=int,
        help="""Number of epochs to train for. You should not
                        need to change this value for your plots.""",
    )
    parser.add_argument(
        "-hidden_size",
        type=int,
        default=200,
        help="""Number of units in hidden layers (needed only
                        for MLP, not perceptron or logistic regression)""",
    )
    parser.add_argument(
        "-learning_rate",
        type=float,
        default=0.001,
        help="""Learning rate for parameter updates (needed for
                        logistic regression and MLP, but not perceptron)""",
    )
    opt = parser.parse_args()

    utils.configure_seed(seed=42)

    add_bias = opt.model != "mlp"
    data = utils.load_oct_data(bias=add_bias)
    train_X, train_y = data["train"]
    dev_X, dev_y = data["dev"]
    test_X, test_y = data["test"]
    n_classes = np.unique(train_y).size
    n_feats = train_X.shape[1]

    # initialize the model
    if opt.model == "perceptron":
        model = Perceptron(n_classes, n_feats)
    elif opt.model == "logistic_regression":
        model = LogisticRegression(n_classes, n_feats)
    else:
        model = MLP(n_classes, n_feats, opt.hidden_size)
    epochs = np.arange(1, opt.epochs + 1)
    train_loss = []
    valid_accs = []
    train_accs = []

    for i in epochs:
        print("Training epoch {}".format(i))
        train_order = np.random.permutation(train_X.shape[0])
        train_X = train_X[train_order]
        train_y = train_y[train_order]
        if opt.model == "mlp":
            loss = model.train_epoch(train_X, train_y, learning_rate=opt.learning_rate)
        else:
            model.train_epoch(train_X, train_y, learning_rate=opt.learning_rate)

        train_accs.append(model.evaluate(train_X, train_y))
        valid_accs.append(model.evaluate(dev_X, dev_y))
        if opt.model == "mlp":
            print(
                "loss: {:.4f} | train acc: {:.4f} | val acc: {:.4f}".format(
                    loss,
                    train_accs[-1],
                    valid_accs[-1],
                )
            )
            train_loss.append(loss)
        else:
            print(
                "train acc: {:.4f} | val acc: {:.4f}".format(
                    train_accs[-1],
                    valid_accs[-1],
                )
            )
    print("Final test acc: {:.4f}".format(model.evaluate(test_X, test_y)))

    # plot
    plot(epochs, train_accs, valid_accs)
    if opt.model == "mlp":
        plot_loss(epochs, train_loss)


if __name__ == "__main__":
    main()
