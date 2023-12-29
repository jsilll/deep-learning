#!/usr/bin/env python

# Deep Learning Homework 2

import argparse

import torch
from torch.utils.data import DataLoader
import torch.nn as nn
from torch import optim
import torch.nn.functional as F
import torchvision
from matplotlib import pyplot as plt
import numpy as np

import utils

class CNN(nn.Module):
    def __init__(self, dropout_prob, no_maxpool=False):
        super(CNN, self).__init__()
        self.no_maxpool = no_maxpool

        # Hint: use nn.Conv2d, nn.MaxPool2d
        # and nn.Linear and nn.Dropout and nn.ReLU
        # and nn.Sequential

        if not no_maxpool:
            # A convolutional layer with:
            # - 8 output channels
            # - kernel size of 3x3
            # - stride of 1
            # - appropriate padding to preserve the original image size
            # - assign it to self.conv1
            #   - Given an N × N × D image, F × F × D filters, K channels, and stride S,
            #     the resulting output will be of size M × M × K , where
            #   - N is the input size - in your case 28
            #   - K is the Kernel size - in your case 3
            #   - P is the padding - in your case 0 i believe
            #   - S is the stride - which you have not provided.
            #   - (N * N − K + 2P) / S + 1
            #   - M = (N − K) / S + 1
            #   - Common padding size: (F − 1)/2, which preserves spatial size: M = N.
            self.conv1 = nn.Conv2d(1, 8, 3, stride=1, padding=1) # Correct
            # 28x28 -> 30x30 -> 3x3 stride 1 -> 28x28 
            # A ReLU activation layer
            self.relu1 = nn.ReLU()
            # A max-pooling layer with:
            # - kernel size of 2x2
            # - stride of 2
            self.maxpool1 = nn.MaxPool2d(kernel_size=2, stride=2)
            # 28x28 -> 2x2 stride 2 -> 14x14

            # A convolutional layer with:
            # - 16 output channels
            # - kernel size of 3x3
            # - stride of 1
            # - padding of 0??
            self.conv2 = nn.Conv2d(8, 16, 3, stride=1, padding=0)
            # 14x14 -> 3x3 stride 1 -> 12x12 
            # A ReLU activation layer
            self.relu2 = nn.ReLU()
            # A max-pooling layer with:
            # - kernel size of 2x2
            # - stride of 2
            self.maxpool2 = nn.MaxPool2d(kernel_size=2, stride=2)
            # 12x12 -> 2x2 stride 2 -> 6x6

            # An affine transformation with:
            # - 320 output features
            # - inp = out_channels * out_width * out_height
            self.fc1_input_features = 16*6*6
            self.fc1 = nn.Linear(self.fc1_input_features, 320)
            # A ReLU activation layer
            self.relu3 = nn.ReLU()

            # A dropout layer with probability dropout_prob
            self.drop = nn.Dropout(dropout_prob)

            # An affine transformation with: 
            # (also known as a fully-connected (FC) layer)
            # - 120 output features
            self.fc2 = nn.Linear(320, 120)

            # A ReLU activation layer
            self.relu4 = nn.ReLU()

            # An affine transformation with:
            # - #Classes output features
            self.fc3 = nn.Linear(120, 10)

            # LogSoftmax layer
            self.logsoftmax = nn.LogSoftmax(dim=1)
        else:
            self.conv1 = nn.Conv2d(1, 8, 3, stride=2, padding=1) # Correct
            self.relu1 = nn.ReLU()
            # 28x28 -> 3x3 stride 2 -> 14x14
            # 28 - 3 + 2 * 1 / 2 + 1 = 14
            self.conv2 = nn.Conv2d(8, 16, 3, stride=2, padding=0)
            self.relu2 = nn.ReLU()
            self.fc1_input_features = 16*6*6
            self.fc1 = nn.Linear(self.fc1_input_features, 320)
            # A ReLU activation layer
            self.relu3 = nn.ReLU()

            # A dropout layer with probability dropout_prob
            self.drop = nn.Dropout(dropout_prob)

            # An affine transformation with: 
            # (also known as a fully-connected (FC) layer)
            # - 120 output features
            self.fc2 = nn.Linear(320, 120)

            # A ReLU activation layer
            self.relu4 = nn.ReLU()

            # An affine transformation with:
            # - #Classes output features
            self.fc3 = nn.Linear(120, 10)

            # LogSoftmax layer
            self.logsoftmax = nn.LogSoftmax(dim=1)
        
        # Implementation for Q2.1 and Q2.2
        
    def forward(self, x):
        # input should be of shape [b, channels, width, height] but we have [b, width * height * channels]
        # so we need to reshape it to [b, channels, width, height]
        x = x.view(-1, 1, 28, 28)
        if not self.no_maxpool:
            x = self.maxpool1(self.relu1(self.conv1(x)))
            x = self.maxpool2(self.relu2(self.conv2(x)))
            x = x.view(-1, self.fc1_input_features)
            x = self.drop(self.relu3(self.fc1(x)))
            x = self.relu4(self.fc2(x))
            x = self.logsoftmax(self.fc3(x))

            return x

        
        # conv and relu layers
        

        # max-pool layer if using it
        if not self.no_maxpool:
            raise NotImplementedError
        
        # prep for fully connected layer + relu
        
        # drop out
        x = self.drop(x)

        # second fully connected layer + relu
        
        # last fully connected layer
        x = self.fc3(x)
        
        return F.log_softmax(x,dim=1)

def train_batch(X, y, model, optimizer, criterion, device, **kwargs):
    """
    X (n_examples x n_features)
    y (n_examples): gold labels
    model: a PyTorch defined model
    optimizer: optimizer used in gradient step
    criterion: loss function
    """
    optimizer.zero_grad()
    X, y = X.to(device), y.to(device)
    out = model(X, **kwargs)
    loss = criterion(out, y)
    loss.backward()
    optimizer.step()
    return loss.item()


def predict(model, X):
    """X (n_examples x n_features)"""
    scores = model(X)  # (n_examples x n_classes)
    predicted_labels = scores.argmax(dim=-1)  # (n_examples)
    return predicted_labels


def evaluate(model, X, y, device):
    """
    X (n_examples x n_features)
    y (n_examples): gold labels
    """
    model.eval()
    X, y = X.to(device), y.to(device)
    y_hat = predict(model, X)
    n_correct = (y == y_hat).sum().item()
    n_possible = float(y.shape[0])
    model.train()
    return n_correct / n_possible


def plot(epochs, plottable, ylabel='', name=''):
    plt.clf()
    plt.xlabel('Epoch')
    plt.ylabel(ylabel)
    plt.plot(epochs, plottable)
    plt.savefig('%s.png' % (name), bbox_inches='tight')


def get_number_trainable_params(model):
    ## TO IMPLEMENT - REPLACE return 0
    return 0


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-epochs', default=20, type=int,
                        help="""Number of epochs to train for. You should not
                        need to change this value for your plots.""")
    parser.add_argument('-batch_size', default=8, type=int,
                        help="Size of training batch.")
    parser.add_argument('-learning_rate', type=float, default=0.01,
                        help="""Learning rate for parameter updates""")
    parser.add_argument('-l2_decay', type=float, default=0)
    parser.add_argument('-dropout', type=float, default=0.7)
    parser.add_argument('-optimizer',
                        choices=['sgd', 'adam'], default='sgd')
    parser.add_argument('-no_maxpool', action='store_true')
    
    opt = parser.parse_args()

    utils.configure_seed(seed=42)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    data = utils.load_oct_data()
    dataset = utils.ClassificationDataset(data)
    train_dataloader = DataLoader(
        dataset, batch_size=opt.batch_size, shuffle=True)
    dev_X, dev_y = dataset.dev_X, dataset.dev_y
    test_X, test_y = dataset.test_X, dataset.test_y

    # initialize the model
    model = CNN(opt.dropout, no_maxpool=opt.no_maxpool).to(device)
    
    # get an optimizer
    optims = {"adam": torch.optim.Adam, "sgd": torch.optim.SGD}

    optim_cls = optims[opt.optimizer]
    optimizer = optim_cls(
        model.parameters(), lr=opt.learning_rate, weight_decay=opt.l2_decay
    )
    
    # get a loss criterion
    criterion = nn.NLLLoss()
    
    # training loop
    epochs = np.arange(1, opt.epochs + 1)
    train_mean_losses = []
    valid_accs = []
    train_losses = []
    for ii in epochs:
        print('Training epoch {}'.format(ii))
        for X_batch, y_batch in train_dataloader:
            loss = train_batch(
                X_batch, y_batch, model, optimizer, criterion, device)
            train_losses.append(loss)

        mean_loss = torch.tensor(train_losses).mean().item()
        print('Training loss: %.4f' % (mean_loss))

        train_mean_losses.append(mean_loss)
        valid_accs.append(evaluate(model, dev_X, dev_y, device))
        print('Valid acc: %.4f' % (valid_accs[-1]))

    print('Final Test acc: %.4f' % (evaluate(model, test_X, test_y, device)))
    # plot
    config = "{}-{}-{}-{}-{}".format(opt.learning_rate, opt.dropout, opt.l2_decay, opt.optimizer, opt.no_maxpool)

    plot(epochs, train_mean_losses, ylabel='Loss', name='CNN-training-loss-{}'.format(config))
    plot(epochs, valid_accs, ylabel='Accuracy', name='CNN-validation-accuracy-{}'.format(config))
    
    print('Number of trainable parameters: ', get_number_trainable_params(model))

if __name__ == '__main__':
    main()
