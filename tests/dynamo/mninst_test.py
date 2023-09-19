# Copyright 2023 Nod Labs, Inc
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

import logging

import math
import unittest
from dataclasses import dataclass
from typing import Any, Optional, Tuple

import torch
import torch.nn.functional as F
from torch import nn
import torch.optim as optim

import torchvision.transforms as transforms
import torchvision.datasets as datasets
from torch.utils.data import DataLoader


# MNIST Data Loader
class MNISTDataLoader:
    def __init__(self, batch_size, shuffle=True):
        self.batch_size = batch_size
        self.shuffle = shuffle

        # Data Transformations
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5,), (0.5,))
        ])

        # Download MNIST dataset
        self.mnist_trainset = datasets.MNIST(root='../data', train=True, download=True, transform=transform)
        self.mnist_testset = datasets.MNIST(root='../data', train=False, download=True, transform=transform)

    def get_train_loader(self):
        return DataLoader(
            dataset=self.mnist_trainset,
            batch_size=self.batch_size,
            shuffle=self.shuffle
        )

    def get_test_loader(self):
        return DataLoader(
            dataset=self.mnist_testset,
            batch_size=self.batch_size,
            shuffle=False
        )


# Simple CNN Model
class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=5)
        self.relu = nn.ReLU()
        self.maxpool = nn.MaxPool2d(kernel_size=2)
        self.fc1 = nn.Linear(32 * 12 * 12, 10)

    def forward(self, x):
        x = self.conv1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc1(x)
        return x

# Training
def train(model, images, labels, optimizer, criterion):
    model.train()

    total_loss = 0.0
    num_correct = 0.0

    optimizer.zero_grad()
    # images, labels = images.to(device), labels.to(device)
    outputs = model(images)
    loss = criterion(outputs, labels)

    num_correct += int((torch.argmax(outputs, dim=1) == labels).sum())
    total_loss += float(loss.item())

    loss.backward()
    optimizer.step()
    total_loss += loss.item()

    # print('Training finished.')
    # acc = 100 * num_correct / (config['batch_size'] * len(train_loader))
    # total_loss = float(total_loss / len(train_loader))

    # # Save the trained model
    # torch.save(model.state_dict(), 'mnist_cnn.pth')
    # return acc, total_loss

# TODO Implement inference func
"""
def test(model, images, labels, criterion):
    model.eval()
    num_correct = 0.0
    total_loss = 0.0
    with torch.no_grad():

        # images, labels = images.to(device), labels.to(device)
        with torch.inference_mode():
            outputs = model(images)
            loss = criterion(outputs, labels)

        num_correct += int((torch.argmax(outputs, dim=1) == labels).sum())
        total_loss += float(loss.item())

    # acc = 100 * num_correct / (config['batch_size'] * len(test_loader))
    # total_loss = float(total_loss / len(test_loader))
    # return acc, total_loss
"""

def main():
    # Example Hyperparameters
    config = {
        'batch_size': 64,
        'learning_rate': 0.001,
        # 'threshold' : 0.001,
        # 'factor' : 0.1,
        'num_epochs': 10,
    }

    # Data Loader
    custom_data_loader = MNISTDataLoader(config['batch_size'])
    train_loader = custom_data_loader.get_train_loader()
    # test_loader = MNISTDataLoader.get_test_loader()

    # Model, optimizer, loss
    model = CNN()
    optimizer = optim.Adam(model.parameters(), lr=config['learning_rate'])
    criterion = nn.CrossEntropyLoss()

    # Training
    train_opt = torch.compile(train, backend="turbine_cpu")
    for i, (images, labels) in enumerate(train_loader):
        train_opt(model, images, labels, optimizer, criterion)


    # TODO: Inference
    """
    test_opt = torch.compile(test, backend="turbine_cpu", mode="reduce-overhead")
    for i, (images, labels) in enumerate(test_loader):    
        test(model, images, labels, criterion)
    """



class ModelTests(unittest.TestCase):
    @unittest.expectedFailure
    def testMNIST(self):
        # TODO: Fix the below error
        """
        failed to legalize operation 'arith.sitofp' that was explicitly marked illegal
        """
        main()


if __name__ == "__main__":
    logging.basicConfig(level=logging.DEBUG)
    unittest.main()
