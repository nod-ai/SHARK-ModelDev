# Copyright 2023 Nod Labs, Inc
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

import logging
import unittest

import torch
from torch import nn
import torch.optim as optim
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
import torchvision.datasets as datasets


from testutils import *

class MNISTDataLoader:
    def __init__(self, batch_size, shuffle=True):
        self.batch_size = batch_size
        self.shuffle = shuffle

        transform = transforms.Compose(
            [transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))]
        )

        self.mnist_trainset = datasets.MNIST(
            root="../data", train=True, download=True, transform=transform
        )
        self.mnist_testset = datasets.MNIST(
            root="../data", train=False, download=True, transform=transform
        )

    def get_train_loader(self):
        return DataLoader(
            dataset=self.mnist_trainset,
            batch_size=self.batch_size,
            shuffle=self.shuffle,
        )

    def get_test_loader(self):
        return DataLoader(
            dataset=self.mnist_testset, batch_size=self.batch_size, shuffle=False
        )


class LinearModel(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(LinearModel, self).__init__()
        self.linear = nn.Linear(input_dim, output_dim)

    def forward(self, x):
        x = x.view(x.size(0), -1)
        out = self.linear(x)
        return out


# Training
def training_iteration(model, images, labels, optimizer, criterion):
    model.train()
    optimizer.zero_grad()
    outputs = model(images)
    loss = criterion(outputs, labels)
    loss.backward()
    optimizer.step()
    return loss


def test_iteration(model, images, labels, criterion):
    model.eval()
    with torch.inference_mode():
        outputs = model(images)
        loss = criterion(outputs, labels)
        predictions = torch.argmax(outputs, dim=1)
        num_correct = int((predictions == labels).sum())
        accuracy = 100 * num_correct / labels.size(0)

    return accuracy, loss
    # return outputs
def model_forward():
    model = LinearModel(28*28, 10)
    opt = torch.compile(model, backend="turbine_cpu")
    opt(torch.randn(5, 1, 28, 28))

def train():
    # Example Parameters
    config = {
        "batch_size": 64,
        "learning_rate": 0.001,
        "num_epochs": 10,
    }

    custom_data_loader = MNISTDataLoader(config["batch_size"])
    train_loader = custom_data_loader.get_train_loader()

    model = LinearModel(28*28, 10)
    optimizer = optim.SGD(model.parameters(), lr=config["learning_rate"])
    criterion = nn.CrossEntropyLoss()

    train_opt = torch.compile(training_iteration, backend="turbine_cpu")
    for i, (images, labels) in enumerate(train_loader):
        train_opt(model, images, labels, optimizer, criterion)


def infer():
    # Example Parameters
    config = {
        "batch_size": 64,
        "learning_rate": 0.001,
        "num_epochs": 10,
    }

    custom_data_loader = MNISTDataLoader(config["batch_size"])
    test_loader = custom_data_loader.get_test_loader()

    model = LinearModel(28*28, 10)
    criterion = nn.CrossEntropyLoss()

    test_opt = torch.compile(test_iteration, backend="turbine_cpu")
    # test_opt = torch.compile(test_iteration, backend=create_backend())

    for i, (images, labels) in enumerate(test_loader):
        test_opt(model, images, labels, criterion)


class ModelTests(unittest.TestCase):
    # @unittest.expectedFailure
    def testMNIST(self):
        # train()
        infer()
        # model_forward()


if __name__ == "__main__":
    logging.basicConfig(level=logging.DEBUG)
    unittest.main()
