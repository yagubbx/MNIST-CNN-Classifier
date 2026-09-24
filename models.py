"""Randomly initialized networks; no downloaded weights."""
from torch import nn


class CNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 16, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(16, 32, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2))
        self.head = nn.Sequential(nn.Flatten(), nn.Linear(32*7*7, 128),
                                  nn.ReLU(), nn.Linear(128, 10))

    def forward(self, x):
        return self.head(self.features(x))


class MLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = nn.Sequential(nn.Flatten(), nn.Linear(784, 256), nn.ReLU(),
                                    nn.Linear(256, 64), nn.ReLU(), nn.Linear(64, 10))

    def forward(self, x):
        return self.layers(x)


MODELS = {'cnn': CNN, 'mlp': MLP}
