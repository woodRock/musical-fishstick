"""Module containing base deep learning models for feature extraction."""

import torch.nn as nn
import torch.nn.functional as F

class MLPBase(nn.Module):
    """Base MLP feature extractor."""
    def __init__(self, input_size, hidden_size=100):
        """Initializes the MLPBase.

        Args:
            input_size (int): The number of input features.
            hidden_size (int, optional): The size of the hidden layers. Defaults to 100.
        """
        super(MLPBase, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.output_size = hidden_size

    def forward(self, x):
        """Performs a forward pass through the MLPBase.

        Args:
            x (torch.Tensor): The input tensor.

        Returns:
            torch.Tensor: The output features.
        """
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return x

class LinearBase(nn.Module):
    """Base linear feature extractor."""
    def __init__(self, input_size):
        """Initializes the LinearBase.

        Args:
            input_size (int): The number of input features.
        """
        super(LinearBase, self).__init__()
        self.fc1 = nn.Linear(input_size, 1)
        self.output_size = 1

    def forward(self, x):
        """Performs a forward pass through the LinearBase.

        Args:
            x (torch.Tensor): The input tensor.

        Returns:
            torch.Tensor: The output feature.
        """
        x = self.fc1(x)
        return x