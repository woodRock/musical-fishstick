import torch.nn as nn
import torch.nn.functional as F

class MLPBase(nn.Module):
    """Base MLP feature extractor."""
    def __init__(self, input_size, hidden_size=100):
        super(MLPBase, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.output_size = hidden_size

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return x

class LinearBase(nn.Module):
    """Base linear feature extractor."""
    def __init__(self, input_size):
        super(LinearBase, self).__init__()
        self.fc1 = nn.Linear(input_size, 1)
        self.output_size = 1

    def forward(self, x):
        x = self.fc1(x)
        return x