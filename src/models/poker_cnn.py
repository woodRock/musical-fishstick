
import torch
import torch.nn as nn
import torch.nn.functional as F
from .corn_loss import corn_loss
from .coral_head import coral_predict # Import coral_predict for prediction logic

class PokerCNN(nn.Module):
    """1D CNN for Poker Hand classification."""

    def __init__(self, suit_embedding_dim=10, rank_embedding_dim=10, num_classes=10):
        super().__init__()
        self.suit_embedding = nn.Embedding(4, suit_embedding_dim)
        self.rank_embedding = nn.Embedding(13, rank_embedding_dim)

        # The input to the CNN will be the concatenated embeddings
        in_channels = suit_embedding_dim + rank_embedding_dim

        self.conv1 = nn.Conv1d(in_channels, 128, kernel_size=2, padding=1)
        self.conv2 = nn.Conv1d(128, 128, kernel_size=2, padding=1)
        self.pool = nn.MaxPool1d(2)

        # The output size of the conv layers needs to be calculated
        # Input sequence length is 5
        # After conv1 (kernel=2, padding=1): length becomes 5 + 2*1 - 2 + 1 = 6
        # After pool: 6 / 2 = 3
        # After conv2 (kernel=2, padding=1): 3 + 2*1 - 2 + 1 = 4
        # After pool: 4 / 2 = 2
        self.fc1 = nn.Linear(128 * 2, 128)
        self.fc2 = nn.Linear(128, num_classes - 1) # Corrected: num_classes - 1 logits for CORN loss
        self.num_classes = num_classes # Store num_classes for loss_fn

    def forward(self, x):
        # x shape: (batch_size, 5, 2)
        suits = self.suit_embedding(x[:, :, 0])
        ranks = self.rank_embedding(x[:, :, 1])

        # Concatenate suit and rank embeddings for each card
        x = torch.cat([suits, ranks], dim=2)

        # Reshape for Conv1d: (batch_size, channels, sequence_length)
        x = x.permute(0, 2, 1)

        x = F.relu(self.conv1(x))
        x = self.pool(x)
        x = F.relu(self.conv2(x))
        x = self.pool(x)

        x = torch.flatten(x, 1)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x

class PokerCNNModel(nn.Module):
    """Standard PyTorch Module for PokerCNN, compatible with dl_trainer.py manual loops."""

    def __init__(self, suit_embedding_dim=10, rank_embedding_dim=10, num_classes=10):
        super().__init__()
        self.cnn = PokerCNN(suit_embedding_dim, rank_embedding_dim, num_classes)
        self.num_classes = num_classes

    def forward(self, x):
        return self.cnn(x)

    def loss_fn(self, y_hat, y):
        return corn_loss(y_hat, y, self.num_classes)
