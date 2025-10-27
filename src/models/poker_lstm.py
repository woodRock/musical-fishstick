
import torch
import torch.nn as nn
import torch.nn.functional as F
from .corn_loss import corn_loss
from .coral_head import coral_predict # Import coral_predict for prediction logic

class PokerLSTM(nn.Module):
    """LSTM for Poker Hand classification."""

    def __init__(self, suit_embedding_dim=10, rank_embedding_dim=10, hidden_size=128, num_layers=1, num_classes=10):
        super().__init__()
        self.suit_embedding = nn.Embedding(4, suit_embedding_dim)
        self.rank_embedding = nn.Embedding(13, rank_embedding_dim)

        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.num_classes = num_classes

        # The input to the LSTM will be the concatenated embeddings
        input_size = suit_embedding_dim + rank_embedding_dim

        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)

        self.fc1 = nn.Linear(hidden_size, 128)
        self.fc2 = nn.Linear(128, num_classes - 1) # num_classes - 1 logits for CORN loss

    def forward(self, x):
        # x shape: (batch_size, 5, 2)
        suits = self.suit_embedding(x[:, :, 0])
        ranks = self.rank_embedding(x[:, :, 1])

        # Concatenate suit and rank embeddings for each card
        x = torch.cat([suits, ranks], dim=2)

        # LSTM expects input of shape (batch_size, seq_len, input_size)
        lstm_out, _ = self.lstm(x)

        # Use the output of the last time step for classification
        # lstm_out shape: (batch_size, seq_len, hidden_size)
        last_output = lstm_out[:, -1, :]

        x = F.relu(self.fc1(last_output))
        x = self.fc2(x)
        return x

class PokerLSTMModel(nn.Module):
    """Standard PyTorch Module for PokerLSTM, compatible with dl_trainer.py manual loops."""

    def __init__(self, suit_embedding_dim=10, rank_embedding_dim=10, hidden_size=128, num_layers=1, num_classes=10):
        super().__init__()
        self.lstm = PokerLSTM(suit_embedding_dim, rank_embedding_dim, hidden_size, num_layers, num_classes)
        self.num_classes = num_classes

    def forward(self, x):
        return self.lstm(x)

    def loss_fn(self, y_hat, y):
        return corn_loss(y_hat, y, self.num_classes)
