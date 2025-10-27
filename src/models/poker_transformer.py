import torch
import torch.nn as nn
import torch.nn.functional as F
from .corn_loss import corn_loss

class PokerTransformer(nn.Module):
    """Transformer for Poker Hand classification."""

    def __init__(self, suit_embedding_dim=10, rank_embedding_dim=10, num_classes=10,
                 nhead=2, nlayers=2, d_model=None, d_hid=128):
        super().__init__()
        if d_model is None:
            d_model = suit_embedding_dim + rank_embedding_dim

        self.suit_embedding = nn.Embedding(4, suit_embedding_dim)
        self.rank_embedding = nn.Embedding(13, rank_embedding_dim)

        # The input to the transformer will be the concatenated embeddings
        self.d_model = d_model
        self.pos_encoder = nn.Parameter(torch.zeros(1, 5, d_model)) # 5 cards in a hand

        encoder_layers = nn.TransformerEncoderLayer(d_model, nhead, d_hid, dropout=0.1, batch_first=True)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layers, nlayers)

        self.fc = nn.Linear(d_model * 5, num_classes - 1) # Flatten and classify
        self.num_classes = num_classes

    def forward(self, x):
        # x shape: (batch_size, 5, 2)
        suits = self.suit_embedding(x[:, :, 0])
        ranks = self.rank_embedding(x[:, :, 1])

        # Concatenate suit and rank embeddings for each card
        x = torch.cat([suits, ranks], dim=2)

        # Add positional encoding
        x = x + self.pos_encoder

        # Transformer encoder
        x = self.transformer_encoder(x)

        # Flatten and classify
        x = torch.flatten(x, 1)
        x = self.fc(x)
        return x

class PokerTransformerModel(nn.Module):
    """Standard PyTorch Module for PokerTransformer."""

    def __init__(self, suit_embedding_dim=10, rank_embedding_dim=10, num_classes=10,
                 nhead=2, nlayers=2, d_model=None, d_hid=128):
        super().__init__()
        if d_model is None:
            d_model = suit_embedding_dim + rank_embedding_dim
        self.transformer = PokerTransformer(suit_embedding_dim, rank_embedding_dim, num_classes,
                                            nhead, nlayers, d_model, d_hid)
        self.num_classes = num_classes

    def forward(self, x):
        return self.transformer(x)

    def loss_fn(self, y_hat, y):
        return corn_loss(y_hat, y, self.num_classes)
