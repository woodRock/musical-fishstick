
import torch
from torch.utils.data import Dataset
import pandas as pd

class PokerHandDataset(Dataset):
    """PyTorch Dataset for the Poker Hand dataset."""

    def __init__(self, data_path):
        """
        Args:
            data_path (str): Path to the poker hand data file.
        """
        self.data_path = data_path
        cols = ['S1', 'C1', 'S2', 'C2', 'S3', 'C3', 'S4', 'C4', 'S5', 'C5', 'CLASS']
        self.df = pd.read_csv(data_path, header=None, names=cols)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        hand_row = self.df.iloc[idx]
        
        # Reshape the 10 features into 5 cards with 2 features each (suit, rank)
        hand = torch.tensor([
            [hand_row['S1'], hand_row['C1']],
            [hand_row['S2'], hand_row['C2']],
            [hand_row['S3'], hand_row['C3']],
            [hand_row['S4'], hand_row['C4']],
            [hand_row['S5'], hand_row['C5']]
        ], dtype=torch.long)

        # Make suit and rank 0-indexed for embedding layers
        # Suits are 1-4, Ranks are 1-13
        hand[:, 0] = hand[:, 0] - 1
        hand[:, 1] = hand[:, 1] - 1

        label = torch.tensor(hand_row['CLASS'], dtype=torch.long)

        return hand, label
