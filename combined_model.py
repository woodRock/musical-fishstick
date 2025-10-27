import torch.nn as nn

class CombinedModel(nn.Module):
    """Combines a base model and a head model."""
    def __init__(self, base_model, head_model):
        super(CombinedModel, self).__init__()
        self.base_model = base_model
        self.head_model = head_model

    def forward(self, x):
        # Get features from the base model
        features = self.base_model(x)
        # Apply the head model to the features
        output = self.head_model(features)
        return output
