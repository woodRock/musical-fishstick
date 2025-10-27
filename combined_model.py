import torch.nn as nn

class CombinedModel(nn.Module):
    """Combines a base model and a head model."""
    def __init__(self, base_model, head_model):
        """Initializes the CombinedModel.

        Args:
            base_model (nn.Module): The base model (feature extractor).
            head_model (nn.Module): The head model (classifier or regressor).
        """
        super(CombinedModel, self).__init__()
        self.base_model = base_model
        self.head_model = head_model

    def forward(self, x):
        """Performs a forward pass through the CombinedModel.

        Args:
            x (torch.Tensor): The input tensor.

        Returns:
            torch.Tensor: The output of the head model.
        """
        # Get features from the base model
        features = self.base_model(x)
        # Apply the head model to the features
        output = self.head_model(features)
        return output
