import torch.nn as nn

class ClassificationHead(nn.Module):
    """Standard classification head for MLP and EMD loss models."""
    def __init__(self, input_size, num_classes):
        """Initializes the ClassificationHead.

        Args:
            input_size (int): The number of input features.
            num_classes (int): The total number of classes.
        """
        super(ClassificationHead, self).__init__()
        self.fc = nn.Linear(input_size, num_classes)

    def forward(self, x):
        """Performs a forward pass through the ClassificationHead.

        Args:
            x (torch.Tensor): The input tensor.

        Returns:
            torch.Tensor: The output logits.
        """
        logits = self.fc(x)
        return logits
