import torch.nn as nn

class ClassificationHead(nn.Module):
    """Standard classification head for MLP and EMD loss models."""
    def __init__(self, input_size, num_classes):
        super(ClassificationHead, self).__init__()
        self.fc = nn.Linear(input_size, num_classes)

    def forward(self, x):
        logits = self.fc(x)
        return logits
