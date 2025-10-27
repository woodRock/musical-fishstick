import torch
import torch.nn as nn

class OrdinalHead(nn.Module):
    """Head for CORAL and CORN which expect num_classes - 1 outputs."""
    def __init__(self, input_size, num_classes):
        super(OrdinalHead, self).__init__()
        self.fc = nn.Linear(input_size, num_classes - 1)

    def forward(self, x):
        logits = self.fc(x)
        return logits

def coral_loss(logits, levels):
    """Computes the CORAL loss."""
    levels = levels.view(-1, 1)
    g = torch.arange(len(logits[0]), device=logits.device).float()
    levels_one_hot = (g < levels).float()
    loss = torch.nn.functional.logsigmoid(logits) * levels_one_hot + \
             (torch.nn.functional.logsigmoid(logits) - logits) * (1 - levels_one_hot)
    return -loss.mean()

def coral_predict(logits):
    """Makes predictions from CORAL logits."""
    probas = torch.sigmoid(logits)
    predict_levels = torch.sum((probas > 0.5), dim=1)
    return predict_levels
