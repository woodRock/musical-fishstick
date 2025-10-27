"""Module for the custom implementation of Proportional Odds Model (POM) components."""

import torch
import torch.nn as nn

class POMHead(nn.Module):
    """A neural network head for the Proportional Odds Model (POM).

    This head takes features from a base model and applies a scoring function
    and learned thresholds to produce cumulative probabilities for ordinal classes.
    """
    def __init__(self, input_size, num_classes):
        """Initializes the POMHead.

        Args:
            input_size (int): The number of input features from the base model.
            num_classes (int): The total number of ordered classes.
        """
        super(POMHead, self).__init__()
        # The input_size for the head is the output_size of the base model
        if input_size != 1:
            # If the base is not linear, we need a final linear layer
            # to produce a single score for the POM logic.
            self.scorer = nn.Linear(input_size, 1)
        else:
            self.scorer = nn.Identity() # Pass through if base is already linear

        self.thresholds = nn.Parameter(torch.randn(num_classes - 1).sort()[0])

    def forward(self, x):
        """Performs a forward pass through the POMHead.

        Args:
            x (torch.Tensor): The input tensor (features from the base model).

        Returns:
            torch.Tensor: Cumulative probabilities for each class.
        """
        score = self.scorer(x)
        sorted_thresholds = torch.sort(self.thresholds)[0]
        cumulative_probs = torch.sigmoid(sorted_thresholds - score)
        return cumulative_probs

def pom_loss(cumulative_probs, y, num_classes):
    """Calculates the loss for the Proportional Odds Model (POM).

    Args:
        cumulative_probs (torch.Tensor): Cumulative probabilities from the POMHead.
        y (torch.Tensor): True class labels.
        num_classes (int): Total number of ordered classes.

    Returns:
        torch.Tensor: The calculated POM loss.
    """
    p_plus = torch.cat([torch.zeros(cumulative_probs.shape[0], 1, device=cumulative_probs.device), cumulative_probs], dim=1)
    p_minus = torch.cat([cumulative_probs, torch.ones(cumulative_probs.shape[0], 1, device=cumulative_probs.device)], dim=1)
    probs = p_minus - p_plus
    probs = torch.clamp(probs, 1e-7, 1 - 1e-7)
    true_class_probs = probs[torch.arange(len(y)), y]
    loss = -torch.log(true_class_probs).mean()
    return loss

def pom_predict(cumulative_probs):
    """Predicts class labels from cumulative probabilities for POM.

    Args:
        cumulative_probs (torch.Tensor): Cumulative probabilities from the POMHead.

    Returns:
        torch.Tensor: Predicted class labels.
    """
    p_plus = torch.cat([torch.zeros(cumulative_probs.shape[0], 1, device=cumulative_probs.device), cumulative_probs], dim=1)
    p_minus = torch.cat([cumulative_probs, torch.ones(cumulative_probs.shape[0], 1, device=cumulative_probs.device)], dim=1)
    probs = p_minus - p_plus
    return torch.argmax(probs, dim=1)