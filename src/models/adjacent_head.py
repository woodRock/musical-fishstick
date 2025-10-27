import torch
import torch.nn as nn

def adjacent_loss(probs, y):
    """Calculates the adjacent loss for ordinal regression.

    Args:
        probs (torch.Tensor): Predicted probabilities for each class.
        y (torch.Tensor): True class labels.

    Returns:
        torch.Tensor: The calculated adjacent loss.
    """
    probs = torch.clamp(probs, 1e-7, 1 - 1e-7)
    true_class_probs = probs[torch.arange(len(y)), y]
    loss = -torch.log(true_class_probs).mean()
    return loss


def adjacent_predict(probs):
    """Predicts class labels from probabilities.

    Args:
        probs (torch.Tensor): Predicted probabilities for each class.

    Returns:
        torch.Tensor: Predicted class labels.
    """
    return torch.argmax(probs, dim=1)


class AdjacentHead(nn.Module):
    """A neural network head for adjacent category ordinal regression.

    This head transforms input features into probabilities for ordered classes
    using a series of binary logistic regressions.
    """
    def __init__(self, input_size, num_classes):
        """Initializes the AdjacentHead.

        Args:
            input_size (int): The number of input features.
            num_classes (int): The total number of ordered classes.
        """
        super(AdjacentHead, self).__init__()
        self.num_classes = num_classes
        if input_size != 1:
            self.scorer = nn.Linear(input_size, 1)
        else:
            self.scorer = nn.Identity()

        self.intercepts = nn.Parameter(torch.randn(num_classes - 1))

    def forward(self, x):
        """Performs a forward pass through the AdjacentHead.

        Args:
            x (torch.Tensor): The input tensor.

        Returns:
            torch.Tensor: Predicted probabilities for each class.
        """
        score = self.scorer(x)
        log_odds = self.intercepts - score
        exp_log_odds = torch.exp(log_odds)
        
        prob_list = [torch.ones(x.shape[0], 1, device=x.device)]
        for k in range(self.num_classes - 2, -1, -1):
            new_prob = prob_list[0] * exp_log_odds[:, k].unsqueeze(1)
            prob_list.insert(0, new_prob)
        
        unnormalized_probs = torch.cat(prob_list, dim=1)
        sum_probs = torch.sum(unnormalized_probs, dim=1, keepdim=True)
        probs = unnormalized_probs / sum_probs
        return probs