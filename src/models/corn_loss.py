import torch
import torch.nn.functional as F

def corn_loss(logits, y_train, num_classes):
    """Computes the CORN loss.

    Args:
        logits (torch.Tensor): Logits from the model (batch_size, num_classes - 1).
        y_train (torch.Tensor): Ground truth labels (batch_size).
        num_classes (int): Total number of classes.

    Returns:
        torch.Tensor: The CORN loss.
    """
    # Create binary targets for each of the K-1 tasks
    y_onehot = torch.zeros_like(logits)
    for i in range(num_classes - 1):
        y_onehot[:, i] = (y_train > i).float()

    # Calculate binary cross-entropy for each task
    loss = F.binary_cross_entropy_with_logits(logits, y_onehot, reduction='mean')
    return loss
