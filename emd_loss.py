import torch

def emd_loss(predictions, targets):
    """
    Computes the Earth Mover's Distance (EMD) loss, also known as Ranked Probability Score.

    Args:
        predictions: Model predictions (logits), shape (batch_size, num_classes).
        targets: Ground truth labels, shape (batch_size).

    Returns:
        The EMD loss.
    """
    # Convert predictions to probabilities and calculate CDF
    predictions_cdf = torch.cumsum(torch.softmax(predictions, dim=-1), dim=-1)

    # Convert targets to one-hot, then calculate CDF
    targets_one_hot = torch.nn.functional.one_hot(targets, num_classes=predictions.shape[1])
    targets_cdf = torch.cumsum(targets_one_hot, dim=-1)

    # Calculate the EMD loss (sum of squared differences between CDFs)
    loss = torch.mean(torch.sum((predictions_cdf - targets_cdf) ** 2, dim=-1))
    
    return loss
