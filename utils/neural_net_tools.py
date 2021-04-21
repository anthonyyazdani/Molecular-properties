import torch


def count_parameters(model):
    """
    Function to count the number of trainable parameters.

    ARGS:
        - model: Pytorch model.

    OUTPUT:
        - Number of trainable parameters
    """
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def poisson_accuracy(y_hat, y_true):
    """
    Function to compute the poisson accuracy.

    ARGS:
        - y_hat: Predictions.
        - y_true: True labels.

    OUTPUT:
        - Accuracy.
    """
    score = ((torch.round(torch.exp(y_hat)).squeeze() == y_true) * 1).float()
    return float(torch.mean(score).cpu().numpy())*100