#---------------------------------------
#Since : 2024/09/05
#Update: 2024/12/16
# -*- coding: utf-8 -*-
#---------------------------------------
import torch

def Criterion(input, target, metric_type):
    """
    Calculate the specified evaluation function between the input and target tensors.

    Args:
        input (torch.Tensor): The input tensor.
        target (torch.Tensor): The target (ground truth) tensor.
        metric_type (str): The type of evaluation function to calculate. Supported metrics: 'mse', 'accuracy', 'cross_entropy', 'binary_cross_entropy'.

    Returns:
        torch.Tensor: The calculated evaluation function value.
    """
    if metric_type == 'mse':
        # Mean Squared Error (MSE):
        # MSE = 1/n * Σ (y_pred - y_true)^2
        return torch.nn.functional.mse_loss(input, target)
    elif metric_type == 'accuracy':
        # Accuracy:
        # Acc = 1/n * Σ 1(y_pred == y_true)
        return (input.argmax(dim=1) == target).float().mean()
    elif metric_type == 'cross_entropy':
        # Cross Entropy Loss:
        # CE = -1/n * Σ y_true * log(softmax(y_pred))
        return F.cross_entropy(input, target)
    elif metric_type == 'binary_cross_entropy':
        # Binary Cross Entropy Loss:
        # BCE = -1/n * Σ (y_true * log(y_pred) + (1 - y_true) * log(1 - y_pred))
        return F.binary_cross_entropy(input, target)
    else:
        raise ValueError(f"Invalid metric type: {metric_type}")

def evaluate_model(model, data_loader, device, num_classes = 10, dataset=None):
    model.eval()
    loss = 0
    correct = 0
    total = 0

    with torch.no_grad():
        for batch_x, batch_y in data_loader:
            batch_x, batch_y = batch_x.to(device), torch.nn.functional.one_hot(batch_y.to(device), num_classes=num_classes)

            B = batch_x.shape[0]
            outputs = model(batch_x.flatten(1))
            loss += Criterion(outputs.to(device), batch_y, metric_type="mse").item()
            total += batch_y.size(0)

            if dataset != "parity":
                _, predicted = outputs.max(1)
                correct += predicted.to(device).eq(batch_y.argmax(dim=1)).sum().item()
            else:
                predicted = (outputs > 0.5).int()
                correct += predicted.to(device).eq(batch_y).sum().item()


    loss /= len(data_loader)
    accuracy = correct / total
    return loss, accuracy
