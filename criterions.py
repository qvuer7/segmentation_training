import torch
import torch.nn as nn


def dice_coefficient(prediction, target):
    smooth = 1e-5  # Small constant to avoid division by zero
    threshold = 0.5  # Threshold for converting network output to binary mask
    with torch.no_grad():
        predicted_class = torch.argmax(prediction, dim=1)  # Get the predicted class index
        prediction_binary = (predicted_class == 1).float()
        intersection = torch.sum(prediction_binary * target)
        union = torch.sum(prediction_binary) + torch.sum(target)
        dice = (2.0 * intersection + smooth) / (union + smooth)
    return dice.item()

def iou(prediction, target):
    smooth = 1e-5  # Small constant to avoid division by zero
    threshold = 0.5  # Threshold for converting network output to binary mask
    with torch.no_grad():
        predicted_class = torch.argmax(prediction, dim=1)  # Get the predicted class index
        prediction_binary = (predicted_class == 1).float()
        intersection = torch.sum(prediction_binary * target)
        union = torch.sum(prediction_binary) + torch.sum(target) - intersection
        iou = (intersection + smooth) / (union + smooth)
    return iou.item()


def CECriterion(inputs, target, we, device):
    losses = {}

    weight = we.to(device)
    criterion = nn.BCEWithLogitsLoss()
    import numpy as np

    for name, x in inputs.items():

        x = x.squeeze()

        target = target.float()
        losses[name] = criterion(x, target)



    if len(losses) == 1:

        return losses['out']

    return losses['out'] + 0.5 * losses['aux']

def BCECriterion(out, targets):
    predictions = out['out']

    batch_size, num_classes, height, width = predictions.size()

    # Apply argmax to get the segmentation mask
    segmentation_mask = predictions.argmax(dim=1)  # Shape: [batch_size, height, width]

    # Flatten the segmentation mask, output, and target tensors
    flattened_segmentation_mask = segmentation_mask.view(batch_size, -1)
    flattened_segmentation_mask = torch.sigmoid(flattened_segmentation_mask)
    flattened_target = targets.view(batch_size, -1).float()

    criterion = nn.BCELoss(reduction='mean')

    loss = criterion(flattened_segmentation_mask, flattened_target)

    # Optionally compute the average loss across batches and samples
    average_loss = loss.mean()


    return average_loss



