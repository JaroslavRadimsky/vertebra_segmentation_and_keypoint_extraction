import numpy as np
import torch


def binary_dice(pred: torch.Tensor, target_onehot: torch.Tensor, epsilon: float = 1e-6):
    """
    Vstupy:
    - pred: [B, 1, H, W] – predikce tříd (0 nebo 1)
    - target_onehot: [B, C, H, W] – one-hot ground truth

    """
    if pred.dtype != torch.long:
        pred = pred.long()
    pred_mask = (pred == 1).float()  # [B, 1, H, W] float maska pro danou třídu
    target_mask = target_onehot[:, 1, :, :]  # [B, 1, H, W]

    intersection = torch.sum(pred_mask * target_mask)
    union = torch.sum(pred_mask) + torch.sum(target_mask)

    dice = (2. * intersection + epsilon) / (union + epsilon)
    return dice.item()


def binary_jaccard(pred: torch.Tensor, target_onehot: torch.Tensor, epsilon: float = 1e-6):
    """
    Vstupy:
    - pred: [B, 1, H, W] – predikce tříd (0 nebo 1)
    - target_onehot: [B, C, H, W] – one-hot ground truth

    """
    if pred.dtype != torch.long:
        pred = pred.long()
    pred_mask = (pred == 1).float()  # [B, 1, H, W]
    target_mask = target_onehot[:, 1, :, :]  # [B, 1, H, W]

    intersection = torch.sum(pred_mask * target_mask)
    union = torch.sum(pred_mask) + torch.sum(target_mask) - intersection

    jaccard = (intersection + epsilon) / (union + epsilon)
    return jaccard.item()


def dice_per_channel(pred_onehot, target_onehot, epsilon=1e-6):
    C = pred_onehot.shape[1]
    dice_scores = []
    for i in range(C):
        pred_i = pred_onehot[:, i]
        target_i = target_onehot[:, i]
        intersection = torch.sum(pred_i * target_i)
        union = torch.sum(pred_i) + torch.sum(target_i)

        dice = (2. * intersection + epsilon) / (union + epsilon)

        if union < 2 and intersection < 2:
            dice_scores.append(np.nan)
        else:
            dice_scores.append(dice.item())
    return dice_scores


def iou_per_channel(pred_onehot, target_onehot, epsilon=1e-6):
    C = pred_onehot.shape[1]
    iou_scores = []
    for i in range(C):
        pred_i = pred_onehot[:, i]
        target_i = target_onehot[:, i]
        intersection = torch.sum(pred_i * target_i)
        union = torch.sum(pred_i + target_i - pred_i * target_i)
        iou = (intersection + epsilon) / (union + epsilon)
        #print(f"{i}>>{intersection}, {union}, {iou}")
        if union <= 2 and intersection<=2:
            iou_scores.append(np.nan)
        else:
            iou_scores.append(iou.item())
    return iou_scores