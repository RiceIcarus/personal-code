import torch
import torch.nn.functional as F

import config_loader


def masked_smooth_l1_loss(pred, target):
    mask = torch.isfinite(target) & torch.isfinite(pred)
    if not mask.any():
        return pred.new_zeros(())
    return F.smooth_l1_loss(pred[mask], target[mask], beta=config_loader.huber_beta)


def masked_l1_loss(pred, target):
    mask = torch.isfinite(target) & torch.isfinite(pred)
    if not mask.any():
        return pred.new_zeros(())
    return F.l1_loss(pred[mask], target[mask])


def masked_mse_loss(pred, target):
    mask = torch.isfinite(target) & torch.isfinite(pred)
    if not mask.any():
        return pred.new_zeros(())
    return F.mse_loss(pred[mask], target[mask])
