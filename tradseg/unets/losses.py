"""Segmentation losses (faithful subset of my_unet-uncertainty/loss_functions/dice_loss.py).

The spleen training recipe uses SoftDiceLoss(batch_dice=True) on the softmax + a plain
torch CrossEntropyLoss on the logits (see engine.run_epoch). Only the pieces that recipe
needs are copied here.
"""

import numpy as np
import torch
from torch import nn


def sum_tensor(inp, axes, keepdim=False):
    axes = np.unique(axes).astype(int)
    if keepdim:
        for ax in axes:
            inp = inp.sum(int(ax), keepdim=True)
    else:
        for ax in sorted(axes, reverse=True):
            inp = inp.sum(int(ax))
    return inp


def soft_dice(net_output, gt, smooth=1., smooth_in_nom=1.):
    axes = tuple(range(2, len(net_output.size())))
    intersect = sum_tensor(net_output * gt, axes, keepdim=False)
    denom = sum_tensor(net_output + gt, axes, keepdim=False)
    return (- ((2 * intersect + smooth_in_nom) / (denom + smooth))).mean()


def soft_dice_per_batch_2(net_output, gt, smooth=1., smooth_in_nom=1., background_weight=1,
                          rebalance_weights=None):
    if rebalance_weights is not None and len(rebalance_weights) != gt.shape[1]:
        rebalance_weights = rebalance_weights[1:]
    axes = tuple([0] + list(range(2, len(net_output.size()))))
    tp = sum_tensor(net_output * gt, axes, keepdim=False)
    fn = sum_tensor((1 - net_output) * gt, axes, keepdim=False)
    fp = sum_tensor(net_output * (1 - gt), axes, keepdim=False)
    weights = torch.ones(tp.shape, device=net_output.device)
    weights[0] = background_weight
    if rebalance_weights is not None:
        rebalance_weights = torch.from_numpy(rebalance_weights).float().to(net_output.device)
        tp = tp * rebalance_weights
        fn = fn * rebalance_weights
    return (- ((2 * tp + smooth_in_nom) / (2 * tp + fp + fn + smooth)) * weights).mean()


class SoftDiceLoss(nn.Module):
    def __init__(self, smooth=1., apply_nonlin=None, batch_dice=False, do_bg=True,
                 smooth_in_nom=True, background_weight=1, rebalance_weights=None):
        super().__init__()
        if not do_bg:
            assert background_weight == 1
        self.rebalance_weights = rebalance_weights
        self.background_weight = background_weight
        self.smooth_in_nom = smooth if smooth_in_nom else 0
        self.do_bg = do_bg
        self.batch_dice = batch_dice
        self.apply_nonlin = apply_nonlin
        self.smooth = smooth

    def forward(self, x, y):
        with torch.no_grad():
            y = y.long()
        shp_x, shp_y = x.shape, y.shape
        if self.apply_nonlin is not None:
            x = self.apply_nonlin(x)
        if len(shp_x) != len(shp_y):
            y = y.view((shp_y[0], 1, *shp_y[1:]))
        y_onehot = torch.zeros(shp_x, device=x.device)
        y_onehot.scatter_(1, y, 1)
        if not self.do_bg:
            x = x[:, 1:]
            y_onehot = y_onehot[:, 1:]
        if not self.batch_dice:
            if self.background_weight != 1 or (self.rebalance_weights is not None):
                raise NotImplementedError
            return soft_dice(x, y_onehot, self.smooth, self.smooth_in_nom)
        return soft_dice_per_batch_2(x, y_onehot, self.smooth, self.smooth_in_nom,
                                     background_weight=self.background_weight,
                                     rebalance_weights=self.rebalance_weights)
