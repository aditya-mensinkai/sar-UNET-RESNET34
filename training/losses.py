"""Binary segmentation losses for oil-spill UNet (raw-logit interface).

Existing repo code has NO trainable loss (preprocessing/pipeline.compute_metrics
is numpy/binary-only ablation QC - incompatible: no logits, no TN, no torch).
This module is the single training-loss implementation.

Default: 0.5*BCEWithLogits + 0.5*soft-Dice. Rationale: oil covers ~1-3% of
pixels, so pure BCE is background-dominated; Dice restores foreground signal.
Weights configurable; defaults are the standard balanced choice, not tuned.
"""
from __future__ import annotations

import torch
import torch.nn as nn


class BCEDiceLoss(nn.Module):
    """Combined loss on RAW LOGITS. Sigmoid lives here, never in the model."""

    def __init__(self, bce_weight=0.5, dice_weight=0.5, smooth=1.0):
        super().__init__()
        if bce_weight < 0 or dice_weight < 0 or bce_weight + dice_weight <= 0:
            raise ValueError("weights must be non-negative with positive sum")
        self.bce_weight = float(bce_weight)
        self.dice_weight = float(dice_weight)
        self.smooth = float(smooth)
        self.bce = nn.BCEWithLogitsLoss()

    def forward(self, logits, targets):
        if logits.shape != targets.shape:
            raise ValueError(f"logits/targets shape mismatch {tuple(logits.shape)} vs "
                             f"{tuple(targets.shape)}")
        if logits.ndim != 4 or logits.shape[1] != 1:
            raise ValueError(f"expected [B,1,H,W], got {tuple(logits.shape)}")
        logits = logits.float()
        targets = targets.float()
        loss = self.bce_weight * self.bce(logits, targets)
        if self.dice_weight > 0:
            probs = torch.sigmoid(logits)  # continuous probabilities, never thresholded
            inter = (probs * targets).sum()
            dice = 1.0 - (2.0 * inter + self.smooth) / (
                probs.sum() + targets.sum() + self.smooth)
            loss = loss + self.dice_weight * dice
        return loss
