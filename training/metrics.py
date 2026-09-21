"""Binary segmentation metrics on raw logits (torch, device-preserving).

Pipeline: logits -> sigmoid -> >=threshold -> pixel TP/FP/TN/FN -> metrics.
Zero-division policy (deterministic, documented): a metric whose denominator is
zero scores 1.0 when that denotes perfect agreement (e.g. IoU/Dice with no
positives anywhere; Precision with no predicted positives; Recall with no
actual positives) and 0.0 otherwise. Dice and F1 share one definition, so
Dice == F1 by construction. Global aggregation via ConfusionAccumulator
(sum counts first, then divide) - never average per-batch ratios.
"""
from __future__ import annotations

import torch


def _check(logits, targets):
    if logits.shape != targets.shape:
        raise ValueError(f"logits/targets shape mismatch {tuple(logits.shape)} vs "
                         f"{tuple(targets.shape)}")
    if logits.ndim != 4 or logits.shape[1] != 1:
        raise ValueError(f"expected [B,1,H,W], got {tuple(logits.shape)}")
    return logits.float(), targets.float()


def confusion_counts(logits, targets, threshold=0.5):
    """Return dict TP/FP/TN/FN (python ints) from raw logits."""
    logits, targets = _check(logits, targets)
    pred = (torch.sigmoid(logits) >= threshold)
    true = targets >= 0.5
    tp = int((pred & true).sum())
    fp = int((pred & ~true).sum())
    tn = int((~pred & ~true).sum())
    fn = int((~pred & true).sum())
    return {"TP": tp, "FP": fp, "TN": tn, "FN": fn}


def metrics_from_counts(tp, fp, tn, fn):
    """Standard binary definitions with the documented zero-division policy."""
    iou_d = tp + fp + fn
    IoU = tp / iou_d if iou_d else 1.0
    dice_d = 2 * tp + fp + fn
    Dice = 2 * tp / dice_d if dice_d else 1.0
    Precision = tp / (tp + fp) if (tp + fp) else 1.0
    Recall = tp / (tp + fn) if (tp + fn) else 1.0
    F1 = 2 * Precision * Recall / (Precision + Recall) if (Precision + Recall) else 0.0
    return {"TP": tp, "FP": fp, "TN": tn, "FN": fn, "IoU": IoU, "Dice": Dice,
            "Precision": Precision, "Recall": Recall, "F1": F1}


def compute_binary_segmentation_metrics(logits, targets, threshold=0.5):
    """Logits + binary targets -> full metric dict (global over the input)."""
    c = confusion_counts(logits, targets, threshold)
    return metrics_from_counts(c["TP"], c["FP"], c["TN"], c["FN"])


class ConfusionAccumulator:
    """Sum TP/FP/TN/FN across batches, then divide once (global pixel metrics)."""

    def __init__(self):
        self.reset()

    def reset(self):
        self.tp = self.fp = self.tn = self.fn = 0
        self.pixels = 0

    def add(self, logits, targets, threshold=0.5):
        c = confusion_counts(logits, targets, threshold)
        self.tp += c["TP"]
        self.fp += c["FP"]
        self.tn += c["TN"]
        self.fn += c["FN"]
        self.pixels += int(targets.numel())
        return c

    def metrics(self):
        m = metrics_from_counts(self.tp, self.fp, self.tn, self.fn)
        m["pixels"] = self.pixels
        return m
