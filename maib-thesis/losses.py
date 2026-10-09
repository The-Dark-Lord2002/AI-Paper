"""
Loss functions for class imbalance (proposal, step 4).

  ce     plain cross-entropy: every report counts the same            "do nothing" baseline
  wce    cross-entropy, each class weighted by 1 / its frequency       step 4a
  focal  focal loss (Lin et al., 2017): easy reports count less        step 4b
  cb     cross-entropy with class-balanced weights (Cui et al., 2019)  step 4c

Class weights are computed from the TRAINING set only. Using validation or test counts would leak
information about the data we evaluate on.
"""
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

import config

LOSS_NAMES = ["ce", "wce", "focal", "cb"]


class FocalLoss(nn.Module):
    """
    loss = -(1 - p)^gamma * log(p),   p = the probability the model gives to the correct class.

    For a report the model already gets right with confidence (p close to 1), (1 - p)^gamma is close
    to 0, so it hardly contributes. Training time goes to the hard reports, which are mostly the
    rare classes. With gamma = 0 this is ordinary cross-entropy.
    """

    def __init__(self, gamma):
        super().__init__()
        self.gamma = gamma

    def forward(self, logits, target):
        log_p = F.log_softmax(logits, dim=-1).gather(1, target.unsqueeze(1)).squeeze(1)
        p = log_p.exp()
        return (-(1 - p) ** self.gamma * log_p).mean()


def make_loss(name, class_counts):
    """class_counts[i] = number of TRAINING reports of class i."""
    counts = np.asarray(class_counts, dtype=float)
    n_classes = len(counts)

    if name == "ce":
        return nn.CrossEntropyLoss()

    if name == "wce":
        # Inverse frequency: a class with 10x fewer reports gets 10x more weight.
        weights = counts.sum() / (n_classes * counts)
        return nn.CrossEntropyLoss(weight=torch.tensor(weights, dtype=torch.float))

    if name == "focal":
        return FocalLoss(gamma=config.FOCAL_GAMMA)

    if name == "cb":
        # "Effective number" of reports: (1 - beta^n) / (1 - beta). It grows almost linearly for small n
        # and flattens for large n, because many reports of a big class are near-duplicates of each other.
        # Weights are 1 / effective number, a milder version of inverse frequency.
        beta = config.CB_BETA
        effective_number = (1 - beta ** counts) / (1 - beta)
        weights = 1 / effective_number
        weights = weights / weights.sum() * n_classes              # rescale so the average weight is 1
        return nn.CrossEntropyLoss(weight=torch.tensor(weights, dtype=torch.float))

    raise ValueError(f"unknown loss '{name}', choose from {LOSS_NAMES}")
