"""
توابع هزینه‌ی حساس به عدم توازن کلاس (Imbalance-aware losses).

  ce       : Cross-Entropy معمولی (مبنا)
  wce      : Cross-Entropy وزن‌دار با معکوس فراوانی کلاس
  focal    : Focal Loss  — Lin et al., ICCV 2017
  cb       : Class-Balanced Loss با «تعداد مؤثر نمونه» — Cui et al., CVPR 2019
  cb_focal : ترکیب Class-Balanced و Focal (همان مقاله‌ی Cui)
  la       : Logit-Adjusted Loss — Menon et al., ICLR 2021

همه‌ی وزن‌ها فقط از توزیع train محاسبه می‌شوند (نه val/test) تا نشت اطلاعات رخ ندهد.
"""
from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

LOSSES = ["ce", "wce", "focal", "cb", "cb_focal", "la"]


def inverse_freq_weights(counts: np.ndarray) -> torch.Tensor:
    w = counts.sum() / (len(counts) * counts)          # همان "balanced" در sklearn
    return torch.tensor(w, dtype=torch.float)


def class_balanced_weights(counts: np.ndarray, beta: float = 0.999) -> torch.Tensor:
    effective_num = 1.0 - np.power(beta, counts)
    w = (1.0 - beta) / effective_num
    w = w / w.sum() * len(counts)                      # نرمال‌سازی: میانگین وزن = ۱
    return torch.tensor(w, dtype=torch.float)


class FocalLoss(nn.Module):
    def __init__(self, gamma: float = 2.0, weight: torch.Tensor | None = None):
        super().__init__()
        self.gamma = gamma
        self.register_buffer("weight", weight if weight is not None else None)

    def forward(self, logits, target):
        logp = F.log_softmax(logits.float(), dim=-1)
        logp_t = logp.gather(1, target.unsqueeze(1)).squeeze(1)
        p_t = logp_t.exp()
        loss = -((1.0 - p_t) ** self.gamma) * logp_t
        if self.weight is not None:
            w = self.weight[target]
            return (w * loss).sum() / w.sum()
        return loss.mean()


class LogitAdjustedLoss(nn.Module):
    """در آموزش، tau*log(prior) به logitها اضافه می‌شود؛ در استنتاج logit خام استفاده می‌شود."""
    def __init__(self, counts: np.ndarray, tau: float = 1.0):
        super().__init__()
        prior = counts / counts.sum()
        self.register_buffer("adj", tau * torch.log(torch.tensor(prior, dtype=torch.float)))

    def forward(self, logits, target):
        return F.cross_entropy(logits.float() + self.adj, target)


class WeightedCE(nn.Module):
    def __init__(self, weight: torch.Tensor | None = None):
        super().__init__()
        self.register_buffer("weight", weight if weight is not None else None)

    def forward(self, logits, target):
        return F.cross_entropy(logits.float(), target, weight=self.weight)


def build_loss(name: str, counts: np.ndarray, gamma: float = 2.0,
               beta: float = 0.999, tau: float = 1.0) -> nn.Module:
    counts = np.asarray(counts, dtype=np.float64)
    if name == "ce":
        return WeightedCE(None)
    if name == "wce":
        return WeightedCE(inverse_freq_weights(counts))
    if name == "focal":
        return FocalLoss(gamma=gamma)
    if name == "cb":
        return WeightedCE(class_balanced_weights(counts, beta))
    if name == "cb_focal":
        return FocalLoss(gamma=gamma, weight=class_balanced_weights(counts, beta))
    if name == "la":
        return LogitAdjustedLoss(counts, tau=tau)
    raise ValueError(f"Unknown loss '{name}'. Choose from {LOSSES}")
