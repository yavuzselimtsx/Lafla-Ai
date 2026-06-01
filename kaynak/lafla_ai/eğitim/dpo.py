"""
Lafla AI için Direct Preference Optimization kaybı.

DPO, ayrı bir reward modeli kurmadan tercih edilen ve reddedilen cevaplar
arasındaki farkı öğretir. Bu dosya sadece matematiksel kaybı tutar; veri
yükleme, model çağırma ve checkpoint yazma başka katmanların sorumluluğudur.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F


@dataclass(frozen=True)
class DpoBatch:
    """DPO için tercih edilen ve reddedilen örnek tokenlarını taşır."""

    chosen_input_ids: torch.Tensor
    chosen_labels: torch.Tensor
    rejected_input_ids: torch.Tensor
    rejected_labels: torch.Tensor


def sequence_log_probability(logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    """Her örnek için etiketlenmiş tokenların toplam log olasılığını hesaplar."""

    shifted_logits = logits[:, :-1, :]
    shifted_labels = labels[:, 1:]
    log_probs = F.log_softmax(shifted_logits, dim=-1)
    token_log_probs = log_probs.gather(dim=-1, index=shifted_labels.unsqueeze(-1)).squeeze(-1)
    mask = shifted_labels.ne(-100)
    return (token_log_probs * mask).sum(dim=-1)


def dpo_loss(
    policy_chosen_logps: torch.Tensor,
    policy_rejected_logps: torch.Tensor,
    reference_chosen_logps: torch.Tensor,
    reference_rejected_logps: torch.Tensor,
    beta: float,
) -> torch.Tensor:
    """Politika ve referans model marjlarını karşılaştırarak DPO kaybını döndürür."""

    if beta <= 0:
        raise ValueError("beta must be positive")
    policy_margin = policy_chosen_logps - policy_rejected_logps
    reference_margin = reference_chosen_logps - reference_rejected_logps
    logits = beta * (policy_margin - reference_margin)
    return -F.logsigmoid(logits).mean()
