"""
@Dosya: training/lr_schedule.py
@Aciklama: Warmup ve cosine decay ogrenme orani programi.
@Yazar: Lafla Gelistirme Ekibi
@Bilgi: Scheduler saf fonksiyondur; checkpoint resume testleri kolaylasir.
"""

from __future__ import annotations

import math


def cosine_with_warmup_lr(step: int, max_steps: int, warmup_steps: int, base_lr: float, min_lr: float) -> float:
    """Step icin LR hesaplar."""

    if max_steps <= 0:
        raise ValueError("max_steps pozitif olmali")
    if step < 0:
        raise ValueError("step negatif olamaz")
    if not 0 <= warmup_steps < max_steps:
        raise ValueError("warmup_steps hatali")
    if not 0 <= min_lr <= base_lr:
        raise ValueError("lr araligi hatali")
    if warmup_steps and step < warmup_steps:
        return base_lr * float(step + 1) / float(warmup_steps)
    progress = min(1.0, max(0.0, (step - warmup_steps) / float(max_steps - warmup_steps)))
    cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
    return min_lr + (base_lr - min_lr) * cosine
