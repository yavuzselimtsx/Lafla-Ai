"""
@Dosya: training/optimizer_policy.py
@Aciklama: Optimizer hizli yol secimini PyTorch/AMP uyumluluguna gore yapar.
@Yazar: Lafla Gelistirme Ekibi
@Bilgi: fp16 CUDA GradScaler bazi AdamW fused/foreach yollarinda step sirasinda
        coker; bu karar kodu egitim geometrisini degistirmeden guvenli yolu secer.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class AdamWFastPathPolicy:
    """AdamW icin secilen hizli yol politikasini tasir."""

    use_fused: bool
    force_single_tensor: bool
    mode: str


def resolve_adamw_fast_path(
    *,
    prefer_fused_optimizer: bool,
    device_type: str,
    precision: str,
) -> AdamWFastPathPolicy:
    """AdamW fused/foreach secimini AMP uyumlulugunu koruyarak yapar."""

    if device_type != "cuda":
        return AdamWFastPathPolicy(use_fused=False, force_single_tensor=False, mode="adamw")
    if precision == "fp16":
        return AdamWFastPathPolicy(use_fused=False, force_single_tensor=True, mode="adamw_amp_safe")
    if prefer_fused_optimizer:
        return AdamWFastPathPolicy(use_fused=True, force_single_tensor=False, mode="fused_adamw")
    return AdamWFastPathPolicy(use_fused=False, force_single_tensor=False, mode="adamw")
