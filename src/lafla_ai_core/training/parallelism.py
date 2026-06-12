"""
@Dosya: training/parallelism.py
@Aciklama: Tek GPU, coklu GPU ve etkili batch kararlari icin torch bagimsiz
            yardimcilar.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class ParallelismDecision:
    """CUDA DataParallel karari ve health log icin tanilama bilgisi."""

    enabled: bool
    mode: str
    cuda_device_count: int
    reason: str


def resolve_data_parallel(data_parallel: str, device_type: str, cuda_device_count: int) -> ParallelismDecision:
    """Config ve cihaz durumundan DataParallel kararini uretir."""

    if data_parallel == "off":
        return ParallelismDecision(False, "single_device", cuda_device_count, "config_disabled")
    if device_type != "cuda":
        return ParallelismDecision(False, "single_device", cuda_device_count, f"device_is_{device_type}")
    if cuda_device_count < 2:
        return ParallelismDecision(False, "single_device", cuda_device_count, "less_than_two_cuda_devices")
    return ParallelismDecision(True, "data_parallel", cuda_device_count, "multi_cuda_available")


def effective_micro_batch_size(configured_micro_batch_size: int, decision: ParallelismDecision) -> int:
    """DataParallel acikken her GPU'ya en az bir ornek dusecek global micro batch'i secer."""

    if configured_micro_batch_size < 1:
        raise ValueError("configured_micro_batch_size pozitif olmali")
    if not decision.enabled:
        return configured_micro_batch_size
    return max(configured_micro_batch_size, decision.cuda_device_count)
