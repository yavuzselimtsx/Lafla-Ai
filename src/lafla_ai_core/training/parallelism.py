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


@dataclass(frozen=True)
class BatchGeometry:
    """Cihaz sayisindan cozulmus optimizer batch geometrisi."""

    global_micro_batch_size: int
    gradient_accumulation_steps: int
    sequences_per_optimizer_step: int


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


def resolve_batch_geometry(
    *,
    configured_micro_batch_size: int,
    configured_gradient_accumulation_steps: int,
    cuda_micro_batch_size_per_device: int,
    target_sequences_per_optimizer_step: int,
    decision: ParallelismDecision,
) -> BatchGeometry:
    """CUDA hiz profilini toplam optimizer batch'ini koruyarak cozer."""

    if configured_gradient_accumulation_steps < 1:
        raise ValueError("configured_gradient_accumulation_steps pozitif olmali")
    tuned = cuda_micro_batch_size_per_device > 0 or target_sequences_per_optimizer_step > 0
    if not tuned:
        micro_batch = effective_micro_batch_size(configured_micro_batch_size, decision)
        accumulation = configured_gradient_accumulation_steps
        return BatchGeometry(micro_batch, accumulation, micro_batch * accumulation)
    if cuda_micro_batch_size_per_device < 1 or target_sequences_per_optimizer_step < 1:
        raise ValueError("cuda batch tuning alanlari birlikte pozitif olmali")
    if decision.cuda_device_count < 1:
        micro_batch = configured_micro_batch_size
        accumulation = configured_gradient_accumulation_steps
        return BatchGeometry(micro_batch, accumulation, micro_batch * accumulation)
    active_cuda_devices = decision.cuda_device_count if decision.enabled else 1
    micro_batch = cuda_micro_batch_size_per_device * active_cuda_devices
    if target_sequences_per_optimizer_step % micro_batch != 0:
        raise ValueError("target_sequences_per_optimizer_step global micro batch'e tam bolunmeli")
    accumulation = target_sequences_per_optimizer_step // micro_batch
    if accumulation < 1:
        raise ValueError("cozulmus gradient accumulation pozitif olmali")
    return BatchGeometry(micro_batch, accumulation, micro_batch * accumulation)


def resolve_gradient_checkpointing(
    *,
    model_checkpointing_enabled: bool,
    minimum_sequence_length: int,
    active_sequence_length: int,
) -> bool:
    """Aktif context uzunlugu icin activation checkpointing kararini verir."""

    if not model_checkpointing_enabled:
        return False
    if minimum_sequence_length <= 0:
        return True
    return active_sequence_length >= minimum_sequence_length
