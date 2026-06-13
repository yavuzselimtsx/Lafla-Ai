"""
@Dosya: training/parallelism.py
@Aciklama: Tek GPU, coklu GPU ve etkili batch kararlari icin torch bagimsiz
            yardimcilar.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Iterator, TypeVar


T = TypeVar("T")


@dataclass(frozen=True)
class ParallelismDecision:
    """Tek cihaz, DataParallel veya DDP karari icin tanilama bilgisi."""

    enabled: bool
    mode: str
    cuda_device_count: int
    reason: str
    world_size: int = 1
    rank: int = 0
    local_rank: int = 0

    @property
    def distributed(self) -> bool:
        return self.mode == "distributed_data_parallel"


@dataclass(frozen=True)
class BatchGeometry:
    """Cihaz sayisindan cozulmus optimizer batch geometrisi."""

    per_process_micro_batch_size: int
    global_micro_batch_size: int
    gradient_accumulation_steps: int
    sequences_per_optimizer_step: int


def resolve_parallelism(
    data_parallel: str,
    device_type: str,
    cuda_device_count: int,
    distributed_world_size: int = 1,
    rank: int = 0,
    local_rank: int = 0,
) -> ParallelismDecision:
    """Config, cihaz ve torchrun ortamindan paralellik kararini uretir."""

    if distributed_world_size < 1:
        raise ValueError("distributed_world_size pozitif olmali")
    if not 0 <= rank < distributed_world_size:
        raise ValueError("rank distributed_world_size araliginda olmali")
    if local_rank < 0:
        raise ValueError("local_rank negatif olamaz")
    if distributed_world_size > 1:
        if data_parallel == "off":
            raise ValueError("torchrun aktifken data_parallel off olamaz")
        if device_type != "cuda":
            raise ValueError("DDP yolu CUDA cihazi gerektirir")
        if cuda_device_count < distributed_world_size:
            raise ValueError("DDP world_size CUDA cihaz sayisini asamaz")
        return ParallelismDecision(
            True,
            "distributed_data_parallel",
            cuda_device_count,
            "torchrun_environment",
            distributed_world_size,
            rank,
            local_rank,
        )
    if data_parallel == "off":
        return ParallelismDecision(False, "single_device", cuda_device_count, "config_disabled")
    if device_type != "cuda":
        return ParallelismDecision(False, "single_device", cuda_device_count, f"device_is_{device_type}")
    if cuda_device_count < 2:
        return ParallelismDecision(False, "single_device", cuda_device_count, "less_than_two_cuda_devices")
    return ParallelismDecision(True, "data_parallel", cuda_device_count, "multi_cuda_available")


def resolve_data_parallel(data_parallel: str, device_type: str, cuda_device_count: int) -> ParallelismDecision:
    """Eski tek-process cagrilari icin DataParallel kararini korur."""

    return resolve_parallelism(data_parallel, device_type, cuda_device_count)


def effective_micro_batch_size(configured_micro_batch_size: int, decision: ParallelismDecision) -> int:
    """Paralellik acikken her GPU'ya en az bir ornek dusecek global micro batch'i secer."""

    if configured_micro_batch_size < 1:
        raise ValueError("configured_micro_batch_size pozitif olmali")
    if not decision.enabled:
        return configured_micro_batch_size
    if decision.distributed:
        return max(configured_micro_batch_size, decision.world_size)
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
        global_micro_batch = effective_micro_batch_size(configured_micro_batch_size, decision)
        if decision.distributed:
            if global_micro_batch % decision.world_size != 0:
                raise ValueError("global micro batch DDP world_size degerine tam bolunmeli")
            per_process_micro_batch = global_micro_batch // decision.world_size
        else:
            per_process_micro_batch = global_micro_batch
        accumulation = configured_gradient_accumulation_steps
        return BatchGeometry(
            per_process_micro_batch,
            global_micro_batch,
            accumulation,
            global_micro_batch * accumulation,
        )
    if cuda_micro_batch_size_per_device < 1 or target_sequences_per_optimizer_step < 1:
        raise ValueError("cuda batch tuning alanlari birlikte pozitif olmali")
    if decision.cuda_device_count < 1:
        global_micro_batch = configured_micro_batch_size
        accumulation = configured_gradient_accumulation_steps
        return BatchGeometry(
            global_micro_batch,
            global_micro_batch,
            accumulation,
            global_micro_batch * accumulation,
        )
    if decision.distributed:
        active_cuda_devices = decision.world_size
        per_process_micro_batch = cuda_micro_batch_size_per_device
    elif decision.enabled:
        active_cuda_devices = decision.cuda_device_count
        per_process_micro_batch = cuda_micro_batch_size_per_device * active_cuda_devices
    else:
        active_cuda_devices = 1
        per_process_micro_batch = cuda_micro_batch_size_per_device
    global_micro_batch = cuda_micro_batch_size_per_device * active_cuda_devices
    if target_sequences_per_optimizer_step % global_micro_batch != 0:
        raise ValueError("target_sequences_per_optimizer_step global micro batch'e tam bolunmeli")
    accumulation = target_sequences_per_optimizer_step // global_micro_batch
    if accumulation < 1:
        raise ValueError("cozulmus gradient accumulation pozitif olmali")
    return BatchGeometry(
        per_process_micro_batch,
        global_micro_batch,
        accumulation,
        global_micro_batch * accumulation,
    )


def should_sync_gradients(micro_step: int, accumulation_steps: int) -> bool:
    """Yalniz son accumulation microstep'inde gradient senkronizasyonu ister."""

    if accumulation_steps < 1:
        raise ValueError("accumulation_steps pozitif olmali")
    if not 0 <= micro_step < accumulation_steps:
        raise ValueError("micro_step accumulation araliginda olmali")
    return micro_step == accumulation_steps - 1


def iter_rank_positions(source: Iterable[T], *, rank: int, world_size: int) -> Iterator[T]:
    """Deterministik global akisi rank'ler arasinda cakismadan boler."""

    if world_size < 1:
        raise ValueError("world_size pozitif olmali")
    if not 0 <= rank < world_size:
        raise ValueError("rank world_size araliginda olmali")
    for index, value in enumerate(source):
        if index % world_size == rank:
            yield value


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
