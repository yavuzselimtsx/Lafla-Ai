"""
@Dosya: training/distributed.py
@Aciklama: torchrun ortam kesfi ve DDP process-group yasam dongusu.
@Bilgi: Rank karari standart torchrun degiskenlerinden gelir; GPU urun adina
        veya Kaggle dosya yollarina baglanmaz.
@Uyari: Eksik rank ortami ya da NCCL eksigi tek-process fallback ile gizlenmez.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any, Mapping


_TORCHRUN_KEYS = ("WORLD_SIZE", "RANK", "LOCAL_RANK")


@dataclass(frozen=True)
class DistributedEnvironment:
    """Standart torchrun rank sozlesmesi."""

    world_size: int = 1
    rank: int = 0
    local_rank: int = 0

    @property
    def is_distributed(self) -> bool:
        return self.world_size > 1

    @property
    def is_primary(self) -> bool:
        return self.rank == 0


@dataclass
class DistributedRuntime:
    """Baslatilmis process-group ve rank yardimcilarini tasir."""

    environment: DistributedEnvironment
    backend: str | None
    torch_module: Any
    initialized: bool = False
    _closed: bool = False

    @property
    def world_size(self) -> int:
        return self.environment.world_size

    @property
    def rank(self) -> int:
        return self.environment.rank

    @property
    def local_rank(self) -> int:
        return self.environment.local_rank

    @property
    def is_distributed(self) -> bool:
        return self.environment.is_distributed

    @property
    def is_primary(self) -> bool:
        return self.environment.is_primary

    def barrier(self) -> None:
        if self.initialized:
            self.torch_module.distributed.barrier()

    def mean_tensor(self, tensor):
        """Rank'ler arasinda detached aritmetik ortalama hesaplar."""

        result = tensor.detach().clone()
        if not self.initialized:
            return result
        self.torch_module.distributed.all_reduce(
            result,
            op=self.torch_module.distributed.ReduceOp.SUM,
        )
        return result / self.world_size

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        if self.initialized:
            self.torch_module.distributed.destroy_process_group()
            self.initialized = False


def read_distributed_environment(environ: Mapping[str, str] | None = None) -> DistributedEnvironment:
    """torchrun degiskenlerini eksiksiz ve fail-closed okur."""

    source = os.environ if environ is None else environ
    present = tuple(key for key in _TORCHRUN_KEYS if key in source)
    if not present:
        return DistributedEnvironment()
    if len(present) != len(_TORCHRUN_KEYS):
        raise ValueError("torchrun ortami WORLD_SIZE, RANK ve LOCAL_RANK alanlarini birlikte tasimali")
    try:
        world_size = int(source["WORLD_SIZE"])
        rank = int(source["RANK"])
        local_rank = int(source["LOCAL_RANK"])
    except (TypeError, ValueError) as exc:
        raise ValueError("torchrun rank alanlari tam sayi olmali") from exc
    if world_size < 1:
        raise ValueError("WORLD_SIZE pozitif olmali")
    if not 0 <= rank < world_size:
        raise ValueError("RANK WORLD_SIZE araliginda olmali")
    if local_rank < 0:
        raise ValueError("LOCAL_RANK negatif olamaz")
    return DistributedEnvironment(world_size=world_size, rank=rank, local_rank=local_rank)


def resolve_distributed_backend(configured_backend: str, device_type: str) -> str:
    """Auto backend'i cihaz tipine gore kararlastirir."""

    if configured_backend not in {"auto", "nccl", "gloo"}:
        raise ValueError(f"desteklenmeyen distributed backend: {configured_backend}")
    backend = "nccl" if configured_backend == "auto" and device_type == "cuda" else configured_backend
    if backend == "auto":
        backend = "gloo"
    if backend == "nccl" and device_type != "cuda":
        raise ValueError("NCCL yalniz CUDA ile kullanilabilir")
    return backend


def initialize_distributed(
    *,
    configured_backend: str,
    device_type: str,
    environ: Mapping[str, str] | None = None,
    torch_module: Any | None = None,
) -> DistributedRuntime:
    """Gerekliyse process group kurar, tek process yolunu dokunmadan birakir."""

    environment = read_distributed_environment(environ)
    if torch_module is None:
        try:
            import torch as torch_module
        except ModuleNotFoundError as exc:  # pragma: no cover - runner Torch gerektirir
            raise ModuleNotFoundError("DDP baslatmak icin torch kurulu olmali") from exc
    if not environment.is_distributed:
        return DistributedRuntime(environment, None, torch_module, initialized=False)

    backend = resolve_distributed_backend(configured_backend, device_type)
    distributed = torch_module.distributed
    if not distributed.is_available():
        raise RuntimeError("PyTorch distributed kullanilabilir degil")
    if backend == "nccl":
        if not distributed.is_nccl_available():
            raise RuntimeError("NCCL kullanilabilir degil")
        cuda_device_count = int(torch_module.cuda.device_count())
        if environment.local_rank >= cuda_device_count:
            raise RuntimeError(
                f"LOCAL_RANK CUDA cihaz sayisini asiyor: local_rank={environment.local_rank}, devices={cuda_device_count}"
            )
        torch_module.cuda.set_device(environment.local_rank)
    distributed.init_process_group(backend=backend, init_method="env://")
    return DistributedRuntime(environment, backend, torch_module, initialized=True)
