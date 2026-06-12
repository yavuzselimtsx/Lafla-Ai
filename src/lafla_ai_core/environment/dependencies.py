"""
@Dosya: environment/dependencies.py
@Aciklama: Egitim oncesi Python paketlerinin varligini raporlar.
@Yazar: Lafla Gelistirme Ekibi
@Bilgi: Colab'da saatler kaybetmemek icin eksik paketler egitim baslamadan
        yakalanir.
@Uyari: Bu modul paket kurmaz; yalnizca eksigi acik raporlar.
@Calisma-Semasi: requirements -> importlib probe -> DependencyReport
"""

from __future__ import annotations

import importlib.util
import json
from dataclasses import asdict, dataclass
from typing import Iterable


@dataclass(frozen=True)
class ModuleRequirement:
    """Bir Python modul gereksinimini tasir."""

    module: str
    purpose: str
    install_hint: str = ""


@dataclass(frozen=True)
class DependencyReport:
    """Bagimlilik kontrol sonucunu tasir."""

    ok: bool
    present: tuple[ModuleRequirement, ...]
    missing: tuple[ModuleRequirement, ...]

    def to_json(self) -> str:
        return json.dumps(asdict(self), ensure_ascii=False, indent=2, sort_keys=True)


def check_required_modules(requirements: Iterable[ModuleRequirement]) -> DependencyReport:
    """Verilen modullerin import edilebilir olup olmadigini kontrol eder."""

    present: list[ModuleRequirement] = []
    missing: list[ModuleRequirement] = []
    for requirement in requirements:
        if importlib.util.find_spec(requirement.module) is None:
            missing.append(requirement)
        else:
            present.append(requirement)
    return DependencyReport(ok=not missing, present=tuple(present), missing=tuple(missing))


def colab_training_requirements(optimizer: str = "adamw8bit", accelerator: str = "cuda") -> tuple[ModuleRequirement, ...]:
    """Colab egitimi icin gerekli Python modullerini dondurur."""

    requirements = [
        ModuleRequirement("torch", "model ve training dongusu", "Colab runtime PyTorch GPU imaji kullanin"),
        ModuleRequirement("tokenizers", "BPE tokenizer egitimi ve encode", "pip install -r requirements/colab.txt"),
        ModuleRequirement("yaml", "YAML config okuma", "pip install -r requirements/colab.txt"),
    ]
    if accelerator == "xla":
        requirements.append(ModuleRequirement("torch_xla", "TPU/XLA PyTorch backend", "pip install torch torch_xla[tpu]"))
    if optimizer == "adamw8bit":
        requirements.append(ModuleRequirement("bitsandbytes", "8-bit AdamW optimizer", "pip install -r requirements/colab.txt"))
    return tuple(requirements)
