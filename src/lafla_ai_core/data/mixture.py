"""
@Dosya: data/mixture.py
@Aciklama: Egitim veri kaynaklari icin normalize edilmis mixture ve sample planlari.
@Yazar: Lafla Gelistirme Ekibi
@Bilgi: GPT-NeoX tarzi agirlik normalizasyonu ve kucuk overbuild payi veri
        eksilmesini sessizce release'e tasimaz.
@Uyari: Duplicate source id, sifir/negatif agirlik ve bos usage fail-closed davranir.
"""

from __future__ import annotations

from dataclasses import dataclass
from math import ceil
from typing import Sequence

from lafla_ai_core.data.manifest import SourceSpec


@dataclass(frozen=True)
class MixtureSource:
    """Manifest veya config tarafindan bildirilen ham veri kaynagi agirligi."""

    source_id: str
    usage: str
    weight: float


@dataclass(frozen=True)
class NormalizedMixtureSource:
    """Toplam agirligi 1.0 olacak sekilde normalize edilmis kaynak."""

    source_id: str
    usage: str
    weight: float
    normalized_weight: float


@dataclass(frozen=True)
class MixturePlanEntry:
    """Tek kaynak icin hesaplanan sample butcesi."""

    source_id: str
    usage: str
    weight: float
    normalized_weight: float
    sample_budget: int


@dataclass(frozen=True)
class MixturePlan:
    """Belirli usage icin egitim veri karisim plani."""

    usage: str
    total_samples: int
    overbuild_factor: float
    entries: tuple[MixturePlanEntry, ...]


def normalize_weights(sources: Sequence[MixtureSource]) -> tuple[NormalizedMixtureSource, ...]:
    """Kaynak agirliklarini toplam 1.0 olacak sekilde normalize eder."""

    if not sources:
        raise ValueError("en az bir mixture source gerekli")
    validated = tuple(_validate_source(source) for source in sources)
    _ensure_unique_source_ids(validated)
    total_weight = sum(source.weight for source in validated)
    if total_weight <= 0:
        raise ValueError("toplam source agirligi pozitif olmali")
    return tuple(
        NormalizedMixtureSource(
            source_id=source.source_id,
            usage=source.usage,
            weight=source.weight,
            normalized_weight=source.weight / total_weight,
        )
        for source in validated
    )


def build_mixture_plan(
    sources: Sequence[MixtureSource],
    *,
    usage: str,
    total_samples: int,
    overbuild_factor: float = 1.005,
) -> MixturePlan:
    """Usage'a gore filtrelenmis normalize kaynaklardan sample butcesi uretir."""

    usage_key = str(usage).strip()
    if not usage_key:
        raise ValueError("usage bos olamaz")
    if total_samples <= 0:
        raise ValueError("total_samples pozitif olmali")
    if overbuild_factor < 1.0:
        raise ValueError("overbuild_factor en az 1.0 olmali")
    selected = tuple(source for source in sources if str(source.usage).strip() == usage_key)
    if not selected:
        raise ValueError(f"usage icin mixture source yok: {usage_key}")
    normalized = normalize_weights(selected)
    entries = tuple(
        MixturePlanEntry(
            source_id=source.source_id,
            usage=source.usage,
            weight=source.weight,
            normalized_weight=source.normalized_weight,
            sample_budget=ceil(total_samples * source.normalized_weight * overbuild_factor),
        )
        for source in normalized
    )
    return MixturePlan(
        usage=usage_key,
        total_samples=total_samples,
        overbuild_factor=overbuild_factor,
        entries=entries,
    )


def mixture_sources_from_manifest(sources: Sequence[SourceSpec]) -> tuple[MixtureSource, ...]:
    """Dataset manifest SourceSpec kayitlarini MixtureSource sozlesmesine indirger."""

    return tuple(MixtureSource(source.source_id, source.usage, source.weight) for source in sources)


def _validate_source(source: MixtureSource) -> MixtureSource:
    source_id = str(source.source_id).strip()
    usage = str(source.usage).strip()
    weight = float(source.weight)
    if not source_id:
        raise ValueError("source_id bos olamaz")
    if not usage:
        raise ValueError("usage bos olamaz")
    if weight <= 0:
        raise ValueError(f"source agirligi pozitif olmali: {source_id}")
    return MixtureSource(source_id=source_id, usage=usage, weight=weight)


def _ensure_unique_source_ids(sources: Sequence[MixtureSource]) -> None:
    seen: set[str] = set()
    for source in sources:
        if source.source_id in seen:
            raise ValueError(f"duplicate source_id: {source.source_id}")
        seen.add(source.source_id)
