"""
@Dosya: data/manifest.py
@Aciklama: LaflaAi-Core veri manifesti icin typed sozlesmeleri tanimlar.
@Yazar: Lafla Gelistirme Ekibi
@Bilgi: Manifest egitimin kaynak, lisans, PII ve eval guven siniridir. Ham JSON
        egitim koduna dogrudan verilmez.
@Uyari: Lisans veya kaynak URL belirsizligi fail-closed davranir.
@Calisma-Semasi: json -> DatasetManifest -> audit -> shard plan
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping

from lafla_ai_core.config.schema import ConfigError


@dataclass(frozen=True)
class SourceSpec:
    """Tek veri kaynagini temsil eder."""

    source_id: str
    loader: str
    subset: str | None
    language: str
    license: str
    weight: float
    usage: str
    trust_tier: str
    source_url: str
    pii_cleaned: bool | None = None
    notes: str = ""

    @classmethod
    def from_mapping(cls, data: Mapping[str, Any]) -> "SourceSpec":
        """Mapping verisinden SourceSpec uretir."""

        return cls(
            source_id=str(data["sourceId"]),
            loader=str(data["loader"]),
            subset=None if data.get("subset") is None else str(data.get("subset")),
            language=str(data.get("language", "")),
            license=str(data.get("license", "")),
            weight=float(data.get("weight", 0.0)),
            usage=str(data.get("usage", "")),
            trust_tier=str(data.get("trustTier", "")),
            source_url=str(data.get("sourceUrl", "")),
            pii_cleaned=None if "piiCleaned" not in data else bool(data.get("piiCleaned")),
            notes=str(data.get("notes", "")),
        )


@dataclass(frozen=True)
class EvaluationSpec:
    """Tek degerlendirme kaynagini temsil eder."""

    source_id: str
    loader: str
    language: str
    usage: str
    source_url: str

    @classmethod
    def from_mapping(cls, data: Mapping[str, Any]) -> "EvaluationSpec":
        """Mapping verisinden EvaluationSpec uretir."""

        return cls(
            source_id=str(data["sourceId"]),
            loader=str(data["loader"]),
            language=str(data.get("language", "")),
            usage=str(data.get("usage", "")),
            source_url=str(data.get("sourceUrl", "")),
        )


@dataclass(frozen=True)
class ManifestPolicy:
    """Manifest icindeki veri politikasi alanlarini tasir."""

    unknown_license_allowed: bool
    pii_required_for_instruction_and_preference: bool
    synthetic_data_requires_teacher_and_source: bool
    minimum_turkish_conversation_tokens: int
    minimum_evaluation_sets: int

    @classmethod
    def from_mapping(cls, data: Mapping[str, Any]) -> "ManifestPolicy":
        """Mapping verisinden ManifestPolicy uretir."""

        return cls(
            unknown_license_allowed=bool(data.get("unknownLicenseAllowed", False)),
            pii_required_for_instruction_and_preference=bool(data.get("piiRequiredForInstructionAndPreference", True)),
            synthetic_data_requires_teacher_and_source=bool(data.get("syntheticDataRequiresTeacherAndSource", True)),
            minimum_turkish_conversation_tokens=int(data.get("minimumTurkishConversationTokens", 0)),
            minimum_evaluation_sets=int(data.get("minimumEvaluationSets", 0)),
        )


@dataclass(frozen=True)
class DatasetManifest:
    """Lafla egitim veri manifesti kok nesnesi."""

    dataset_version: str
    target_tokens: int
    policy: ManifestPolicy
    sources: tuple[SourceSpec, ...] = field(default_factory=tuple)
    evaluation_sets: tuple[EvaluationSpec, ...] = field(default_factory=tuple)
    filters: Mapping[str, Any] = field(default_factory=dict)

    @classmethod
    def from_mapping(cls, data: Mapping[str, Any]) -> "DatasetManifest":
        """Mapping verisinden DatasetManifest uretir."""

        policy_data = data.get("policy")
        if not isinstance(policy_data, Mapping):
            raise ConfigError("manifest policy zorunlu")
        source_items = data.get("sources", ())
        eval_items = data.get("evaluationSets", ())
        if not isinstance(source_items, list):
            raise ConfigError("sources liste olmali")
        if not isinstance(eval_items, list):
            raise ConfigError("evaluationSets liste olmali")
        return cls(
            dataset_version=str(data["datasetVersion"]),
            target_tokens=int(data["targetTokens"]),
            policy=ManifestPolicy.from_mapping(policy_data),
            sources=tuple(SourceSpec.from_mapping(item) for item in source_items),
            evaluation_sets=tuple(EvaluationSpec.from_mapping(item) for item in eval_items),
            filters=data.get("filters", {}) if isinstance(data.get("filters", {}), Mapping) else {},
        )


def load_manifest(path: str | Path) -> DatasetManifest:
    """JSON manifest dosyasini yukler ve typed manifest dondurur."""

    manifest_path = Path(path)
    if not manifest_path.exists():
        raise ConfigError(f"manifest bulunamadi: {manifest_path}")
    data = json.loads(manifest_path.read_text(encoding="utf-8"))
    if not isinstance(data, Mapping):
        raise ConfigError("manifest kok nesnesi mapping olmali")
    return DatasetManifest.from_mapping(data)

