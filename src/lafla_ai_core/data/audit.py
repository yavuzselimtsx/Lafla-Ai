"""
@Dosya: data/audit.py
@Aciklama: Veri manifestini lisans, kaynak, PII, eval ve agirlik kapilarindan
            geciren audit katmanini uygular.
@Yazar: Lafla Gelistirme Ekibi
@Bilgi: OLMo/GPT-NeoX kaynak karisimi fikri burada Lafla'nin lisans ve guven
        politikasiyla birlestirilir.
@Uyari: Audit basarisizsa tokenizer ve egitim baslamaz.
@Calisma-Semasi: manifest -> checks -> AuditReport -> pass/fail
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from typing import Iterable

from .manifest import DatasetManifest, SourceSpec


UNKNOWN_LICENSE_MARKERS = {"", "unknown", "belirsiz", "none", "n/a"}
INSTRUCTION_USAGES = {"instruction", "preference"}


@dataclass(frozen=True)
class AuditFinding:
    """Tek audit bulgusunu tasir."""

    code: str
    message: str
    source_id: str | None = None


@dataclass(frozen=True)
class AuditReport:
    """Manifest audit sonucunu tasir."""

    ok: bool
    dataset_version: str
    source_count: int
    evaluation_set_count: int
    total_weight: float
    findings: tuple[AuditFinding, ...]

    def to_json(self) -> str:
        """Raporu JSON olarak dondurur."""

        return json.dumps(asdict(self), ensure_ascii=False, indent=2, sort_keys=True)


def audit_manifest(manifest: DatasetManifest) -> AuditReport:
    """Manifest icin fail-closed audit kosar."""

    findings: list[AuditFinding] = []
    if not manifest.dataset_version.strip():
        findings.append(AuditFinding("dataset_version_missing", "datasetVersion bos olamaz"))
    if manifest.target_tokens <= 0:
        findings.append(AuditFinding("target_tokens_invalid", "targetTokens pozitif olmali"))
    _audit_sources(manifest.sources, manifest, findings)
    _audit_evals(manifest, findings)
    total_weight = round(sum(source.weight for source in manifest.sources), 6)
    if manifest.sources and abs(total_weight - 1.0) > 0.001:
        findings.append(AuditFinding("source_weight_sum_invalid", f"source weights toplami 1.0 olmali, bulundu {total_weight}"))
    return AuditReport(
        ok=not findings,
        dataset_version=manifest.dataset_version,
        source_count=len(manifest.sources),
        evaluation_set_count=len(manifest.evaluation_sets),
        total_weight=total_weight,
        findings=tuple(findings),
    )


def _audit_sources(sources: Iterable[SourceSpec], manifest: DatasetManifest, findings: list[AuditFinding]) -> None:
    """Kaynak seviyesindeki audit kapilarini uygular."""

    seen: set[str] = set()
    for source in sources:
        if source.source_id in seen:
            findings.append(AuditFinding("duplicate_source_id", "sourceId tekrar ediyor", source.source_id))
        seen.add(source.source_id)
        if source.weight < 0:
            findings.append(AuditFinding("negative_source_weight", "source weight negatif olamaz", source.source_id))
        if not source.source_url.strip():
            findings.append(AuditFinding("source_url_missing", "sourceUrl zorunlu", source.source_id))
        if not manifest.policy.unknown_license_allowed and source.license.strip().lower() in UNKNOWN_LICENSE_MARKERS:
            findings.append(AuditFinding("unknown_license", "lisansi belirsiz kaynak reddedildi", source.source_id))
        if source.usage in INSTRUCTION_USAGES and manifest.policy.pii_required_for_instruction_and_preference:
            if source.pii_cleaned is not True:
                findings.append(AuditFinding("pii_cleaning_required", "instruction/preference kaynagi piiCleaned=true tasimali", source.source_id))
        if source.loader.lower().startswith("synthetic") and manifest.policy.synthetic_data_requires_teacher_and_source:
            if "teacher" not in source.notes.lower() or not source.source_url:
                findings.append(AuditFinding("synthetic_teacher_required", "sentetik veri teacher ve source belirtmeli", source.source_id))


def _audit_evals(manifest: DatasetManifest, findings: list[AuditFinding]) -> None:
    """Eval seti kapilarini uygular."""

    if len(manifest.evaluation_sets) < manifest.policy.minimum_evaluation_sets:
        findings.append(
            AuditFinding(
                "minimum_eval_sets",
                f"en az {manifest.policy.minimum_evaluation_sets} eval seti gerekli",
            )
        )
    for eval_set in manifest.evaluation_sets:
        if not eval_set.source_url.strip():
            findings.append(AuditFinding("eval_source_url_missing", "eval sourceUrl zorunlu", eval_set.source_id))

