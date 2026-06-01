"""
Veri manifestini lisans, kaynak güveni ve kullanım amacı açısından doğrular.

İlk eğitimde halüsinasyon riskini azaltmanın en ucuz yolu kötü veriyi içeri
almamaktır. Bu doğrulayıcı; bilinmeyen lisans, kaynaksız veri, PII temizliği
eksik instruction/preference verisi ve yetersiz değerlendirme setini reddeder.
"""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any


ACCEPTED_LICENSE_PREFIXES = (
    "Apache-2.0",
    "ODC-By-1.0",
    "ODC-By-1.0-with-subset-review",
    "lafla-owned",
    "only-approved-local-licenses",
)

REQUIRES_PII_CLEANING = {"instruction", "preference"}


@dataclass(frozen=True)
class ManifestIssue:
    """Manifest içinde bulunan tek kalite veya güvenlik sorunu."""

    code: str
    source_id: str
    message: str


def load_manifest(path: Path) -> dict[str, Any]:
    """UTF-8 JSON manifestini okur."""

    return json.loads(path.read_text(encoding="utf-8"))


def validate_manifest(manifest: dict[str, Any]) -> list[ManifestIssue]:
    """Manifest sorunlarını liste olarak döndürür; boş liste geçer demektir."""

    issues: list[ManifestIssue] = []
    sources = manifest.get("sources")
    if not isinstance(sources, list) or not sources:
        return [ManifestIssue("sources_missing", "manifest", "sources listesi boş")]
    total_weight = 0.0
    for source in sources:
        source_id = require_source_id(source)
        total_weight += float(source.get("weight", 0.0))
        issues.extend(validate_source(source_id, source))
    if abs(total_weight - 1.0) > 0.001:
        issues.append(ManifestIssue("weight_sum_invalid", "manifest", f"kaynak ağırlıkları 1.0 değil: {total_weight:.3f}"))
    evaluation_sets = manifest.get("evaluationSets")
    minimum_eval_sets = int(manifest.get("policy", {}).get("minimumEvaluationSets", 0))
    if not isinstance(evaluation_sets, list) or len(evaluation_sets) < minimum_eval_sets:
        issues.append(ManifestIssue("evaluation_sets_missing", "manifest", "değerlendirme seti sayısı yetersiz"))
    return issues


def validate_source(source_id: str, source: dict[str, Any]) -> list[ManifestIssue]:
    """Tek veri kaynağı için doğrulama kurallarını uygular."""

    issues: list[ManifestIssue] = []
    license_name = str(source.get("license", "unknown"))
    if not license_name.startswith(ACCEPTED_LICENSE_PREFIXES):
        issues.append(ManifestIssue("license_rejected", source_id, f"lisans kabul edilmiyor: {license_name}"))
    source_url = str(source.get("sourceUrl", ""))
    if not source_url.startswith(("https://", "file://", "s3://", "gs://")):
        issues.append(ManifestIssue("source_url_missing", source_id, "geçerli sourceUrl yok"))
    usage = str(source.get("usage", ""))
    if usage in REQUIRES_PII_CLEANING and source.get("piiCleaned") is not True:
        issues.append(ManifestIssue("pii_required", source_id, "instruction/preference için PII temizliği zorunlu"))
    if float(source.get("weight", 0.0)) <= 0:
        issues.append(ManifestIssue("weight_invalid", source_id, "kaynak ağırlığı pozitif olmalı"))
    if str(source.get("trustTier", "")) == "secondary_review_required" and "subset-review" not in license_name:
        issues.append(ManifestIssue("subset_review_missing", source_id, "ikincil veri için alt lisans incelemesi işaretlenmeli"))
    return issues


def require_source_id(source: object) -> str:
    if not isinstance(source, dict):
        raise ValueError("source must be object")
    source_id = source.get("sourceId")
    if not isinstance(source_id, str) or not source_id.strip():
        raise ValueError("sourceId missing")
    return source_id.strip()


def assert_manifest_is_release_ready(path: Path) -> None:
    """Manifest üretim eğitimine hazır değilse açıklayıcı hata fırlatır."""

    issues = validate_manifest(load_manifest(path))
    if issues:
        detail = "; ".join(f"{issue.source_id}:{issue.code}" for issue in issues)
        raise ValueError(f"veri manifesti geçemedi: {detail}")
