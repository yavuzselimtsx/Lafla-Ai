"""
@Dosya: post_training/sft_mixture.py
@Aciklama: Thinking-chat ve safety SFT kaynaklarini dengeli, karisik JSONL'e cevirir.
@Yazar: Lafla Gelistirme Ekibi
@Bilgi: Safety/refusal orani kucuk modellerde davranisi kolayca domine eder;
        bu modul safety kayitlarini oran ve template tekrar siniriyle secer.
@Uyari: Bu katman egitim verisi hazirlar; model runtime'ina cevap hardcode etmez.
@Calisma-Semasi: thinking jsonl + safety jsonl -> template-aware sample -> mixed jsonl + report
"""

from __future__ import annotations

import json
import math
import random
import re
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Iterable, Mapping, Sequence

from lafla_ai_core.post_training.thinking_dataset import iter_thinking_jsonl_records
from lafla_ai_core.post_training.thinking_sft import ThinkingSftRecord


_VARIANT_RE = re.compile(r"\b(varyant|variant|variante)\s*\d+\b", flags=re.IGNORECASE)
_NUMBER_RE = re.compile(r"\d+")
_MOJIBAKE_MARKERS = (chr(0x00C3), chr(0x00C4), chr(0x00C5), chr(0x00EF) + chr(0x00BF) + chr(0x00BD), "\ufffd")
DEFAULT_MAX_SAFETY_RATIO = 0.10
DEFAULT_MAX_IDENTITY_RATIO = 0.18
DEFAULT_MAX_UNCERTAINTY_RATIO = 0.25
DEFAULT_MAX_DOMINANT_CATEGORY_RATIO = 0.45


@dataclass(frozen=True)
class SftMixtureSummary:
    """Hazirlanan SFT mixture raporu."""

    thinking_input: int
    safety_input: int
    thinking_selected: int
    safety_selected: int
    total_selected: int
    target_safety_ratio: float
    actual_safety_ratio: float
    max_safety_ratio: float
    max_identity_ratio: float
    max_uncertainty_ratio: float
    max_dominant_category_ratio: float
    safety_template_count: int
    max_safety_per_template: int
    output_jsonl: str
    report_json: str | None = None
    thinking_dropped_by_category: dict[str, int] = field(default_factory=dict)
    quality_ok: bool = True
    quality_findings: tuple[str, ...] = ()
    category_counts: dict[str, int] = field(default_factory=dict)

    def to_json(self) -> str:
        return json.dumps(asdict(self), ensure_ascii=False, indent=2, sort_keys=True)


def build_thinking_sft_mixture(
    *,
    thinking_jsonl: str | Path,
    safety_jsonl: str | Path,
    output_jsonl: str | Path,
    report_json: str | Path | None = None,
    target_safety_ratio: float = 0.08,
    max_safety_per_template: int = 12,
    max_safety_ratio: float = DEFAULT_MAX_SAFETY_RATIO,
    max_identity_ratio: float = DEFAULT_MAX_IDENTITY_RATIO,
    max_uncertainty_ratio: float = DEFAULT_MAX_UNCERTAINTY_RATIO,
    max_dominant_category_ratio: float = DEFAULT_MAX_DOMINANT_CATEGORY_RATIO,
    seed: int = 1337,
) -> SftMixtureSummary:
    """Sohbet agirlikli, safety kontrollu SFT JSONL dosyasi yazar."""

    if not 0.0 <= target_safety_ratio < 0.5:
        raise ValueError("target_safety_ratio 0 ile 0.5 arasinda olmali")
    if max_safety_per_template < 1:
        raise ValueError("max_safety_per_template pozitif olmali")
    for name, ratio in (
        ("max_safety_ratio", max_safety_ratio),
        ("max_identity_ratio", max_identity_ratio),
        ("max_uncertainty_ratio", max_uncertainty_ratio),
        ("max_dominant_category_ratio", max_dominant_category_ratio),
    ):
        if not 0.0 < ratio < 1.0:
            raise ValueError(f"{name} 0 ile 1 arasinda olmali")
    thinking_input_records = [record for _line, record in iter_thinking_jsonl_records(thinking_jsonl)]
    safety_records = [record for _line, record in iter_thinking_jsonl_records(safety_jsonl)]
    if not thinking_input_records:
        raise ValueError("thinking SFT kaynagi bos olamaz")
    thinking_records, thinking_dropped_by_category = _select_balanced_thinking_records(
        thinking_input_records,
        category_final_caps={
            "identity": max_identity_ratio,
            "uncertainty": max_uncertainty_ratio,
        },
        seed=seed,
    )
    target_safety_count = _target_safety_count(len(thinking_records), target_safety_ratio)
    selected_safety = _select_diverse_safety_records(
        safety_records,
        target_count=target_safety_count,
        max_per_template=max_safety_per_template,
        seed=seed,
    )
    rows: list[tuple[str, ThinkingSftRecord]] = [("thinking", record) for record in thinking_records]
    rows.extend(("safety", record) for record in selected_safety)
    random.Random(seed).shuffle(rows)
    quality_ok, quality_findings, category_counts = _assess_mixture_quality(
        rows,
        max_safety_ratio=max_safety_ratio,
        max_identity_ratio=max_identity_ratio,
        max_uncertainty_ratio=max_uncertainty_ratio,
        max_dominant_category_ratio=max_dominant_category_ratio,
    )

    output = Path(output_jsonl)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8") as handle:
        for source, record in rows:
            payload = {
                "_sft_category": _category_for_record(source, record),
                "_sft_source": source,
                "system": record.system,
                "user": record.user,
                "thinking": record.thinking,
                "assistant": record.assistant,
            }
            handle.write(json.dumps(payload, ensure_ascii=False, sort_keys=True) + "\n")

    safety_templates = {_template_key(record) for record in safety_records}
    actual_ratio = len(selected_safety) / len(rows) if rows else 0.0
    summary = SftMixtureSummary(
        thinking_input=len(thinking_input_records),
        safety_input=len(safety_records),
        thinking_selected=len(thinking_records),
        safety_selected=len(selected_safety),
        total_selected=len(rows),
        target_safety_ratio=target_safety_ratio,
        actual_safety_ratio=actual_ratio,
        max_safety_ratio=max_safety_ratio,
        max_identity_ratio=max_identity_ratio,
        max_uncertainty_ratio=max_uncertainty_ratio,
        max_dominant_category_ratio=max_dominant_category_ratio,
        safety_template_count=len(safety_templates),
        max_safety_per_template=max_safety_per_template,
        output_jsonl=str(output),
        report_json=None if report_json is None else str(report_json),
        thinking_dropped_by_category=thinking_dropped_by_category,
        quality_ok=quality_ok,
        quality_findings=quality_findings,
        category_counts=category_counts,
    )
    if report_json is not None:
        report = Path(report_json)
        report.parent.mkdir(parents=True, exist_ok=True)
        report.write_text(summary.to_json(), encoding="utf-8")
    return summary


def _select_diverse_safety_records(
    records: Sequence[ThinkingSftRecord],
    *,
    target_count: int,
    max_per_template: int,
    seed: int,
) -> tuple[ThinkingSftRecord, ...]:
    if target_count <= 0 or not records:
        return ()
    rng = random.Random(seed)
    buckets: dict[str, list[ThinkingSftRecord]] = {}
    for record in records:
        buckets.setdefault(_template_key(record), []).append(record)
    for bucket in buckets.values():
        rng.shuffle(bucket)
    keys = list(buckets)
    rng.shuffle(keys)
    selected: list[ThinkingSftRecord] = []
    selected_per_template = {key: 0 for key in keys}
    while keys and len(selected) < target_count:
        next_keys: list[str] = []
        for key in keys:
            bucket = buckets[key]
            if selected_per_template[key] >= max_per_template:
                continue
            if bucket:
                selected.append(bucket.pop())
                selected_per_template[key] += 1
            if bucket and selected_per_template[key] < max_per_template:
                next_keys.append(key)
            if len(selected) >= target_count:
                break
        keys = next_keys
    return tuple(selected)


def _select_balanced_thinking_records(
    records: Sequence[ThinkingSftRecord],
    *,
    category_final_caps: Mapping[str, float],
    seed: int,
) -> tuple[tuple[ThinkingSftRecord, ...], dict[str, int]]:
    if not records:
        return (), {}
    rng = random.Random(seed)
    total_input = len(records)
    buckets: dict[str, list[ThinkingSftRecord]] = {}
    for record in records:
        buckets.setdefault(_category_for_record("thinking", record), []).append(record)
    selected: list[ThinkingSftRecord] = []
    dropped: dict[str, int] = {}
    for category in sorted(buckets):
        bucket = list(buckets[category])
        rng.shuffle(bucket)
        cap_ratio = category_final_caps.get(category)
        if cap_ratio is None:
            selected.extend(bucket)
            continue
        other_count = total_input - len(bucket)
        cap = max(1, int((cap_ratio / (1.0 - cap_ratio)) * other_count))
        selected.extend(bucket[:cap])
        if len(bucket) > cap:
            dropped[category] = len(bucket) - cap
    rng.shuffle(selected)
    return tuple(selected), dropped


def _target_safety_count(thinking_count: int, ratio: float) -> int:
    if ratio <= 0.0:
        return 0
    return int(math.floor((thinking_count * ratio) / (1.0 - ratio)))


def _template_key(record: ThinkingSftRecord) -> str:
    surface = f"{record.thinking}\n{record.assistant}".casefold()
    surface = _VARIANT_RE.sub("variant #", surface)
    surface = _NUMBER_RE.sub("#", surface)
    return " ".join(surface.split())


def _assess_mixture_quality(
    rows: Sequence[tuple[str, ThinkingSftRecord]],
    *,
    max_safety_ratio: float,
    max_identity_ratio: float,
    max_uncertainty_ratio: float,
    max_dominant_category_ratio: float,
) -> tuple[bool, tuple[str, ...], dict[str, int]]:
    counts: dict[str, int] = {}
    mojibake_count = 0
    for source, record in rows:
        category = _category_for_record(source, record)
        counts[category] = counts.get(category, 0) + 1
        if _contains_mojibake(record):
            mojibake_count += 1
    total = len(rows)
    findings: list[str] = []
    if total == 0:
        return False, ("empty_mixture",), counts
    dominant_category, dominant_count = max(counts.items(), key=lambda item: item[1])
    dominant_ratio = dominant_count / total
    safety_ratio = counts.get("safety", 0) / total
    identity_ratio = counts.get("identity", 0) / total
    uncertainty_ratio = counts.get("uncertainty", 0) / total
    if dominant_ratio > max_dominant_category_ratio:
        findings.append(f"dominant_category_ratio:{dominant_category}:{dominant_ratio:.3f}")
    if safety_ratio > max_safety_ratio:
        findings.append(f"safety_ratio:{safety_ratio:.3f}")
    if identity_ratio > max_identity_ratio:
        findings.append(f"identity_ratio:{identity_ratio:.3f}")
    if uncertainty_ratio > max_uncertainty_ratio:
        findings.append(f"uncertainty_ratio:{uncertainty_ratio:.3f}")
    if total >= 20 and len(counts) < 3:
        findings.append(f"low_category_diversity:{len(counts)}")
    if mojibake_count:
        findings.append(f"mojibake_like_text:{mojibake_count}")
    return not findings, tuple(findings), counts


def _category_for_record(source: str, record: ThinkingSftRecord) -> str:
    if source == "safety":
        return "safety"
    surface = f"{record.system}\n{record.user}\n{record.thinking}\n{record.assistant}".casefold()
    if any(marker in surface for marker in ("laflagpt", "gpt-5.5", "yavuz selim", "parametre", "parameter")):
        return "identity"
    if any(marker in surface for marker in ("2+2", "kaç eder", "yüzde", "toplama", "denklem", "matematik")):
        return "reasoning_math"
    if any(marker in surface for marker in ("başkent", "başkenti", "ankara", "hauptstadt", "capital")):
        return "factual_anchor"
    if any(marker in surface for marker in ("bilmiyorum", "doğrulayam", "kaynak gerekir", "nicht sicher", "weiß ich")):
        return "uncertainty"
    if any(marker in surface for marker in ("instagram", "discord", "bot", "context", "bağlam")):
        return "bot_context"
    return "general_chat"


def _contains_mojibake(record: ThinkingSftRecord) -> bool:
    surface = f"{record.system}\n{record.user}\n{record.thinking}\n{record.assistant}"
    return any(marker in surface for marker in _MOJIBAKE_MARKERS)
