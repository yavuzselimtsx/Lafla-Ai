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
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable, Sequence

from lafla_ai_core.post_training.thinking_dataset import iter_thinking_jsonl_records
from lafla_ai_core.post_training.thinking_sft import ThinkingSftRecord


_VARIANT_RE = re.compile(r"\b(varyant|variant|variante)\s*\d+\b", flags=re.IGNORECASE)
_NUMBER_RE = re.compile(r"\d+")


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
    safety_template_count: int
    max_safety_per_template: int
    output_jsonl: str
    report_json: str | None = None

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
    seed: int = 1337,
) -> SftMixtureSummary:
    """Sohbet agirlikli, safety kontrollu SFT JSONL dosyasi yazar."""

    if not 0.0 <= target_safety_ratio < 0.5:
        raise ValueError("target_safety_ratio 0 ile 0.5 arasinda olmali")
    if max_safety_per_template < 1:
        raise ValueError("max_safety_per_template pozitif olmali")
    thinking_records = [record for _line, record in iter_thinking_jsonl_records(thinking_jsonl)]
    safety_records = [record for _line, record in iter_thinking_jsonl_records(safety_jsonl)]
    if not thinking_records:
        raise ValueError("thinking SFT kaynagi bos olamaz")
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

    output = Path(output_jsonl)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8") as handle:
        for source, record in rows:
            payload = {
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
        thinking_input=len(thinking_records),
        safety_input=len(safety_records),
        thinking_selected=len(thinking_records),
        safety_selected=len(selected_safety),
        total_selected=len(rows),
        target_safety_ratio=target_safety_ratio,
        actual_safety_ratio=actual_ratio,
        safety_template_count=len(safety_templates),
        max_safety_per_template=max_safety_per_template,
        output_jsonl=str(output),
        report_json=None if report_json is None else str(report_json),
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


def _target_safety_count(thinking_count: int, ratio: float) -> int:
    if ratio <= 0.0:
        return 0
    return int(math.floor((thinking_count * ratio) / (1.0 - ratio)))


def _template_key(record: ThinkingSftRecord) -> str:
    surface = f"{record.thinking}\n{record.assistant}".casefold()
    surface = _VARIANT_RE.sub("variant #", surface)
    surface = _NUMBER_RE.sub("#", surface)
    return " ".join(surface.split())
