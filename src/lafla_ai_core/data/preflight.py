"""
@Dosya: data/preflight.py
@Aciklama: Gercek pretraining JSONL dosyalarini GPU baslamadan once tarar.
@Bilgi: Dogrulama streaming calisir; buyuk corpus tumden RAM'e alinmaz.
@Uyari: Tek bozuk veya bos dosya egitimi fail-closed durdurur.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from typing import Sequence

from lafla_ai_core.data.packing import iter_jsonl_texts
from lafla_ai_core.data.routing import assert_pretraining_inputs


@dataclass(frozen=True)
class PretrainingDataFileReport:
    path: str
    records: int


@dataclass(frozen=True)
class PretrainingDataReport:
    ok: bool
    total_records: int
    files: tuple[PretrainingDataFileReport, ...]

    def to_json(self) -> str:
        return json.dumps(asdict(self), ensure_ascii=False, indent=2, sort_keys=True)


def validate_pretraining_jsonl(paths: Sequence[str]) -> PretrainingDataReport:
    """Tum kayitlari parse, format ve temiz metin kurallariyla dogrular."""

    if not paths:
        raise ValueError("en az bir pretraining JSONL dosyasi gerekli")
    assert_pretraining_inputs(paths)
    files: list[PretrainingDataFileReport] = []
    total_records = 0
    for path in paths:
        records = sum(1 for _ in iter_jsonl_texts((path,)))
        if records < 1:
            raise ValueError(f"pretraining JSONL bos: {path}")
        files.append(PretrainingDataFileReport(path=path, records=records))
        total_records += records
    return PretrainingDataReport(ok=True, total_records=total_records, files=tuple(files))
