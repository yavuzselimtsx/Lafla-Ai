"""
@Dosya: post_training/thinking_dataset.py
@Aciklama: Thinking SFT JSONL kayitlarini okur ve veri kapilarindan gecirir.
@Yazar: Lafla Gelistirme Ekibi
@Bilgi: Bu modul egitim dongusunden bagimsizdir; Colab'da zaman kaybetmeden
        once veri sozlesmesini fail-closed dogrular.
@Calisma-Semasi: jsonl -> ThinkingSftRecord -> validation report
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable, Mapping

from lafla_ai_core.post_training.thinking_sft import ThinkingFinding, ThinkingSftRecord, validate_thinking_record


REQUIRED_FIELDS = ("system", "user", "thinking", "assistant")


@dataclass(frozen=True)
class ThinkingDatasetFinding:
    """Thinking SFT veri setindeki tek bulgu."""

    path: str
    line: int
    code: str
    detail: str


@dataclass(frozen=True)
class ThinkingDatasetReport:
    """Thinking SFT veri seti dogrulama raporu."""

    ok: bool
    total_records: int
    findings: tuple[ThinkingDatasetFinding, ...]

    def to_json(self) -> str:
        return json.dumps(asdict(self), ensure_ascii=False, indent=2, sort_keys=True)


def iter_thinking_jsonl_records(path: str | Path) -> Iterable[tuple[int, ThinkingSftRecord]]:
    """JSONL dosyasindan satir numarali ThinkingSftRecord akisi uretir."""

    source = Path(path)
    if not source.exists():
        raise FileNotFoundError(f"thinking SFT jsonl bulunamadi: {source}")
    for line_number, line in enumerate(source.read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        try:
            payload = json.loads(line)
        except json.JSONDecodeError as exc:
            raise ValueError(f"{source}:{line_number} JSONL gecersiz: {exc}") from exc
        if not isinstance(payload, Mapping):
            raise ValueError(f"{source}:{line_number} JSONL kaydi nesne olmali")
        missing = [field for field in REQUIRED_FIELDS if field not in payload]
        if missing:
            raise KeyError(f"{source}:{line_number} eksik alanlar: {missing}")
        yield line_number, ThinkingSftRecord(
            system=str(payload["system"]),
            user=str(payload["user"]),
            thinking=str(payload["thinking"]),
            assistant=str(payload["assistant"]),
        )


def validate_thinking_jsonl_file(path: str | Path, max_thinking_chars: int = 4000) -> ThinkingDatasetReport:
    """Tek thinking SFT JSONL dosyasini dogrular."""

    source = Path(path)
    findings: list[ThinkingDatasetFinding] = []
    total_records = 0
    if not source.exists():
        return ThinkingDatasetReport(
            ok=False,
            total_records=0,
            findings=(ThinkingDatasetFinding(str(source), 0, "file_not_found", "dosya bulunamadi"),),
        )
    for line_number, line in enumerate(source.read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        total_records += 1
        record = _parse_record_line(source, line_number, line, findings)
        if record is None:
            continue
        validation = validate_thinking_record(record, max_thinking_chars=max_thinking_chars)
        findings.extend(_lift_record_findings(source, line_number, validation.findings))
    if total_records == 0:
        findings.append(ThinkingDatasetFinding(str(source), 0, "empty_dataset", "en az bir kayit olmali"))
    return ThinkingDatasetReport(ok=not findings, total_records=total_records, findings=tuple(findings))


def _parse_record_line(
    source: Path,
    line_number: int,
    line: str,
    findings: list[ThinkingDatasetFinding],
) -> ThinkingSftRecord | None:
    try:
        payload = json.loads(line)
    except json.JSONDecodeError as exc:
        findings.append(ThinkingDatasetFinding(str(source), line_number, "invalid_json", str(exc)))
        return None
    if not isinstance(payload, Mapping):
        findings.append(ThinkingDatasetFinding(str(source), line_number, "record_not_object", "JSONL kaydi nesne olmali"))
        return None
    missing = [field for field in REQUIRED_FIELDS if field not in payload]
    if missing:
        findings.append(ThinkingDatasetFinding(str(source), line_number, "missing_field", f"eksik alanlar: {missing}"))
        return None
    return ThinkingSftRecord(
        system=str(payload["system"]),
        user=str(payload["user"]),
        thinking=str(payload["thinking"]),
        assistant=str(payload["assistant"]),
    )


def _lift_record_findings(
    source: Path,
    line_number: int,
    findings: Iterable[ThinkingFinding],
) -> list[ThinkingDatasetFinding]:
    return [ThinkingDatasetFinding(str(source), line_number, finding.code, finding.detail) for finding in findings]
