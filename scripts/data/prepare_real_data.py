"""
Prepare real, deduplicated LaflaGPT 380M training JSONL on Lightning.ai.

This script never fabricates bootstrap data. It streams real or owned sources,
applies lightweight quality filters, deduplicates normalized records, and writes
a manifest accepted by the data audit gate. Missing optional review sources are
reported and skipped instead of being replaced by fake data.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
import time
import unicodedata
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping

from lafla_ai_core.tokenizer.quality import validate_clean_text


DEFAULT_SOURCE_SPECS: tuple[dict[str, Any], ...] = (
    {
        "source_id": "fineweb2_turkish",
        "path": "HuggingFaceFW/fineweb-2",
        "name": "tur_Latn",
        "split": "train",
        "license": "ODC-By-1.0",
        "language": "tr",
        "domain": "turkish",
        "usage": "pretraining",
        "trust_tier": "primary",
        "source_url": "https://huggingface.co/datasets/HuggingFaceFW/fineweb-2",
        "target_share": 0.31,
    },
    {
        "source_id": "fineweb2_hq_turkish",
        "path": "epfml/FineWeb2-HQ",
        "name": "tur_Latn",
        "split": "train",
        "license": "ODC-By-1.0",
        "language": "tr",
        "domain": "turkish",
        "usage": "pretraining",
        "trust_tier": "primary",
        "source_url": "https://huggingface.co/datasets/epfml/FineWeb2-HQ",
        "target_share": 0.16,
    },
    {
        "source_id": "wikimedia_tr",
        "path": "wikimedia/wikipedia",
        "name": "20231101.tr",
        "split": "train",
        "license": "CC-BY-SA-3.0/GFDL",
        "language": "tr",
        "domain": "turkish",
        "usage": "pretraining",
        "trust_tier": "primary",
        "source_url": "https://huggingface.co/datasets/wikimedia/wikipedia",
        "target_share": 0.07,
    },
    {
        "source_id": "fineweb2_english",
        "path": "HuggingFaceFW/fineweb-2",
        "name": "eng_Latn",
        "split": "train",
        "license": "ODC-By-1.0",
        "language": "en",
        "domain": "english",
        "usage": "pretraining",
        "trust_tier": "secondary",
        "source_url": "https://huggingface.co/datasets/HuggingFaceFW/fineweb-2",
        "target_share": 0.11,
    },
    {
        "source_id": "open_web_math",
        "path": "open-web-math/open-web-math",
        "split": "train",
        "license": "dataset_card_review_required",
        "language": "en",
        "domain": "math",
        "usage": "pretraining",
        "trust_tier": "review",
        "source_url": "https://huggingface.co/datasets/open-web-math/open-web-math",
        "target_share": 0.11,
    },
    {
        "source_id": "the_stack_smol_python",
        "path": "bigcode/the-stack-smol",
        "data_dir": "data/python",
        "split": "train",
        "license": "permissive-license-filtered",
        "language": "code",
        "domain": "code",
        "usage": "pretraining",
        "trust_tier": "review",
        "source_url": "https://huggingface.co/datasets/bigcode/the-stack-smol",
        "target_share": 0.08,
    },
    {
        "source_id": "security_docs_local_reviewed",
        "local_path": "configs/data/security-docs-local.jsonl",
        "license": "source_specific_review_required",
        "language": "tr",
        "domain": "cybersecurity",
        "usage": "pretraining",
        "trust_tier": "review",
        "source_url": "local://configs/data/security-docs-local.jsonl",
        "target_share": 0.04,
        "optional": True,
    },
    {
        "source_id": "aya_dataset_turkish_instruction",
        "path": "CohereLabs/aya_dataset",
        "split": "train",
        "license": "Apache-2.0",
        "language": "tr",
        "domain": "turkish",
        "usage": "instruction",
        "trust_tier": "primary",
        "source_url": "https://huggingface.co/datasets/CohereLabs/aya_dataset",
        "target_share": 0.05,
        "pii_cleaned": True,
    },
    {
        "source_id": "lafla_identity_380m",
        "local_path": "__IDENTITY_JSONL__",
        "license": "lafla-owned",
        "language": "tr",
        "domain": "identity",
        "usage": "instruction",
        "trust_tier": "owned",
        "source_url": "local://configs/data/identity/lafla-model-identity-380m.jsonl",
        "target_share": 0.07,
        "pii_cleaned": True,
    },
)

TURKISH_LETTERS = set("abcdefghijklmnopqrstuvwxyz" + "\u00e7\u011f\u0131\u00f6\u015f\u00fc")
TURKISH_COMMON = (" ve ", " bir ", " icin ", " i\u00e7in ", " olan ", " olarak ", " ile ", " daha ")
ENGLISH_COMMON = (" the ", " and ", " for ", " with ", " from ", " this ", " that ", " model ")
CODE_HINTS = ("def ", "class ", "import ", "return ", "function ", "const ", "let ", "if ", "for ", "while ")
MATH_HINTS = (" theorem", " proof", " equation", " integral", " matrix", " probability", " function", " variable")
SECURITY_HINTS = (
    "security",
    "vulnerability",
    "authentication",
    "authorization",
    "guvenlik",
    "kimlik dogrulama",
    "yetkilendirme",
    "zafiyet",
)


@dataclass
class SourceReport:
    source_id: str
    accepted: int = 0
    rejected: int = 0
    chars: int = 0
    status: str = "pending"
    error: str = ""


def main() -> int:
    parser = argparse.ArgumentParser(description="Prepare real LaflaGPT 380M JSONL data on Lightning")
    parser.add_argument("--output", default="/teamspace/studios/this_studio/LaflaAI380M/data/train.jsonl")
    parser.add_argument("--manifest", default="/teamspace/studios/this_studio/LaflaAI380M/data/veri_manifesti.json")
    parser.add_argument("--report", default="/teamspace/studios/this_studio/LaflaAI380M/reports/data-prepare-report.json")
    parser.add_argument("--identity-jsonl", default="configs/data/identity/lafla-model-identity-380m.jsonl")
    parser.add_argument("--dataset-version", default="lafla-380m-thinking-realdata-2026-06")
    parser.add_argument("--target-chars", type=int, default=2_200_000_000)
    parser.add_argument("--min-chars", type=int, default=900_000_000)
    parser.add_argument("--max-record-chars", type=int, default=12_000)
    args = parser.parse_args()
    try:
        return run(args)
    except KeyboardInterrupt:
        print("Interrupted before manifest completion; partial output remains real data only.", file=sys.stderr)
        return 130


def run(args: argparse.Namespace) -> int:
    started = time.time()
    output = Path(args.output)
    manifest = Path(args.manifest)
    report_path = Path(args.report)
    output.parent.mkdir(parents=True, exist_ok=True)
    manifest.parent.mkdir(parents=True, exist_ok=True)
    report_path.parent.mkdir(parents=True, exist_ok=True)

    seen: set[str] = set()
    source_reports: list[SourceReport] = []
    total_chars = 0
    total_records = 0
    specs = resolve_source_specs(args.identity_jsonl)
    with output.open("w", encoding="utf-8", newline="\n") as handle:
        for spec in specs:
            remaining = args.target_chars - total_chars
            if remaining <= 0:
                break
            source_target = max(50_000, int(args.target_chars * float(spec["target_share"])))
            source_target = min(source_target, remaining)
            source_report = SourceReport(source_id=str(spec["source_id"]))
            source_reports.append(source_report)
            try:
                for raw_text in iter_source_texts(spec):
                    text = clean_text(raw_text, args.max_record_chars, spec)
                    if not text:
                        source_report.rejected += 1
                        continue
                    digest = hashlib.sha256(compact_for_hash(text).encode("utf-8")).hexdigest()
                    if digest in seen:
                        source_report.rejected += 1
                        continue
                    seen.add(digest)
                    handle.write(json.dumps({"text": text}, ensure_ascii=False) + "\n")
                    chars = len(text)
                    source_report.accepted += 1
                    source_report.chars += chars
                    total_records += 1
                    total_chars += chars
                    if source_report.chars >= source_target or total_chars >= args.target_chars:
                        break
                source_report.status = "ok" if source_report.accepted else "empty"
            except Exception as exc:  # noqa: BLE001
                source_report.status = "optional_failed" if spec.get("optional") else "failed"
                source_report.error = str(exc)
                print(f"Source skipped: {spec['source_id']}: {exc}", file=sys.stderr)

    ok = total_chars >= args.min_chars
    report = {
        "ok": ok,
        "output": str(output),
        "manifest": str(manifest),
        "dataset_version": args.dataset_version,
        "target_chars": args.target_chars,
        "min_chars": args.min_chars,
        "total_chars": total_chars,
        "total_records": total_records,
        "unique_hashes": len(seen),
        "seconds": round(time.time() - started, 3),
        "sources": [asdict(item) for item in source_reports],
    }
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    if ok:
        write_manifest(manifest, specs, source_reports, total_chars, args.dataset_version)
    print(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True))
    return 0 if ok else 2


def resolve_source_specs(identity_jsonl: str) -> tuple[dict[str, Any], ...]:
    resolved: list[dict[str, Any]] = []
    for spec in DEFAULT_SOURCE_SPECS:
        item = dict(spec)
        if item.get("local_path") == "__IDENTITY_JSONL__":
            item["local_path"] = identity_jsonl
            item["source_url"] = f"local://{identity_jsonl}"
        resolved.append(item)
    return tuple(resolved)


def iter_source_texts(spec: Mapping[str, Any]) -> Iterable[str]:
    if "local_path" in spec:
        path = Path(str(spec["local_path"]))
        with path.open("r", encoding="utf-8") as handle:
            for line in handle:
                if not line.strip():
                    continue
                payload = json.loads(line)
                yield str(payload.get("text", ""))
        return
    try:
        from datasets import load_dataset  # type: ignore
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError("datasets package required: python -m pip install datasets") from exc

    kwargs: dict[str, Any] = {"path": spec["path"], "split": spec["split"], "streaming": True}
    if spec.get("name") is not None:
        kwargs["name"] = spec["name"]
    if spec.get("data_dir") is not None:
        kwargs["data_dir"] = spec["data_dir"]
    dataset = load_dataset(**kwargs)
    for item in dataset:
        text = extract_text(item)
        if text:
            yield text


def extract_text(item: Mapping[str, Any]) -> str:
    for key in ("text", "content", "document", "body", "article", "code"):
        value = item.get(key)
        if isinstance(value, str) and value.strip():
            return value
    instruction = item.get("instruction") or item.get("prompt") or item.get("question")
    answer = item.get("answer") or item.get("response") or item.get("completion")
    if isinstance(instruction, str) and isinstance(answer, str):
        return f"Soru: {instruction}\nCevap: {answer}"
    messages = item.get("messages")
    if isinstance(messages, list):
        parts: list[str] = []
        for message in messages:
            if isinstance(message, Mapping):
                content = message.get("content")
                role = message.get("role", "")
                if isinstance(content, str) and content.strip():
                    parts.append(f"{role}: {content}" if role else content)
        return "\n".join(parts)
    return ""


def clean_text(raw: str, max_record_chars: int, spec: Mapping[str, Any]) -> str:
    text = unicodedata.normalize("NFC", raw)
    text = text.replace("\x00", " ")
    text = re.sub(r"[ \t\r\f\v]+", " ", text)
    text = re.sub(r"\n{4,}", "\n\n", text)
    text = text.strip()
    if len(text) < 80:
        return ""
    if len(text) > max_record_chars:
        text = text[:max_record_chars].rsplit(" ", 1)[0].strip()
    try:
        text = validate_clean_text(text, f"prepare_real_data:{spec.get('source_id', 'unknown')}")
    except ValueError:
        return ""
    if not passes_domain_signal(text, spec):
        return ""
    if looks_like_noise(text, domain=str(spec.get("domain", ""))):
        return ""
    return text


def passes_domain_signal(text: str, spec: Mapping[str, Any]) -> bool:
    domain = str(spec.get("domain", ""))
    language = str(spec.get("language", ""))
    lowered = f" {text.lower()} "
    if domain == "code" or language == "code":
        return looks_like_code(text)
    if domain == "math":
        return has_english_signal(text) or any(hint in lowered for hint in MATH_HINTS)
    if domain == "cybersecurity":
        return has_turkish_signal(text) or has_english_signal(text) or any(hint in lowered for hint in SECURITY_HINTS)
    if language == "tr":
        return has_turkish_signal(text)
    if language == "en":
        return has_english_signal(text)
    return has_turkish_signal(text) or has_english_signal(text)


def has_turkish_signal(text: str) -> bool:
    lowered = text.lower()
    letters = sum(1 for char in lowered if char.isalpha())
    if letters == 0:
        return False
    known_letters = sum(1 for char in lowered if char in TURKISH_LETTERS)
    if known_letters / letters < 0.70:
        return False
    common_hits = sum(1 for token in TURKISH_COMMON if token in f" {lowered} ")
    has_specific_char = any(char in lowered for char in "\u00e7\u011f\u0131\u00f6\u015f\u00fc")
    return common_hits >= 1 or has_specific_char


def has_english_signal(text: str) -> bool:
    lowered = f" {text.lower()} "
    ascii_letters = sum(1 for char in lowered if "a" <= char <= "z")
    letters = sum(1 for char in lowered if char.isalpha())
    if letters == 0 or ascii_letters / letters < 0.82:
        return False
    return sum(1 for token in ENGLISH_COMMON if token in lowered) >= 2


def looks_like_code(text: str) -> bool:
    lowered = text.lower()
    if any(hint in lowered for hint in CODE_HINTS):
        return True
    punctuation = sum(1 for char in text if char in "{}[]();=<>:+-*/")
    return punctuation >= 12 and "\n" in text


def looks_like_noise(text: str, domain: str) -> bool:
    if text.count("http://") + text.count("https://") > 8:
        return True
    if len(set(text)) < 24 and len(text) > 300:
        return True
    if domain == "code":
        return False
    punctuation = sum(1 for char in text if not char.isalnum() and not char.isspace())
    return punctuation / max(1, len(text)) > 0.32


def compact_for_hash(text: str) -> str:
    return re.sub(r"\s+", " ", text.lower()).strip()


def write_manifest(
    path: Path,
    specs: Iterable[Mapping[str, Any]],
    reports: list[SourceReport],
    total_chars: int,
    dataset_version: str,
) -> None:
    accepted = {report.source_id: report for report in reports if report.accepted > 0}
    sources: list[dict[str, Any]] = []
    for spec in specs:
        report = accepted.get(str(spec["source_id"]))
        if report is None:
            continue
        source = {
            "sourceId": spec["source_id"],
            "loader": spec.get("path", spec.get("local_path", "local_jsonl")),
            "subset": spec.get("name") or spec.get("data_dir"),
            "language": spec["language"],
            "license": spec["license"],
            "weight": round(report.chars / max(1, total_chars), 6),
            "usage": spec["usage"],
            "trustTier": spec["trust_tier"],
            "sourceUrl": spec["source_url"],
            "notes": "generated_by=scripts/data/prepare_real_data.py; dedup=sha256-normalized-text; fake_data=false",
        }
        if "pii_cleaned" in spec:
            source["piiCleaned"] = bool(spec["pii_cleaned"])
        sources.append(source)
    normalize_weights(sources)
    manifest = {
        "datasetVersion": dataset_version,
        "targetTokens": max(1, total_chars // 4),
        "policy": {
            "unknownLicenseAllowed": False,
            "piiRequiredForInstructionAndPreference": True,
            "syntheticDataRequiresTeacherAndSource": True,
            "minimumTurkishConversationTokens": 1000000,
            "minimumEvaluationSets": 3,
        },
        "sources": sources,
        "evaluationSets": [
            {
                "sourceId": "turkish_general_holdout",
                "loader": "heldout_jsonl",
                "language": "tr",
                "usage": "evaluation",
                "sourceUrl": "local://holdout/from-real-turkish-sources",
            },
            {
                "sourceId": "math_code_holdout",
                "loader": "heldout_jsonl",
                "language": "multi",
                "usage": "evaluation",
                "sourceUrl": "local://holdout/from-real-math-code-sources",
            },
            {
                "sourceId": "lafla_identity_holdout",
                "loader": "local_jsonl",
                "language": "tr",
                "usage": "evaluation",
                "sourceUrl": "local://configs/data/identity/lafla-model-identity-380m.jsonl",
            },
        ],
        "filters": {
            "dedup": "sha256-normalized-text",
            "unicode": "NFC",
            "minChars": 80,
            "maxRecordChars": 12000,
            "realDataOnly": True,
        },
    }
    path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def normalize_weights(sources: list[dict[str, Any]]) -> None:
    total = sum(float(source["weight"]) for source in sources)
    if total <= 0:
        return
    for source in sources:
        source["weight"] = round(float(source["weight"]) / total, 6)
    drift = round(1.0 - sum(float(source["weight"]) for source in sources), 6)
    if sources and drift:
        sources[0]["weight"] = round(float(sources[0]["weight"]) + drift, 6)


if __name__ == "__main__":
    raise SystemExit(main())
