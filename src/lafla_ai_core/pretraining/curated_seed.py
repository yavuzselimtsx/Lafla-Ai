"""Kaynak metinden tekrarsız, doküman biçimli continued-pretraining seed üretir."""

from __future__ import annotations

import hashlib
import itertools
import json
import re
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable

from lafla_ai_core.tokenizer.quality import validate_clean_text


DEFAULT_OUTPUT_PATH = Path("datasets/pretraining/curated/jsonl/lafla-prompt-curated-pretraining-seed-300.jsonl")
DEFAULT_MANIFEST_PATH = Path("datasets/pretraining/curated/manifests/lafla-prompt-curated-pretraining-seed-300.manifest.json")
DEFAULT_SOURCE_PATH = Path("../../Prompt.md")
DEFAULT_COUNT = 300
DATASET_VERSION = "lafla-curated-pretraining-seed-2026-06-v1"


@dataclass(frozen=True)
class CuratedPretrainingSeedReport:
    output_path: str
    manifest_path: str
    source_path: str
    records_written: int
    candidate_pool_size: int
    sha256: str
    dataset_version: str = DATASET_VERSION
    data_kind: str = "curated_document_pretraining_seed"

    def to_json(self) -> str:
        return json.dumps(asdict(self), ensure_ascii=False, indent=2, sort_keys=True)


@dataclass(frozen=True)
class _SourceBlock:
    block_id: str
    order: int
    domain: str
    heading: str
    text: str


@dataclass(frozen=True)
class _Candidate:
    domain: str
    blocks: tuple[_SourceBlock, ...]
    text: str


def generate_curated_pretraining_seed(
    *,
    source_path: str | Path = DEFAULT_SOURCE_PATH,
    output_path: str | Path = DEFAULT_OUTPUT_PATH,
    manifest_path: str | Path = DEFAULT_MANIFEST_PATH,
    count: int = DEFAULT_COUNT,
) -> CuratedPretrainingSeedReport:
    """Kaynak dışına çıkmadan tekli/ikili/üçlü teknik dokümanlar üretir."""

    if count <= 0:
        raise ValueError("count pozitif olmalı")
    source = Path(source_path)
    if not source.is_file():
        raise FileNotFoundError(f"source bulunamadı: {source}")
    source_bytes = source.read_bytes()
    source_text = validate_clean_text(source_bytes.decode("utf-8"), str(source))
    source_sha256 = hashlib.sha256(source_bytes).hexdigest()
    blocks = tuple(_extract_blocks(source_text))
    if not blocks:
        raise ValueError("source içinden eğitim metni çıkarılamadı")
    candidates = _build_candidates(blocks, source_sha256)
    if count > len(candidates):
        raise ValueError(f"istenen {count} kayıt için yalnız {len(candidates)} benzersiz aday var")
    selected = candidates[:count]

    output = Path(output_path)
    manifest = Path(manifest_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    manifest.parent.mkdir(parents=True, exist_ok=True)
    payloads = tuple(_candidate_payload(candidate, source) for candidate in selected)
    serialized = "".join(json.dumps(payload, ensure_ascii=False, sort_keys=True) + "\n" for payload in payloads)
    output.write_text(serialized, encoding="utf-8", newline="\n")
    output_sha256 = hashlib.sha256(output.read_bytes()).hexdigest()

    texts = [candidate.text for candidate in selected]
    block_usage = Counter(block.block_id for candidate in selected for block in candidate.blocks)
    domain_counts = Counter(candidate.domain for candidate in selected)
    metrics = {
        "exact_duplicate_count": len(texts) - len(set(texts)),
        "max_source_block_reuse_count": max(block_usage.values()),
        "max_source_block_reuse_ratio": round(max(block_usage.values()) / count, 6),
        "source_block_count": len(blocks),
        "source_block_coverage_ratio": round(len(block_usage) / len(blocks), 6),
        "unique_text_ratio": round(len(set(texts)) / len(texts), 6),
    }
    if metrics["exact_duplicate_count"] != 0 or metrics["unique_text_ratio"] != 1.0:
        raise ValueError("curated pretraining çıktısında tekrar bulundu")

    manifest_payload = {
        "allowed_for_pretraining": True,
        "allowed_for_post_training": False,
        "data_kind": "curated_document_pretraining_seed",
        "dataset_version": DATASET_VERSION,
        "records": count,
        "candidate_pool_size": len(candidates),
        "format": "jsonl",
        "fields": ["text", "source", "source_sections", "category"],
        "source": str(source),
        "source_sha256": source_sha256,
        "sha256": output_sha256,
        "domain_counts": dict(sorted(domain_counts.items())),
        "quality_metrics": metrics,
        "quality_policy": {
            "composition": "extractive_source_blocks_only",
            "exact_duplicate_count_max": 0,
            "generated_factual_claims": False,
            "record_style": "document_not_chat",
            "recommended_usage": "low_weight_continued_pretraining_seed",
        },
    }
    manifest.write_text(
        json.dumps(manifest_payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    return CuratedPretrainingSeedReport(
        str(output),
        str(manifest),
        str(source),
        count,
        len(candidates),
        output_sha256,
    )


def _extract_blocks(source_text: str) -> Iterable[_SourceBlock]:
    cleaned = re.sub(r"<!--.*?-->", "", source_text, flags=re.DOTALL)
    current_heading = "Genel İlkeler"
    current_top_level: int | None = None
    body: list[str] = []
    order = 0

    def flush() -> Iterable[_SourceBlock]:
        nonlocal order
        raw = "\n".join(body).strip()
        if not raw:
            return ()
        domain = _domain_for(current_top_level)
        output: list[_SourceBlock] = []
        for paragraph_index, paragraph in enumerate(re.split(r"\n\s*\n", raw), start=1):
            text = paragraph.strip()
            if len(text) < 30 or text.startswith("```") and text.endswith("```") and len(text) < 60:
                continue
            order += 1
            output.append(
                _SourceBlock(
                    block_id=f"b{order:04d}",
                    order=order,
                    domain=domain,
                    heading=current_heading,
                    text=text,
                )
            )
        return tuple(output)

    for line in cleaned.splitlines():
        match = re.match(r"^(#{1,6})\s+(.+?)\s*$", line)
        if match:
            yield from flush()
            body.clear()
            current_heading = match.group(2).strip()
            number_match = re.match(r"^(\d+)(?:\.|\s)", current_heading)
            if len(match.group(1)) <= 2 and number_match:
                current_top_level = int(number_match.group(1))
            continue
        body.append(line)
    yield from flush()


def _domain_for(top_level: int | None) -> str:
    if top_level in {1, 2, 3, 4, 13, 18, 19}:
        return "product_behavior"
    if top_level in {5, 6, 7, 8, 9}:
        return "architecture_security"
    if top_level in {10, 11, 12, 14, 15, 16, 17}:
        return "engineering_quality"
    return "foundation"


def _build_candidates(blocks: tuple[_SourceBlock, ...], source_sha256: str) -> tuple[_Candidate, ...]:
    grouped: defaultdict[str, list[_SourceBlock]] = defaultdict(list)
    for block in blocks:
        grouped[block.domain].append(block)

    singles: list[_Candidate] = []
    combinations: list[_Candidate] = []
    seen_texts: set[str] = set()
    for domain in sorted(grouped):
        domain_blocks = tuple(grouped[domain])
        for size in (1, 2, 3):
            for selected in itertools.combinations(domain_blocks, size):
                text = _compose_document(domain, selected)
                if text in seen_texts:
                    continue
                seen_texts.add(text)
                candidate = _Candidate(domain, selected, text)
                if size == 1:
                    singles.append(candidate)
                else:
                    combinations.append(candidate)
    combinations.sort(key=lambda item: _candidate_rank(item, source_sha256))
    singles.sort(key=lambda item: item.blocks[0].order)
    return tuple((*singles, *combinations))


def _compose_document(domain: str, blocks: tuple[_SourceBlock, ...]) -> str:
    domain_titles = {
        "architecture_security": "Mimari ve Güvenlik İlkeleri",
        "engineering_quality": "Mühendislik Kalitesi İlkeleri",
        "foundation": "Temel Proje İlkeleri",
        "product_behavior": "Ürün ve Davranış İlkeleri",
    }
    parts = [f"# {domain_titles[domain]}"]
    for block in blocks:
        parts.append(f"## {block.heading}\n{block.text}")
    return validate_clean_text("\n\n".join(parts), "curated document")


def _candidate_rank(candidate: _Candidate, source_sha256: str) -> str:
    signature = ":".join(block.block_id for block in candidate.blocks)
    return hashlib.sha256(f"{source_sha256}:{candidate.domain}:{signature}".encode("utf-8")).hexdigest()


def _candidate_payload(candidate: _Candidate, source: Path) -> dict[str, object]:
    return {
        "category": candidate.domain,
        "source": str(source),
        "source_sections": [block.heading for block in candidate.blocks],
        "text": candidate.text,
    }
