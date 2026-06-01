"""
 _        _    _____ _        _
| |      / \\  |  ___| |      / \\
| |     / _ \\ | |_  | |     / _ \\
| |___ / ___ \\|  _| | |___ / ___ \\
|_____/_/   \\_\\_|   |_____/_/   \\_\\

@Dosya: veri_hazirla.py
@Açıklama: Lafla AI eğitim verisini lisans, dil ve PII filtreleriyle JSONL shardlara hazırlar.
@Yazar: Lafla Geliştirme Ekibi
@Bilgi: Kimlik öğretim kayıtları ayrı kimlik yapılandırmasından üretilir.
@Uyarı: Bu dosya veri lisansı doğrulanmamış kaynakları üretim eğitimine onaylamaz.
@Çalışma-Şeması: manifest -> streaming source -> clean/filter/dedupe -> jsonl shard
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
from pathlib import Path
from typing import Iterable

from datasets import load_dataset
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parent))

from lafla_persona import iter_instruction_records, load_identity


PII_PATTERNS = [
    re.compile(r"\b[\w.+-]+@[\w-]+\.[\w.-]+\b"),
    re.compile(r"\b(?:\+90|0)?\s?5\d{2}\s?\d{3}\s?\d{2}\s?\d{2}\b"),
    re.compile(r"\b\d{11}\b"),
]


def main() -> None:
    """Komut satırından veri hazırlama akışını çalıştırır."""

    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--identity-config", default=str(Path(__file__).resolve().parents[1] / "konfigurasyon" / "lafla-ai-kimlik.json"))
    parser.add_argument("--max-records", type=int, default=10000)
    args = parser.parse_args()

    manifest = json.loads(Path(args.manifest).read_text(encoding="utf-8"))
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)

    seen: set[str] = set()
    written = 0
    shard_path = output / "lafla-tr-shard-00000.jsonl"
    with shard_path.open("w", encoding="utf-8") as shard:
        for record in iter_identity_records(args.identity_config):
            shard.write(json.dumps(record, ensure_ascii=False) + "\n")
            written += 1
        for source in manifest["sources"]:
            for text in iter_source_texts(source):
                cleaned = clean_text(text)
                if not accepted(cleaned, manifest["filters"]):
                    continue
                digest = hashlib.sha256(cleaned.encode("utf-8")).hexdigest()
                if digest in seen:
                    continue
                seen.add(digest)
                record = {
                    "id": digest,
                    "source_id": source["sourceId"],
                    "language": source["language"],
                    "license": source["license"],
                    "usage": source["usage"],
                    "text": cleaned,
                }
                shard.write(json.dumps(record, ensure_ascii=False) + "\n")
                written += 1
                if written >= args.max_records:
                    print(json.dumps({"written": written, "output": str(shard_path)}, ensure_ascii=False))
                    return

    print(json.dumps({"written": written, "output": str(shard_path)}, ensure_ascii=False))


def iter_source_texts(source: dict) -> Iterable[str]:
    """Manifest kaynağını metin akışına çevirir."""

    if source["loader"] == "local_jsonl":
        yield from iter_local_jsonl(source)
    elif source["sourceId"] == "fineweb2_hq_turkish":
        dataset = load_dataset(source["loader"], source["subset"], split="train", streaming=True)
        for row in tqdm(dataset, desc=source["sourceId"]):
            text = row.get("text")
            if isinstance(text, str):
                yield text
    elif source["sourceId"] == "aya_dataset_turkish":
        dataset = load_dataset(source["loader"], split="train", streaming=True)
        for row in tqdm(dataset, desc=source["sourceId"]):
            if row.get("language_code") != "tur":
                continue
            prompt = row.get("inputs")
            answer = row.get("targets")
            if isinstance(prompt, str) and isinstance(answer, str):
                yield f"Kullanıcı: {prompt}\nAsistan: {answer}"


def iter_identity_records(identity_config: str) -> Iterable[dict[str, str]]:
    """Lafla AI kimliğini instruction kayıtları olarak veri shardına ekler."""

    identity = load_identity(identity_config)
    yield from iter_instruction_records(identity)


def iter_local_jsonl(source: dict) -> Iterable[str]:
    """Yerel JSONL kaynaktan text veya sohbet alanlarını okur."""

    path = Path(source["path"])
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            record = json.loads(line)
            text = record.get("text")
            if isinstance(text, str):
                yield text
                continue
            user = record.get("user")
            assistant = record.get("assistant")
            if isinstance(user, str) and isinstance(assistant, str):
                yield f"Kullanıcı: {user}\nAsistan: {assistant}"


def clean_text(text: str) -> str:
    """Satır sonu ve boşlukları eğitim için kararlı hale getirir."""

    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def accepted(text: str, filters: dict) -> bool:
    """Metnin uzunluk, PII ve dil filtrelerinden geçip geçmediğini döndürür."""

    length = filters["length"]
    if len(text) < length["minChars"] or len(text) > length["maxChars"]:
        return False
    if filters["pii"] and any(pattern.search(text) for pattern in PII_PATTERNS):
        return False
    if filters["languageId"] and not looks_turkish(text):
        return False
    return True


def looks_turkish(text: str) -> bool:
    """Basit Türkçe işaretleriyle hızlı dil filtresi uygular."""

    lowered = text.lower()
    markers = [" ve ", " bir ", " için ", " değil", " olarak", " daha ", " ile ", "ş", "ğ", "ı"]
    return sum(1 for marker in markers if marker in lowered) >= 2


if __name__ == "__main__":
    main()
