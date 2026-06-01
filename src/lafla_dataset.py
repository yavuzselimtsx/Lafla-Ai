"""
 _        _    _____ _        _
| |      / \\  |  ___| |      / \\
| |     / _ \\ | |_  | |     / _ \\
| |___ / ___ \\|  _| | |___ / ___ \\
|_____/_/   \\_\\_|   |_____/_/   \\_\\

@Dosya: lafla_dataset.py
@Açıklama: Lafla AI JSONL verisini tokenizer ile sabit uzunluklu eğitim bloklarına paketler.
@Yazar: Lafla Geliştirme Ekibi
@Bilgi: Smoke modu dışında rastgele veri kullanılmaz; eğitim batch'i gerçek shardlardan gelir.
@Uyarı: Bu dosya veri indirme yapmaz, yalnızca yerel işlenmiş shardları okur.
@Çalışma-Şeması: jsonl -> token ids -> fixed blocks -> torch dataset
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Iterator

import torch
from torch.utils.data import Dataset
from tokenizers import Tokenizer


class PackedJsonlDataset(Dataset[torch.Tensor]):
    """Tokenize edilmiş metni context_length uzunluğunda örneklere böler."""

    def __init__(self, paths: list[Path], tokenizer_path: Path, context_length: int) -> None:
        if context_length < 8:
            raise ValueError("context_length too small")
        self.context_length = context_length
        self.tokenizer = Tokenizer.from_file(str(tokenizer_path))
        self.blocks = list(pack_token_blocks(iter_records(paths), self.tokenizer, context_length))
        if not self.blocks:
            raise ValueError("training dataset is empty")

    def __len__(self) -> int:
        """Dataset içindeki sabit uzunluklu blok sayısını döndürür."""

        return len(self.blocks)

    def __getitem__(self, index: int) -> torch.Tensor:
        """Tek eğitim bloğunu long tensor olarak döndürür."""

        return torch.tensor(self.blocks[index], dtype=torch.long)


def iter_records(paths: list[Path]) -> Iterator[str]:
    """JSONL kayıtlarından metin alanlarını sırayla üretir."""

    for path in paths:
        with path.open("r", encoding="utf-8") as handle:
            for line in handle:
                if not line.strip():
                    continue
                record = json.loads(line)
                text = record_to_training_text(record)
                if text:
                    yield text


def pack_token_blocks(texts: Iterator[str], tokenizer: Tokenizer, context_length: int) -> Iterator[list[int]]:
    """Token akışını context_length uzunluklu bloklara paketler."""

    eos_id = tokenizer.token_to_id("<|eos|>")
    if eos_id is None:
        raise ValueError("tokenizer missing <|eos|>")
    buffer: list[int] = []
    for text in texts:
        ids = tokenizer.encode(text).ids
        if not ids:
            continue
        buffer.extend(ids)
        buffer.append(eos_id)
        while len(buffer) >= context_length:
            yield buffer[:context_length]
            buffer = buffer[context_length:]


def record_to_training_text(record: object) -> str:
    """Tek JSONL kaydını model eğitimi için normalize edilmiş metne çevirir."""

    if not isinstance(record, dict):
        return ""
    system = record.get("system")
    user = record.get("user")
    assistant = record.get("assistant")
    if isinstance(user, str) and isinstance(assistant, str):
        prefix = f"<|system|>{system}" if isinstance(system, str) and system else ""
        return f"{prefix}<|user|>{user}<|assistant|>{assistant}"
    text = record.get("text")
    return text if isinstance(text, str) else ""
