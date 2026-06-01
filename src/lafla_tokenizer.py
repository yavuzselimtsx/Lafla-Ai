"""
 _        _    _____ _        _
| |      / \\  |  ___| |      / \\
| |     / _ \\ | |_  | |     / _ \\
| |___ / ___ \\|  _| | |___ / ___ \\
|_____/_/   \\_\\_|   |_____/_/   \\_\\

@Dosya: lafla_tokenizer.py
@Açıklama: Lafla AI için hazır tokenizer kopyalamadan BPE tokenizer eğitir.
@Yazar: Lafla Geliştirme Ekibi
@Bilgi: Eğitim girdisi JSONL kayıtlarında text veya user/assistant alanları olabilir.
@Uyarı: Tokenizer çıktısı veri lisans raporu geçmeden üretim eğitiminde kullanılmamalıdır.
@Çalışma-Şeması: jsonl text iterator -> BPE trainer -> tokenizer json
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Iterable

from tokenizers import Tokenizer
from tokenizers.models import BPE
from tokenizers.normalizers import NFKC, Sequence
from tokenizers.pre_tokenizers import ByteLevel
from tokenizers.trainers import BpeTrainer


SPECIAL_TOKENS = ["<|pad|>", "<|bos|>", "<|eos|>", "<|system|>", "<|user|>", "<|assistant|>"]


def main() -> None:
    """Komut satırından tokenizer eğitimini başlatır."""

    parser = argparse.ArgumentParser()
    parser.add_argument("--input", action="append", required=True, help="JSONL eğitim shard yolu. Birden çok kez verilebilir.")
    parser.add_argument("--output", required=True, help="Tokenizer JSON çıktı yolu.")
    parser.add_argument("--vocab-size", type=int, default=50304)
    args = parser.parse_args()

    tokenizer = train_tokenizer([Path(item) for item in args.input], args.vocab_size)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    tokenizer.save(str(output))
    print(json.dumps({"tokenizer": str(output), "vocab_size": tokenizer.get_vocab_size()}, ensure_ascii=False))


def train_tokenizer(paths: list[Path], vocab_size: int) -> Tokenizer:
    """JSONL metinlerinden byte-level BPE tokenizer üretir."""

    tokenizer = Tokenizer(BPE(unk_token="<|pad|>"))
    tokenizer.normalizer = Sequence([NFKC()])
    tokenizer.pre_tokenizer = ByteLevel(add_prefix_space=False)
    trainer = BpeTrainer(vocab_size=vocab_size, special_tokens=SPECIAL_TOKENS, min_frequency=2)
    tokenizer.train_from_iterator(iter_training_text(paths), trainer=trainer)
    return tokenizer


def iter_training_text(paths: list[Path]) -> Iterable[str]:
    """JSONL shard kayıtlarını eğitim metnine çevirir."""

    for path in paths:
        with path.open("r", encoding="utf-8") as handle:
            for line in handle:
                if not line.strip():
                    continue
                record = json.loads(line)
                text = record_to_text(record)
                if text:
                    yield text


def record_to_text(record: object) -> str:
    """Tek kayıt içinden tokenizer eğitimi için metin çıkarır."""

    if not isinstance(record, dict):
        return ""
    text = record.get("text")
    if isinstance(text, str):
        return text
    user = record.get("user")
    assistant = record.get("assistant")
    if isinstance(user, str) and isinstance(assistant, str):
        return f"<|user|>{user}<|assistant|>{assistant}<|eos|>"
    return ""


if __name__ == "__main__":
    main()
