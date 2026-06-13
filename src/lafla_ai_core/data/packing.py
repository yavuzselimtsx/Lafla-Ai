"""
@Dosya: data/packing.py
@Aciklama: JSONL metin kayitlarini sabit uzunluklu token bloklarina paketler.
@Yazar: Lafla Gelistirme Ekibi
@Bilgi: Egitim dongusu veri formatini bilmez; sadece PackedTokenDataset okur.
@Uyari: Bos veya tokenizer'siz veri sessiz gecilmez, acik hata uretir.
@Calisma-Semasi: jsonl -> tokenizer encode -> eos join -> fixed blocks
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Protocol

from lafla_ai_core.tokenizer.chat_template import ChatTurn, render_chat_transcript
from lafla_ai_core.tokenizer.quality import validate_clean_text


class TokenCodec(Protocol):
    """Tokenizer icin gereken en kucuk arayuz."""

    def encode(self, text: str) -> list[int]:
        """Metni token id listesine cevirir."""


@dataclass(frozen=True)
class PackedSequence:
    """Bir egitim blogunu tasir."""

    input_ids: tuple[int, ...]
    labels: tuple[int, ...]


def iter_jsonl_texts(paths: Iterable[str | Path]) -> Iterable[str]:
    """JSONL dosyalarindan text/prompt-response alanlarini okur."""

    for raw_path in paths:
        path = Path(raw_path)
        if not path.exists():
            raise FileNotFoundError(f"data jsonl bulunamadi: {path}")
        with path.open("r", encoding="utf-8") as handle:
            for line_no, line in enumerate(handle, start=1):
                if not line.strip():
                    continue
                try:
                    record = json.loads(line)
                except json.JSONDecodeError as exc:
                    raise ValueError(f"{path}:{line_no} JSONL gecersiz: {exc}") from exc
                text = _record_to_text(record, f"{path}:{line_no}")
                if text:
                    yield text


def pack_token_sequences(
    texts: Iterable[str],
    tokenizer: TokenCodec,
    sequence_length: int,
    eos_id: int,
) -> list[PackedSequence]:
    """Metinleri EOS ile birlestirip sabit uzunluklu egitim bloklari uretir."""

    if sequence_length < 2:
        raise ValueError("sequence_length en az 2 olmali")
    buffer: list[int] = []
    packed: list[PackedSequence] = []
    for text in texts:
        token_ids = tokenizer.encode(text)
        if not token_ids:
            continue
        buffer.extend(token_ids)
        buffer.append(eos_id)
        while len(buffer) >= sequence_length:
            block = tuple(buffer[:sequence_length])
            packed.append(PackedSequence(input_ids=block, labels=block))
            del buffer[:sequence_length]
    if not packed:
        raise ValueError("paketlenecek yeterli token yok")
    return packed


def iter_packed_token_blocks(
    texts: Iterable[str],
    tokenizer: TokenCodec,
    sequence_length: int,
    eos_id: int,
) -> Iterable[tuple[int, ...]]:
    """Buyuk veri icin bloklari bellekte biriktirmeden uretir."""

    if sequence_length < 2:
        raise ValueError("sequence_length en az 2 olmali")
    buffer: list[int] = []
    yielded = False
    for text in texts:
        token_ids = tokenizer.encode(text)
        if not token_ids:
            continue
        buffer.extend(token_ids)
        buffer.append(eos_id)
        while len(buffer) >= sequence_length:
            yielded = True
            block = tuple(buffer[:sequence_length])
            del buffer[:sequence_length]
            yield block
    if not yielded:
        raise ValueError("paketlenecek yeterli token yok")


def resolve_special_token_id(tokenizer: object, token: str) -> int:
    """Tokenizer icinden special token id'sini fail-closed cozer."""

    token_to_id = getattr(tokenizer, "token_to_id", None)
    if not callable(token_to_id):
        raise TypeError("tokenizer token_to_id(token) arayuzunu saglamali")
    token_id = token_to_id(token)
    if token_id is None:
        raise ValueError(f"special token tokenizer icinde yok: {token}")
    return int(token_id)


class TokenizersCodec:
    """Hugging Face tokenizers JSON dosyasini TokenCodec'e uyarlar."""

    def __init__(self, tokenizer_path: str | Path) -> None:
        try:
            from tokenizers import Tokenizer  # type: ignore
        except ModuleNotFoundError as exc:
            raise ModuleNotFoundError("tokenizers paketi gerekli: pip install tokenizers") from exc
        self._tokenizer = Tokenizer.from_file(str(tokenizer_path))

    def encode(self, text: str) -> list[int]:
        return list(self._tokenizer.encode(text).ids)

    def token_to_id(self, token: str) -> int | None:
        return self._tokenizer.token_to_id(token)

    def vocab_size(self) -> int:
        return int(self._tokenizer.get_vocab_size())


def _record_to_text(record: object, context: str = "record") -> str:
    if not isinstance(record, dict):
        raise ValueError("JSONL kaydi nesne olmali")
    if isinstance(record.get("text"), str):
        return validate_clean_text(str(record["text"]).strip(), f"{context}:text")
    if isinstance(record.get("prompt"), str) and isinstance(record.get("response"), str):
        prompt = validate_clean_text(str(record["prompt"]).strip(), f"{context}:prompt")
        response = validate_clean_text(str(record["response"]).strip(), f"{context}:response")
        return render_chat_transcript(
            (ChatTurn("user", prompt), ChatTurn("assistant", response)),
            include_bos=True,
            include_eos=False,
            validate_content_tokens=True,
        )
    if isinstance(record.get("system"), str) and isinstance(record.get("user"), str) and isinstance(record.get("assistant"), str):
        system = validate_clean_text(str(record["system"]).strip(), f"{context}:system")
        user = validate_clean_text(str(record["user"]).strip(), f"{context}:user")
        assistant = validate_clean_text(str(record["assistant"]).strip(), f"{context}:assistant")
        return render_chat_transcript(
            (
                ChatTurn("system", system),
                ChatTurn("user", user),
                ChatTurn("assistant", assistant),
            ),
            include_bos=True,
            include_eos=False,
            validate_content_tokens=True,
        )
    raise ValueError("JSONL kaydinda text veya sohbet alanlari yok")
