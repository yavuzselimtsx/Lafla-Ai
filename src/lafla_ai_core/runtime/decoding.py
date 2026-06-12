"""
@Dosya: runtime/decoding.py
@Aciklama: Model token çıktısını kullanıcıya gösterilecek temiz metne dönüştürür.
@Yazar: Lafla Geliştirme Ekibi
@Bilgi: Byte-level BPE decoder eksikse ham token yüzeyi Ġ/Ċ olarak görünür; runtime bunu
        açıkça yakalar ve onarır.
@Calisma-Semasi: token ids -> tokenizer.decode -> byte-level cleanup -> display text
"""

from __future__ import annotations

import re
from collections.abc import Iterable

from lafla_ai_core.tokenizer.quality import repair_mojibake_for_display


_SPECIAL_TOKEN_RE = re.compile(r"<\|[^|]+?\|>")


def decode_token_ids(tokenizer: object, token_ids: Iterable[int], skip_special_tokens: bool = False) -> str:
    """Tokenizer çıktısını güvenli runtime metnine çevirir."""

    ids = [int(token_id) for token_id in token_ids]
    _attach_bytelevel_decoder_if_possible(tokenizer)
    decode = getattr(tokenizer, "decode", None)
    if not callable(decode):
        raise TypeError("tokenizer decode(token_ids) arayüzünü sağlamalı")
    try:
        decoded = decode(ids, skip_special_tokens=skip_special_tokens)
    except TypeError:
        decoded = decode(ids)
    if not isinstance(decoded, str):
        raise TypeError("tokenizer.decode str döndürmeli")
    return clean_decoded_text(decoded, strip_special_tokens=skip_special_tokens)


def clean_decoded_text(text: str, strip_special_tokens: bool = True) -> str:
    """Byte-level yüzey artıkları ve görüntüleme mojibake'ini temizler."""

    bytelevel_cleaned = text.replace("Ċ", "\n").replace("Ġ", " ")
    repaired = repair_mojibake_for_display(bytelevel_cleaned)
    if strip_special_tokens:
        repaired = _SPECIAL_TOKEN_RE.sub("", repaired)
    repaired = re.sub(r"[ \t]+\n", "\n", repaired)
    repaired = re.sub(r"\n{3,}", "\n\n", repaired)
    repaired = re.sub(r"[ \t]{2,}", " ", repaired)
    return repaired.strip()


def _attach_bytelevel_decoder_if_possible(tokenizer: object) -> None:
    try:
        current = getattr(tokenizer, "decoder", None)
    except Exception:
        return
    if current is not None:
        return
    try:
        from tokenizers import decoders  # type: ignore
    except Exception:
        return
    try:
        setattr(tokenizer, "decoder", decoders.ByteLevel())
    except Exception:
        return
