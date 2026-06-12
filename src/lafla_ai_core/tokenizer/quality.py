"""
@Dosya: tokenizer/quality.py
@Aciklama: Tokenizer egitimi, veri girisi ve runtime decode icin metin
            kalite kapilari.
@Yazar: Lafla Gelistirme Ekibi
@Bilgi: Byte-level tokenizer ham token yuzeyi mojibake veriyle
        karistirilmamalidir; ikisi ayri kapilardan gecer.
@Uyari: Bu kapilar gecmeden model egitimi baslatilamaz.
@Calisma-Semasi: text -> encoding gate -> tokenizer gate -> report
"""

from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass
from typing import Iterable


MOJIBAKE_MARKERS = (
    "\u00c3",
    "\u00c4",
    "\u00c5",
    "\u00c2",
    "\ufffd",
    "\u00ef\u00bf\u00bd",
    "\u00e2\u20ac",
    "\u00f0\u0178",
)
BYTELEVEL_SURFACE_MARKERS = ("\u0120", "\u010a")
TURKISH_CHARS = set("\u00e7\u011f\u0131\u00f6\u015f\u00fc\u00c7\u011e\u0130\u00d6\u015e\u00dc")
_CONTROL_RE = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]")


@dataclass(frozen=True)
class TokenizerQualityReport:
    """Tokenizer kalite raporunu tasir."""

    sample_count: int
    mojibake_count: int
    turkish_sample_count: int
    missing_special_tokens: tuple[str, ...]
    passed: bool


def normalize_text(text: str) -> str:
    """Metni Unicode NFC bicimine ceker."""

    return unicodedata.normalize("NFC", text)


def has_mojibake(text: str) -> bool:
    """Metinde yaygin UTF-8/Windows-125x bozulma izlerini arar."""

    return any(marker in text for marker in MOJIBAKE_MARKERS)


def has_bytelevel_surface(text: str) -> bool:
    """Decode edilmemis byte-level token yuzeyini yakalar."""

    return any(marker in text for marker in BYTELEVEL_SURFACE_MARKERS)


def contains_turkish(text: str) -> bool:
    """Metnin Turkceye ozgu harf icerip icermedigini denetler."""

    return any(char in TURKISH_CHARS for char in text)


def validate_clean_text(text: str, context: str) -> str:
    """Egitime girecek metni fail-closed encoding kapisindan gecirir."""

    normalized = normalize_text(text)
    if _CONTROL_RE.search(normalized):
        raise ValueError(f"{context}: kontrol karakteri bulundu")
    if has_mojibake(normalized):
        raise ValueError(f"{context}: mojibake encoding bozulmasi bulundu")
    return normalized


def repair_mojibake_for_display(text: str) -> str:
    """Runtime goruntulemede yaygin mojibake bicimlerini onarmayi dener."""

    if not has_mojibake(text):
        return normalize_text(text)
    ftfy_fixed = _try_ftfy(text)
    if ftfy_fixed is not None and _is_better_repair(text, ftfy_fixed):
        return normalize_text(ftfy_fixed)
    latin1_fixed = _try_latin1_utf8(text)
    if latin1_fixed is not None and _is_better_repair(text, latin1_fixed):
        return normalize_text(latin1_fixed)
    return normalize_text(text)


def analyze_tokenizer_texts(
    texts: Iterable[str],
    required_special_tokens: Iterable[str],
    tokenizer_vocab: Iterable[str],
) -> TokenizerQualityReport:
    """Metin ve vocab uzerinden temel tokenizer kapilarini raporlar."""

    text_list = [normalize_text(text) for text in texts]
    vocab = set(tokenizer_vocab)
    missing = tuple(token for token in required_special_tokens if token not in vocab)
    mojibake_count = sum(1 for text in text_list if has_mojibake(text))
    turkish_count = sum(1 for text in text_list if contains_turkish(text))
    passed = bool(text_list) and mojibake_count == 0 and turkish_count > 0 and not missing
    return TokenizerQualityReport(
        sample_count=len(text_list),
        mojibake_count=mojibake_count,
        turkish_sample_count=turkish_count,
        missing_special_tokens=missing,
        passed=passed,
    )


def _try_ftfy(text: str) -> str | None:
    try:
        import ftfy  # type: ignore
    except Exception:
        return None
    return str(ftfy.fix_text(text))


def _try_latin1_utf8(text: str) -> str | None:
    try:
        return text.encode("latin-1").decode("utf-8")
    except UnicodeError:
        return None


def _is_better_repair(original: str, candidate: str) -> bool:
    return _mojibake_score(candidate) < _mojibake_score(original) and "\ufffd" not in candidate


def _mojibake_score(text: str) -> int:
    return sum(text.count(marker) for marker in MOJIBAKE_MARKERS) + sum(text.count(marker) for marker in BYTELEVEL_SURFACE_MARKERS)
