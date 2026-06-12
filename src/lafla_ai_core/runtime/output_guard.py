"""
@Dosya: runtime/output_guard.py
@Aciklama: Ham model completion metnini public/runtime sinirlarina sokar.
@Yazar: Lafla Gelistirme Ekibi
@Bilgi: OLMo completion-only ve GPT-NeoX stop-token dersleri Lafla runtime
        cikis korumasina ayrica kodlanir.
@Uyari: Bu modul model kalitesini uydurmaz; prompt/role/decode sizintisini
        fail-closed sekilde temizler ve uyarir.
@Calisma-Semasi: raw completion -> role boundary -> decode repair -> prompt echo guard
"""

from __future__ import annotations

import re
from dataclasses import dataclass

from lafla_ai_core.runtime.decoding import clean_decoded_text
from lafla_ai_core.tokenizer.quality import has_mojibake


ROLE_STOP_SEQUENCES = ("<|eos|>", "<|user|>", "<|system|>", "<|assistant|>", "<|bos|>", "<|pad|>")
_ROLE_MARKERS = ("<|system|>", "<|user|>", "<|assistant|>", "<|bos|>", "<|pad|>")
_BYTELEVEL_SURFACE_MARKERS = ("\u0120", "\u010a", "\u00c4\u00a0", "\u00c4\u0160")
_PUNCT_SPACE_RE = re.compile(r"\s+([?.!,;:])")


@dataclass(frozen=True)
class OutputGuardResult:
    """Runtime output guard sonucunu tasir."""

    text: str
    warnings: tuple[str, ...]


def sanitize_completion(raw_text: str, *, prompt_text: str | None = None, system_text: str | None = None) -> OutputGuardResult:
    """Ham uretilen metni completion-only public sinira yaklastirir."""

    warnings: list[str] = []
    if _has_bytelevel_surface(raw_text):
        warnings.append("bytelevel_surface_repaired")
    had_mojibake = has_mojibake(raw_text)

    completion, boundary_warning = _slice_after_generation_prompt(raw_text)
    if boundary_warning:
        warnings.append(boundary_warning)

    completion, stopped = _truncate_at_later_stop_sequence(completion)
    if stopped:
        warnings.append("control_token_stop")

    cleaned = clean_decoded_text(completion, strip_special_tokens=False)
    if had_mojibake and not has_mojibake(cleaned):
        warnings.append("mojibake_repaired")

    cleaned = _strip_role_markers(cleaned)
    cleaned, echo_removed = _remove_prompt_echoes(cleaned, system_text=system_text, prompt_text=prompt_text)
    if echo_removed:
        warnings.append("prompt_echo_removed")

    cleaned, stopped_after_echo = _truncate_at_later_stop_sequence(cleaned)
    if stopped_after_echo:
        warnings.append("control_token_stop")
    cleaned = _normalize_display_text(_strip_role_markers(cleaned))
    if cleaned and _is_low_information_completion(cleaned):
        warnings.append("low_information_completion")
        cleaned = ""
    if not cleaned:
        warnings.append("empty_after_output_guard")

    return OutputGuardResult(text=cleaned, warnings=_dedupe(warnings))


def _slice_after_generation_prompt(text: str) -> tuple[str, str | None]:
    assistant = "<|assistant|>"
    last_assistant = text.rfind(assistant)
    if last_assistant == -1:
        return text, None
    prefix = text[:last_assistant]
    if any(marker in prefix for marker in ("<|system|>", "<|user|>", "<|bos|>")):
        return text[last_assistant + len(assistant) :], "role_boundary_trimmed"
    return text, None


def _truncate_at_later_stop_sequence(text: str) -> tuple[str, bool]:
    positions = [index for token in ROLE_STOP_SEQUENCES if (index := text.find(token)) > 0]
    if not positions:
        return text, False
    return text[: min(positions)], True


def _strip_role_markers(text: str) -> str:
    stripped = text
    for marker in _ROLE_MARKERS:
        stripped = stripped.replace(marker, " ")
    stripped = stripped.replace("<|eos|>", " ")
    return stripped


def _remove_prompt_echoes(text: str, *, system_text: str | None, prompt_text: str | None) -> tuple[str, bool]:
    cleaned = text
    removed = False
    for candidate in (system_text, prompt_text, system_text, prompt_text):
        if not candidate:
            continue
        cleaned, did_remove = _remove_loose_prefix(cleaned, candidate)
        removed = removed or did_remove
    return cleaned, removed


def _remove_loose_prefix(text: str, prefix: str) -> tuple[str, bool]:
    canonical_prefix = _canonical_echo_text(prefix)
    if not canonical_prefix:
        return text, False
    seen = ""
    started = False
    for index, char in enumerate(text):
        if char.isalnum():
            started = True
            seen += char.casefold()
            if not canonical_prefix.startswith(seen):
                return text, False
            if seen == canonical_prefix:
                return text[index + 1 :].lstrip(" \t\r\n.:;-,?!"), True
        elif not started:
            continue
    return text, False


def _canonical_echo_text(text: str) -> str:
    cleaned = clean_decoded_text(text, strip_special_tokens=True).casefold()
    return "".join(char for char in cleaned if char.isalnum())


def _normalize_display_text(text: str) -> str:
    cleaned = clean_decoded_text(text, strip_special_tokens=False)
    cleaned = _PUNCT_SPACE_RE.sub(r"\1", cleaned)
    return " ".join(cleaned.split())


def _has_bytelevel_surface(text: str) -> bool:
    return any(marker in text for marker in _BYTELEVEL_SURFACE_MARKERS)


def _is_low_information_completion(text: str) -> bool:
    compact_surface = "".join(char for char in text if not char.isspace())
    if not compact_surface:
        return True
    if not any(char.isalnum() for char in compact_surface):
        return len(compact_surface) >= 3

    compact_alnum = "".join(char.casefold() for char in text if char.isalnum())
    if len(compact_alnum) >= 8 and len(set(compact_alnum)) == 1:
        return True
    if len(compact_alnum) >= 12 and _is_repeated_compact_pattern(compact_alnum):
        return True

    tokens = _alnum_tokens(text)
    if len(tokens) >= 5 and len(set(tokens)) <= 2:
        return True
    if len(tokens) >= 6 and _is_repeated_token_pattern(tokens):
        return True
    return False


def _alnum_tokens(text: str) -> list[str]:
    tokens: list[str] = []
    current: list[str] = []
    for char in text.casefold():
        if char.isalnum():
            current.append(char)
            continue
        if current:
            tokens.append("".join(current))
            current = []
    if current:
        tokens.append("".join(current))
    return tokens


def _is_repeated_compact_pattern(text: str) -> bool:
    max_pattern = min(8, len(text) // 2)
    for size in range(1, max_pattern + 1):
        if len(text) % size != 0:
            continue
        pattern = text[:size]
        if len(text) // size >= 3 and pattern * (len(text) // size) == text:
            return True
    return False


def _is_repeated_token_pattern(tokens: list[str]) -> bool:
    max_pattern = min(4, len(tokens) // 2)
    for size in range(1, max_pattern + 1):
        if len(tokens) % size != 0:
            continue
        pattern = tokens[:size]
        if len(tokens) // size >= 3 and pattern * (len(tokens) // size) == tokens:
            return True
    return False


def _dedupe(values: list[str]) -> tuple[str, ...]:
    seen: set[str] = set()
    output: list[str] = []
    for value in values:
        if value not in seen:
            seen.add(value)
            output.append(value)
    return tuple(output)
