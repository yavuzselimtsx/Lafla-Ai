"""
@Dosya: runtime/generation_contract.py
@Aciklama: Inference motorlari icin Lafla sohbet promptu ve completion-only token siniri.
@Yazar: Lafla Gelistirme Ekibi
@Bilgi: OLMo'nun completions_only ve GPT-NeoX'un stop-token davranisi burada
        tokenizer seviyesinde fail-closed sozlesmeye cevrilir.
@Uyari: Public runtime ham generated_ids icinden prompt tokenlarini asla cevap
        olarak decode etmemelidir.
@Calisma-Semasi: ChatMessage -> prompt ids + stop ids -> generated ids -> guarded completion
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol, Sequence

from lafla_ai_core.runtime.context import ChatMessage, VALID_ROLES
from lafla_ai_core.runtime.output_guard import ROLE_STOP_SEQUENCES, OutputGuardResult, sanitize_completion
from lafla_ai_core.tokenizer.chat_template import ChatTurn, render_generation_prompt


class GenerationTokenizer(Protocol):
    """Generation sozlesmesi icin gereken en kucuk tokenizer arayuzu."""

    def encode(self, text: str) -> Sequence[int]:
        """Metni token id dizisine cevirir."""

    def decode(self, token_ids: Sequence[int], skip_special_tokens: bool = False) -> str:
        """Token id dizisini metne cevirir."""


@dataclass(frozen=True)
class GenerationRequest:
    """Inference motoruna verilecek prompt ve stop-token sozlesmesi."""

    prompt_text: str
    prompt_token_ids: tuple[int, ...]
    stop_sequences: tuple[str, ...]
    stop_token_ids: tuple[tuple[int, ...], ...]


def build_generation_request(
    messages: Sequence[ChatMessage],
    tokenizer: GenerationTokenizer,
    *,
    stop_sequences: Sequence[str] = ROLE_STOP_SEQUENCES,
    max_prompt_tokens: int | None = None,
) -> GenerationRequest:
    """Sohbet mesajlarindan assistant generation promptu ve stop token id'leri uretir."""

    if not messages:
        raise ValueError("en az bir mesaj gerekli")
    prompt_text = render_generation_prompt(_to_chat_turns(messages))
    prompt_token_ids = tuple(_encode_text(tokenizer, prompt_text))
    if not prompt_token_ids:
        raise ValueError("prompt tokenizer tarafindan bos kodlandi")
    if max_prompt_tokens is not None and len(prompt_token_ids) > max_prompt_tokens:
        raise ValueError("prompt token butcesini asiyor")
    return GenerationRequest(
        prompt_text=prompt_text,
        prompt_token_ids=prompt_token_ids,
        stop_sequences=tuple(stop_sequences),
        stop_token_ids=resolve_stop_token_ids(tokenizer, stop_sequences),
    )


def resolve_stop_token_ids(
    tokenizer: GenerationTokenizer,
    stop_sequences: Sequence[str] = ROLE_STOP_SEQUENCES,
) -> tuple[tuple[int, ...], ...]:
    """Stop sequence metinlerini tokenizer-id dizilerine cevirir."""

    if not stop_sequences:
        raise ValueError("en az bir stop sequence gerekli")
    resolved: list[tuple[int, ...]] = []
    seen: set[tuple[int, ...]] = set()
    for sequence in stop_sequences:
        if not isinstance(sequence, str) or not sequence:
            raise ValueError("stop sequence bos olamaz")
        token_ids = tuple(_encode_text(tokenizer, sequence))
        if not token_ids:
            raise ValueError(f"stop sequence tokenizer tarafindan bos kodlandi: {sequence}")
        if token_ids not in seen:
            seen.add(token_ids)
            resolved.append(token_ids)
    return tuple(resolved)


def trim_completion_token_ids(
    generated_ids: Sequence[int],
    *,
    prompt_token_count: int,
    stop_token_ids: Sequence[Sequence[int]] = (),
) -> tuple[int, ...]:
    """Full generated token dizisinden promptu ve ilk stop-token sonrasini ayiklar."""

    completion = _completion_slice(generated_ids, prompt_token_count)
    stop_index = _find_earliest_stop_index(completion, stop_token_ids)
    if stop_index is None:
        return completion
    return completion[:stop_index]


def decode_completion_from_ids(
    tokenizer: GenerationTokenizer,
    generated_ids: Sequence[int],
    *,
    prompt_token_count: int,
    stop_token_ids: Sequence[Sequence[int]] = (),
    prompt_text: str | None = None,
    system_text: str | None = None,
) -> OutputGuardResult:
    """Generated id'lerden yalniz completion kismini decode edip output guard uygular."""

    completion = _completion_slice(generated_ids, prompt_token_count)
    stopped = _find_earliest_stop_index(completion, stop_token_ids) is not None
    trimmed = trim_completion_token_ids(generated_ids, prompt_token_count=prompt_token_count, stop_token_ids=stop_token_ids)
    raw_text = _decode_ids(tokenizer, trimmed)
    guarded = sanitize_completion(raw_text, prompt_text=prompt_text, system_text=system_text)
    if stopped:
        return _append_warning(guarded, "control_token_stop")
    return guarded


def _to_chat_turns(messages: Sequence[ChatMessage]) -> tuple[ChatTurn, ...]:
    turns: list[ChatTurn] = []
    for message in messages:
        role = str(message.role).strip()
        content = str(message.content).strip()
        if role not in VALID_ROLES:
            raise ValueError(f"desteklenmeyen mesaj rolu: {role}")
        if not content:
            raise ValueError("bos mesaj icerigi")
        turns.append(ChatTurn(role, content))
    return tuple(turns)


def _encode_text(tokenizer: GenerationTokenizer, text: str) -> tuple[int, ...]:
    encode = tokenizer.encode
    try:
        raw_ids = encode(text, add_special_tokens=False)  # type: ignore[call-arg]
    except TypeError:
        raw_ids = encode(text)
    return tuple(int(token_id) for token_id in raw_ids)


def _decode_ids(tokenizer: GenerationTokenizer, token_ids: Sequence[int]) -> str:
    try:
        return tokenizer.decode(tuple(token_ids), skip_special_tokens=False)
    except TypeError:
        return tokenizer.decode(tuple(token_ids))  # type: ignore[call-arg]


def _completion_slice(generated_ids: Sequence[int], prompt_token_count: int) -> tuple[int, ...]:
    if prompt_token_count < 0:
        raise ValueError("prompt_token_count negatif olamaz")
    ids = tuple(int(token_id) for token_id in generated_ids)
    if prompt_token_count > len(ids):
        raise ValueError("prompt_token_count generated_ids uzunlugunu asiyor")
    return ids[prompt_token_count:]


def _find_earliest_stop_index(
    completion_ids: Sequence[int],
    stop_token_ids: Sequence[Sequence[int]],
) -> int | None:
    earliest: int | None = None
    completion = tuple(int(token_id) for token_id in completion_ids)
    for stop_sequence in stop_token_ids:
        stop = tuple(int(token_id) for token_id in stop_sequence)
        if not stop:
            continue
        index = _find_subsequence(completion, stop)
        if index is not None and (earliest is None or index < earliest):
            earliest = index
    return earliest


def _find_subsequence(values: Sequence[int], needle: Sequence[int]) -> int | None:
    if len(needle) > len(values):
        return None
    for index in range(0, len(values) - len(needle) + 1):
        if tuple(values[index : index + len(needle)]) == tuple(needle):
            return index
    return None


def _append_warning(result: OutputGuardResult, warning: str) -> OutputGuardResult:
    if warning in result.warnings:
        return result
    return OutputGuardResult(text=result.text, warnings=(*result.warnings, warning))
