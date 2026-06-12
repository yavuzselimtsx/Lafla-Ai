"""
@Dosya: runtime/context.py
@Aciklama: Context dolduğunda sistem mesajını koruyan deterministik pencereleme.
@Yazar: Lafla Geliştirme Ekibi
@Bilgi: Tokenizer bağlı değilken karakter bütçesiyle çalışır; tokenizer bağlı
        sürüm bu sözleşmenin yerine değil üstüne gelir.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence


VALID_ROLES = {"system", "user", "assistant"}


@dataclass(frozen=True)
class ChatMessage:
    """Runtime sohbet mesajı."""

    role: str
    content: str


@dataclass(frozen=True)
class ContextWindow:
    """Context pencereleme sonucu."""

    messages: tuple[ChatMessage, ...]
    dropped_count: int
    warnings: tuple[str, ...]


def prepare_context_window(
    messages: Sequence[ChatMessage],
    max_chars: int,
    strategy: str = "sliding_window",
) -> ContextWindow:
    """Sistem mesajlarını koruyup en yeni non-system mesajları bütçeye sığdırır."""

    if max_chars < 16:
        raise ValueError("max_chars cok kucuk")
    if strategy not in {"truncate_oldest", "sliding_window", "summarize_then_slide"}:
        raise ValueError(f"desteklenmeyen context stratejisi: {strategy}")
    normalized = tuple(_validate_message(message) for message in messages)
    system_messages = tuple(message for message in normalized if message.role == "system")
    non_system = tuple(message for message in normalized if message.role != "system")
    system_cost = _messages_cost(system_messages)
    if system_cost > max_chars:
        raise ValueError("system mesajlari context butcesine sigmiyor")

    warnings: list[str] = []
    if strategy == "summarize_then_slide":
        warnings.append("summary_not_generated")

    remaining = max_chars - system_cost
    kept_reversed: list[ChatMessage] = []
    used = 0
    dropped = 0
    for message in reversed(non_system):
        cost = _message_cost(message)
        if used + cost <= remaining:
            kept_reversed.append(message)
            used += cost
            continue
        if not kept_reversed and remaining > 0:
            trimmed = _trim_message_to_budget(message, remaining)
            kept_reversed.append(trimmed)
            used += _message_cost(trimmed)
        dropped += 1
    if dropped:
        warnings.append("context_trimmed")
    return ContextWindow(messages=system_messages + tuple(reversed(kept_reversed)), dropped_count=dropped, warnings=tuple(warnings))


def _validate_message(message: ChatMessage) -> ChatMessage:
    if message.role not in VALID_ROLES:
        raise ValueError(f"desteklenmeyen mesaj rolu: {message.role}")
    content = message.content.strip()
    if not content:
        raise ValueError("bos mesaj icerigi")
    return ChatMessage(message.role, content)


def _message_cost(message: ChatMessage) -> int:
    return len(message.role) + len(message.content) + 4


def _messages_cost(messages: Sequence[ChatMessage]) -> int:
    return sum(_message_cost(message) for message in messages)


def _trim_message_to_budget(message: ChatMessage, budget: int) -> ChatMessage:
    overhead = len(message.role) + 4
    content_budget = max(0, budget - overhead)
    if content_budget <= 0:
        raise ValueError("tek mesaj context butcesine sigmiyor")
    return ChatMessage(message.role, message.content[-content_budget:])
