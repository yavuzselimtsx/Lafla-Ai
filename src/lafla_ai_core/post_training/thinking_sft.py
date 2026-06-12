"""
@Dosya: post_training/thinking_sft.py
@Aciklama: Lafla thinking SFT kayitlari, chat render ve label-mask uretimi.
@Yazar: Lafla Gelistirme Ekibi
@Bilgi: Thinking davranisi pretraining motoruna gomulmez; post-training veri
        sozlesmesiyle ogretilir ve runtime'da ayrica filtrelenir.
@Uyari: Gizli dusunceyi public ciktiya sizdirmamak runtime politikasinin gorevidir.
@Calisma-Semasi: structured turns -> rendered chat -> token ids + labels
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Iterable, Protocol

from lafla_ai_core.tokenizer.chat_template import (
    CONTROL_TOKENS,
    ROLE_TOKENS,
    THINK_CLOSE,
    THINK_OPEN,
    ChatTurn,
    render_chat_transcript,
)


class TokenCodec(Protocol):
    def encode(self, text: str) -> list[int]:
        """Metni token id listesine cevirir."""


@dataclass(frozen=True)
class ThinkingSftRecord:
    """Tek thinking SFT kaydini tasir."""

    system: str
    user: str
    thinking: str
    assistant: str


@dataclass(frozen=True)
class ThinkingFinding:
    """Thinking veri kapisi bulgusu."""

    code: str
    detail: str


@dataclass(frozen=True)
class ThinkingValidationReport:
    """Thinking kaydi dogrulama raporu."""

    ok: bool
    findings: tuple[ThinkingFinding, ...]


@dataclass(frozen=True)
class SupervisedChatExample:
    """Tokenize edilmis chat ve loss label'larini tasir."""

    input_ids: tuple[int, ...]
    labels: tuple[int, ...]
    rendered_text: str


def render_thinking_record(record: ThinkingSftRecord) -> str:
    """Thinking SFT kaydini Lafla sohbet formatina cevirir."""

    turns = (
        ChatTurn("system", record.system),
        ChatTurn("user", record.user),
        ChatTurn("assistant", f"{THINK_OPEN}{record.thinking.strip()}{THINK_CLOSE}\n{record.assistant.strip()}"),
    )
    return render_chat_turns(turns)


def render_chat_turns(turns: Iterable[ChatTurn]) -> str:
    """Chat turn listesini special-token metnine cevirir."""

    return render_chat_transcript(turns, include_bos=True, include_eos=True, validate_content_tokens=False)


def build_supervised_chat_example(
    turns: Iterable[ChatTurn],
    tokenizer: TokenCodec,
    *,
    only_last_assistant: bool = True,
    supervise_thinking: bool = True,
) -> SupervisedChatExample:
    """Sohbeti token ids ve -100 maskeli labels olarak hazirlar."""

    turn_list = tuple(turns)
    if not turn_list:
        raise ValueError("en az bir chat turn gerekli")
    assistant_indices = [index for index, turn in enumerate(turn_list) if turn.role == "assistant"]
    last_assistant = assistant_indices[-1] if assistant_indices else -1
    input_ids: list[int] = tokenizer.encode("<|bos|>\n")
    labels: list[int] = [-100] * len(input_ids)
    rendered_parts = ["<|bos|>"]

    for index, turn in enumerate(turn_list):
        prefix = f"{ROLE_TOKENS.get(turn.role, '')}\n"
        if turn.role not in ROLE_TOKENS:
            raise ValueError(f"desteklenmeyen chat role: {turn.role}")
        content = turn.content.strip()
        if not content:
            raise ValueError(f"bos chat icerigi: {turn.role}")
        supervise = turn.role == "assistant" and (not only_last_assistant or index == last_assistant)
        chunk_ids, chunk_labels = _encode_chat_chunk(prefix, content, tokenizer, supervise, supervise_thinking)
        input_ids.extend(chunk_ids)
        labels.extend(chunk_labels)
        rendered_parts.append(f"{ROLE_TOKENS[turn.role]}\n{content}")

    eos_ids = tokenizer.encode("<|eos|>")
    input_ids.extend(eos_ids)
    labels.extend(eos_ids if last_assistant >= 0 else [-100] * len(eos_ids))
    rendered_parts.append("<|eos|>")
    return SupervisedChatExample(tuple(input_ids), tuple(labels), "\n".join(rendered_parts))


def validate_thinking_record(record: ThinkingSftRecord, max_thinking_chars: int = 4000) -> ThinkingValidationReport:
    """Thinking SFT kaydinin temel kalite kapilarini uygular."""

    findings: list[ThinkingFinding] = []
    if not record.system.strip():
        findings.append(ThinkingFinding("system_empty", "system bos olamaz"))
    if not record.user.strip():
        findings.append(ThinkingFinding("user_empty", "user bos olamaz"))
    if not record.thinking.strip():
        findings.append(ThinkingFinding("thinking_empty", "thinking bos olamaz"))
    if not record.assistant.strip():
        findings.append(ThinkingFinding("assistant_empty", "assistant bos olamaz"))
    if len(record.thinking) > max_thinking_chars:
        findings.append(ThinkingFinding("thinking_too_long", "thinking metni izin verilen siniri asti"))
    for field_name, value in (("system", record.system), ("user", record.user)):
        if any(token in value for token in CONTROL_TOKENS):
            findings.append(ThinkingFinding(f"{field_name}_contains_control_token", f"{field_name} alani special token tasimamali"))
    if any(token in record.thinking for token in ROLE_TOKENS.values()):
        findings.append(ThinkingFinding("thinking_contains_role_token", "thinking alani role token tasimamali"))
    if THINK_OPEN in record.assistant or THINK_CLOSE in record.assistant:
        findings.append(ThinkingFinding("assistant_contains_think_tag", "assistant alani think token tasimamali"))
    return ThinkingValidationReport(ok=not findings, findings=tuple(findings))


def strip_thinking_for_public(text: str) -> str:
    """Public runtime icin think bloklarini kaldirir."""

    stripped = re.sub(r"<\|think\|>.*?<\|/think\|>", " ", text, flags=re.DOTALL)
    stripped = stripped.replace(THINK_OPEN, " ").replace(THINK_CLOSE, " ")
    return " ".join(stripped.split())


def _encode_chat_chunk(
    prefix: str,
    content: str,
    tokenizer: TokenCodec,
    supervise: bool,
    supervise_thinking: bool,
) -> tuple[list[int], list[int]]:
    if not supervise:
        chunk_ids = tokenizer.encode(f"{prefix}{content}\n")
        return chunk_ids, [-100] * len(chunk_ids)

    if supervise_thinking or THINK_OPEN not in content:
        chunk_ids = tokenizer.encode(f"{prefix}{content}\n")
        return chunk_ids, list(chunk_ids)

    input_ids: list[int] = []
    labels: list[int] = []
    for segment, visible_to_loss in _split_thinking_segments(f"{prefix}{content}\n"):
        segment_ids = tokenizer.encode(segment)
        input_ids.extend(segment_ids)
        labels.extend(segment_ids if visible_to_loss else [-100] * len(segment_ids))
    return input_ids, labels


def _split_thinking_segments(text: str) -> Iterable[tuple[str, bool]]:
    cursor = 0
    while True:
        start = text.find(THINK_OPEN, cursor)
        if start == -1:
            if cursor < len(text):
                yield text[cursor:], True
            return
        if start > cursor:
            yield text[cursor:start], True
        end = text.find(THINK_CLOSE, start + len(THINK_OPEN))
        if end == -1:
            yield text[start:], False
            return
        end += len(THINK_CLOSE)
        yield text[start:end], False
        cursor = end
