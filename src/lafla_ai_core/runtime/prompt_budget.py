"""
@Dosya: runtime/prompt_budget.py
@Aciklama: Sistem, ozet, retrieval, son mesajlar ve cikti icin ortak token butcesi.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Sequence

from lafla_ai_core.runtime.conversation_memory import (
    MessageRecord,
    StructuredSummary,
    message_token_count,
)


TokenCounter = Callable[[str], int]


@dataclass(frozen=True)
class PromptBudget:
    total_tokens: int
    output_reserve_tokens: int
    retrieval_max_tokens: int
    recent_max_tokens: int

    def validate(self) -> None:
        if self.total_tokens <= 0:
            raise ValueError("total_tokens pozitif olmali")
        for name, value in (
            ("output_reserve_tokens", self.output_reserve_tokens),
            ("retrieval_max_tokens", self.retrieval_max_tokens),
            ("recent_max_tokens", self.recent_max_tokens),
        ):
            if value < 0:
                raise ValueError(f"{name} negatif olamaz")
        if self.output_reserve_tokens >= self.total_tokens:
            raise ValueError("output reserve tum context'i tuketemez")


@dataclass(frozen=True)
class AssembledPrompt:
    system_prompt: str
    summary: StructuredSummary | None
    retrieval_records: tuple[MessageRecord, ...]
    recent_messages: tuple[MessageRecord, ...]
    input_tokens: int
    output_reserve_tokens: int
    retrieval_tokens: int
    recent_tokens: int


def assemble_prompt(
    *,
    system_prompt: str,
    summary: StructuredSummary | None,
    retrieval_records: Sequence[MessageRecord],
    recent_messages: Sequence[MessageRecord],
    token_counter: TokenCounter,
    budget: PromptBudget,
) -> AssembledPrompt:
    budget.validate()
    if not system_prompt.strip():
        raise ValueError("system_prompt bos olamaz")
    system_tokens = token_counter(system_prompt)
    summary_tokens = token_counter(summary.render()) if summary is not None else 0
    fixed_tokens = system_tokens + summary_tokens + budget.output_reserve_tokens
    if fixed_tokens > budget.total_tokens:
        raise ValueError("system + summary + output reserve context sinirini asiyor")
    remaining = budget.total_tokens - fixed_tokens

    packed_retrieval: list[MessageRecord] = []
    retrieval_tokens = 0
    retrieval_limit = min(budget.retrieval_max_tokens, remaining)
    for record in retrieval_records:
        cost = message_token_count(record, token_counter)
        if cost > retrieval_limit or retrieval_tokens + cost > retrieval_limit:
            continue
        packed_retrieval.append(record)
        retrieval_tokens += cost
    remaining -= retrieval_tokens

    recent_limit = min(budget.recent_max_tokens, remaining)
    packed_recent_reversed: list[MessageRecord] = []
    recent_tokens = 0
    for message in reversed(tuple(recent_messages)):
        cost = message_token_count(message, token_counter)
        if recent_tokens + cost > recent_limit:
            break
        packed_recent_reversed.append(message)
        recent_tokens += cost
    packed_recent_reversed.reverse()
    input_tokens = system_tokens + summary_tokens + retrieval_tokens + recent_tokens
    return AssembledPrompt(
        system_prompt=system_prompt,
        summary=summary,
        retrieval_records=tuple(packed_retrieval),
        recent_messages=tuple(packed_recent_reversed),
        input_tokens=input_tokens,
        output_reserve_tokens=budget.output_reserve_tokens,
        retrieval_tokens=retrieval_tokens,
        recent_tokens=recent_tokens,
    )
