"""
@Dosya: runtime/conversation_memory.py
@Aciklama: Token butceli konusma ozeti planini ve atomik state degisimini yonetir.
@Uyari: Gecersiz veya kaynak kimligi eksik ozet, mevcut state'i degistiremez.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable


TokenCounter = Callable[[str], int]
MESSAGE_FRAME_TOKENS = 8


@dataclass(frozen=True)
class MessageRecord:
    message_id: str
    role: str
    content: str

    def validate(self) -> None:
        if not self.message_id.strip():
            raise ValueError("message_id bos olamaz")
        if self.role not in {"system", "user", "assistant", "tool", "document"}:
            raise ValueError(f"desteklenmeyen message role: {self.role}")
        if not self.content.strip():
            raise ValueError("message content bos olamaz")


@dataclass(frozen=True)
class StructuredSummary:
    facts: tuple[str, ...]
    decisions: tuple[str, ...]
    open_questions: tuple[str, ...]
    user_preferences: tuple[str, ...]
    source_message_ids: tuple[str, ...]
    open_tasks: tuple[str, ...] = ()
    uncertainties: tuple[str, ...] = ()
    safety_context: tuple[str, ...] = ()

    def render(self) -> str:
        sections = (
            ("Olgular", self.facts),
            ("Kararlar", self.decisions),
            ("Acik sorular", self.open_questions),
            ("Kullanici tercihleri", self.user_preferences),
            ("Acik gorevler", self.open_tasks),
            ("Belirsizlikler", self.uncertainties),
            ("Guvenlik baglami", self.safety_context),
        )
        lines: list[str] = []
        for title, values in sections:
            if values:
                lines.append(f"{title}:")
                lines.extend(f"- {value}" for value in values)
        return "\n".join(lines)

    def validate(self) -> None:
        if not self.source_message_ids:
            raise ValueError("summary source_message_ids bos olamaz")
        if len(set(self.source_message_ids)) != len(self.source_message_ids):
            raise ValueError("summary source_message_ids tekrarli olamaz")
        if not self.render().strip():
            raise ValueError("summary en az bir dolu bolum icermeli")


@dataclass(frozen=True)
class ConversationState:
    summary: StructuredSummary | None
    messages: tuple[MessageRecord, ...]


@dataclass(frozen=True)
class SummaryPlan:
    required: bool
    total_tokens: int
    source_message_ids: tuple[str, ...]
    preserved_message_ids: tuple[str, ...]
    preserved_tokens: int


@dataclass(frozen=True)
class SummaryTransactionResult:
    state: ConversationState
    committed: bool
    reason: str


def message_token_count(message: MessageRecord, token_counter: TokenCounter) -> int:
    message.validate()
    return token_counter(message.content) + MESSAGE_FRAME_TOKENS


def build_summary_plan(
    state: ConversationState,
    *,
    token_counter: TokenCounter,
    trigger_tokens: int,
    preserve_recent_tokens: int,
) -> SummaryPlan:
    if trigger_tokens <= 0 or preserve_recent_tokens <= 0:
        raise ValueError("summary token sinirlari pozitif olmali")
    message_costs = tuple(message_token_count(message, token_counter) for message in state.messages)
    summary_tokens = token_counter(state.summary.render()) if state.summary is not None else 0
    total_tokens = summary_tokens + sum(message_costs)
    if total_tokens <= trigger_tokens:
        return SummaryPlan(False, total_tokens, (), tuple(message.message_id for message in state.messages), sum(message_costs))
    if state.messages and message_costs[-1] > preserve_recent_tokens:
        raise ValueError("son mesaj preserve_recent_tokens butcesini asiyor; sessizce ozetlenemez")

    preserved: list[MessageRecord] = []
    preserved_tokens = 0
    for message, cost in zip(reversed(state.messages), reversed(message_costs)):
        if preserved_tokens + cost > preserve_recent_tokens:
            break
        preserved.append(message)
        preserved_tokens += cost
    preserved.reverse()
    preserved_ids = tuple(message.message_id for message in preserved)
    preserved_set = set(preserved_ids)
    source_ids = tuple(state.summary.source_message_ids if state.summary is not None else ()) + tuple(
        message.message_id for message in state.messages if message.message_id not in preserved_set
    )
    return SummaryPlan(True, total_tokens, source_ids, preserved_ids, preserved_tokens)


def apply_summary(
    state: ConversationState,
    plan: SummaryPlan,
    candidate: StructuredSummary,
    *,
    token_counter: TokenCounter,
    summary_max_tokens: int,
) -> SummaryTransactionResult:
    if not plan.required:
        return SummaryTransactionResult(state, False, "summary_not_required")
    try:
        candidate.validate()
    except ValueError as exc:
        return SummaryTransactionResult(state, False, str(exc))
    if candidate.source_message_ids != plan.source_message_ids:
        return SummaryTransactionResult(state, False, "summary source_message_ids plan ile ayni olmali")
    if token_counter(candidate.render()) > summary_max_tokens:
        return SummaryTransactionResult(state, False, "summary token sinirini asti")
    preserved = tuple(
        message for message in state.messages if message.message_id in set(plan.preserved_message_ids)
    )
    return SummaryTransactionResult(ConversationState(candidate, preserved), True, "committed")
