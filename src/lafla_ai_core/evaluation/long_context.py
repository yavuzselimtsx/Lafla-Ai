"""
@Dosya: evaluation/long_context.py
@Aciklama: Model calistirmadan passkey/needle cevaplarini deterministik puanlar.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class LongContextCase:
    case_id: str
    kind: str
    context_tokens: int
    needle_position: float
    expected_answer: str

    def validate(self) -> None:
        if self.kind not in {"passkey", "needle"}:
            raise ValueError("long-context kind passkey veya needle olmali")
        if self.context_tokens < 512:
            raise ValueError("context_tokens cok kucuk")
        if not 0.0 <= self.needle_position <= 1.0:
            raise ValueError("needle_position 0-1 araliginda olmali")
        if not self.expected_answer.strip():
            raise ValueError("expected_answer bos olamaz")


@dataclass(frozen=True)
class LongContextResult:
    case_id: str
    kind: str
    context_tokens: int
    needle_position: float
    passed: bool
    detail: str


def score_long_context_case(case: LongContextCase, response: str) -> LongContextResult:
    case.validate()
    expected = _normalize(case.expected_answer)
    actual = _normalize(response)
    passed = expected in actual
    return LongContextResult(
        case_id=case.case_id,
        kind=case.kind,
        context_tokens=case.context_tokens,
        needle_position=case.needle_position,
        passed=passed,
        detail="expected_answer_found" if passed else "expected_answer_missing",
    )


def _normalize(text: str) -> str:
    return " ".join(text.casefold().split())
