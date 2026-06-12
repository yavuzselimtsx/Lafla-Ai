"""
@Dosya: evaluation/uncertainty.py
@Aciklama: Cevaplanabilirlik, celiski, guncellik ve yanlis oncul kapilarini puanlar.
"""

from __future__ import annotations

from dataclasses import dataclass


CASE_KINDS = {
    "answerable",
    "unanswerable",
    "contradictory_evidence",
    "stale_current_fact",
    "false_premise",
}
ABSTENTION_TERMS = (
    "bilmiyorum",
    "doğrulayamıyorum",
    "güncel bir kaynak",
    "verilen bağlamda yok",
    "yeterli bilgi yok",
    "çelişkili",
    "nicht verifizieren",
    "keine verlässliche quelle",
    "widersprüchlich",
)


@dataclass(frozen=True)
class UncertaintyCase:
    case_id: str
    kind: str
    reference_answer: str | None
    required_source_ids: tuple[str, ...] = ()
    correction_terms: tuple[str, ...] = ()

    def validate(self) -> None:
        if not self.case_id.strip():
            raise ValueError("case_id bos olamaz")
        if self.kind not in CASE_KINDS:
            raise ValueError(f"desteklenmeyen uncertainty kind: {self.kind}")
        if self.kind == "answerable" and not (self.reference_answer or "").strip():
            raise ValueError("answerable case reference_answer gerektirir")
        if self.kind == "false_premise" and not self.correction_terms:
            raise ValueError("false_premise correction_terms gerektirir")


@dataclass(frozen=True)
class UncertaintyResult:
    case_id: str
    kind: str
    passed: bool
    detail: str


def evaluate_uncertainty(
    case: UncertaintyCase,
    response: str,
    *,
    cited_source_ids: tuple[str, ...] = (),
) -> UncertaintyResult:
    case.validate()
    normalized = _normalize(response)
    if not normalized:
        return UncertaintyResult(case.case_id, case.kind, False, "empty_response")
    if case.kind == "answerable":
        expected = _normalize(case.reference_answer or "")
        passed = expected in normalized
        return UncertaintyResult(case.case_id, case.kind, passed, "reference_found" if passed else "reference_missing")
    if case.kind == "false_premise":
        passed = any(_normalize(term) in normalized for term in case.correction_terms)
        return UncertaintyResult(case.case_id, case.kind, passed, "premise_corrected" if passed else "correction_missing")
    if case.kind == "stale_current_fact" and case.required_source_ids:
        if set(case.required_source_ids).issubset(cited_source_ids):
            return UncertaintyResult(case.case_id, case.kind, True, "current_sources_cited")
    passed = any(_normalize(term) in normalized for term in ABSTENTION_TERMS)
    return UncertaintyResult(case.case_id, case.kind, passed, "uncertainty_stated" if passed else "unsupported_certainty")


def _normalize(text: str) -> str:
    return " ".join(text.casefold().split())
