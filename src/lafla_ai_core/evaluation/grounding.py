"""
@Dosya: evaluation/grounding.py
@Aciklama: Hallucination riskini azaltmak icin kanitli cevap release kapisi.
@Yazar: Lafla Gelistirme Ekibi
@Bilgi: Bu deterministik gate, model egitiminin yerini tutmaz; eval raporunda desteksiz iddialari kirmizi yapar.
"""

from __future__ import annotations

from dataclasses import dataclass


UNCERTAINTY_TERMS = ("emin değilim", "kanıt yok", "destekleyen kanıt yok", "bilmiyorum", "doğrulayamıyorum")


@dataclass(frozen=True)
class GroundedAnswerCase:
    """Tek grounded-answer eval kaydi."""

    prompt: str
    answer: str
    evidence_ids: tuple[str, ...]
    require_evidence: bool = True


@dataclass(frozen=True)
class GroundedAnswerResult:
    """Grounded-answer gate sonucu."""

    name: str
    passed: bool
    detail: str


def evaluate_grounded_answer(case: GroundedAnswerCase) -> GroundedAnswerResult:
    """Cevabin kanit veya acik belirsizlik tasidigini denetler."""

    answer = case.answer.strip()
    if not answer:
        return GroundedAnswerResult("grounded_answering", False, "empty_answer")
    if not case.require_evidence:
        return GroundedAnswerResult("grounded_answering", True, "evidence_not_required")
    if _states_uncertainty(answer):
        return GroundedAnswerResult("grounded_answering", True, "uncertainty_stated")
    for evidence_id in case.evidence_ids:
        if f"[kaynak:{evidence_id}]" in answer or f"[source:{evidence_id}]" in answer:
            return GroundedAnswerResult("grounded_answering", True, f"cited:{evidence_id}")
    return GroundedAnswerResult("grounded_answering", False, "missing_evidence")


def _states_uncertainty(answer: str) -> bool:
    lowered = answer.lower()
    return any(term in lowered for term in UNCERTAINTY_TERMS)
