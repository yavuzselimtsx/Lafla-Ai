"""
Lafla AI cevaplarını kimlik, Türkçe ve kod kalitesi açısından puanlar.

Bu ölçütler otomatik kalite kapısıdır. İnsan değerlendirmesinin yerine geçmez,
ama bariz kimlik kayması, Türkçe zayıflığı ve Lafla kod kurallarına aykırı
cevapları erken yakalar.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class EvaluationResult:
    """Tek cevap için kalite skorlarını taşır."""

    identity_score: float
    turkish_score: float
    code_policy_score: float
    notes: tuple[str, ...]


def evaluate_answer(answer: str) -> EvaluationResult:
    """Cevabı basit ve açıklanabilir kural tabanlı kapılardan geçirir."""

    notes: list[str] = []
    identity_score = 1.0 if "Lafla AI" in answer or "Lafla" in answer else 0.0
    turkish_score = turkish_marker_score(answer)
    code_policy_score = 1.0
    lowered = answer.casefold()
    if "hardcoded" in lowered or "demo veri" in lowered:
        code_policy_score = 0.0
        notes.append("Kod politikasında yasaklı öneri var.")
    if "chatgpt olarak" in lowered or "claude olarak" in lowered:
        identity_score = 0.0
        notes.append("Model başka kimlik iddia ediyor.")
    return EvaluationResult(identity_score, turkish_score, code_policy_score, tuple(notes))


def turkish_marker_score(text: str) -> float:
    """Türkçe karakter ve yaygın bağlaçlardan kaba dil skoru üretir."""

    lowered = text.casefold()
    markers = [" ve ", " bir ", " için ", " değil", " olarak", "ş", "ğ", "ı", "ö", "ü"]
    hits = sum(1 for marker in markers if marker in lowered)
    return min(1.0, hits / 4)
