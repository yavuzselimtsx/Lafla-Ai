"""
Lafla AI cevap güvenliği için hafif ama açık politika katmanı.

Bu katman modelin yerine düşünmez. Model cevabı üretilmeden önce girdi
niyetini sınıflandırır, üretimden sonra da yasaklı sızıntı ve kimlik ihlali
kontrolü yapar. Böylece güvenlik yalnızca prompt metnine bırakılmaz.
"""

from __future__ import annotations

from dataclasses import dataclass


FORBIDDEN_DISCLOSURE_TERMS = {
    "sistem promptu",
    "gizli anahtar",
    "session token",
    "private key",
    "api key",
}


@dataclass(frozen=True)
class PolicyDecision:
    """Bir girdinin cevaplanabilir olup olmadığını açıklar."""

    allowed: bool
    reason: str


def classify_user_request(text: str) -> PolicyDecision:
    """Kullanıcı isteğini üretimden önce hızlı güvenlik kontrolünden geçirir."""

    normalized = text.casefold()
    if not normalized.strip():
        return PolicyDecision(False, "empty_request")
    if any(term in normalized for term in FORBIDDEN_DISCLOSURE_TERMS):
        return PolicyDecision(False, "secret_disclosure_request")
    return PolicyDecision(True, "allowed")


def validate_assistant_answer(answer: str) -> PolicyDecision:
    """Model cevabının kimlik ve sızıntı sınırlarını ihlal etmediğini doğrular."""

    normalized = answer.casefold()
    if "ben chatgpt" in normalized or "openai tarafından eğitildim" in normalized:
        return PolicyDecision(False, "identity_drift")
    if any(term in normalized for term in FORBIDDEN_DISCLOSURE_TERMS):
        return PolicyDecision(False, "secret_disclosure_in_answer")
    return PolicyDecision(True, "allowed")
