"""
Checkpoint yayın kararını kalite metriklerinden üretir.

Bir model tek bir smoke test geçti diye iyi sayılmaz. Bu kapı; Türkçe konuşma,
kimlik, kanıt disiplini ve güvenlik metrikleri eşiğin altındaysa checkpoint'i
reddeder.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class ReleaseMetrics:
    identity_score: float
    turkish_score: float
    conversation_helpfulness: float
    evidence_discipline: float
    unsafe_acceptance_rate: float
    unsupported_claim_rate: float


@dataclass(frozen=True)
class ReleaseDecision:
    accepted: bool
    reasons: tuple[str, ...]


def decide_release(metrics: ReleaseMetrics, gates: dict[str, float]) -> ReleaseDecision:
    """Kalite eşiğini geçmeyen checkpoint için net ret gerekçeleri üretir."""

    reasons: list[str] = []
    if metrics.identity_score < gates["minimumIdentityScore"]:
        reasons.append("kimlik skoru yetersiz")
    if metrics.turkish_score < gates["minimumTurkishScore"]:
        reasons.append("Türkçe skoru yetersiz")
    if metrics.conversation_helpfulness < gates["minimumConversationHelpfulness"]:
        reasons.append("konuşma yararlılığı yetersiz")
    if metrics.evidence_discipline < gates["minimumEvidenceDiscipline"]:
        reasons.append("kanıt disiplini yetersiz")
    if metrics.unsafe_acceptance_rate > gates["maximumUnsafeAcceptanceRate"]:
        reasons.append("güvensiz kabul oranı yüksek")
    if metrics.unsupported_claim_rate > gates["maximumUnsupportedClaimRate"]:
        reasons.append("kanıtsız iddia oranı yüksek")
    return ReleaseDecision(accepted=not reasons, reasons=tuple(reasons))
