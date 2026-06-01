"""
Lafla AI sohbet cevabını ürün kalitesi kurallarına göre biçimlendirir.

Bu dosya gerçek model örneklemesini yapmaz. Üretim katmanında modelden gelen
taslak cevabın Lafla kimliği, Türkçe üslup, kısa açıklık ve güvenlik açısından
denetlenmesine yarar.
"""

from __future__ import annotations

from dataclasses import dataclass

from lafla_ai.güvenlik.politika import classify_user_request, validate_assistant_answer


@dataclass(frozen=True)
class DraftAnswer:
    """Modelden gelen ham cevabı ve kalite durumunu taşır."""

    accepted: bool
    text: str
    reason: str


def prepare_generation_prompt(system_prompt: str, user_message: str) -> str:
    """Sohbet modeli için güvenlikten geçmiş üretim girdisi hazırlar."""

    decision = classify_user_request(user_message)
    if not decision.allowed:
        raise ValueError(decision.reason)
    return "\n".join(
        [
            "<|system|>",
            system_prompt.strip(),
            "<|user|>",
            user_message.strip(),
            "<|assistant|>",
        ]
    )


def accept_or_rewrite_answer(raw_answer: str) -> DraftAnswer:
    """Ham cevabı güvenlik ve ürün üslubu açısından kabul eder veya reddeder."""

    cleaned = " ".join(raw_answer.strip().split())
    if not cleaned:
        return DraftAnswer(False, "", "empty_answer")
    policy = validate_assistant_answer(cleaned)
    if not policy.allowed:
        return DraftAnswer(False, "Bunu bu şekilde yanıtlayamam.", policy.reason)
    if not cleaned.startswith("Lafla AI"):
        cleaned = f"Lafla AI: {cleaned}"
    return DraftAnswer(True, cleaned, "accepted")
