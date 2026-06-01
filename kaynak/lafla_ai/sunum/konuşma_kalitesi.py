"""
Lafla AI cevaplarını kullanıcıyla doğal Türkçe konuşmaya uygun hale getirir.

Amaç modeli rol yapmaya zorlamak değil; cevapların kısa, doğru, bağlama sadık
ve belirsizlikleri açık söyleyen bir üsluba sahip olmasını sağlamaktır.
"""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class ConversationQualityConfig:
    language: str
    tone: str
    max_intro_sentences: int
    prefer_actionable_answer: bool
    admit_uncertainty: bool
    dialogue_capabilities: tuple[str, ...]
    anti_hallucination_rules: tuple[str, ...]


def load_conversation_quality(path: Path) -> ConversationQualityConfig:
    """Konuşma kalitesi yapılandırmasını okur."""

    payload = json.loads(path.read_text(encoding="utf-8"))
    style = require_mapping(payload.get("defaultStyle"), "defaultStyle")
    return ConversationQualityConfig(
        language=require_text(style, "language"),
        tone=require_text(style, "tone"),
        max_intro_sentences=int(style.get("maxIntroSentences", 1)),
        prefer_actionable_answer=bool(style.get("preferActionableAnswer", True)),
        admit_uncertainty=bool(style.get("admitUncertainty", True)),
        dialogue_capabilities=tuple(require_text_list(payload, "dialogueCapabilities")),
        anti_hallucination_rules=tuple(require_text_list(payload, "antiHallucinationRules")),
    )


def render_conversation_policy(config: ConversationQualityConfig) -> str:
    """Sistem yönergesine eklenebilir konuşma politikası üretir."""

    capabilities = "\n".join(f"- {item}" for item in config.dialogue_capabilities)
    hallucination_rules = "\n".join(f"- {item}" for item in config.anti_hallucination_rules)
    return (
        f"Konuşma dili: {config.language}\n"
        f"Üslup: {config.tone}\n"
        f"Azami giriş cümlesi: {config.max_intro_sentences}\n"
        f"Konuşma becerileri:\n{capabilities}\n"
        f"Halüsinasyon karşıtı kurallar:\n{hallucination_rules}"
    )


def require_mapping(value: Any, field_name: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ValueError(f"{field_name} must be an object")
    return value


def require_text(mapping: dict[str, Any], field_name: str) -> str:
    value = mapping.get(field_name)
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{field_name} must be a non-empty string")
    return value.strip()


def require_text_list(mapping: dict[str, Any], field_name: str) -> list[str]:
    value = mapping.get(field_name)
    if not isinstance(value, list) or not value:
        raise ValueError(f"{field_name} must be a non-empty list")
    result = [item.strip() for item in value if isinstance(item, str) and item.strip()]
    if len(result) != len(value):
        raise ValueError(f"{field_name} contains invalid items")
    return result
