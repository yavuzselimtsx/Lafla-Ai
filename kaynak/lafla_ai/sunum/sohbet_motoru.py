"""
Lafla AI sohbet oturumunu ve sistem yönergesini yönetir.

Bu modül model ağırlığına bağlı değildir. Üretim tarafında yapılacak gerçek
örnekleme çağrısına temiz, denetlenebilir ve kimliği açık bir prompt verir.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import json
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class LaflaIdentity:
    """Kimlik yapılandırmasından okunan model davranış sözleşmesi."""

    name: str
    product: str
    default_language: str
    short_description: str
    behavior_rules: list[str]


@dataclass
class ConversationTurn:
    """Sohbet geçmişindeki tek kullanıcı veya asistan mesajı."""

    role: str
    content: str


@dataclass
class LaflaChatSession:
    """Lafla AI konuşma oturumunu sistem yönergesiyle birlikte tutar."""

    identity_config: Path
    turns: list[ConversationTurn] = field(default_factory=list)

    def system_prompt(self) -> str:
        """Oturum için güncel sistem yönergesini üretir."""

        identity = load_identity(self.identity_config)
        rules = "\n".join(f"- {rule}" for rule in identity.behavior_rules)
        return (
            f"Sen {identity.name} adlı yardımcı modelsin.\n"
            f"Ürün: {identity.product}\n"
            f"Varsayılan dil: {identity.default_language}\n"
            f"Tanım: {identity.short_description}\n"
            f"Kurallar:\n{rules}"
        )

    def add_user_message(self, content: str) -> None:
        """Kullanıcı mesajını boş olmayan içerikle oturuma ekler."""

        self.turns.append(ConversationTurn(role="user", content=clean_content(content)))

    def add_assistant_message(self, content: str) -> None:
        """Asistan cevabını boş olmayan içerikle oturuma ekler."""

        self.turns.append(ConversationTurn(role="assistant", content=clean_content(content)))

    def render_prompt(self) -> str:
        """Model giriş metnini sistem yönergesi ve konuşma geçmişinden üretir."""

        lines = [f"<|system|>{self.system_prompt()}"]
        for turn in self.turns:
            marker = "<|user|>" if turn.role == "user" else "<|assistant|>"
            lines.append(f"{marker}{turn.content}")
        lines.append("<|assistant|>")
        return "\n".join(lines)


def clean_content(content: str) -> str:
    """Boş veya yalnız boşluk içeren sohbet mesajlarını reddeder."""

    cleaned = content.strip()
    if not cleaned:
        raise ValueError("content is empty")
    return cleaned


def load_identity(path: Path) -> LaflaIdentity:
    """JSON kimlik yapılandırmasını okur ve gerekli alanları doğrular."""

    payload = json.loads(path.read_text(encoding="utf-8"))
    identity = require_mapping(payload.get("model_identity"), "model_identity")
    behavior_rules = payload.get("behavior_rules")
    if not isinstance(behavior_rules, list) or not all(isinstance(rule, str) for rule in behavior_rules):
        raise ValueError("behavior_rules must be a string list")
    return LaflaIdentity(
        name=require_text(identity, "name"),
        product=require_text(identity, "product"),
        default_language=require_text(identity, "default_language"),
        short_description=require_text(identity, "short_description"),
        behavior_rules=behavior_rules,
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
