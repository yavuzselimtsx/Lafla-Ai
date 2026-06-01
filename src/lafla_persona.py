"""
 _        _    _____ _        _
| |      / \\  |  ___| |      / \\
| |     / _ \\ | |_  | |     / _ \\
| |___ / ___ \\|  _| | |___ / ___ \\
|_____/_/   \\_\\_|   |_____/_/   \\_\\

@Dosya: lafla_persona.py
@Açıklama: Lafla AI kimliğini, davranış kurallarını ve öğretim örneklerini üretir.
@Yazar: Lafla Geliştirme Ekibi
@Bilgi: Kimlik bilgisi model ağırlığına sihirli şekilde gömülmez; veri ve sistem yönergesiyle öğretilir.
@Uyarı: Bu dosya ürün sırrı veya üretim anahtarı içermez.
@Çalışma-Şeması: identity json -> validation -> system prompt / instruction records
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class LaflaIdentity:
    """Model kimliği ve davranış kurallarını tipli şekilde taşır."""

    name: str
    product: str
    default_language: str
    short_description: str
    behavior_rules: tuple[str, ...]
    instruction_examples: tuple[dict[str, str], ...]


def load_identity(path: str | Path) -> LaflaIdentity:
    """Kimlik yapılandırmasını okur ve gerekli alanlar yoksa hata verir."""

    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    identity = payload.get("model_identity")
    if not isinstance(identity, dict):
        raise ValueError("model_identity missing")
    behavior_rules = require_string_list(payload, "behavior_rules")
    examples = payload.get("instruction_examples")
    if not isinstance(examples, list) or not examples:
        raise ValueError("instruction_examples missing")
    normalized_examples = tuple(normalize_example(item) for item in examples)
    return LaflaIdentity(
        name=require_string(identity, "name"),
        product=require_string(identity, "product"),
        default_language=require_string(identity, "default_language"),
        short_description=require_string(identity, "short_description"),
        behavior_rules=tuple(behavior_rules),
        instruction_examples=normalized_examples,
    )


def build_system_prompt(identity: LaflaIdentity) -> str:
    """Modelin çalışma zamanı sistem yönergesini kararlı bir metne çevirir."""

    rules = "\n".join(f"- {rule}" for rule in identity.behavior_rules)
    return (
        f"Sen {identity.name} adlı yardımcı modelsin.\n"
        f"Ürün: {identity.product}\n"
        f"Tanım: {identity.short_description}\n"
        f"Varsayılan dil: {identity.default_language}\n"
        "Kurallar:\n"
        f"{rules}"
    )


def iter_instruction_records(identity: LaflaIdentity) -> list[dict[str, str]]:
    """Kimliği öğretmek için instruction-tuning kayıtlarını üretir."""

    system = build_system_prompt(identity)
    return [
        {
            "system": system,
            "user": example["user"],
            "assistant": example["assistant"],
        }
        for example in identity.instruction_examples
    ]


def require_string(payload: dict[str, Any], key: str) -> str:
    """Sözlükten boş olmayan string alan okur."""

    value = payload.get(key)
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{key} missing")
    return value.strip()


def require_string_list(payload: dict[str, Any], key: str) -> list[str]:
    """Sözlükten boş olmayan string listesi okur."""

    value = payload.get(key)
    if not isinstance(value, list) or not value:
        raise ValueError(f"{key} missing")
    result = [item.strip() for item in value if isinstance(item, str) and item.strip()]
    if len(result) != len(value):
        raise ValueError(f"{key} contains invalid item")
    return result


def normalize_example(value: object) -> dict[str, str]:
    """Tek instruction örneğini user/assistant alanlarına indirger."""

    if not isinstance(value, dict):
        raise ValueError("instruction example must be object")
    return {
        "user": require_string(value, "user"),
        "assistant": require_string(value, "assistant"),
    }
