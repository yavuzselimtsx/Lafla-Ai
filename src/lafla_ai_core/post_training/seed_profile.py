"""
@Dosya: post_training/seed_profile.py
@Aciklama: Sentetik SFT seed ureticileri icin dis profil yukleyici.
@Bilgi: Model kimligi, dil odagi ve davranis metinleri Python koduna gomulmez;
        profil dosyasi degistirilerek yeni model ailesi uretilebilir.
"""

from __future__ import annotations

import json
from pathlib import Path
from string import Formatter
from typing import Any, Mapping, Sequence


DEFAULT_SEED_PROFILE_PATH = Path("configs/post_training/lafla-100m-seed-profile.json")


class SeedProfileError(ValueError):
    """Seed profil sozlesmesi bozuldugunda yukseltir."""


def load_seed_profile(path: str | Path = DEFAULT_SEED_PROFILE_PATH) -> dict[str, Any]:
    """JSON seed profilini fail-closed okur."""

    source = Path(path)
    payload = json.loads(source.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise SeedProfileError("seed profili JSON nesnesi olmali")
    for section in ("model", "chat_seed", "safety_seed"):
        if not isinstance(payload.get(section), dict):
            raise SeedProfileError(f"seed profilinde {section!r} bolumu olmali")
    return payload


def section_count(profile: Mapping[str, Any], section: str) -> int:
    value = _section(profile, section).get("count")
    if not isinstance(value, int) or value <= 0:
        raise SeedProfileError(f"{section}.count pozitif int olmali")
    return value


def section_path(profile: Mapping[str, Any], section: str, key: str) -> Path:
    value = _section(profile, section).get(key)
    if not isinstance(value, str) or not value:
        raise SeedProfileError(f"{section}.{key} bos olmayan string olmali")
    return Path(value)


def section_text(profile: Mapping[str, Any], section: str, key: str) -> str:
    value = _section(profile, section).get(key)
    if not isinstance(value, str) or not value:
        raise SeedProfileError(f"{section}.{key} bos olmayan string olmali")
    return value


def model_context(profile: Mapping[str, Any]) -> dict[str, str]:
    model = _section(profile, "model")
    context: dict[str, str] = {}
    for key, value in model.items():
        if isinstance(value, str):
            context[key] = value
    if not context.get("identity_statement"):
        raise SeedProfileError("model.identity_statement bos olamaz")
    return context


def render_template(template: str, context: Mapping[str, str]) -> str:
    """Profil template stringini guvenli formatlar."""

    fields = [field_name for _, field_name, _, _ in Formatter().parse(template) if field_name]
    missing = [field for field in fields if field not in context]
    if missing:
        raise SeedProfileError(f"template eksik model alani istiyor: {missing}")
    return template.format_map(context)


def scenario_items(profile: Mapping[str, Any], section: str) -> tuple[Mapping[str, Any], ...]:
    scenarios = _section(profile, section).get("scenarios")
    if not isinstance(scenarios, list) or not scenarios:
        raise SeedProfileError(f"{section}.scenarios bos olmayan liste olmali")
    for scenario in scenarios:
        if not isinstance(scenario, dict):
            raise SeedProfileError(f"{section}.scenarios nesne listesi olmali")
    return tuple(scenarios)


def modifier_items(profile: Mapping[str, Any], section: str) -> tuple[Mapping[str, Any], ...]:
    modifiers = _section(profile, section).get("modifiers")
    if not isinstance(modifiers, list) or not modifiers:
        raise SeedProfileError(f"{section}.modifiers bos olmayan liste olmali")
    for modifier in modifiers:
        if not isinstance(modifier, dict):
            raise SeedProfileError(f"{section}.modifiers nesne listesi olmali")
    return tuple(modifiers)


def localized_modifier(modifier: Mapping[str, Any], language: str) -> str:
    value = modifier.get(language) or modifier.get("default")
    if not isinstance(value, str) or not value:
        raise SeedProfileError(f"modifier {modifier.get('id', '<no-id>')} icin {language} metni yok")
    return value


def localized_suffixes(profile: Mapping[str, Any], section: str, language: str) -> Mapping[str, str]:
    suffixes = _section(profile, section).get("suffixes", {})
    if not isinstance(suffixes, dict):
        raise SeedProfileError(f"{section}.suffixes nesne olmali")
    raw = suffixes.get(language) or suffixes.get("default")
    if not isinstance(raw, dict):
        raise SeedProfileError(f"{section}.suffixes icinde {language} veya default olmali")
    output: dict[str, str] = {}
    for key, value in raw.items():
        if isinstance(value, str):
            output[str(key)] = value
    return output


def require_keys(item: Mapping[str, Any], keys: Sequence[str], context: str) -> None:
    missing = [key for key in keys if key not in item]
    if missing:
        raise SeedProfileError(f"{context} eksik alanlar: {missing}")


def _section(profile: Mapping[str, Any], section: str) -> Mapping[str, Any]:
    value = profile.get(section)
    if not isinstance(value, Mapping):
        raise SeedProfileError(f"{section!r} bolumu nesne olmali")
    return value

