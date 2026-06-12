"""
@Dosya: runtime/policy.py
@Aciklama: LaflaGPT çıktısını geliştirici ve üretim profillerine göre işler.
@Yazar: Lafla Geliştirme Ekibi
@Bilgi: Model ağırlıkları sabit kalsa bile thinking görünürlüğü, düşünme bütçesi
        ve prompt sızıntı denetimi runtime config ile yönetilir.
@Uyari: Research profili ham thinking'i gösterir; üretim profili private bloğu
        public cevaptan ayıklar.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

from lafla_ai_core.config.schema import RuntimeConfig
from lafla_ai_core.post_training.thinking_sft import THINK_CLOSE, THINK_OPEN, strip_thinking_for_public
from lafla_ai_core.runtime.decoding import clean_decoded_text
from lafla_ai_core.runtime.output_guard import ROLE_STOP_SEQUENCES, sanitize_completion


ROLE_MARKERS = ("<|system|>", "<|user|>", "<|assistant|>", "<|bos|>", "<|eos|>", "<|pad|>")
PROMPT_LEAK_TERMS = ("system prompt", "sistem prompt", "sistem istemi", "<|system|>")
MOJIBAKE_MARKERS = ("Ã", "Ä", "Å", "Â", "�", "ï¿½")


@dataclass(frozen=True)
class RuntimeOutput:
    """Runtime politika işleminden geçmiş model cevabı."""

    public_text: str
    raw_thinking: str | None
    warnings: tuple[str, ...]


@dataclass(frozen=True)
class GenerationSettings:
    """Inference sırasında modele verilecek ayarlanabilir üretim değerleri."""

    max_new_tokens: int
    temperature: float
    top_p: float
    repetition_penalty: float
    thinking_effort: str
    thinking_budget_tokens: int
    context_overflow_strategy: str
    stop_sequences: tuple[str, ...]


def render_runtime_output(
    raw_text: str,
    config: RuntimeConfig,
    *,
    prompt_text: str | None = None,
    system_text: str | None = None,
) -> RuntimeOutput:
    """Ham model çıktısını public cevap ve opsiyonel developer thinking'e ayırır."""

    config.validate()
    guarded = sanitize_completion(raw_text, prompt_text=prompt_text, system_text=system_text)
    thinking_sections = tuple(_extract_thinking(guarded.text))
    public_text = _normalize_public_text(clean_decoded_text(strip_thinking_for_public(guarded.text), strip_special_tokens=True))
    warnings = _merge_warnings(_runtime_warnings(raw_text, public_text, config), guarded.warnings)
    raw_thinking = (
        "\n\n".join(clean_decoded_text(section, strip_special_tokens=False) for section in thinking_sections).strip()
        if config.raw_thinking_visible and thinking_sections
        else None
    )
    return RuntimeOutput(public_text=public_text, raw_thinking=raw_thinking, warnings=warnings)


def build_generation_settings(config: RuntimeConfig) -> GenerationSettings:
    """Runtime config'i inference parametrelerine çevirir."""

    config.validate()
    return GenerationSettings(
        max_new_tokens=config.max_new_tokens,
        temperature=config.temperature,
        top_p=config.top_p,
        repetition_penalty=config.repetition_penalty,
        thinking_effort=config.thinking_effort,
        thinking_budget_tokens=_resolved_thinking_budget(config),
        context_overflow_strategy=config.context_overflow_strategy,
        stop_sequences=ROLE_STOP_SEQUENCES,
    )


def _extract_thinking(text: str) -> list[str]:
    pattern = re.escape(THINK_OPEN) + r"(.*?)" + re.escape(THINK_CLOSE)
    return [match.strip() for match in re.findall(pattern, text, flags=re.DOTALL) if match.strip()]


def _normalize_public_text(text: str) -> str:
    cleaned = text
    for marker in ROLE_MARKERS:
        cleaned = cleaned.replace(marker, " ")
    return " ".join(cleaned.split())


def _runtime_warnings(raw_text: str, public_text: str, config: RuntimeConfig) -> tuple[str, ...]:
    warnings: list[str] = []
    lowered = raw_text.lower()
    if config.system_prompt_leak_guard and any(term in lowered for term in PROMPT_LEAK_TERMS):
        warnings.append("possible_prompt_leak")
    if not config.raw_thinking_visible and (THINK_OPEN in raw_text or THINK_CLOSE in raw_text):
        warnings.append("private_thinking_stripped")
    if config.turkish_quality_guard and _looks_like_mojibake(public_text):
        warnings.append("possible_mojibake")
    return tuple(warnings)


def _resolved_thinking_budget(config: RuntimeConfig) -> int:
    if config.thinking_budget_tokens > 0:
        return config.thinking_budget_tokens
    return {"low": 64, "medium": 192, "high": 512}[config.thinking_effort]


def _looks_like_mojibake(text: str) -> bool:
    return any(marker in text for marker in MOJIBAKE_MARKERS)


def _merge_warnings(*groups: tuple[str, ...]) -> tuple[str, ...]:
    merged: list[str] = []
    seen: set[str] = set()
    for group in groups:
        for warning in group:
            if warning not in seen:
                seen.add(warning)
                merged.append(warning)
    return tuple(merged)
