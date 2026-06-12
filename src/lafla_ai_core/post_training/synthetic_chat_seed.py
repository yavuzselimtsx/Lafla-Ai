"""
@Dosya: post_training/synthetic_chat_seed.py
@Aciklama: Profil dosyasindan deterministik sohbet/thinking SFT seed uretir.
@Bilgi: Model kimligi ve davranis metinleri kodda degil, seed profilindedir.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping

from lafla_ai_core.post_training.seed_profile import (
    DEFAULT_SEED_PROFILE_PATH,
    load_seed_profile,
    localized_modifier,
    localized_suffixes,
    model_context,
    modifier_items,
    render_template,
    require_keys,
    scenario_items,
    section_count,
    section_path,
    section_text,
)
from lafla_ai_core.post_training.thinking_sft import ThinkingSftRecord


DEFAULT_OUTPUT_PATH = Path("datasets/post_training/thinking/jsonl/lafla-100m-thinking-chat-seed-20k.jsonl")
DEFAULT_MANIFEST_PATH = Path("datasets/post_training/thinking/manifests/lafla-100m-thinking-chat-seed-20k.manifest.json")
DEFAULT_COUNT = 20_000


@dataclass(frozen=True)
class SyntheticChatSeedOptions:
    """Sentetik chat seed uretim ayarlarini tasir."""

    count: int = DEFAULT_COUNT
    output_path: Path = DEFAULT_OUTPUT_PATH
    manifest_path: Path = DEFAULT_MANIFEST_PATH
    profile_path: Path = DEFAULT_SEED_PROFILE_PATH


@dataclass(frozen=True)
class SyntheticChatSeedReport:
    """Dataset uretim raporu."""

    output_path: str
    manifest_path: str
    records_written: int
    dataset_version: str
    data_kind: str

    def to_json(self) -> str:
        return json.dumps(asdict(self), ensure_ascii=False, indent=2, sort_keys=True)


@dataclass(frozen=True)
class _Scenario:
    family: str
    language: str
    system: str
    user: str
    thinking: str
    assistant: str


def generate_synthetic_chat_seed(
    *,
    output_path: str | Path = DEFAULT_OUTPUT_PATH,
    manifest_path: str | Path = DEFAULT_MANIFEST_PATH,
    count: int = DEFAULT_COUNT,
    profile_path: str | Path = DEFAULT_SEED_PROFILE_PATH,
) -> SyntheticChatSeedReport:
    """Varsayilan sentetik thinking chat seed dosyasini yazar."""

    return write_synthetic_chat_seed(
        SyntheticChatSeedOptions(
            count=count,
            output_path=Path(output_path),
            manifest_path=Path(manifest_path),
            profile_path=Path(profile_path),
        )
    )


def write_synthetic_chat_seed(options: SyntheticChatSeedOptions) -> SyntheticChatSeedReport:
    """JSONL dataset ve manifest dosyasini uretir."""

    profile = load_seed_profile(options.profile_path)
    if options.count <= 0:
        raise ValueError("count pozitif olmali")
    output = Path(options.output_path)
    manifest = Path(options.manifest_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    manifest.parent.mkdir(parents=True, exist_ok=True)

    records_written = 0
    with output.open("w", encoding="utf-8", newline="\n") as handle:
        for record in iter_synthetic_chat_seed_records(options):
            handle.write(json.dumps(asdict(record), ensure_ascii=False, sort_keys=True) + "\n")
            records_written += 1

    manifest.write_text(
        json.dumps(_build_manifest(profile, records_written), ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return SyntheticChatSeedReport(
        output_path=str(output),
        manifest_path=str(manifest),
        records_written=records_written,
        dataset_version=section_text(profile, "chat_seed", "dataset_version"),
        data_kind=section_text(profile, "chat_seed", "data_kind"),
    )


def iter_synthetic_chat_seed_records(options: SyntheticChatSeedOptions) -> Iterable[ThinkingSftRecord]:
    """Deterministik ve tekrar uretilebilir ThinkingSftRecord akisi verir."""

    profile = load_seed_profile(options.profile_path)
    context = model_context(profile)
    scenarios = _load_scenarios(profile, context)
    modifiers = modifier_items(profile, "chat_seed")
    for index in range(options.count):
        scenario = scenarios[index % len(scenarios)]
        modifier = modifiers[(index // len(scenarios)) % len(modifiers)]
        variant = (index // (len(scenarios) * len(modifiers))) + 1
        yield _compose_record(profile, scenario, modifier, f"{variant:03d}")


def profile_default_options(profile_path: str | Path = DEFAULT_SEED_PROFILE_PATH) -> SyntheticChatSeedOptions:
    """Profildeki varsayilan chat seed ayarlarini options'a cevirir."""

    profile = load_seed_profile(profile_path)
    return SyntheticChatSeedOptions(
        count=section_count(profile, "chat_seed"),
        output_path=section_path(profile, "chat_seed", "output_path"),
        manifest_path=section_path(profile, "chat_seed", "manifest_path"),
        profile_path=Path(profile_path),
    )


def _load_scenarios(profile: Mapping[str, Any], context: Mapping[str, str]) -> tuple[_Scenario, ...]:
    scenarios: list[_Scenario] = []
    for index, raw in enumerate(scenario_items(profile, "chat_seed"), start=1):
        require_keys(raw, ("family", "language", "system", "user", "thinking", "assistant"), f"chat_seed.scenarios[{index}]")
        scenarios.append(
            _Scenario(
                family=str(raw["family"]),
                language=str(raw["language"]),
                system=render_template(str(raw["system"]), context),
                user=render_template(str(raw["user"]), context),
                thinking=render_template(str(raw["thinking"]), context),
                assistant=render_template(str(raw["assistant"]), context),
            )
        )
    return tuple(scenarios)


def _compose_record(profile: Mapping[str, Any], scenario: _Scenario, modifier: Mapping[str, Any], variant: str) -> ThinkingSftRecord:
    modifier_text = localized_modifier(modifier, scenario.language)
    suffixes = localized_suffixes(profile, "chat_seed", scenario.language)
    system = f"{scenario.system} {_format_suffix(suffixes, 'system_extra', modifier_text, variant)}"
    user = f"{scenario.user} {_format_suffix(suffixes, 'user_extra', modifier_text, variant)}"
    thinking = (
        f"{suffixes['thinking_prefix']} {scenario.thinking} "
        f"{_format_suffix(suffixes, 'thinking_extra', modifier_text, variant)}"
    )
    assistant = _compose_assistant(scenario, modifier, modifier_text, variant, suffixes)
    return ThinkingSftRecord(system=system, user=user, thinking=thinking, assistant=assistant)


def _compose_assistant(
    scenario: _Scenario,
    modifier: Mapping[str, Any],
    modifier_text: str,
    variant: str,
    suffixes: Mapping[str, str],
) -> str:
    if modifier.get("id") == "bullets":
        return f"- {scenario.assistant}\n- {suffixes['bullet_boundary']}\n- {variant}."
    return f"{scenario.assistant} {_format_suffix(suffixes, 'assistant_suffix', modifier_text, variant)}"


def _format_suffix(suffixes: Mapping[str, str], key: str, modifier: str, variant: str) -> str:
    return suffixes[key].format(modifier=modifier, variant=variant)


def _build_manifest(profile: Mapping[str, Any], records_written: int) -> dict[str, object]:
    section = profile["chat_seed"]
    manifest = dict(section.get("manifest", {})) if isinstance(section.get("manifest"), dict) else {}
    manifest.update(
        {
            "dataset_version": section_text(profile, "chat_seed", "dataset_version"),
            "data_kind": section_text(profile, "chat_seed", "data_kind"),
            "records": records_written,
            "format": "jsonl",
            "fields": ["system", "user", "thinking", "assistant"],
            "identity_policy": model_context(profile)["identity_statement"],
        }
    )
    return manifest

