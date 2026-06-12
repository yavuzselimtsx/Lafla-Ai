"""
@Dosya: model/growth.py
@Aciklama: Ayni genislikteki Lafla modelleri icin function-preserving depth growth.
@Uyari: Donusum tek basina 200M kalite saglamaz; devam pretraining zorunludur.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Mapping

from lafla_ai_core.config.schema import ModelConfig
from lafla_ai_core.model.size import estimate_decoder_parameters


@dataclass(frozen=True)
class DepthGrowthPlan:
    source_model: str
    target_model: str
    source_parameters: int
    target_parameters: int
    source_to_target: tuple[tuple[int, int], ...]
    inserted_target_layers: tuple[int, ...]
    continued_pretraining_required: bool = True

    def to_mapping(self) -> dict[str, Any]:
        return asdict(self)


def validate_growth_compatibility(source: ModelConfig, target: ModelConfig) -> None:
    comparable_fields = (
        "family",
        "vocab_size",
        "context_length",
        "hidden_size",
        "intermediate_size",
        "num_attention_heads",
        "num_key_value_heads",
        "activation",
        "norm",
        "rope",
        "qk_norm",
        "rope_theta",
        "norm_eps",
        "tie_word_embeddings",
        "use_bias",
        "attention_pattern",
        "sliding_window",
        "rope_scaling",
    )
    for field_name in comparable_fields:
        source_value = getattr(source, field_name)
        target_value = getattr(target, field_name)
        if source_value != target_value:
            raise ValueError(
                f"depth growth uyumsuz {field_name}: source={source_value!r}, target={target_value!r}"
            )
    if target.num_layers <= source.num_layers:
        raise ValueError("target num_layers source num_layers degerinden buyuk olmali")
    if target.parameter_target <= source.parameter_target:
        raise ValueError("target parameter_target source degerinden buyuk olmali")


def build_depth_growth_plan(source: ModelConfig, target: ModelConfig) -> DepthGrowthPlan:
    validate_growth_compatibility(source, target)
    return DepthGrowthPlan(
        source_model=source.name,
        target_model=target.name,
        source_parameters=estimate_decoder_parameters(source),
        target_parameters=estimate_decoder_parameters(target),
        source_to_target=tuple((index, index) for index in range(source.num_layers)),
        inserted_target_layers=tuple(range(source.num_layers, target.num_layers)),
    )


def grow_state_dict(
    source_state: Mapping[str, Any],
    target_initial_state: Mapping[str, Any],
    source: ModelConfig,
    target: ModelConfig,
) -> dict[str, Any]:
    """Kaynak state'i target state'e tasir ve ek residual bloklari identity-compatible yapar."""

    plan = build_depth_growth_plan(source, target)
    inserted = set(plan.inserted_target_layers)
    grown: dict[str, Any] = {}
    for key, target_value in target_initial_state.items():
        layer_index = _block_index(key)
        if layer_index is None:
            grown[key] = _copy_source_or_target(key, source_state, target_value)
            continue
        if layer_index < source.num_layers:
            if key not in source_state:
                raise KeyError(f"source state key eksik: {key}")
            grown[key] = source_state[key].detach().clone()
            continue
        value = target_value.detach().clone()
        if layer_index in inserted and _is_residual_output_parameter(key):
            value.zero_()
        grown[key] = value
    return grown


def _copy_source_or_target(key: str, source_state: Mapping[str, Any], target_value: Any) -> Any:
    if key not in source_state:
        return target_value.detach().clone()
    source_value = source_state[key]
    if tuple(source_value.shape) != tuple(target_value.shape):
        raise ValueError(f"state shape uyumsuz: {key}")
    return source_value.detach().clone()


def _block_index(key: str) -> int | None:
    if not key.startswith("blocks."):
        return None
    parts = key.split(".", 2)
    if len(parts) < 3:
        return None
    try:
        return int(parts[1])
    except ValueError:
        return None


def _is_residual_output_parameter(key: str) -> bool:
    return ".attn.out_proj." in key or ".ffn.down_proj." in key
