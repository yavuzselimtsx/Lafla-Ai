"""
@Dosya: runtime/memory_budget.py
@Aciklama: Model ve runtime configlerinden agirlik, KV cache ve headroom baytlarini turetir.
@Uyari: Bu hesap kabul kaniti degildir; gercek kabul process-tree peak RSS ile yapilir.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

from lafla_ai_core.config.schema import ModelConfig, RuntimeConfig
from lafla_ai_core.model.size import estimate_decoder_parameters


MIB = 1024 * 1024
DTYPE_BYTES = {
    "fp32": 4.0,
    "fp16": 2.0,
    "bf16": 2.0,
    "int8": 1.0,
    "int4": 0.5,
}


@dataclass(frozen=True)
class RuntimeMemoryEstimate:
    parameter_count: int
    weight_bytes: int
    kv_cache_bytes: int
    runtime_overhead_bytes: int
    allocator_headroom_bytes: int
    total_bytes: int
    configured_limit_bytes: int
    within_configured_limit: bool


def dtype_size_bytes(dtype: str) -> float:
    try:
        return DTYPE_BYTES[dtype]
    except KeyError as exc:
        raise ValueError(f"desteklenmeyen dtype: {dtype}") from exc


def estimate_weight_bytes(model: ModelConfig, runtime: RuntimeConfig) -> int:
    return math.ceil(estimate_decoder_parameters(model) * dtype_size_bytes(runtime.weight_dtype))


def estimate_hybrid_kv_cache_bytes(
    model: ModelConfig,
    runtime: RuntimeConfig,
    *,
    context_tokens: int | None = None,
) -> int:
    context = runtime.context_length if context_tokens is None else context_tokens
    if not 1 <= context <= model.context_length:
        raise ValueError("context_tokens model context araliginda olmali")
    head_dim = model.hidden_size // model.num_attention_heads
    bytes_per_token_layer = (
        2
        * model.num_key_value_heads
        * head_dim
        * dtype_size_bytes(runtime.cache_dtype)
        * runtime.batch_size
    )
    cached_token_layers = 0
    for mode in model.resolved_attention_pattern():
        cached_token_layers += context if mode == "global" else min(context, model.sliding_window)
    return math.ceil(cached_token_layers * bytes_per_token_layer)


def estimate_runtime_memory(
    model: ModelConfig,
    runtime: RuntimeConfig,
    *,
    context_tokens: int | None = None,
) -> RuntimeMemoryEstimate:
    model.validate()
    runtime.validate()
    parameter_count = estimate_decoder_parameters(model)
    weight_bytes = estimate_weight_bytes(model, runtime)
    kv_cache_bytes = estimate_hybrid_kv_cache_bytes(model, runtime, context_tokens=context_tokens)
    runtime_overhead_bytes = runtime.runtime_overhead_mib * MIB
    subtotal = weight_bytes + kv_cache_bytes + runtime_overhead_bytes
    allocator_headroom_bytes = math.ceil(subtotal * runtime.allocator_headroom_ratio)
    total_bytes = subtotal + allocator_headroom_bytes
    configured_limit_bytes = runtime.peak_rss_limit_mib * MIB
    return RuntimeMemoryEstimate(
        parameter_count=parameter_count,
        weight_bytes=weight_bytes,
        kv_cache_bytes=kv_cache_bytes,
        runtime_overhead_bytes=runtime_overhead_bytes,
        allocator_headroom_bytes=allocator_headroom_bytes,
        total_bytes=total_bytes,
        configured_limit_bytes=configured_limit_bytes,
        within_configured_limit=bool(configured_limit_bytes and total_bytes <= configured_limit_bytes),
    )
