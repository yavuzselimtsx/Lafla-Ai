"""
@Dosya: export/hf_remote_code.py
@Aciklama: HF reposunda LaflaAi-Core olmadan yuklenebilen model kodu sablonlari.
@Yazar: Lafla Gelistirme Ekibi
@Bilgi: Remote-code dosyalari mimariyi repo icine tasir; agirlik dosyasi ve
        checkpoint donusumu yine ayri artifact sorumlulugudur.
"""

from __future__ import annotations

from typing import Any

from lafla_ai_core.config.schema import ModelConfig


def build_hf_config_payload(model_config: ModelConfig, vocab: dict[str, int]) -> dict[str, Any]:
    """ModelConfig'i Transformers config.json sozlesmesine cevirir."""

    model_config.validate()
    return {
        "model_type": "lafla",
        "architectures": ["LaflaForCausalLM"],
        "auto_map": {
            "AutoConfig": "configuration_lafla.LaflaConfig",
            "AutoModelForCausalLM": "modeling_lafla.LaflaForCausalLM",
        },
        "model_name": model_config.name,
        "display_name": model_config.display_name,
        "creator_name": model_config.creator_name,
        "identity_statement": model_config.identity_statement,
        "family": model_config.family,
        "parameter_target": model_config.parameter_target,
        "vocab_size": model_config.vocab_size,
        "max_position_embeddings": model_config.context_length,
        "context_length": model_config.context_length,
        "hidden_size": model_config.hidden_size,
        "intermediate_size": model_config.intermediate_size,
        "num_hidden_layers": model_config.num_layers,
        "num_attention_heads": model_config.num_attention_heads,
        "num_key_value_heads": model_config.num_key_value_heads,
        "hidden_act": model_config.activation,
        "rms_norm_eps": model_config.norm_eps,
        "rope_theta": model_config.rope_theta,
        "rope_scaling": {
            "type": model_config.rope_scaling.type,
            "factor": model_config.rope_scaling.factor,
            "original_context_length": model_config.rope_scaling.original_context_length,
        },
        "attention_pattern": list(model_config.attention_pattern),
        "sliding_window": model_config.sliding_window,
        "qk_norm": model_config.qk_norm,
        "dropout": model_config.dropout,
        "tie_word_embeddings": model_config.tie_word_embeddings,
        "use_bias": model_config.use_bias,
        "initializer_range": model_config.initializer_std,
        "use_cache": True,
        "bos_token_id": vocab["<|bos|>"],
        "eos_token_id": vocab["<|eos|>"],
        "pad_token_id": vocab["<|pad|>"],
        "torch_dtype": "float16",
    }


def remote_code_files() -> dict[str, str]:
    """HF repo kokune yazilacak self-contained Python dosyalarini dondurur."""

    return {
        "configuration_lafla.py": _CONFIGURATION_LAFLA,
        "modeling_lafla.py": _MODELING_LAFLA,
    }


_CONFIGURATION_LAFLA = r'''"""Hugging Face configuration for LaflaForCausalLM."""

from transformers import PretrainedConfig


class LaflaConfig(PretrainedConfig):
    model_type = "lafla"

    def __init__(
        self,
        vocab_size=32768,
        max_position_embeddings=2048,
        context_length=None,
        model_name="lafla",
        display_name="Lafla",
        creator_name="",
        identity_statement="",
        parameter_target=0,
        hidden_size=768,
        intermediate_size=2048,
        num_hidden_layers=12,
        num_attention_heads=12,
        num_key_value_heads=2,
        hidden_act="swiglu",
        rms_norm_eps=1e-5,
        rope_theta=10000.0,
        rope_scaling=None,
        attention_pattern=("global",),
        sliding_window=0,
        qk_norm=True,
        dropout=0.0,
        use_bias=False,
        initializer_range=0.02,
        use_cache=True,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.model_name = model_name
        self.display_name = display_name
        self.creator_name = creator_name
        self.identity_statement = identity_statement
        self.parameter_target = parameter_target
        self.vocab_size = vocab_size
        self.max_position_embeddings = max_position_embeddings
        self.context_length = context_length or max_position_embeddings
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.num_key_value_heads = num_key_value_heads
        self.hidden_act = hidden_act
        self.rms_norm_eps = rms_norm_eps
        self.rope_theta = rope_theta
        self.rope_scaling = rope_scaling or {
            "type": "none",
            "factor": 1.0,
            "original_context_length": 0,
        }
        self.attention_pattern = tuple(attention_pattern)
        self.sliding_window = sliding_window
        self.qk_norm = qk_norm
        self.dropout = dropout
        self.use_bias = use_bias
        self.initializer_range = initializer_range
        self.use_cache = use_cache
'''


_MODELING_LAFLA = r'''"""Self-contained Hugging Face model code for LaflaForCausalLM."""

from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import PreTrainedModel
from transformers.modeling_outputs import CausalLMOutputWithPast

from .configuration_lafla import LaflaConfig


class RmsNorm(nn.Module):
    def __init__(self, hidden_size: int, eps: float) -> None:
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(hidden_size))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        variance = x.pow(2).mean(dim=-1, keepdim=True)
        return self.weight * x * torch.rsqrt(variance + self.eps)


class RotaryEmbedding(nn.Module):
    def __init__(self, head_dim: int, theta: float, rope_scaling: dict) -> None:
        super().__init__()
        if head_dim % 2 != 0:
            raise ValueError("RoPE head_dim must be even")
        self.head_dim = head_dim
        self.theta = theta
        self.rope_scaling = rope_scaling
        self.register_buffer("inv_freq", self._build_inv_freq(theta), persistent=False)

    def _build_inv_freq(self, theta: float) -> torch.Tensor:
        return 1.0 / (theta ** (torch.arange(0, self.head_dim, 2, dtype=torch.float32) / self.head_dim))

    def forward(
        self,
        position_ids: torch.Tensor,
        device: torch.device,
        dtype: torch.dtype,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        positions = position_ids[0].to(device=device, dtype=self.inv_freq.dtype)
        scaling_type = self.rope_scaling.get("type", "none")
        factor = float(self.rope_scaling.get("factor", 1.0))
        inv_freq = self.inv_freq
        if scaling_type == "linear":
            positions = positions / factor
        elif scaling_type == "dynamic":
            original = int(self.rope_scaling.get("original_context_length", 0))
            seq_len = int(position_ids.max().item()) + 1
            if original > 0 and seq_len > original:
                ratio = (factor * seq_len / original) - (factor - 1.0)
                dynamic_theta = self.theta * (ratio ** (self.head_dim / (self.head_dim - 2)))
                inv_freq = self._build_inv_freq(dynamic_theta)
        freqs = torch.outer(positions, inv_freq.to(device=device))
        emb = torch.cat((freqs, freqs), dim=-1)
        return emb.cos().to(dtype=dtype), emb.sin().to(dtype=dtype)


def apply_rotary(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    cos = cos[None, None, :, :]
    sin = sin[None, None, :, :]
    half = x.shape[-1] // 2
    rotated = torch.cat((-x[..., half:], x[..., :half]), dim=-1)
    return (x * cos) + (rotated * sin)


class FeedForward(nn.Module):
    def __init__(self, config: LaflaConfig) -> None:
        super().__init__()
        self.gate_proj = nn.Linear(config.hidden_size, config.intermediate_size, bias=config.use_bias)
        self.up_proj = nn.Linear(config.hidden_size, config.intermediate_size, bias=config.use_bias)
        self.down_proj = nn.Linear(config.intermediate_size, config.hidden_size, bias=config.use_bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.down_proj(F.silu(self.gate_proj(x)) * self.up_proj(x))


class GroupedQueryAttention(nn.Module):
    def __init__(self, config: LaflaConfig, attention_mode: str) -> None:
        super().__init__()
        if config.hidden_size % config.num_attention_heads != 0:
            raise ValueError("hidden_size must divide num_attention_heads")
        if config.num_attention_heads % config.num_key_value_heads != 0:
            raise ValueError("num_attention_heads must divide num_key_value_heads")
        if attention_mode not in {"local", "global"}:
            raise ValueError("attention_mode must be local or global")
        self.attention_mode = attention_mode
        self.sliding_window = config.sliding_window
        self.num_heads = config.num_attention_heads
        self.num_kv_heads = config.num_key_value_heads
        self.head_dim = config.hidden_size // config.num_attention_heads
        self.repeat_factor = config.num_attention_heads // config.num_key_value_heads
        self.q_proj = nn.Linear(config.hidden_size, config.num_attention_heads * self.head_dim, bias=config.use_bias)
        self.k_proj = nn.Linear(config.hidden_size, config.num_key_value_heads * self.head_dim, bias=config.use_bias)
        self.v_proj = nn.Linear(config.hidden_size, config.num_key_value_heads * self.head_dim, bias=config.use_bias)
        self.out_proj = nn.Linear(config.hidden_size, config.hidden_size, bias=config.use_bias)
        self.q_norm = RmsNorm(self.head_dim, config.rms_norm_eps) if config.qk_norm else nn.Identity()
        self.k_norm = RmsNorm(self.head_dim, config.rms_norm_eps) if config.qk_norm else nn.Identity()
        self.rotary = RotaryEmbedding(self.head_dim, config.rope_theta, config.rope_scaling)
        self.dropout_p = config.dropout

    def forward(
        self,
        x: torch.Tensor,
        position_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor],
        past_key_value: Optional[tuple[torch.Tensor, torch.Tensor]],
        use_cache: bool,
    ) -> tuple[torch.Tensor, Optional[tuple[torch.Tensor, torch.Tensor]]]:
        batch, query_len, hidden = x.shape
        q = self.q_proj(x).view(batch, query_len, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(batch, query_len, self.num_kv_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(batch, query_len, self.num_kv_heads, self.head_dim).transpose(1, 2)
        q = self.q_norm(q)
        k = self.k_norm(k)
        cos, sin = self.rotary(position_ids, x.device, q.dtype)
        q = apply_rotary(q, cos, sin)
        k = apply_rotary(k, cos, sin)
        if past_key_value is not None:
            past_k, past_v = past_key_value
            k = torch.cat((past_k, k), dim=-2)
            v = torch.cat((past_v, v), dim=-2)

        present = None
        if use_cache:
            if self.attention_mode == "local":
                present = (k[:, :, -self.sliding_window :, :], v[:, :, -self.sliding_window :, :])
            else:
                present = (k, v)

        expanded_k = k.repeat_interleave(self.repeat_factor, dim=1) if self.repeat_factor > 1 else k
        expanded_v = v.repeat_interleave(self.repeat_factor, dim=1) if self.repeat_factor > 1 else v
        if self.attention_mode == "local":
            attn = self._local_attention(q, expanded_k, expanded_v, attention_mask)
        else:
            attn = self._global_attention(q, expanded_k, expanded_v, attention_mask)
        output = self.out_proj(attn.transpose(1, 2).contiguous().view(batch, query_len, hidden))
        return output, present

    def _local_attention(self, q, k, v, attention_mask):
        if self.sliding_window <= 0:
            raise ValueError("local attention requires sliding_window")
        query_len = q.shape[-2]
        key_len = k.shape[-2]
        query_base = key_len - query_len
        chunk_size = min(512, self.sliding_window, query_len)
        scale = self.head_dim ** -0.5
        outputs = []
        key_padding = None if attention_mask is None else attention_mask[:, -key_len:].bool()
        for start in range(0, query_len, chunk_size):
            end = min(start + chunk_size, query_len)
            absolute_start = query_base + start
            absolute_end = query_base + end
            key_start = max(0, absolute_start - self.sliding_window + 1)
            q_chunk = q[:, :, start:end, :]
            k_chunk = k[:, :, key_start:absolute_end, :]
            v_chunk = v[:, :, key_start:absolute_end, :]
            scores = torch.matmul(q_chunk, k_chunk.transpose(-2, -1)) * scale
            query_positions = torch.arange(absolute_start, absolute_end, device=q.device)[:, None]
            key_positions = torch.arange(key_start, absolute_end, device=q.device)[None, :]
            allowed = (key_positions <= query_positions) & (
                key_positions >= query_positions - self.sliding_window + 1
            )
            allowed = allowed[None, None, :, :]
            if key_padding is not None:
                allowed = allowed & key_padding[:, None, None, key_start:absolute_end]
            scores = scores.masked_fill(~allowed, torch.finfo(scores.dtype).min)
            probs = torch.softmax(scores, dim=-1)
            if self.training and self.dropout_p > 0.0:
                probs = F.dropout(probs, p=self.dropout_p)
            outputs.append(torch.matmul(probs, v_chunk))
        return torch.cat(outputs, dim=-2)

    def _global_attention(self, q, k, v, attention_mask):
        query_len = q.shape[-2]
        key_len = k.shape[-2]
        dropout = self.dropout_p if self.training else 0.0
        if query_len > 4096 and q.device.type == "cpu":
            return self._chunked_global_attention(q, k, v, attention_mask)
        if attention_mask is not None and bool(torch.all(attention_mask[:, -key_len:]).item()):
            attention_mask = None
        if attention_mask is None and query_len == key_len:
            return F.scaled_dot_product_attention(q, k, v, dropout_p=dropout, is_causal=True)
        query_base = key_len - query_len
        query_positions = torch.arange(query_base, key_len, device=q.device)[:, None]
        key_positions = torch.arange(key_len, device=q.device)[None, :]
        allowed = (key_positions <= query_positions)[None, None, :, :]
        if attention_mask is not None:
            allowed = allowed & attention_mask[:, None, None, -key_len:].bool()
        mask = torch.zeros((), dtype=q.dtype, device=q.device).expand(allowed.shape)
        mask = mask.masked_fill(~allowed, torch.finfo(q.dtype).min)
        return F.scaled_dot_product_attention(q, k, v, attn_mask=mask, dropout_p=dropout, is_causal=False)

    def _chunked_global_attention(self, q, k, v, attention_mask):
        query_len = q.shape[-2]
        key_len = k.shape[-2]
        query_base = key_len - query_len
        scale = self.head_dim ** -0.5
        key_padding = None if attention_mask is None else attention_mask[:, -key_len:].bool()
        outputs = []
        for start in range(0, query_len, 128):
            end = min(start + 128, query_len)
            absolute_start = query_base + start
            absolute_end = query_base + end
            q_chunk = q[:, :, start:end, :]
            k_chunk = k[:, :, :absolute_end, :]
            v_chunk = v[:, :, :absolute_end, :]
            scores = torch.matmul(q_chunk, k_chunk.transpose(-2, -1)) * scale
            query_positions = torch.arange(absolute_start, absolute_end, device=q.device)[:, None]
            key_positions = torch.arange(absolute_end, device=q.device)[None, :]
            allowed = (key_positions <= query_positions)[None, None, :, :]
            if key_padding is not None:
                allowed = allowed & key_padding[:, None, None, :absolute_end]
            scores = scores.masked_fill(~allowed, torch.finfo(scores.dtype).min)
            probs = torch.softmax(scores, dim=-1)
            if self.training and self.dropout_p > 0.0:
                probs = F.dropout(probs, p=self.dropout_p)
            outputs.append(torch.matmul(probs, v_chunk))
        return torch.cat(outputs, dim=-2)


class DecoderBlock(nn.Module):
    def __init__(self, config: LaflaConfig, attention_mode: str) -> None:
        super().__init__()
        self.attn_norm = RmsNorm(config.hidden_size, config.rms_norm_eps)
        self.attn = GroupedQueryAttention(config, attention_mode)
        self.ffn_norm = RmsNorm(config.hidden_size, config.rms_norm_eps)
        self.ffn = FeedForward(config)

    def forward(self, x, position_ids, attention_mask, past_key_value, use_cache):
        attn_output, present = self.attn(
            self.attn_norm(x),
            position_ids,
            attention_mask,
            past_key_value,
            use_cache,
        )
        x = x + attn_output
        x = x + self.ffn(self.ffn_norm(x))
        return x, present


class LaflaForCausalLM(PreTrainedModel):
    config_class = LaflaConfig
    base_model_prefix = "lafla"
    supports_gradient_checkpointing = False

    def __init__(self, config: LaflaConfig) -> None:
        super().__init__(config)
        self.token_embeddings = nn.Embedding(config.vocab_size, config.hidden_size)
        self.dropout = nn.Dropout(config.dropout)
        pattern = tuple(config.attention_pattern)
        modes = tuple(pattern[index % len(pattern)] for index in range(config.num_hidden_layers))
        self.blocks = nn.ModuleList(DecoderBlock(config, mode) for mode in modes)
        self.final_norm = RmsNorm(config.hidden_size, config.rms_norm_eps)
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        if config.tie_word_embeddings:
            self.lm_head.weight = self.token_embeddings.weight
        self.post_init()

    def forward(
        self,
        input_ids: torch.LongTensor,
        attention_mask: Optional[torch.Tensor] = None,
        labels: Optional[torch.LongTensor] = None,
        past_key_values=None,
        use_cache: Optional[bool] = None,
        cache_position: Optional[torch.LongTensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        num_logits_to_keep: int = 1,
        **kwargs,
    ) -> CausalLMOutputWithPast:
        if input_ids.ndim != 2:
            raise ValueError("input_ids must be [batch, seq]")
        use_cache = bool(self.config.use_cache if use_cache is None else use_cache)
        use_cache = bool(use_cache and labels is None)
        if past_key_values is not None and hasattr(past_key_values, "to_legacy_cache"):
            past_key_values = past_key_values.to_legacy_cache()
        past_key_values = past_key_values or (None,) * len(self.blocks)
        past_seen = max(
            (layer[0].shape[-2] for layer in past_key_values if layer is not None),
            default=0,
        )
        if cache_position is None:
            cache_position = torch.arange(
                past_seen,
                past_seen + input_ids.shape[1],
                device=input_ids.device,
                dtype=torch.long,
            )
        if cache_position.ndim != 1 or cache_position.shape[0] != input_ids.shape[1]:
            raise ValueError("cache_position must match the current input length")
        if int(cache_position[-1].item()) + 1 > self.config.context_length:
            raise ValueError("input_ids exceed context_length")
        if position_ids is None:
            position_ids = cache_position.unsqueeze(0).expand(input_ids.shape[0], -1)
        x = self.dropout(self.token_embeddings(input_ids))
        presents = []
        for block, layer_past in zip(self.blocks, past_key_values):
            x, present = block(x, position_ids, attention_mask, layer_past, use_cache)
            if use_cache:
                presents.append(present)
        hidden = self.final_norm(x)
        if labels is None and num_logits_to_keep > 0:
            logits = self.lm_head(hidden[:, -num_logits_to_keep:, :])
        else:
            logits = self.lm_head(hidden)
        loss = None
        if labels is not None:
            loss = F.cross_entropy(
                logits[:, :-1, :].contiguous().view(-1, logits.size(-1)),
                labels[:, 1:].contiguous().view(-1),
                ignore_index=-100,
            )
        return CausalLMOutputWithPast(
            loss=loss,
            logits=logits,
            past_key_values=tuple(presents) if use_cache else None,
        )

    def prepare_inputs_for_generation(
        self,
        input_ids,
        past_key_values=None,
        attention_mask=None,
        cache_position=None,
        **kwargs,
    ):
        position_ids = None
        if attention_mask is not None:
            position_ids = attention_mask.long().cumsum(-1) - 1
            position_ids.masked_fill_(attention_mask == 0, 0)
        if past_key_values is not None:
            input_ids = input_ids[:, -1:]
            if position_ids is not None:
                position_ids = position_ids[:, -1:]
            if cache_position is not None:
                cache_position = cache_position[-1:]
        return {
            "input_ids": input_ids,
            "past_key_values": past_key_values,
            "attention_mask": attention_mask,
            "cache_position": cache_position,
            "position_ids": position_ids,
            "use_cache": kwargs.get("use_cache", True),
        }

    @staticmethod
    def _reorder_cache(past_key_values, beam_idx):
        return tuple(
            tuple(state.index_select(0, beam_idx.to(state.device)) for state in layer)
            for layer in past_key_values
        )

    def _init_weights(self, module: nn.Module) -> None:
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=self.config.initializer_range)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=self.config.initializer_range)
'''
