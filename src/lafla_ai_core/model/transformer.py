"""
@Dosya: model/transformer.py
@Aciklama: LaflaAi-Core icin decoder-only Transformer cekirdegi.
@Yazar: Lafla Gelistirme Ekibi
@Bilgi: RoPE, RMSNorm, SwiGLU ve grouped-query attention fikirleri modern acik
        LLM egitim sistemlerinden temiz oda olarak Lafla sozlesmesine uyarlandi.
@Uyari: Bu modul ham dict kabul etmez; ModelConfig validate edilmis olmalidir.
@Calisma-Semasi: token ids -> embeddings -> decoder blocks -> logits/loss
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    import torch.utils.checkpoint
except ModuleNotFoundError as exc:  # pragma: no cover - exercised on machines without torch
    raise ModuleNotFoundError("lafla_ai_core.model.transformer icin torch kurulu olmali") from exc

from lafla_ai_core.config.schema import ModelConfig, RopeScalingConfig


@dataclass(frozen=True)
class ModelOutput:
    """Model ileri gecis sonucunu tasir."""

    logits: torch.Tensor
    loss: Optional[torch.Tensor] = None


class RmsNorm(nn.Module):
    """Kucuk ve kararli RMSNorm uygulamasi."""

    def __init__(self, hidden_size: int, eps: float) -> None:
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(hidden_size))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        variance = x.pow(2).mean(dim=-1, keepdim=True)
        return self.weight * x * torch.rsqrt(variance + self.eps)


class RotaryEmbedding(nn.Module):
    """RoPE cos/sin cache uretir."""

    def __init__(self, head_dim: int, theta: float, scaling: RopeScalingConfig) -> None:
        super().__init__()
        if head_dim % 2 != 0:
            raise ValueError("RoPE head_dim cift olmali")
        self.head_dim = head_dim
        self.theta = theta
        self.scaling = scaling
        self.register_buffer("inv_freq", self._build_inv_freq(theta), persistent=False)

    def forward(self, seq_len: int, device: torch.device, dtype: torch.dtype) -> tuple[torch.Tensor, torch.Tensor]:
        positions = torch.arange(seq_len, device=device, dtype=self.inv_freq.dtype)
        inv_freq = self.inv_freq
        if self.scaling.type == "linear":
            positions = positions / self.scaling.factor
        elif self.scaling.type == "dynamic" and seq_len > self.scaling.original_context_length:
            if self.head_dim <= 2:
                raise ValueError("dynamic RoPE icin head_dim 2'den buyuk olmali")
            ratio = (self.scaling.factor * seq_len / self.scaling.original_context_length) - (
                self.scaling.factor - 1.0
            )
            dynamic_theta = self.theta * (ratio ** (self.head_dim / (self.head_dim - 2)))
            inv_freq = self._build_inv_freq(dynamic_theta)
        freqs = torch.outer(positions, inv_freq.to(device=device))
        emb = torch.cat((freqs, freqs), dim=-1)
        return emb.cos().to(dtype=dtype), emb.sin().to(dtype=dtype)

    def _build_inv_freq(self, theta: float) -> torch.Tensor:
        return 1.0 / (theta ** (torch.arange(0, self.head_dim, 2, dtype=torch.float32) / self.head_dim))


def apply_rotary(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    """RoPE donusumunu attention tensore uygular."""

    cos = cos[None, None, :, :]
    sin = sin[None, None, :, :]
    half = x.shape[-1] // 2
    rotated = torch.cat((-x[..., half:], x[..., :half]), dim=-1)
    return (x * cos) + (rotated * sin)


def checkpoint_decoder_block(block: nn.Module, x: torch.Tensor) -> torch.Tensor:
    """Aktivasyon checkpointing'i cihaz uyumlu fonksiyonla calistirir."""

    if x.device.type == "xla":
        try:
            from torch_xla.utils.checkpoint import checkpoint as xla_checkpoint  # type: ignore
        except ModuleNotFoundError as exc:  # pragma: no cover - TPU ortaminda dogrulanir
            raise ModuleNotFoundError("XLA checkpointing icin torch_xla kurulu olmali") from exc
        return xla_checkpoint(block, x, use_reentrant=True)
    return torch.utils.checkpoint.checkpoint(block, x, use_reentrant=False)


class FeedForward(nn.Module):
    """SwiGLU MLP blogu."""

    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        bias = config.use_bias
        self.gate_proj = nn.Linear(config.hidden_size, config.intermediate_size, bias=bias)
        self.up_proj = nn.Linear(config.hidden_size, config.intermediate_size, bias=bias)
        self.down_proj = nn.Linear(config.intermediate_size, config.hidden_size, bias=bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.down_proj(F.silu(self.gate_proj(x)) * self.up_proj(x))


class GroupedQueryAttention(nn.Module):
    """Grouped-query causal self-attention."""

    def __init__(self, config: ModelConfig, attention_mode: str = "global") -> None:
        super().__init__()
        if config.hidden_size % config.num_attention_heads != 0:
            raise ValueError("hidden_size num_attention_heads ile bolunmeli")
        if config.num_attention_heads % config.num_key_value_heads != 0:
            raise ValueError("num_attention_heads num_key_value_heads ile bolunmeli")
        self.num_heads = config.num_attention_heads
        self.num_kv_heads = config.num_key_value_heads
        self.head_dim = config.hidden_size // config.num_attention_heads
        self.repeat_factor = config.num_attention_heads // config.num_key_value_heads
        if attention_mode not in {"local", "global"}:
            raise ValueError(f"desteklenmeyen attention_mode: {attention_mode}")
        self.attention_mode = attention_mode
        self.sliding_window = config.sliding_window
        bias = config.use_bias
        self.q_proj = nn.Linear(config.hidden_size, config.num_attention_heads * self.head_dim, bias=bias)
        self.k_proj = nn.Linear(config.hidden_size, config.num_key_value_heads * self.head_dim, bias=bias)
        self.v_proj = nn.Linear(config.hidden_size, config.num_key_value_heads * self.head_dim, bias=bias)
        self.out_proj = nn.Linear(config.hidden_size, config.hidden_size, bias=bias)
        self.q_norm = RmsNorm(self.head_dim, config.norm_eps) if config.qk_norm else nn.Identity()
        self.k_norm = RmsNorm(self.head_dim, config.norm_eps) if config.qk_norm else nn.Identity()
        self.rotary = RotaryEmbedding(self.head_dim, config.rope_theta, config.rope_scaling)
        self.dropout_p = config.dropout

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch, seq_len, hidden = x.shape
        q = self.q_proj(x).view(batch, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(batch, seq_len, self.num_kv_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(batch, seq_len, self.num_kv_heads, self.head_dim).transpose(1, 2)
        q = self.q_norm(q)
        k = self.k_norm(k)
        cos, sin = self.rotary(seq_len, x.device, q.dtype)
        q = apply_rotary(q, cos, sin)
        k = apply_rotary(k, cos, sin)
        if self.repeat_factor > 1:
            k = k.repeat_interleave(self.repeat_factor, dim=1)
            v = v.repeat_interleave(self.repeat_factor, dim=1)
        if self.attention_mode == "local":
            attn = self._chunked_local_attention(q, k, v)
        elif x.device.type == "xla":
            attn = self._xla_global_attention(q, k, v)
        else:
            attn = F.scaled_dot_product_attention(
                q,
                k,
                v,
                attn_mask=None,
                dropout_p=self.dropout_p if self.training else 0.0,
                is_causal=True,
            )
        attn = attn.transpose(1, 2).contiguous().view(batch, seq_len, hidden)
        return self.out_proj(attn)

    def _chunked_local_attention(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
        """Tam NxN maske ayirmadan yerel causal attention hesaplar."""

        if self.sliding_window <= 0:
            raise ValueError("local attention icin sliding_window pozitif olmali")
        seq_len = q.shape[-2]
        chunk_size = min(512, self.sliding_window, seq_len)
        scale = q.shape[-1] ** -0.5
        outputs: list[torch.Tensor] = []
        for start in range(0, seq_len, chunk_size):
            end = min(start + chunk_size, seq_len)
            key_start = max(0, start - self.sliding_window + 1)
            q_chunk = q[:, :, start:end, :]
            k_chunk = k[:, :, key_start:end, :]
            v_chunk = v[:, :, key_start:end, :]
            scores = torch.matmul(q_chunk, k_chunk.transpose(-2, -1)) * scale
            query_positions = torch.arange(start, end, device=q.device)[:, None]
            key_positions = torch.arange(key_start, end, device=q.device)[None, :]
            allowed = (key_positions <= query_positions) & (
                key_positions >= (query_positions - self.sliding_window + 1)
            )
            scores = scores.masked_fill(~allowed[None, None, :, :], torch.finfo(scores.dtype).min)
            probs = torch.softmax(scores, dim=-1)
            if self.training and self.dropout_p > 0.0:
                probs = F.dropout(probs, p=self.dropout_p)
            outputs.append(torch.matmul(probs, v_chunk))
        return torch.cat(outputs, dim=-2)

    def _xla_global_attention(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
        """TPU'da uzun global attention icin XLA FlashAttention kullanir."""

        try:
            from torch_xla.experimental.custom_kernel import flash_attention  # type: ignore
        except (ImportError, ModuleNotFoundError):
            if q.shape[-2] > 4096:
                raise RuntimeError(
                    "4096 token ustu XLA global attention icin torch_xla FlashAttention zorunlu"
                )
            return self._manual_causal_attention(q, k, v)
        return flash_attention(q, k, v, causal=True, sm_scale=q.shape[-1] ** -0.5)

    def _manual_causal_attention(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
        scale = q.shape[-1] ** -0.5
        scores = torch.matmul(q, k.transpose(-2, -1)) * scale
        seq_len = scores.shape[-1]
        mask = torch.ones((seq_len, seq_len), dtype=torch.bool, device=scores.device).triu(1)
        scores = scores.masked_fill(mask, torch.finfo(scores.dtype).min)
        probs = torch.softmax(scores, dim=-1)
        if self.training and self.dropout_p > 0.0:
            probs = F.dropout(probs, p=self.dropout_p)
        return torch.matmul(probs, v)


class DecoderBlock(nn.Module):
    """Pre-norm decoder blogu."""

    def __init__(self, config: ModelConfig, attention_mode: str) -> None:
        super().__init__()
        self.attn_norm = RmsNorm(config.hidden_size, config.norm_eps)
        self.attn = GroupedQueryAttention(config, attention_mode=attention_mode)
        self.ffn_norm = RmsNorm(config.hidden_size, config.norm_eps)
        self.ffn = FeedForward(config)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.attn_norm(x))
        x = x + self.ffn(self.ffn_norm(x))
        return x


class LaflaDecoderModel(nn.Module):
    """Lafla ailesinin egitimde kullanilan decoder-only modeli."""

    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        config.validate()
        self.config = config
        self.token_embeddings = nn.Embedding(config.vocab_size, config.hidden_size)
        self.dropout = nn.Dropout(config.dropout)
        self.blocks = nn.ModuleList(
            DecoderBlock(config, attention_mode=mode) for mode in config.resolved_attention_pattern()
        )
        self.final_norm = RmsNorm(config.hidden_size, config.norm_eps)
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        if config.tie_word_embeddings:
            self.lm_head.weight = self.token_embeddings.weight
        self.apply(self._init_weights)
        # Rezidüel akisa yazan projeksiyonlar derinlikle olceklenir (GPT-2/NeoX
        # tarzi std/sqrt(2*num_layers)); derin modellerde aktivasyon buyumesini
        # ve erken loss spike'larini azaltir.
        residual_std = config.initializer_std / math.sqrt(2.0 * config.num_layers)
        for block in self.blocks:
            nn.init.normal_(block.attn.out_proj.weight, mean=0.0, std=residual_std)
            nn.init.normal_(block.ffn.down_proj.weight, mean=0.0, std=residual_std)

    def forward(self, input_ids: torch.Tensor, labels: torch.Tensor | None = None) -> ModelOutput:
        """Input tokenlarindan logits ve opsiyonel next-token loss uretir."""

        if input_ids.ndim != 2:
            raise ValueError("input_ids [batch, seq] olmali")
        if input_ids.shape[1] > self.config.context_length:
            raise ValueError("input_ids context_length sinirini asti")
        x = self.dropout(self.token_embeddings(input_ids))
        for block in self.blocks:
            if self.config.gradient_checkpointing and self.training:
                x = checkpoint_decoder_block(block, x)
            else:
                x = block(x)
        hidden = self.final_norm(x)
        loss = None
        if labels is not None:
            if labels.shape != input_ids.shape:
                raise ValueError("labels input_ids ile ayni sekilde olmali")
            if hidden.device.type == "xla":
                loss = self._chunked_xla_loss(hidden, labels)
                logits = self.lm_head(hidden[:, -1:, :])
                return ModelOutput(logits=logits, loss=loss)
        logits = self.lm_head(hidden)
        if labels is not None:
            loss = F.cross_entropy(
                logits[:, :-1, :].contiguous().view(-1, logits.size(-1)),
                labels[:, 1:].contiguous().view(-1),
                ignore_index=-100,
            )
        return ModelOutput(logits=logits, loss=loss)

    def _chunked_xla_loss(self, hidden: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        """TPU/XLA icin dev [batch, seq, vocab] logits tensorunu parcalara boler."""

        total_loss = hidden.new_zeros(())
        total_count = hidden.new_zeros(())
        shift_labels = labels[:, 1:]
        chunk_size = min(128, shift_labels.shape[1])
        for start in range(0, shift_labels.shape[1], chunk_size):
            end = min(start + chunk_size, shift_labels.shape[1])
            logits = self.lm_head(hidden[:, start:end, :])
            chunk_labels = shift_labels[:, start:end].contiguous().view(-1)
            total_loss = total_loss + F.cross_entropy(
                logits.contiguous().view(-1, logits.size(-1)),
                chunk_labels,
                ignore_index=-100,
                reduction="sum",
            )
            total_count = total_count + (chunk_labels != -100).sum().to(dtype=total_loss.dtype)
        return total_loss / total_count.clamp_min(1.0)

    def _init_weights(self, module: nn.Module) -> None:
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=self.config.initializer_std)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=self.config.initializer_std)
