"""
 _        _    _____ _        _
| |      / \\  |  ___| |      / \\
| |     / _ \\ | |_  | |     / _ \\
| |___ / ___ \\|  _| | |___ / ___ \\
|_____/_/   \\_\\_|   |_____/_/   \\_\\

@Dosya: lafla_model.py
@Açıklama: Lafla AI için sıfırdan eğitilecek decoder-only transformer çekirdeği.
@Yazar: Lafla Geliştirme Ekibi
@Bilgi: Hazır model ağırlığı veya kapalı mimari bağımlılığı içermez.
@Uyarı: Bu dosya kimlik öğretmez; kimlik veri ve sistem yönergesi katmanında verilir.
@Çalışma-Şeması: config -> transformer blocks -> logits/loss
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch
from torch import nn
import torch.nn.functional as F


@dataclass(frozen=True)
class LaflaModelConfig:
    vocab_size: int
    context_length: int
    hidden_size: int
    num_layers: int
    num_attention_heads: int
    num_key_value_heads: int
    intermediate_size: int
    rope_theta: float = 10000.0
    rms_norm_eps: float = 1e-5
    dropout: float = 0.0
    tie_word_embeddings: bool = False
    use_bias: bool = False


class RMSNorm(nn.Module):
    """RMSNorm katmanı, büyük dil modeli bloklarında LayerNorm yerine kullanılır."""

    def __init__(self, hidden_size: int, eps: float) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        scale = torch.rsqrt(x.pow(2).mean(dim=-1, keepdim=True) + self.eps)
        return self.weight * x * scale


class RotaryEmbedding(nn.Module):
    """RoPE frekans tablolarını üretir ve attention sorgu/anahtar tensörlerine uygular."""

    def __init__(self, head_dim: int, max_position: int, theta: float) -> None:
        super().__init__()
        inv_freq = 1.0 / (theta ** (torch.arange(0, head_dim, 2).float() / head_dim))
        positions = torch.arange(max_position).float()
        freqs = torch.einsum("i,j->ij", positions, inv_freq)
        self.register_buffer("cos", freqs.cos(), persistent=False)
        self.register_buffer("sin", freqs.sin(), persistent=False)

    def forward(self, q: torch.Tensor, k: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        seq_len = q.shape[-2]
        cos = self.cos[:seq_len].to(dtype=q.dtype, device=q.device)
        sin = self.sin[:seq_len].to(dtype=q.dtype, device=q.device)
        return apply_rope(q, cos, sin), apply_rope(k, cos, sin)


def apply_rope(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    """Çift ve tek boyutları döndürerek rotary positional embedding uygular."""

    x_even = x[..., 0::2]
    x_odd = x[..., 1::2]
    rotated_even = x_even * cos - x_odd * sin
    rotated_odd = x_even * sin + x_odd * cos
    return torch.stack((rotated_even, rotated_odd), dim=-1).flatten(-2)


class SwiGLU(nn.Module):
    """SwiGLU ileri besleme katmanı, parametre verimliliği için kapılı aktivasyon kullanır."""

    def __init__(self, config: LaflaModelConfig) -> None:
        super().__init__()
        self.gate = nn.Linear(config.hidden_size, config.intermediate_size, bias=config.use_bias)
        self.up = nn.Linear(config.hidden_size, config.intermediate_size, bias=config.use_bias)
        self.down = nn.Linear(config.intermediate_size, config.hidden_size, bias=config.use_bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.down(F.silu(self.gate(x)) * self.up(x))


class GroupedQueryAttention(nn.Module):
    """Grouped-query attention uygular; KV başlıklarını sorgu başlıklarına tekrarlar."""

    def __init__(self, config: LaflaModelConfig) -> None:
        super().__init__()
        if config.num_attention_heads % config.num_key_value_heads != 0:
            raise ValueError("num_attention_heads must be divisible by num_key_value_heads")
        self.num_heads = config.num_attention_heads
        self.num_kv_heads = config.num_key_value_heads
        self.head_dim = config.hidden_size // config.num_attention_heads
        self.kv_repeat = self.num_heads // self.num_kv_heads
        self.q_proj = nn.Linear(config.hidden_size, self.num_heads * self.head_dim, bias=config.use_bias)
        self.k_proj = nn.Linear(config.hidden_size, self.num_kv_heads * self.head_dim, bias=config.use_bias)
        self.v_proj = nn.Linear(config.hidden_size, self.num_kv_heads * self.head_dim, bias=config.use_bias)
        self.o_proj = nn.Linear(config.hidden_size, config.hidden_size, bias=config.use_bias)
        self.rope = RotaryEmbedding(self.head_dim, config.context_length, config.rope_theta)
        self.dropout = config.dropout

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch, seq_len, _ = x.shape
        q = self.q_proj(x).view(batch, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(batch, seq_len, self.num_kv_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(batch, seq_len, self.num_kv_heads, self.head_dim).transpose(1, 2)
        q, k = self.rope(q, k)
        k = k.repeat_interleave(self.kv_repeat, dim=1)
        v = v.repeat_interleave(self.kv_repeat, dim=1)
        y = F.scaled_dot_product_attention(q, k, v, is_causal=True, dropout_p=self.dropout if self.training else 0.0)
        y = y.transpose(1, 2).contiguous().view(batch, seq_len, self.num_heads * self.head_dim)
        return self.o_proj(y)


class LaflaBlock(nn.Module):
    """Attention ve feed-forward alt bloklarını artık bağlantılarla birleştirir."""

    def __init__(self, config: LaflaModelConfig) -> None:
        super().__init__()
        self.attn_norm = RMSNorm(config.hidden_size, config.rms_norm_eps)
        self.attn = GroupedQueryAttention(config)
        self.ffn_norm = RMSNorm(config.hidden_size, config.rms_norm_eps)
        self.ffn = SwiGLU(config)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.attn_norm(x))
        x = x + self.ffn(self.ffn_norm(x))
        return x


class LaflaTransformer(nn.Module):
    """Lafla AI dil modeli gövdesini ve dil modelleme kafasını içerir."""

    def __init__(self, config: LaflaModelConfig) -> None:
        super().__init__()
        self.config = config
        self.embed = nn.Embedding(config.vocab_size, config.hidden_size)
        self.blocks = nn.ModuleList([LaflaBlock(config) for _ in range(config.num_layers)])
        self.norm = RMSNorm(config.hidden_size, config.rms_norm_eps)
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        if config.tie_word_embeddings:
            self.lm_head.weight = self.embed.weight
        self.apply(self._init_weights)

    def _init_weights(self, module: nn.Module) -> None:
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, input_ids: torch.Tensor, labels: torch.Tensor | None = None) -> dict[str, torch.Tensor]:
        if input_ids.shape[1] > self.config.context_length:
            raise ValueError("sequence length exceeds context_length")
        x = self.embed(input_ids)
        for block in self.blocks:
            x = block(x)
        logits = self.lm_head(self.norm(x))
        result = {"logits": logits}
        if labels is not None:
            result["loss"] = F.cross_entropy(logits[:, :-1].reshape(-1, logits.size(-1)), labels[:, 1:].reshape(-1))
        return result


def estimate_parameters(config: LaflaModelConfig) -> int:
    """Verilen yapılandırma için eğitilebilir parametre sayısını hesaplar."""

    model = LaflaTransformer(config)
    return sum(parameter.numel() for parameter in model.parameters())
