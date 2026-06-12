"""
@Dosya: model/size.py
@Aciklama: Decoder-only model configleri icin parametre sayisi tahmini.
@Yazar: Lafla Gelistirme Ekibi
@Bilgi: Model adi hedef sinifi belirtse bile gercek boyut hesaplanmadan Colab kosusu
        baslatilmaz.
@Uyari: Bu tahmin mimari karar denetimi icindir; torch modeli olusturmaz.
"""

from __future__ import annotations

from lafla_ai_core.config.schema import ModelConfig


def estimate_decoder_parameters(config: ModelConfig) -> int:
    """Lafla decoder mimarisinin yaklasik egitilebilir parametre sayisini hesaplar."""

    head_dim = config.hidden_size // config.num_attention_heads
    kv_width = config.num_key_value_heads * head_dim
    embedding = config.vocab_size * config.hidden_size
    lm_head = 0 if config.tie_word_embeddings else config.vocab_size * config.hidden_size
    attention = (
        config.hidden_size * config.hidden_size
        + config.hidden_size * kv_width
        + config.hidden_size * kv_width
        + config.hidden_size * config.hidden_size
    )
    feed_forward = config.hidden_size * config.intermediate_size * 3
    norms = config.hidden_size * 2
    if config.qk_norm:
        norms += head_dim * 2
    per_layer = attention + feed_forward + norms
    final_norm = config.hidden_size
    return int(embedding + lm_head + (per_layer * config.num_layers) + final_norm)
