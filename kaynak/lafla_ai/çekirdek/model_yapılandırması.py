"""
Lafla AI model boyutunu ve mimari kararlarını tek yerde doğrular.

Bu dosya gerçek eğitim kodundan bağımsızdır. Amaç, Colab veya üretim
eğitim betiği başlamadan önce gizli varsayımları yakalamaktır: baş sayısı,
GQA grupları, bağlam uzunluğu ve parametre bütçesi burada açıkça kontrol edilir.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class TransformerConfig:
    """Decoder-only transformer için taşınabilir yapılandırma."""

    attention_heads: int
    context_length: int
    embedding_size: int
    feed_forward_multiple: float
    grouped_query_heads: int
    layers: int
    vocabulary_size: int

    def validate(self) -> None:
        """Eğitime başlamadan önce kırılacak mimari seçimleri reddeder."""

        if self.layers < 8:
            raise ValueError("layers must be at least 8 for a serious base model")
        if self.embedding_size % self.attention_heads != 0:
            raise ValueError("embedding_size must be divisible by attention_heads")
        if self.attention_heads % self.grouped_query_heads != 0:
            raise ValueError("attention_heads must be divisible by grouped_query_heads")
        if self.context_length < 2048:
            raise ValueError("context_length must be at least 2048")
        if self.vocabulary_size < 32000:
            raise ValueError("vocabulary_size must be at least 32000")
        if self.feed_forward_multiple < 2.0:
            raise ValueError("feed_forward_multiple is too small for SwiGLU-style blocks")

    def estimated_parameters(self) -> int:
        """Yaklaşık parametre sayısını döndürür.

        Bu sayı kesin checkpoint boyutu değildir; embedding, attention ve MLP
        katmanlarının kaba toplamıdır. Planlama ve Colab bellek hesabı için
        yeterince açıklayıcıdır.
        """

        self.validate()
        embeddings = self.vocabulary_size * self.embedding_size
        attention = self.layers * 4 * self.embedding_size * self.embedding_size
        feed_forward_width = int(self.embedding_size * self.feed_forward_multiple)
        feed_forward = self.layers * 3 * self.embedding_size * feed_forward_width
        norms = self.layers * 2 * self.embedding_size
        output_head = self.vocabulary_size * self.embedding_size
        return embeddings + attention + feed_forward + norms + output_head


def default_lafla_1b_config() -> TransformerConfig:
    """Tek GPU/Colab odaklı, büyütülebilir 1B sınıfı başlangıç ayarı."""

    return TransformerConfig(
        attention_heads=16,
        context_length=4096,
        embedding_size=2048,
        feed_forward_multiple=2.75,
        grouped_query_heads=4,
        layers=24,
        vocabulary_size=64000,
    )
