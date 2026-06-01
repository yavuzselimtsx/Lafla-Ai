"""
Lafla AI model boyutunu ve mimari kararlarını doğrular.

Bu dosya yalnızca modelin eğitimden önce kırılmayacak ve 2B parametre sınırını
aşmayacak şekilde tanımlanmasını sağlar.
"""

from __future__ import annotations

from dataclasses import dataclass


MAXIMUM_MODEL_PARAMETERS = 2_000_000_000


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
        if self.estimated_parameters(skip_validate=True) > MAXIMUM_MODEL_PARAMETERS:
            raise ValueError("model exceeds 2B parameter limit")

    def estimated_parameters(self, *, skip_validate: bool = False) -> int:
        """Yaklaşık parametre sayısını döndürür."""

        if not skip_validate:
            self.validate()
        embeddings = self.vocabulary_size * self.embedding_size
        attention = self.layers * 4 * self.embedding_size * self.embedding_size
        feed_forward_width = int(self.embedding_size * self.feed_forward_multiple)
        feed_forward = self.layers * 3 * self.embedding_size * feed_forward_width
        norms = self.layers * 2 * self.embedding_size
        output_head = self.vocabulary_size * self.embedding_size
        return embeddings + attention + feed_forward + norms + output_head


def default_lafla_1b_config() -> TransformerConfig:
    """2B sınırını aşmadan büyütülebilir 1B sınıfı başlangıç ayarı."""

    return TransformerConfig(
        attention_heads=16,
        context_length=4096,
        embedding_size=2048,
        feed_forward_multiple=2.75,
        grouped_query_heads=4,
        layers=24,
        vocabulary_size=64000,
    )
