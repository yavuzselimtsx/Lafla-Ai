"""
Lafla AI veri kaynaklarını lisans, dil ve kalite bakımından denetler.

Modeli güçlü yapan şey sadece mimari değildir. Veri kataloğu; Türkçe konuşma,
kod, güvenli tercih verisi ve genel metin kaynaklarını birbirinden ayırır.
Bu modül bilinmeyen lisanslı veya PII temizliği yapılmamış veriyi eğitime almaz.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum


class DatasetKind(str, Enum):
    """Eğitim hattının ayırt ettiği veri türleri."""

    PRETRAIN = "pretrain"
    INSTRUCTION = "instruction"
    PREFERENCE = "preference"
    EVALUATION = "evaluation"


@dataclass(frozen=True)
class DatasetRecord:
    """Tek veri kaynağına ait denetlenebilir manifest kaydı."""

    kind: DatasetKind
    language: str
    license_name: str
    name: str
    pii_cleaned: bool
    source: str
    token_estimate: int

    def validate(self) -> None:
        if not self.name.strip():
            raise ValueError("dataset name is empty")
        if not self.source.startswith(("https://", "s3://", "gs://", "file://")):
            raise ValueError(f"unsupported dataset source: {self.source}")
        if self.token_estimate <= 0:
            raise ValueError("token_estimate must be positive")
        if self.license_name.lower() in {"unknown", "belirsiz", "proprietary"}:
            raise ValueError(f"dataset license is not acceptable: {self.name}")
        if not self.pii_cleaned and self.kind in {DatasetKind.INSTRUCTION, DatasetKind.PREFERENCE}:
            raise ValueError(f"dataset must be PII-cleaned before supervised training: {self.name}")


def total_tokens(records: list[DatasetRecord], *, language: str | None = None) -> int:
    """İsteğe bağlı dil filtresiyle toplam token bütçesini hesaplar."""

    for record in records:
        record.validate()
    selected = [record for record in records if language is None or record.language == language]
    return sum(record.token_estimate for record in selected)


def require_minimum_turkish_conversation_tokens(records: list[DatasetRecord], minimum: int) -> None:
    """Türkçe konuşma verisi hedefin altındaysa eğitimi durdurur."""

    turkish_instruction_tokens = sum(
        record.token_estimate
        for record in records
        if record.language == "tr" and record.kind in {DatasetKind.INSTRUCTION, DatasetKind.PREFERENCE}
    )
    if turkish_instruction_tokens < minimum:
        raise ValueError(
            f"Türkçe konuşma verisi yetersiz: {turkish_instruction_tokens} < {minimum}"
        )
