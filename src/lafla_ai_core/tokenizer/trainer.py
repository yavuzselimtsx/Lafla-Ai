"""
@Dosya: tokenizer/trainer.py
@Aciklama: Lafla tokenizer egitim arayuzunu ve cikis kalite raporunu yonetir.
@Yazar: Lafla Gelistirme Ekibi
@Bilgi: Gercek egitim icin HuggingFace tokenizers paketi kullanilir; paket yoksa
        sessiz fallback yapilmaz, acik kurulum hatasi verilir.
@Uyari: Tokenizer kalite raporu gecmeden model egitimi baslatilamaz.
@Calisma-Semasi: jsonl -> train BPE -> save tokenizer -> quality report
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable, Iterator, Mapping

from lafla_ai_core.config.schema import TokenizerConfig
from lafla_ai_core.tokenizer.quality import TokenizerQualityReport, analyze_tokenizer_texts, validate_clean_text


TEXT_FIELDS = ("text", "content", "prompt", "chosen", "rejected", "assistant", "system")


@dataclass(frozen=True)
class TokenizerTrainingResult:
    """Tokenizer egitim sonucunu tasir."""

    tokenizer_path: str
    report_path: str
    quality_report: TokenizerQualityReport

    def to_json(self) -> str:
        """Sonucu JSON olarak dondurur."""

        return json.dumps(asdict(self), ensure_ascii=False, indent=2, sort_keys=True)


class SamplingTextIterator:
    """Metinleri stream ederken kalite raporu icin sinirli ornek tutar."""

    def __init__(self, source: Iterable[str], sample_limit: int) -> None:
        if sample_limit <= 0:
            raise ValueError("sample_limit pozitif olmali")
        self._source = iter(source)
        self._sample_limit = sample_limit
        self._sample: list[str] = []
        self.count = 0

    def __iter__(self) -> "SamplingTextIterator":
        return self

    def __next__(self) -> str:
        text = next(self._source)
        self.count += 1
        if len(self._sample) < self._sample_limit:
            self._sample.append(text)
        return text

    @property
    def sample(self) -> tuple[str, ...]:
        """Toplanan kalite orneklerini dondurur."""

        return tuple(self._sample)


def iter_jsonl_texts(paths: Iterable[str | Path]) -> Iterator[str]:
    """JSONL dosyalarindan egitime uygun metin alanlarini uretir."""

    for item in paths:
        path = Path(item)
        with path.open("r", encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, start=1):
                if not line.strip():
                    continue
                try:
                    record = json.loads(line)
                except json.JSONDecodeError as exc:
                    raise ValueError(f"JSONL parse hatasi {path}:{line_number}: {exc}") from exc
                if isinstance(record, str):
                    yield validate_clean_text(record, f"{path}:{line_number}")
                    continue
                if not isinstance(record, Mapping):
                    raise ValueError(f"JSONL kaydi mapping veya string olmali: {path}:{line_number}")
                for field in TEXT_FIELDS:
                    value = record.get(field)
                    if isinstance(value, str) and value.strip():
                        yield validate_clean_text(value, f"{path}:{line_number}:{field}")


def train_bpe_tokenizer(
    input_paths: Iterable[str | Path],
    output_path: str | Path,
    report_path: str | Path,
    config: TokenizerConfig,
    sample_limit: int = 2000,
) -> TokenizerTrainingResult:
    """BPE tokenizer egitir, kaydeder ve kalite raporu yazar."""

    config.validate()
    tokenizers = _load_tokenizers()
    tokenizer = tokenizers.Tokenizer(tokenizers.models.BPE(unk_token="<|unk|>"))
    tokenizer.normalizer = tokenizers.normalizers.NFC()
    tokenizer.pre_tokenizer = tokenizers.pre_tokenizers.ByteLevel(add_prefix_space=False)
    tokenizer.decoder = tokenizers.decoders.ByteLevel()
    trainer = tokenizers.trainers.BpeTrainer(
        vocab_size=config.vocab_size,
        special_tokens=list(config.required_special_tokens) + ["<|unk|>"],
        min_frequency=2,
    )
    text_stream = SamplingTextIterator(iter_jsonl_texts(input_paths), sample_limit=sample_limit)
    tokenizer.train_from_iterator(text_stream, trainer=trainer)
    if text_stream.count == 0:
        raise ValueError("tokenizer egitimi icin metin bulunamadi")
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    tokenizer.save(str(output))
    vocab = tokenizer.get_vocab().keys()
    report = analyze_tokenizer_texts(text_stream.sample, config.required_special_tokens, vocab)
    report_output = Path(report_path)
    report_output.parent.mkdir(parents=True, exist_ok=True)
    report_output.write_text(json.dumps(asdict(report), ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    if not report.passed:
        raise ValueError(f"tokenizer kalite kapisi gecmedi: {report}")
    return TokenizerTrainingResult(str(output), str(report_output), report)


def _load_tokenizers():
    """tokenizers paketini yukler veya acik hata verir."""

    try:
        import tokenizers
    except Exception as exc:
        raise RuntimeError("tokenizer egitimi icin `pip install tokenizers` gerekli") from exc
    return tokenizers
