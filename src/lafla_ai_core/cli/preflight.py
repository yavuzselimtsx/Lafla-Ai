"""
@Dosya: cli/preflight.py
@Aciklama: LaflaAi-Core config ve ortam preflight kapilarini calistiran CLI
            girisini saglar.
@Yazar: Lafla Gelistirme Ekibi
@Bilgi: Egitim komutlari bu preflight gecmeden calismamalidir. CLI JSON cikti
        uretir ki Colab ve yerel otomasyon ayni sonucu okuyabilsin.
@Uyari: Preflight hatasi bastirilamaz; hata kodu egitimi durdurur.
@Calisma-Semasi: args -> load configs -> validate -> report
"""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Sequence

from lafla_ai_core.config.loader import load_mapping
from lafla_ai_core.config.schema import ConfigError, ModelConfig, PostTrainingConfig, RuntimeConfig, TokenizerConfig, TrainingConfig


@dataclass(frozen=True)
class PreflightReport:
    """Preflight sonucunu tasir."""

    ok: bool
    checked_files: tuple[str, ...]
    errors: tuple[str, ...]

    def to_json(self) -> str:
        """Raporu JSON olarak dondurur."""

        return json.dumps(asdict(self), ensure_ascii=False, indent=2, sort_keys=True)


def run_preflight(paths: Sequence[str | Path]) -> PreflightReport:
    """Verilen config dosyalari icin uygun schema dogrulamalarini kosar."""

    errors: list[str] = []
    checked: list[str] = []
    configs: list[object] = []
    for path in paths:
        checked.append(str(path))
        try:
            data = load_mapping(path)
            configs.append(_validate_by_root_key(data))
        except Exception as exc:
            errors.append(f"{path}: {exc}")
    if not errors:
        errors.extend(_cross_validate_configs(configs))
    return PreflightReport(ok=not errors, checked_files=tuple(checked), errors=tuple(errors))


def main(argv: Sequence[str] | None = None) -> int:
    """CLI ana fonksiyonu."""

    parser = argparse.ArgumentParser(description="LaflaAi-Core preflight")
    parser.add_argument("configs", nargs="+", help="Dogrulanacak config dosyalari")
    args = parser.parse_args(argv)
    report = run_preflight(args.configs)
    print(report.to_json())
    return 0 if report.ok else 2


def _validate_by_root_key(data: dict[str, object]) -> object:
    """Config kok anahtarina gore dogru schema validatorunu secer."""

    validators = {
        "model": ModelConfig,
        "training": TrainingConfig,
        "tokenizer": TokenizerConfig,
        "runtime": RuntimeConfig,
        "post_training": PostTrainingConfig,
    }
    matched = [key for key in validators if key in data]
    if len(matched) != 1:
        raise ConfigError(f"tek kok config bolumu beklenir, bulunan: {matched}")
    config = validators[matched[0]].from_mapping(data)  # type: ignore[attr-defined]
    config.validate()
    return config


def _cross_validate_configs(configs: Sequence[object]) -> list[str]:
    """Birlikte verilen configlerin birbirleriyle uyumunu dogrular."""

    model = next((item for item in configs if isinstance(item, ModelConfig)), None)
    training = next((item for item in configs if isinstance(item, TrainingConfig)), None)
    tokenizer = next((item for item in configs if isinstance(item, TokenizerConfig)), None)
    runtime = next((item for item in configs if isinstance(item, RuntimeConfig)), None)
    post_training = next((item for item in configs if isinstance(item, PostTrainingConfig)), None)
    errors: list[str] = []
    if model is not None and training is not None:
        if training.sequence_length > model.context_length:
            errors.append(
                f"sequence_length model context_length degerini asamaz: training={training.sequence_length}, model={model.context_length}"
            )
        if training.sequence_curriculum and max(training.sequence_curriculum) > model.context_length:
            errors.append(
                "sequence_curriculum model context_length degerini asamaz: "
                f"curriculum={max(training.sequence_curriculum)}, model={model.context_length}"
            )
    if model is not None and tokenizer is not None:
        if tokenizer.vocab_size != model.vocab_size:
            errors.append(
                f"tokenizer vocab_size model vocab_size ile ayni olmali: tokenizer={tokenizer.vocab_size}, model={model.vocab_size}"
            )
    if model is not None and runtime is not None:
        if runtime.context_length > model.context_length:
            errors.append(
                f"runtime context_length model context_length degerini asamaz: runtime={runtime.context_length}, model={model.context_length}"
            )
    if model is not None and post_training is not None:
        if post_training.sequence_length > model.context_length:
            errors.append(
                f"post_training sequence_length model context_length degerini asamaz: post_training={post_training.sequence_length}, model={model.context_length}"
            )
    return errors


if __name__ == "__main__":
    raise SystemExit(main())
