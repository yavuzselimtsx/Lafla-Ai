"""
@Dosya: cli/train_pretrain.py
@Aciklama: LaflaAi-Core pretraining komut satiri girisi.
@Yazar: Lafla Gelistirme Ekibi
@Bilgi: Colab notebook bu CLI'yi cagirir; egitim mantigi notebook hucrelerine
        gomulmez.
@Uyari: --smoke disinda veri ve tokenizer eksigi egitimi baslatmadan durdurur.
@Calisma-Semasi: args -> configs -> runner -> JSON summary
"""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from pathlib import Path
from typing import Sequence

from lafla_ai_core.config.loader import load_mapping
from lafla_ai_core.config.schema import ModelConfig, TrainingConfig
from lafla_ai_core.data.routing import assert_pretraining_inputs
from lafla_ai_core.training.runner import TrainingPaths, run_pretraining


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="LaflaAi-Core pretraining")
    parser.add_argument("--model-config", required=True)
    parser.add_argument("--training-config", required=True)
    parser.add_argument("--tokenizer-path", required=True)
    parser.add_argument("--checkpoint-dir", required=True)
    parser.add_argument("--health-log", required=True)
    parser.add_argument("--resume-from")
    parser.add_argument("--data-jsonl", action="append", default=[])
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args(argv)

    model_config = ModelConfig.from_mapping(load_mapping(args.model_config))
    training_config = TrainingConfig.from_mapping(load_mapping(args.training_config))
    model_config.validate()
    training_config.validate()
    if not args.smoke:
        _require_existing(args.tokenizer_path, "tokenizer")
        for data_path in args.data_jsonl:
            _require_existing(data_path, "data_jsonl")
        assert_pretraining_inputs(tuple(args.data_jsonl))
    paths = TrainingPaths(
        data_jsonl=tuple(args.data_jsonl),
        tokenizer_path=args.tokenizer_path,
        checkpoint_dir=args.checkpoint_dir,
        health_log_path=args.health_log,
        resume_from=args.resume_from,
    )
    summary = run_pretraining(model_config, training_config, paths, smoke=args.smoke)
    print(json.dumps(asdict(summary), ensure_ascii=False, indent=2, sort_keys=True))
    return 0


def _require_existing(path: str, label: str) -> None:
    if not Path(path).exists():
        raise FileNotFoundError(f"{label} bulunamadi: {path}")


if __name__ == "__main__":
    raise SystemExit(main())
