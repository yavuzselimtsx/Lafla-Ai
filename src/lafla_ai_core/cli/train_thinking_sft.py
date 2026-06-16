"""
@Dosya: cli/train_thinking_sft.py
@Aciklama: LaflaAi-Core thinking SFT komut satiri girisi.
@Yazar: Lafla Gelistirme Ekibi
@Bilgi: Bu CLI kaynak checkpointi yerinde degistirmez; --output-dir ayri bir
        SFT checkpoint klasorudur.
@Uyari: Tokenizer degistirilirse checkpoint embedding sozlesmesi bozulur.
@Calisma-Semasi: args -> post_training config -> SFT runner -> JSON summary
"""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from pathlib import Path
from typing import Sequence

from lafla_ai_core.config.loader import load_mapping
from lafla_ai_core.config.schema import PostTrainingConfig
from lafla_ai_core.post_training.thinking_trainer import (
    ThinkingSftTrainingPaths,
    run_thinking_sft,
)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="LaflaAi-Core thinking SFT")
    parser.add_argument("--checkpoint-dir", required=True, help="Kaynak pretrain checkpoint klasoru")
    parser.add_argument("--tokenizer-path", required=True)
    parser.add_argument("--config", required=True, help="configs/post_training/*.yaml")
    parser.add_argument("--output-dir", required=True, help="Ayri yazilacak SFT checkpoint klasoru")
    parser.add_argument("--health-log", help="SFT JSONL health log yolu")
    parser.add_argument("--data-jsonl", action="append", default=[], help="Thinking SFT JSONL dosyasi")
    parser.add_argument("--device", default="auto", choices=("auto", "cuda", "cpu"))
    parser.add_argument("--precision", default="fp16", choices=("fp16", "bf16", "fp32"))
    parser.add_argument("--micro-batch-size", type=int, default=1)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=1)
    parser.add_argument("--max-steps", type=int, default=0, help="0 ise config epoch'lari tamamlanir")
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--grad-clip-norm", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=1337)
    parser.add_argument("--shuffle-buffer-size", type=int, default=4096)
    args = parser.parse_args(argv)

    _require_existing(args.checkpoint_dir, "checkpoint")
    _require_existing(Path(args.checkpoint_dir) / "READY.json", "checkpoint READY")
    _require_existing(args.tokenizer_path, "tokenizer")
    for data_path in args.data_jsonl:
        _require_existing(data_path, "data_jsonl")

    config = PostTrainingConfig.from_mapping(load_mapping(args.config))
    config.validate()
    output_dir = Path(args.output_dir)
    health_log = Path(args.health_log) if args.health_log else output_dir / "sft-health.jsonl"
    paths = ThinkingSftTrainingPaths(
        source_checkpoint=args.checkpoint_dir,
        tokenizer_path=args.tokenizer_path,
        data_jsonl=tuple(args.data_jsonl),
        output_dir=str(output_dir),
        health_log_path=str(health_log),
    )
    summary = run_thinking_sft(
        config,
        paths,
        device=args.device,
        precision=args.precision,
        micro_batch_size=args.micro_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        max_steps=args.max_steps,
        weight_decay=args.weight_decay,
        grad_clip_norm=args.grad_clip_norm,
        seed=args.seed,
        shuffle_buffer_size=args.shuffle_buffer_size,
    )
    print(json.dumps(asdict(summary), ensure_ascii=False, indent=2, sort_keys=True))
    return 0


def _require_existing(path: str | Path, label: str) -> None:
    if not Path(path).exists():
        raise FileNotFoundError(f"{label} bulunamadi: {path}")


if __name__ == "__main__":
    raise SystemExit(main())
