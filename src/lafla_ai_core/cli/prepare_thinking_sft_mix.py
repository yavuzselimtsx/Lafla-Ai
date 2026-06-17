"""
@Dosya: cli/prepare_thinking_sft_mix.py
@Aciklama: Thinking-chat agirlikli SFT mixture hazirlar.
@Yazar: Lafla Gelistirme Ekibi
"""

from __future__ import annotations

import argparse
from typing import Sequence

from lafla_ai_core.post_training.sft_mixture import build_thinking_sft_mixture


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Lafla thinking SFT mixture builder")
    parser.add_argument("--thinking-jsonl", required=True)
    parser.add_argument("--safety-jsonl", required=True)
    parser.add_argument("--output-jsonl", required=True)
    parser.add_argument("--report-json")
    parser.add_argument("--target-safety-ratio", type=float, default=0.08)
    parser.add_argument("--max-safety-per-template", type=int, default=12)
    parser.add_argument("--seed", type=int, default=1337)
    args = parser.parse_args(argv)
    summary = build_thinking_sft_mixture(
        thinking_jsonl=args.thinking_jsonl,
        safety_jsonl=args.safety_jsonl,
        output_jsonl=args.output_jsonl,
        report_json=args.report_json,
        target_safety_ratio=args.target_safety_ratio,
        max_safety_per_template=args.max_safety_per_template,
        seed=args.seed,
    )
    print(summary.to_json())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
