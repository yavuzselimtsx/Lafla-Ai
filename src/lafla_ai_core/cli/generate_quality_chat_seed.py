"""Lafla kalite odaklı sohbet SFT seed üretim CLI'ı."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Sequence

from lafla_ai_core.post_training.quality_chat_seed import (
    DEFAULT_COUNT,
    DEFAULT_MANIFEST_PATH,
    DEFAULT_OUTPUT_PATH,
    generate_quality_chat_seed,
)
from lafla_ai_core.post_training.seed_profile import DEFAULT_SEED_PROFILE_PATH


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Lafla kalite odaklı sohbet SFT seed üretici")
    parser.add_argument("--profile", default=str(DEFAULT_SEED_PROFILE_PATH))
    parser.add_argument("--count", type=int, default=DEFAULT_COUNT)
    parser.add_argument("--output", default=str(DEFAULT_OUTPUT_PATH))
    parser.add_argument("--manifest", default=str(DEFAULT_MANIFEST_PATH))
    args = parser.parse_args(argv)
    report = generate_quality_chat_seed(
        output_path=Path(args.output),
        manifest_path=Path(args.manifest),
        count=args.count,
        profile_path=Path(args.profile),
    )
    print(report.to_json())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
