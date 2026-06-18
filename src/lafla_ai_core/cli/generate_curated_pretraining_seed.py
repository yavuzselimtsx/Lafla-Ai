"""Kaynak temelli curated continued-pretraining seed CLI'ı."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Sequence

from lafla_ai_core.pretraining.curated_seed import (
    DEFAULT_COUNT,
    DEFAULT_MANIFEST_PATH,
    DEFAULT_OUTPUT_PATH,
    DEFAULT_SOURCE_PATH,
    generate_curated_pretraining_seed,
)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Kaynak temelli curated continued-pretraining seed üretici")
    parser.add_argument("--source", default=str(DEFAULT_SOURCE_PATH))
    parser.add_argument("--count", type=int, default=DEFAULT_COUNT)
    parser.add_argument("--output", default=str(DEFAULT_OUTPUT_PATH))
    parser.add_argument("--manifest", default=str(DEFAULT_MANIFEST_PATH))
    args = parser.parse_args(argv)
    report = generate_curated_pretraining_seed(
        source_path=Path(args.source),
        output_path=Path(args.output),
        manifest_path=Path(args.manifest),
        count=args.count,
    )
    print(report.to_json())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
