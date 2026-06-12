"""
@Dosya: cli/generate_safety_jailbreak_seed.py
@Aciklama: Lafla 100M guvenlik/jailbreak thinking SFT seed dosyasini uretir.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Sequence

from lafla_ai_core.post_training.safety_jailbreak_seed import (
    generate_safety_jailbreak_seed,
    profile_default_options,
)
from lafla_ai_core.post_training.seed_profile import DEFAULT_SEED_PROFILE_PATH


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Lafla 100M synthetic safety jailbreak seed generator")
    parser.add_argument("--profile", default=str(DEFAULT_SEED_PROFILE_PATH), help="Seed profil JSON yolu")
    parser.add_argument("--count", type=int, help="Yazilacak JSONL kayit sayisi")
    parser.add_argument("--output", help="JSONL cikti yolu")
    parser.add_argument("--manifest", help="Manifest JSON cikti yolu")
    args = parser.parse_args(argv)
    defaults = profile_default_options(args.profile)

    report = generate_safety_jailbreak_seed(
        output_path=Path(args.output) if args.output else defaults.output_path,
        manifest_path=Path(args.manifest) if args.manifest else defaults.manifest_path,
        count=args.count if args.count is not None else defaults.count,
        profile_path=args.profile,
    )
    print(report.to_json())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
