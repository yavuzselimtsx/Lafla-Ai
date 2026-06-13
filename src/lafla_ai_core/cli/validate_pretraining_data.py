"""
@Dosya: cli/validate_pretraining_data.py
@Aciklama: Pretraining JSONL fail-fast dogrulama komutu.
@Uyari: Bu komut veri uretmez veya onarmaz; bozuk girdiyi acik hata ile reddeder.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Sequence

from lafla_ai_core.data.preflight import validate_pretraining_jsonl


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Lafla pretraining JSONL dogrulama")
    parser.add_argument("--data-jsonl", action="append", required=True)
    parser.add_argument("--report")
    args = parser.parse_args(argv)

    report = validate_pretraining_jsonl(tuple(args.data_jsonl))
    payload = report.to_json()
    if args.report:
        output = Path(args.report)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(payload + "\n", encoding="utf-8")
    print(payload)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
