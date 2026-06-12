"""
@Dosya: cli/validate_thinking_sft.py
@Aciklama: Thinking SFT JSONL veri kapisini komut satirindan calistirir.
@Yazar: Lafla Gelistirme Ekibi
@Bilgi: Colab'da uzun post-training kosusundan once veri hatalarini yakalamak
        icin JSON raporu uretir.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Sequence

from lafla_ai_core.post_training.thinking_dataset import validate_thinking_jsonl_file


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="LaflaAi-Core thinking SFT validator")
    parser.add_argument("--input", required=True, help="Thinking SFT JSONL dosyasi")
    parser.add_argument("--report", help="JSON rapor yolu")
    parser.add_argument("--max-thinking-chars", type=int, default=4000)
    args = parser.parse_args(argv)

    report = validate_thinking_jsonl_file(args.input, max_thinking_chars=args.max_thinking_chars)
    payload = report.to_json()
    if args.report:
        report_path = Path(args.report)
        report_path.parent.mkdir(parents=True, exist_ok=True)
        report_path.write_text(payload + "\n", encoding="utf-8")
    print(payload)
    return 0 if report.ok else 2


if __name__ == "__main__":
    raise SystemExit(main())
