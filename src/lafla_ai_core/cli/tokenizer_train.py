"""
@Dosya: cli/tokenizer_train.py
@Aciklama: Lafla tokenizer egitimi icin CLI girisi saglar.
@Yazar: Lafla Gelistirme Ekibi
@Bilgi: CLI, config dogrulama ve kalite raporunu zorunlu tutar.
@Uyari: Kalite kapisi gecmezse cikis kodu basarisizdir.
@Calisma-Semasi: args -> config -> train -> report
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Sequence

from lafla_ai_core.config.loader import load_mapping
from lafla_ai_core.config.schema import TokenizerConfig
from lafla_ai_core.tokenizer.trainer import train_bpe_tokenizer


def main(argv: Sequence[str] | None = None) -> int:
    """CLI ana fonksiyonu."""

    parser = argparse.ArgumentParser(description="Lafla tokenizer train")
    parser.add_argument("--config", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--report", required=True)
    parser.add_argument("input_jsonl", nargs="+")
    args = parser.parse_args(argv)
    config = TokenizerConfig.from_mapping(load_mapping(args.config))
    result = train_bpe_tokenizer(
        input_paths=[Path(item) for item in args.input_jsonl],
        output_path=args.output,
        report_path=args.report,
        config=config,
    )
    print(result.to_json())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

