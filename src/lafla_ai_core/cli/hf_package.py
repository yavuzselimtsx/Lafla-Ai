"""
@Dosya: cli/hf_package.py
@Aciklama: Hugging Face tokenizer/runtime metadata paketi uretir.
@Yazar: Lafla Gelistirme Ekibi
@Bilgi: CLI yalniz export modulunu cagirir; paketleme mantigi burada tutulmaz.
"""

from __future__ import annotations

import argparse
from typing import Sequence

from lafla_ai_core.config.loader import load_mapping
from lafla_ai_core.config.schema import ModelConfig
from lafla_ai_core.export.hf_package import write_hf_tokenizer_package


def main(argv: Sequence[str] | None = None) -> int:
    """HF paketleme CLI girisi."""

    parser = argparse.ArgumentParser(description="LaflaAi-Core Hugging Face package writer")
    parser.add_argument("--tokenizer-json", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--model-name", required=True)
    parser.add_argument("--model-config", help="Opsiyonel HF remote-code config.json icin model yaml")
    args = parser.parse_args(argv)
    model_config = ModelConfig.from_mapping(load_mapping(args.model_config)) if args.model_config else None
    output = write_hf_tokenizer_package(args.tokenizer_json, args.output_dir, model_name=args.model_name, model_config=model_config)
    print(f"hf package written: {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
