"""
@Dosya: cli/test_checkpoint.py
@Aciklama: Egitilmis checkpoint icin kisa generation smoke testi.
@Yazar: Lafla Gelistirme Ekibi
@Bilgi: Colab 100-step kosusundan sonra checkpoint public cevabi JSON raporlanir.
"""

from __future__ import annotations

import argparse
from typing import Sequence

from lafla_ai_core.config.loader import load_mapping
from lafla_ai_core.config.schema import RuntimeConfig
from lafla_ai_core.runtime.checkpoint_inference import generate_from_checkpoint


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="LaflaAi-Core checkpoint smoke generation")
    parser.add_argument("--checkpoint-dir", required=True)
    parser.add_argument("--tokenizer-path", required=True)
    parser.add_argument("--prompt", default="2+2 kac eder? Kisa cevap ver.")
    parser.add_argument("--system-text")
    parser.add_argument("--max-new-tokens", type=int, default=64)
    parser.add_argument("--device")
    parser.add_argument("--runtime-config")
    args = parser.parse_args(argv)
    runtime_config = RuntimeConfig.from_mapping(load_mapping(args.runtime_config)) if args.runtime_config else None
    if runtime_config is not None:
        runtime_config.validate()
    result = generate_from_checkpoint(
        checkpoint_dir=args.checkpoint_dir,
        tokenizer_path=args.tokenizer_path,
        user_text=args.prompt,
        system_text=args.system_text,
        max_new_tokens=args.max_new_tokens,
        device=args.device,
        runtime_config=runtime_config,
    )
    print(result.to_json())
    return 0 if result.quality_ok else 2


if __name__ == "__main__":
    raise SystemExit(main())
