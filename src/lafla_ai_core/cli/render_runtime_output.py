"""
@Dosya: cli/render_runtime_output.py
@Aciklama: Ham model çıktısını runtime profiline göre JSON olarak işler.
@Yazar: Lafla Geliştirme Ekibi
@Bilgi: Developer research profilinde raw thinking alanını görünür kılar.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from typing import Sequence

from lafla_ai_core.config.loader import load_mapping
from lafla_ai_core.config.schema import RuntimeConfig
from lafla_ai_core.runtime.policy import build_generation_settings, render_runtime_output


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="LaflaAi-Core runtime output renderer")
    parser.add_argument("--runtime-config", required=True)
    parser.add_argument("--raw-text", required=True)
    parser.add_argument("--prompt-text", default=None)
    parser.add_argument("--system-text", default=None)
    args = parser.parse_args(argv)

    config = RuntimeConfig.from_mapping(load_mapping(args.runtime_config))
    config.validate()
    output = render_runtime_output(args.raw_text, config, prompt_text=args.prompt_text, system_text=args.system_text)
    payload = {
        "public_text": output.public_text,
        "raw_thinking": output.raw_thinking,
        "warnings": output.warnings,
        "generation_settings": asdict(build_generation_settings(config)),
    }
    print(json.dumps(payload, ensure_ascii=True, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
