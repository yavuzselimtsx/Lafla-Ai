"""
@Dosya: cli/artifact_manifest.py
@Aciklama: Artifact klasoru icin hashli manifest CLI girisi.
@Yazar: Lafla Gelistirme Ekibi
@Bilgi: Colab final arsivinden once tokenizer, checkpoint ve rapor dosyalari izlenir.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Sequence

from lafla_ai_core.observability.artifact_manifest import write_manifest


def main(argv: Sequence[str] | None = None) -> int:
    """Artifact manifesti yazar."""

    parser = argparse.ArgumentParser(description="LaflaAi-Core artifact manifest writer")
    parser.add_argument("--root", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args(argv)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    write_manifest(args.root, output)
    print(f"artifact manifest written: {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
