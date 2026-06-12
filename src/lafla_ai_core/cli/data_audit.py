"""
@Dosya: cli/data_audit.py
@Aciklama: Lafla veri manifestini audit eden CLI girisi.
@Yazar: Lafla Gelistirme Ekibi
@Bilgi: Colab ve yerel egitim bu komutu gecmeden veri shard/tokenizer uretimine
        baslamaz.
@Uyari: Audit hatasi egitim baslangicini durdurur.
@Calisma-Semasi: manifest -> audit -> json report -> exit code
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Sequence

from lafla_ai_core.data.audit import audit_manifest
from lafla_ai_core.data.manifest import load_manifest


def main(argv: Sequence[str] | None = None) -> int:
    """CLI ana fonksiyonu."""

    parser = argparse.ArgumentParser(description="Lafla data manifest audit")
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--report", required=False)
    args = parser.parse_args(argv)
    report = audit_manifest(load_manifest(args.manifest))
    output = report.to_json()
    if args.report:
        Path(args.report).parent.mkdir(parents=True, exist_ok=True)
        Path(args.report).write_text(output + "\n", encoding="utf-8")
    print(output)
    return 0 if report.ok else 2


if __name__ == "__main__":
    raise SystemExit(main())

