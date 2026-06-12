"""
@Dosya: cli/quality_scan.py
@Aciklama: LaflaAi-Core statik kalite kapisi CLI girisi.
@Yazar: Lafla Gelistirme Ekibi
@Bilgi: Colab paketlemeden once bilinen risk kaliplarini fail-closed yakalar.
"""

from __future__ import annotations

import argparse
from typing import Sequence

from lafla_ai_core.quality.static_scan import collect_project_text_files, run_static_scan


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="LaflaAi-Core static quality scan")
    parser.add_argument("--root", default=".")
    args = parser.parse_args(argv)
    report = run_static_scan(collect_project_text_files(args.root))
    print(report.to_json())
    return 0 if report.ok else 2


if __name__ == "__main__":
    raise SystemExit(main())
