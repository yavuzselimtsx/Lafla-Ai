"""
@Dosya: cli/check_environment.py
@Aciklama: Colab/yerel egitim ortami bagimlilik kontrolu CLI girisi.
@Yazar: Lafla Gelistirme Ekibi
@Bilgi: Notebook hucreleri bu komutu calistirarak eksik paketleri egitim
        baslamadan gorur.
"""

from __future__ import annotations

import argparse
from typing import Sequence

from lafla_ai_core.environment.dependencies import check_required_modules, colab_training_requirements


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="LaflaAi-Core environment check")
    parser.add_argument("--optimizer", default="adamw8bit", choices=("adamw", "adamw8bit", "lion"))
    parser.add_argument("--accelerator", default="cuda", choices=("cuda", "xla", "cpu", "auto"))
    args = parser.parse_args(argv)
    report = check_required_modules(colab_training_requirements(args.optimizer, accelerator=args.accelerator))
    print(report.to_json())
    return 0 if report.ok else 2


if __name__ == "__main__":
    raise SystemExit(main())
