"""
@Dosya: cli/training_phase_plan.py
@Aciklama: Lafla varsayilan asamali egitim planini JSON olarak basar.
@Yazar: Lafla Gelistirme Ekibi
@Bilgi: Colab ve yerel operator komutlari ayni phase/gate sozlesmesini okur.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from typing import Sequence

from lafla_ai_core.training.phase_plan import default_lafla_100m_thinking_plan, validate_phase_plan


def main(argv: Sequence[str] | None = None) -> int:
    """CLI ana fonksiyonu."""

    parser = argparse.ArgumentParser(description="LaflaAi-Core training phase plan")
    parser.parse_args(argv)

    plan = default_lafla_100m_thinking_plan()
    report = validate_phase_plan(plan)
    payload = asdict(plan)
    payload["validation"] = asdict(report)
    print(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True))
    return 0 if report.ok else 2


if __name__ == "__main__":
    raise SystemExit(main())
