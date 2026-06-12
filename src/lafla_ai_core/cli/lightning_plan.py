"""
@Dosya: cli/lightning_plan.py
@Aciklama: Lightning.ai H200 calistirma plani ureten CLI girisi.
@Yazar: Lafla Gelistirme Ekibi
"""

from __future__ import annotations

import argparse
from typing import Sequence

from lafla_ai_core.lightning.run_plan import LightningPaths, build_lightning_run_plan


def main(argv: Sequence[str] | None = None) -> int:
    """CLI ana fonksiyonu."""

    parser = argparse.ArgumentParser(description="Lafla Lightning H200 run plan")
    parser.add_argument("--data-jsonl", required=True)
    parser.add_argument("--manifest", default="/teamspace/studios/this_studio/LaflaAI380M/data/veri_manifesti.json")
    parser.add_argument("--repo-dir", default="/teamspace/studios/this_studio/LaflaAi-Core")
    parser.add_argument("--workspace-root", default="/teamspace/studios/this_studio/LaflaAI380M")
    parser.add_argument("--artifact-name", default="lafla-380m-thinking-h200-50000-step-run.tar.gz")
    parser.add_argument("--thinking-jsonl", help="Opsiyonel thinking SFT JSONL denetimi")
    parser.add_argument("--resume-from", help="Opsiyonel checkpoint yolu")
    args = parser.parse_args(argv)
    plan = build_lightning_run_plan(
        LightningPaths.for_workspace(repo_dir=args.repo_dir, workspace_root=args.workspace_root, artifact_name=args.artifact_name),
        data_jsonl=args.data_jsonl,
        manifest_path=args.manifest,
        thinking_jsonl=args.thinking_jsonl,
        resume_from=args.resume_from,
    )
    for index, command in enumerate(plan.commands, start=1):
        print(f"# Step {index}")
        print(command)
        print()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
