"""
@Dosya: cli/kaggle_plan.py
@Aciklama: Kaggle GPU calistirma plani ureten CLI girisi.
@Yazar: Lafla Gelistirme Ekibi
@Bilgi: Notebook hucreleri yerine repo tarafindan tekrar uretilebilir komut
        listesi basar.
@Uyari: Komut plani egitim mantiginin yerine gecmez.
@Calisma-Semasi: args -> KagglePaths -> KaggleRunPlan -> output
"""

from __future__ import annotations

import argparse
from typing import Sequence

from lafla_ai_core.kaggle.run_plan import KagglePaths, KaggleTrainingProfile, build_kaggle_run_plan


def main(argv: Sequence[str] | None = None) -> int:
    """CLI ana fonksiyonu."""

    parser = argparse.ArgumentParser(description="Lafla Kaggle GPU run plan")
    parser.add_argument("--data-jsonl", required=True)
    parser.add_argument("--manifest", default="/kaggle/working/LaflaAI100M/data/veri_manifesti.json")
    parser.add_argument("--repo-dir", default="/kaggle/working/LaflaAi-Core")
    parser.add_argument("--work-root", default="/kaggle/working/LaflaAI100M")
    parser.add_argument("--artifact-name", default="lafla-100m-thinking-kaggle-gpu-run.tar.gz")
    parser.add_argument("--model-config", default="configs/model/lafla-100m-thinking.yaml")
    parser.add_argument("--training-config", default="configs/training/kaggle/kaggle-gpu-100m.yaml")
    parser.add_argument("--tokenizer-config", default="configs/tokenizer/turkish-german-thinking-bpe.yaml")
    parser.add_argument("--runtime-config", default="configs/runtime/desktop-i3-int8-100m.yaml")
    parser.add_argument("--post-training-config", default="configs/post_training/lafla-thinking-sft.yaml")
    parser.add_argument("--identity-data", default="configs/data/identity/lafla-model-identity-100m.jsonl")
    parser.add_argument("--source-plan", default="configs/data/source-plans/lafla-100m-source-plan.json")
    parser.add_argument("--model-name", default="lafla-100m-thinking")
    parser.add_argument("--thinking-jsonl", action="append", help="Ek thinking SFT JSONL denetimi; birden fazla kez verilebilir")
    args = parser.parse_args(argv)
    plan = build_kaggle_run_plan(
        KagglePaths(
            repo_dir=args.repo_dir,
            work_root=args.work_root,
            artifact_name=args.artifact_name,
            checkpoint_dir=f"{args.work_root}/checkpoints",
            tokenizer_path=f"{args.work_root}/tokenizer/lafla-tokenizer.json",
            tokenizer_report_path=f"{args.work_root}/reports/tokenizer-quality.json",
        ),
        data_jsonl=args.data_jsonl,
        manifest_path=args.manifest,
        thinking_jsonls=args.thinking_jsonl,
        profile=KaggleTrainingProfile(
            model_config=args.model_config,
            training_config=args.training_config,
            tokenizer_config=args.tokenizer_config,
            runtime_config=args.runtime_config,
            post_training_config=args.post_training_config,
            identity_data=args.identity_data,
            source_plan=args.source_plan,
            model_name=args.model_name,
        ),
    )
    for index, command in enumerate(plan.commands, start=1):
        print(f"# Step {index}")
        print(command)
        print()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
