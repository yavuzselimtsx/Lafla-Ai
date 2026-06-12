"""
@Dosya: cli/colab_plan.py
@Aciklama: Colab calistirma plani ureten CLI girisi.
@Yazar: Lafla Gelistirme Ekibi
@Bilgi: Kullaniciya yapistirilacak komutlar repo tarafindan uretilir.
@Uyari: Komut plani egitim mantiginin yerine gecmez.
@Calisma-Semasi: args -> ColabPaths -> ColabRunPlan -> output
"""

from __future__ import annotations

import argparse
from typing import Sequence

from lafla_ai_core.colab.run_plan import ColabPaths, ColabTrainingProfile, build_colab_run_plan


def main(argv: Sequence[str] | None = None) -> int:
    """CLI ana fonksiyonu."""

    parser = argparse.ArgumentParser(description="Lafla Colab run plan")
    parser.add_argument("--data-jsonl", required=True)
    parser.add_argument("--manifest", default="/content/LaflaAI100M/data/veri_manifesti.json")
    parser.add_argument("--repo-dir", default="/content/LaflaAi-Core")
    parser.add_argument("--drive-root", default="/content/gdrive/MyDrive/LaflaAI100M")
    parser.add_argument("--work-root", default="/content/LaflaAI100M")
    parser.add_argument("--artifact-name", default="lafla-100m-thinking-colab-tpu-v5e-run.tar.gz")
    parser.add_argument("--model-config", default="configs/model/lafla-100m-thinking.yaml")
    parser.add_argument("--training-config", default="configs/training/colab-tpu-v5e-100m.yaml")
    parser.add_argument("--tokenizer-config", default="configs/tokenizer/turkish-german-thinking-bpe.yaml")
    parser.add_argument("--runtime-config", default="configs/runtime/desktop-i3-int8-100m.yaml")
    parser.add_argument("--post-training-config", default="configs/post_training/lafla-thinking-sft.yaml")
    parser.add_argument("--identity-data", default="configs/data/lafla-model-identity-100m.jsonl")
    parser.add_argument("--source-plan", default="configs/data/lafla-100m-source-plan.json")
    parser.add_argument("--model-name", default="lafla-100m-thinking")
    parser.add_argument("--thinking-jsonl", action="append", help="Ek thinking SFT JSONL denetimi; birden fazla kez verilebilir")
    args = parser.parse_args(argv)
    plan = build_colab_run_plan(
        ColabPaths(
            repo_dir=args.repo_dir,
            drive_root=args.drive_root,
            work_root=args.work_root,
            artifact_name=args.artifact_name,
            checkpoint_dir=f"{args.work_root}/checkpoints",
            tokenizer_path=f"{args.work_root}/tokenizer/lafla-tokenizer.json",
            tokenizer_report_path=f"{args.work_root}/reports/tokenizer-quality.json",
        ),
        data_jsonl=args.data_jsonl,
        manifest_path=args.manifest,
        thinking_jsonls=args.thinking_jsonl,
        profile=ColabTrainingProfile(
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
