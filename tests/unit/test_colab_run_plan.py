import unittest
import json
from pathlib import Path

from lafla_ai_core.colab.run_plan import ColabPaths, ColabTrainingProfile, build_colab_run_plan


class ColabRunPlanTest(unittest.TestCase):
    def test_rejects_non_drive_artifact_root(self):
        with self.assertRaises(ValueError):
            build_colab_run_plan(ColabPaths(drive_root="/content/drive/MyDrive/LaflaAI"), "/content/data.jsonl")

    def test_plan_contains_preflight_and_archive(self):
        plan = build_colab_run_plan(ColabPaths(), "/content/data.jsonl")
        joined = "\n".join(plan.commands)
        self.assertIn("lafla_ai_core.cli.quality_scan", joined)
        self.assertIn("lafla_ai_core.cli.preflight", joined)
        self.assertIn("PJRT_DEVICE=TPU", joined)
        self.assertIn("requirements/colab-tpu.txt", joined)
        self.assertIn("configs/model/lafla-100m-thinking.yaml", joined)
        self.assertIn("configs/training/colab-tpu-v5e-100m.yaml", joined)
        self.assertIn("--model-name lafla-100m-thinking", joined)
        self.assertIn("configs/data/lafla-model-identity-100m.jsonl", joined)
        self.assertIn("configs/post_training/lafla-thinking-sft.yaml", joined)
        self.assertIn("lafla_ai_core.cli.data_audit", joined)
        self.assertIn("pip install -r requirements/colab-tpu.txt", joined)
        self.assertIn("lafla_ai_core.cli.train_pretrain", joined)
        self.assertIn("lafla_ai_core.cli.hf_package", joined)
        self.assertNotIn("TRAINING_ENTRYPOINT_NOT_WIRED_YET", joined)
        self.assertIn("tar -czf", joined)
        self.assertIn("lafla-final", joined)
        self.assertNotIn("-C /content LaflaAI", joined)
        self.assertIn("/content/gdrive/MyDrive/LaflaAI", joined)
        self.assertIn("lafla_ai_core.cli.artifact_manifest", joined)
        self.assertIn("lafla-100m-thinking-colab-tpu-v5e-run.tar.gz", joined)
        self.assertLess(joined.index("tokenizer_train"), joined.index("hf_package"))
        self.assertLess(joined.index("hf_package"), joined.index("train_pretrain"))
        self.assertLess(joined.index("artifact_manifest"), joined.index("tar -czf"))

    def test_profile_supplies_all_active_model_paths_without_380m_constants(self):
        profile = ColabTrainingProfile(
            model_config="configs/model/custom.yaml",
            training_config="configs/training/custom.yaml",
            tokenizer_config="configs/tokenizer/custom.yaml",
            runtime_config="configs/runtime/custom.yaml",
            post_training_config="configs/post_training/custom.yaml",
            identity_data="configs/data/custom-identity.jsonl",
            source_plan="configs/data/custom-source-plan.json",
            model_name="custom-model",
        )

        plan = build_colab_run_plan(ColabPaths(), "/content/data.jsonl", profile=profile)
        joined = "\n".join(plan.commands)

        for value in (
            profile.model_config,
            profile.training_config,
            profile.tokenizer_config,
            profile.runtime_config,
            profile.post_training_config,
            profile.identity_data,
            profile.source_plan,
            profile.model_name,
        ):
            self.assertIn(value, joined)
        self.assertNotIn("lafla-380m", joined)

    def test_plan_can_validate_thinking_sft_before_training(self):
        plan = build_colab_run_plan(
            ColabPaths(),
            "/content/data.jsonl",
            thinking_jsonl="/content/LaflaAI/data/thinking_sft.jsonl",
        )
        joined = "\n".join(plan.commands)
        self.assertIn("lafla_ai_core.cli.validate_thinking_sft", joined)
        self.assertLess(joined.index("validate_thinking_sft"), joined.index("train_pretrain"))

    def test_default_plan_validates_builtin_sft_seed_files_without_mixing_into_pretraining(self):
        plan = build_colab_run_plan(ColabPaths(), "/content/data.jsonl")
        joined = "\n".join(plan.commands)

        self.assertIn("datasets/synthetic/lafla-100m-thinking-chat-seed-20k.jsonl", joined)
        self.assertIn("datasets/synthetic/lafla-100m-safety-jailbreak-seed-10k.jsonl", joined)
        self.assertIn("thinking-sft-audit-001.json", joined)
        self.assertIn("thinking-sft-audit-002.json", joined)
        self.assertLess(joined.index("lafla-100m-thinking-chat-seed-20k.jsonl"), joined.index("train_pretrain"))
        self.assertNotIn("--data-jsonl datasets/synthetic/lafla-100m-thinking-chat-seed-20k.jsonl", joined)
        self.assertNotIn("--data-jsonl datasets/synthetic/lafla-100m-safety-jailbreak-seed-10k.jsonl", joined)

    def test_profile_can_override_default_sft_seed_files(self):
        profile = ColabTrainingProfile(
            thinking_sft_data=("custom-a.jsonl", "custom-b.jsonl"),
        )
        plan = build_colab_run_plan(ColabPaths(), "/content/data.jsonl", profile=profile)
        joined = "\n".join(plan.commands)

        self.assertIn("custom-a.jsonl", joined)
        self.assertIn("custom-b.jsonl", joined)
        self.assertNotIn("lafla-100m-thinking-chat-seed-20k.jsonl", joined)

    def test_100m_launcher_and_notebook_require_real_data_without_bootstrap_generation(self):
        script = Path("scripts/colab_start_tpu_v5e_100m.sh").read_text(encoding="utf-8")
        notebook = json.loads(Path("notebooks/lafla_colab_tpu_100m_training.ipynb").read_text(encoding="utf-8"))
        notebook_text = json.dumps(notebook, ensure_ascii=False)

        self.assertIn("configs/model/lafla-100m-thinking.yaml", script)
        self.assertIn('test -s "$DATA_JSONL"', script)
        self.assertIn('test -s "$MANIFEST"', script)
        self.assertIn("lafla-100m-thinking-chat-seed-20k.jsonl", script)
        self.assertIn("lafla-100m-safety-jailbreak-seed-10k.jsonl", script)
        self.assertIn("validate_thinking_sft", script)
        self.assertNotIn("lightning_prepare_real_data.py", script)
        self.assertIn("colab_start_tpu_v5e_100m.sh", notebook_text)
        self.assertIn("veri_manifesti.json", notebook_text)


if __name__ == "__main__":
    unittest.main()
