import unittest

from lafla_ai_core.kaggle.run_plan import KagglePaths, build_kaggle_run_plan


class KaggleRunPlanTest(unittest.TestCase):
    def test_plan_uses_organized_work_dirs_and_cuda_training_config(self):
        plan = build_kaggle_run_plan(KagglePaths(), "/kaggle/working/LaflaAI100M/data/train.jsonl")
        joined = "\n".join(plan.commands)

        self.assertIn("/kaggle/working/LaflaAI100M/data", joined)
        self.assertIn("/kaggle/working/LaflaAI100M/tokenizer", joined)
        self.assertIn("/kaggle/working/LaflaAI100M/checkpoints", joined)
        self.assertIn("/kaggle/working/LaflaAI100M/reports", joined)
        self.assertIn("/kaggle/working/LaflaAI100M/hf-package", joined)
        self.assertIn("/kaggle/working/LaflaAI100M/archives", joined)
        self.assertIn("configs/training/kaggle/kaggle-gpu-100m.yaml", joined)
        self.assertIn("--accelerator cuda", joined)
        self.assertIn("lafla_ai_core.cli.train_pretrain", joined)
        self.assertIn("lafla_ai_core.cli.validate_pretraining_data", joined)
        self.assertIn("torchrun --standalone --nproc_per_node", joined)
        self.assertIn("CUDA_DEVICE_COUNT", joined)
        self.assertIn("RESUME_FROM", joined)
        self.assertIn("lafla-100m-thinking-kaggle-gpu-run.tar.gz", joined)
        self.assertNotIn("PJRT_DEVICE=TPU", joined)
        self.assertNotIn("torch_xla", joined)
        self.assertLess(joined.index("quality_scan"), joined.index("train_pretrain"))

    def test_plan_requires_real_data_and_manifest_before_training(self):
        plan = build_kaggle_run_plan(KagglePaths(), "/kaggle/working/LaflaAI100M/data/train.jsonl")
        joined = "\n".join(plan.commands)

        self.assertIn("test -s /kaggle/working/LaflaAI100M/data/train.jsonl", joined)
        self.assertIn("test -s /kaggle/working/LaflaAI100M/data/veri_manifesti.json", joined)
        self.assertNotIn("lightning_prepare_real_data.py", joined)
        self.assertLess(joined.index("validate_pretraining_data"), joined.index("train_pretrain"))


if __name__ == "__main__":
    unittest.main()
