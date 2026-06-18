import unittest
from pathlib import Path


class LightningH100LauncherTest(unittest.TestCase):
    def test_shared_launcher_exposes_hardware_profile_overrides(self):
        script = Path("scripts/lightning/start_t4_100m.sh").read_text(encoding="utf-8")

        for variable in (
            "PROFILE_NAME",
            "PYTORCH_INDEX_URL",
            "PYTORCH_VERSION",
            "DEVICE_LABEL",
            "DATASET_VERSION",
            "ARCHIVE_NAME",
            "CUDA_BATCH_SCALE",
        ):
            self.assertIn(variable, script)
        self.assertIn('--cuda-batch-scale "$CUDA_BATCH_SCALE"', script)
        self.assertIn("download.pytorch.org/whl/cu121", script)

    def test_h100_launcher_uses_bf16_capable_runtime_and_oom_only_fallback(self):
        script = Path("scripts/lightning/start_h100_100m.sh").read_text(encoding="utf-8")
        shared = Path("scripts/lightning/start_t4_100m.sh").read_text(encoding="utf-8")

        self.assertIn("lightning-h100-100m-quality-fast.yaml", script)
        self.assertIn("lafla-100m-h100", script)
        self.assertIn("lightning-h100-100m.log", script)
        self.assertIn("download.pytorch.org/whl/cu128", script)
        self.assertIn("2.11.0", script)
        self.assertIn("NVIDIA H100", script)
        self.assertIn("is_bf16_supported", script + shared)
        self.assertIn("for CUDA_BATCH_SCALE in 1.0 0.5 0.25", script)
        self.assertIn("OutOfMemoryError", script)
        self.assertIn("CUDA out of memory", script)
        self.assertIn("start_t4_100m.sh", script)

    def test_h100_background_launcher_prevents_duplicate_training(self):
        script = Path("scripts/lightning/start_h100_100m_background.sh").read_text(encoding="utf-8")

        self.assertIn("nohup bash scripts/lightning/start_h100_100m.sh", script)
        self.assertIn("lightning-h100-100m.pid", script)
        self.assertIn("lightning-h100-100m-nohup.log", script)
        self.assertIn('pgrep -af "lafla_ai_core.cli.train_pretrain"', script)

    def test_h100_monitor_selects_h100_nohup_log(self):
        script = Path("scripts/lightning/monitor_h100_100m.sh").read_text(encoding="utf-8")
        common = Path("scripts/lightning/monitor_100m.sh").read_text(encoding="utf-8")

        self.assertIn("lightning-h100-100m-nohup.log", script)
        self.assertIn("monitor_100m.sh", script)
        self.assertIn('NOHUP_LOG="${NOHUP_LOG:-', common)


if __name__ == "__main__":
    unittest.main()
