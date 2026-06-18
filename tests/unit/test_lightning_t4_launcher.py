import unittest
from pathlib import Path


class LightningT4LauncherTest(unittest.TestCase):
    def test_t4_launcher_bootstraps_cuda_venv_real_data_and_resume(self):
        script = Path("scripts/lightning/start_t4_100m.sh").read_text(encoding="utf-8")

        self.assertIn("/teamspace/studios/this_studio", script)
        self.assertIn("LaflaAI100M", script)
        self.assertIn("python3 -m venv", script)
        self.assertIn("get-pip.py", script)
        self.assertIn("python3 -m virtualenv", script)
        self.assertIn("python3.12-venv", script)
        self.assertIn("python3-pip", script)
        self.assertIn("download.pytorch.org/whl/cu121", script)
        self.assertIn("requirements/kaggle-gpu.txt", script)
        self.assertIn("scripts/data/prepare_real_data.py", script)
        self.assertIn("--identity-jsonl configs/data/identity/lafla-model-identity-100m.jsonl", script)
        self.assertIn("lafla-100m-lightning-t4-realdata-2026-06", script)
        self.assertIn("validate_pretraining_data", script)
        self.assertIn("RESUME_FROM", script)
        self.assertIn("TRAINING_CONFIG", script)
        self.assertIn("LAFLA_TRAINING_CONFIG", script)
        self.assertIn("lafla_ai_core.cli.train_pretrain", script)
        self.assertIn("configs/training/lightning/lightning-t4-100m-quality-fast.yaml", script)
        self.assertIn("CUDA_DEVICE_COUNT", script)
        self.assertNotIn("torchrun --standalone", script)
        self.assertNotIn("synthetic", script.lower())
        self.assertNotIn("post_training/thinking", script)
        self.assertNotIn("post_training/safety", script)

    def test_t4_launcher_does_not_silently_start_without_real_data_contract(self):
        script = Path("scripts/lightning/start_t4_100m.sh").read_text(encoding="utf-8")

        self.assertLess(script.index("prepare_real_data.py"), script.index("data_audit"))
        self.assertLess(script.index("data_audit"), script.index("validate_pretraining_data"))
        self.assertLess(script.index("validate_pretraining_data"), script.index("train_pretrain"))

    def test_t4_background_launcher_keeps_user_terminal_free(self):
        script = Path("scripts/lightning/start_t4_100m_background.sh").read_text(encoding="utf-8")

        self.assertIn("nohup bash scripts/lightning/start_t4_100m.sh", script)
        self.assertIn("lightning-t4-100m.pid", script)
        self.assertIn("lightning-t4-100m-nohup.log", script)
        self.assertIn("pgrep -af \"lafla_ai_core.cli.train_pretrain\"", script)
        self.assertIn("monitor_100m.sh --watch 10", script)

    def test_t4_monitor_reports_process_gpu_logs_and_checkpoints(self):
        script = Path("scripts/lightning/monitor_100m.sh").read_text(encoding="utf-8")

        self.assertIn("nvidia-smi --query-gpu", script)
        self.assertIn("train-health.jsonl", script)
        self.assertIn("lightning-t4-100m-nohup.log", script)
        self.assertIn("CHECKPOINTS", script)
        self.assertIn("CHECKPOINT_BACKUPS", script)
        self.assertIn("CHECKPOINT BACKUPS", script)
        self.assertIn("--watch", script)

    def test_rtx_pro_6000_launcher_selects_dedicated_quality_fast_profile(self):
        script = Path("scripts/lightning/start_rtxp6000_100m.sh").read_text(encoding="utf-8")
        background = Path("scripts/lightning/start_rtxp6000_100m_background.sh").read_text(encoding="utf-8")

        self.assertIn("lightning-rtx-pro-6000-100m-quality-fast.yaml", script)
        self.assertIn("start_t4_100m.sh", script)
        self.assertIn("nohup bash scripts/lightning/start_rtxp6000_100m.sh", background)
        self.assertIn("lightning-rtxp6000-100m.pid", background)
        self.assertIn("lightning-rtxp6000-100m-nohup.log", background)


if __name__ == "__main__":
    unittest.main()
