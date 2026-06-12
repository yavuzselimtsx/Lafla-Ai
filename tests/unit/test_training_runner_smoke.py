import tempfile
import unittest
from pathlib import Path

from lafla_ai_core.config.schema import ModelConfig, TrainingConfig

try:
    import torch
    from lafla_ai_core.training.runner import TrainingPaths, run_pretraining
except ModuleNotFoundError:
    torch = None
    TrainingPaths = None
    run_pretraining = None


@unittest.skipIf(torch is None, "torch kurulu degil")
class TrainingRunnerSmokeTest(unittest.TestCase):
    def test_smoke_training_writes_final_checkpoint(self):
        assert TrainingPaths is not None
        assert run_pretraining is not None
        model_config = ModelConfig(
            name="tiny",
            family="decoder-only",
            parameter_target=100_000_000,
            vocab_size=128,
            context_length=512,
            hidden_size=16,
            intermediate_size=32,
            num_layers=1,
            num_attention_heads=4,
            num_key_value_heads=2,
            activation="swiglu",
            norm="rmsnorm",
            rope=True,
            qk_norm=False,
            gradient_checkpointing=False,
        )
        training_config = TrainingConfig(
            max_steps=2,
            sequence_length=128,
            micro_batch_size=1,
            gradient_accumulation_steps=1,
            precision="fp32",
            optimizer="adamw",
            learning_rate=0.001,
            min_learning_rate=0.0001,
            weight_decay=0.0,
            grad_clip_norm=1.0,
            warmup_steps=1,
            save_every=1,
            keep_last_checkpoints=1,
            checkpoint_min_free_gb=0.0,
            log_every=1,
            seed=7,
            require_drive_or_explicit_local_fallback=False,
        )
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            summary = run_pretraining(
                model_config,
                training_config,
                TrainingPaths(
                    data_jsonl=(),
                    tokenizer_path=str(root / "unused-tokenizer.json"),
                    checkpoint_dir=str(root / "checkpoints"),
                    health_log_path=str(root / "reports" / "health.jsonl"),
                ),
                smoke=True,
            )
            self.assertEqual(summary.final_step, 2)
            self.assertTrue((root / "checkpoints" / "lafla-final" / "READY.json").exists())
            self.assertTrue((root / "reports" / "health.jsonl").exists())


if __name__ == "__main__":
    unittest.main()
