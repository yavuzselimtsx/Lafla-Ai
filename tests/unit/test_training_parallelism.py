import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from lafla_ai_core.config.schema import ModelConfig, TrainingConfig
from lafla_ai_core.training.parallelism import ParallelismDecision, effective_micro_batch_size

try:
    import torch
    from lafla_ai_core.model.checkpoint_io import load_training_checkpoint, save_training_checkpoint
    from lafla_ai_core.training.runner import _resolve_data_parallel
except ModuleNotFoundError:
    torch = None
    load_training_checkpoint = None
    save_training_checkpoint = None
    _resolve_data_parallel = None


def _training_config(**overrides):
    values = {
        "max_steps": 2,
        "sequence_length": 128,
        "micro_batch_size": 1,
        "gradient_accumulation_steps": 1,
        "precision": "fp32",
        "optimizer": "adamw",
        "learning_rate": 0.001,
        "min_learning_rate": 0.0001,
        "weight_decay": 0.0,
        "grad_clip_norm": 1.0,
        "warmup_steps": 1,
        "save_every": 1,
        "keep_last_checkpoints": 1,
        "checkpoint_min_free_gb": 0.0,
        "log_every": 1,
        "seed": 7,
        "require_drive_or_explicit_local_fallback": False,
    }
    values.update(overrides)
    return TrainingConfig(**values)


def _model_config():
    return ModelConfig(
        name="tiny",
        family="decoder-only",
        parameter_target=100_000_000,
        vocab_size=32,
        context_length=128,
        hidden_size=8,
        intermediate_size=16,
        num_layers=1,
        num_attention_heads=2,
        num_key_value_heads=1,
        activation="swiglu",
        norm="rmsnorm",
        rope=True,
        qk_norm=False,
    )


class TrainingParallelismConfigTest(unittest.TestCase):
    def test_training_config_accepts_data_parallel_auto(self):
        config = TrainingConfig.from_mapping(
            {
                "training": {
                    "max_steps": 10,
                    "sequence_length": 128,
                    "micro_batch_size": 1,
                    "gradient_accumulation_steps": 1,
                    "precision": "fp32",
                    "optimizer": "adamw",
                    "learning_rate": 0.001,
                    "warmup_steps": 1,
                    "save_every": 5,
                    "keep_last_checkpoints": 1,
                    "require_drive_or_explicit_local_fallback": False,
                    "accelerator": "cuda",
                    "data_parallel": "auto",
                }
            }
        )

        config.validate()

        self.assertEqual(config.data_parallel, "auto")

    def test_effective_micro_batch_uses_both_cuda_devices_when_data_parallel_is_enabled(self):
        decision = ParallelismDecision(
            enabled=True,
            mode="data_parallel",
            cuda_device_count=2,
            reason="multi_cuda_available",
        )

        self.assertEqual(effective_micro_batch_size(configured_micro_batch_size=1, decision=decision), 2)

    def test_effective_micro_batch_keeps_single_gpu_budget_unchanged(self):
        decision = ParallelismDecision(
            enabled=False,
            mode="single_device",
            cuda_device_count=1,
            reason="less_than_two_cuda_devices",
        )

        self.assertEqual(effective_micro_batch_size(configured_micro_batch_size=1, decision=decision), 1)


@unittest.skipIf(torch is None, "torch kurulu degil")
class TrainingParallelismDecisionTest(unittest.TestCase):
    def test_auto_enables_data_parallel_only_on_multi_cuda(self):
        assert _resolve_data_parallel is not None
        config = _training_config(accelerator="cuda", data_parallel="auto")

        with patch("lafla_ai_core.training.runner.torch.cuda.device_count", return_value=2):
            decision = _resolve_data_parallel(config, torch.device("cuda"))

        self.assertTrue(decision.enabled)
        self.assertEqual(decision.mode, "data_parallel")
        self.assertEqual(decision.cuda_device_count, 2)

    def test_auto_falls_back_to_single_device_when_only_one_gpu_exists(self):
        assert _resolve_data_parallel is not None
        config = _training_config(accelerator="cuda", data_parallel="auto")

        with patch("lafla_ai_core.training.runner.torch.cuda.device_count", return_value=1):
            decision = _resolve_data_parallel(config, torch.device("cuda"))

        self.assertFalse(decision.enabled)
        self.assertEqual(decision.mode, "single_device")
        self.assertEqual(decision.cuda_device_count, 1)


@unittest.skipIf(torch is None, "torch kurulu degil")
class CheckpointParallelismTest(unittest.TestCase):
    def test_checkpoint_loads_data_parallel_prefixed_weights_into_plain_model(self):
        assert save_training_checkpoint is not None
        assert load_training_checkpoint is not None
        model = torch.nn.Linear(4, 2)
        parallel_model = torch.nn.DataParallel(model)
        optimizer = torch.optim.AdamW(parallel_model.parameters(), lr=0.001)

        with tempfile.TemporaryDirectory() as temp_dir:
            checkpoint = Path(temp_dir) / "checkpoint"
            save_training_checkpoint(checkpoint, parallel_model, optimizer, _model_config(), {"step": 1})

            raw_state = torch.load(checkpoint / "model.pt", map_location="cpu")
            self.assertFalse(any(key.startswith("module.") for key in raw_state))

            restored = torch.nn.Linear(4, 2)
            load_training_checkpoint(checkpoint, restored, map_location="cpu")

        for left, right in zip(model.parameters(), restored.parameters()):
            self.assertTrue(torch.equal(left.detach(), right.detach()))


if __name__ == "__main__":
    unittest.main()
