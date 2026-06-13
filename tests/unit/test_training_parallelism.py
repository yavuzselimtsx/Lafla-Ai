import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from lafla_ai_core.config.schema import ModelConfig, TrainingConfig
from lafla_ai_core.training.parallelism import (
    ParallelismDecision,
    effective_micro_batch_size,
    iter_rank_positions,
    resolve_batch_geometry,
    resolve_gradient_checkpointing,
    resolve_parallelism,
    should_sync_gradients,
)

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

    def test_training_config_accepts_safe_distributed_fast_paths(self):
        config = _training_config(
            data_parallel="auto",
            distributed_backend="auto",
            gradient_sync="final_microstep",
            pin_memory=True,
            prefer_fused_optimizer=True,
            prefer_native_gqa=True,
        )

        config.validate()

        self.assertEqual(config.distributed_backend, "auto")
        self.assertEqual(config.gradient_sync, "final_microstep")
        self.assertTrue(config.pin_memory)
        self.assertTrue(config.prefer_fused_optimizer)
        self.assertTrue(config.prefer_native_gqa)

    def test_ddp_geometry_keeps_32_sequences_with_rank_local_batches(self):
        decision = resolve_parallelism(
            data_parallel="auto",
            device_type="cuda",
            cuda_device_count=2,
            distributed_world_size=2,
            rank=1,
            local_rank=1,
        )

        geometry = resolve_batch_geometry(
            configured_micro_batch_size=1,
            configured_gradient_accumulation_steps=16,
            cuda_micro_batch_size_per_device=2,
            target_sequences_per_optimizer_step=32,
            decision=decision,
        )

        self.assertEqual(decision.mode, "distributed_data_parallel")
        self.assertEqual(geometry.per_process_micro_batch_size, 2)
        self.assertEqual(geometry.global_micro_batch_size, 4)
        self.assertEqual(geometry.gradient_accumulation_steps, 8)
        self.assertEqual(geometry.sequences_per_optimizer_step, 32)

    def test_rank_positions_are_disjoint_and_reconstruct_source(self):
        source = tuple(range(8))
        rank_zero = tuple(iter_rank_positions(source, rank=0, world_size=2))
        rank_one = tuple(iter_rank_positions(source, rank=1, world_size=2))

        self.assertEqual(rank_zero, (0, 2, 4, 6))
        self.assertEqual(rank_one, (1, 3, 5, 7))
        self.assertEqual(tuple(value for pair in zip(rank_zero, rank_one) for value in pair), source)

    def test_gradient_sync_occurs_only_on_final_accumulation_microstep(self):
        self.assertFalse(should_sync_gradients(0, 8))
        self.assertFalse(should_sync_gradients(6, 8))
        self.assertTrue(should_sync_gradients(7, 8))

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

    def test_safe_speed_geometry_keeps_32_sequences_on_two_cuda_devices(self):
        decision = ParallelismDecision(
            enabled=True,
            mode="data_parallel",
            cuda_device_count=2,
            reason="multi_cuda_available",
        )

        geometry = resolve_batch_geometry(
            configured_micro_batch_size=1,
            configured_gradient_accumulation_steps=16,
            cuda_micro_batch_size_per_device=2,
            target_sequences_per_optimizer_step=32,
            decision=decision,
        )

        self.assertEqual(geometry.global_micro_batch_size, 4)
        self.assertEqual(geometry.gradient_accumulation_steps, 8)
        self.assertEqual(geometry.sequences_per_optimizer_step, 32)

    def test_safe_speed_geometry_keeps_32_sequences_on_single_cuda_device(self):
        decision = ParallelismDecision(
            enabled=False,
            mode="single_device",
            cuda_device_count=1,
            reason="less_than_two_cuda_devices",
        )

        geometry = resolve_batch_geometry(
            configured_micro_batch_size=1,
            configured_gradient_accumulation_steps=16,
            cuda_micro_batch_size_per_device=2,
            target_sequences_per_optimizer_step=32,
            decision=decision,
        )

        self.assertEqual(geometry.global_micro_batch_size, 2)
        self.assertEqual(geometry.gradient_accumulation_steps, 16)
        self.assertEqual(geometry.sequences_per_optimizer_step, 32)

    def test_batch_geometry_preserves_existing_profiles_when_tuning_is_disabled(self):
        decision = ParallelismDecision(
            enabled=True,
            mode="data_parallel",
            cuda_device_count=2,
            reason="multi_cuda_available",
        )

        geometry = resolve_batch_geometry(
            configured_micro_batch_size=1,
            configured_gradient_accumulation_steps=16,
            cuda_micro_batch_size_per_device=0,
            target_sequences_per_optimizer_step=0,
            decision=decision,
        )

        self.assertEqual(geometry.global_micro_batch_size, 2)
        self.assertEqual(geometry.gradient_accumulation_steps, 16)
        self.assertEqual(geometry.sequences_per_optimizer_step, 32)

    def test_gradient_checkpointing_starts_at_configured_sequence_threshold(self):
        self.assertFalse(
            resolve_gradient_checkpointing(
                model_checkpointing_enabled=True,
                minimum_sequence_length=4096,
                active_sequence_length=2048,
            )
        )
        self.assertTrue(
            resolve_gradient_checkpointing(
                model_checkpointing_enabled=True,
                minimum_sequence_length=4096,
                active_sequence_length=4096,
            )
        )


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
