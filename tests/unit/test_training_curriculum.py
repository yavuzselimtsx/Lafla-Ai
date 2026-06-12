import unittest

from lafla_ai_core.config.loader import load_mapping
from lafla_ai_core.config.schema import TrainingConfig
from lafla_ai_core.training.curriculum import resolve_curriculum_stage, tokens_per_optimizer_step


class TrainingCurriculumTest(unittest.TestCase):
    def test_resolves_sequence_length_from_cumulative_token_boundaries(self):
        config = TrainingConfig.from_mapping(load_mapping("configs/training/colab/colab-tpu-v5e-100m.yaml"))

        self.assertEqual(resolve_curriculum_stage(config, 0).sequence_length, 2048)
        self.assertEqual(resolve_curriculum_stage(config, 3_600_000_000).sequence_length, 4096)
        self.assertEqual(resolve_curriculum_stage(config, 4_800_000_000).sequence_length, 8192)
        self.assertEqual(resolve_curriculum_stage(config, 5_400_000_000).sequence_length, 12288)
        self.assertEqual(resolve_curriculum_stage(config, 5_700_000_000).sequence_length, 16384)
        self.assertEqual(resolve_curriculum_stage(config, 5_900_000_000).sequence_length, 20480)

    def test_non_curriculum_config_uses_base_sequence_length(self):
        config = TrainingConfig.from_mapping(load_mapping("configs/training/colab/colab-tpu-v5e-380m-50000.yaml"))
        stage = resolve_curriculum_stage(config, 123)
        self.assertEqual(stage.index, 0)
        self.assertEqual(stage.sequence_length, config.sequence_length)

    def test_optimizer_step_token_count_uses_active_stage_length(self):
        config = TrainingConfig.from_mapping(load_mapping("configs/training/colab/colab-tpu-v5e-100m.yaml"))
        stage = resolve_curriculum_stage(config, 4_800_000_000)
        self.assertEqual(
            tokens_per_optimizer_step(config, stage),
            8192 * config.micro_batch_size * config.gradient_accumulation_steps,
        )

    def test_optimizer_step_token_count_accepts_resolved_batch_geometry(self):
        config = TrainingConfig.from_mapping(load_mapping("configs/training/kaggle/kaggle-gpu-100m.yaml"))
        stage = resolve_curriculum_stage(config, 0)

        self.assertEqual(
            tokens_per_optimizer_step(
                config,
                stage,
                micro_batch_size=4,
                gradient_accumulation_steps=8,
            ),
            2048 * 4 * 8,
        )


if __name__ == "__main__":
    unittest.main()
