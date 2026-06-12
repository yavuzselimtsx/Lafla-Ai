import tempfile
import unittest
from pathlib import Path

from lafla_ai_core.cli.preflight import run_preflight
from lafla_ai_core.config.loader import load_mapping
from lafla_ai_core.config.schema import ModelConfig, PostTrainingConfig, RuntimeConfig, TokenizerConfig, TrainingConfig


class ConfigPreflightTest(unittest.TestCase):
    def test_lafla_100m_profile_loads_with_long_context_runtime_contracts(self):
        model_path = Path("configs/model/lafla-100m-thinking.yaml")
        training_path = Path("configs/training/colab/colab-tpu-v5e-100m.yaml")
        tokenizer_path = Path("configs/tokenizer/turkish-german-thinking-bpe.yaml")
        runtime_path = Path("configs/runtime/desktop-i3-int8-100m.yaml")

        model = ModelConfig.from_mapping(load_mapping(model_path))
        training = TrainingConfig.from_mapping(load_mapping(training_path))
        tokenizer = TokenizerConfig.from_mapping(load_mapping(tokenizer_path))
        runtime = RuntimeConfig.from_mapping(load_mapping(runtime_path))
        report = run_preflight([model_path, training_path, tokenizer_path, runtime_path])

        self.assertTrue(report.ok, report.errors)
        self.assertEqual(model.num_layers, 12)
        self.assertEqual(model.attention_pattern, ("local", "local", "local", "global"))
        self.assertEqual(model.sliding_window, 4096)
        self.assertEqual(model.rope_scaling.type, "linear")
        self.assertEqual(model.context_length, 20_480)
        self.assertEqual(tokenizer.vocab_size, 32_768)
        self.assertEqual(training.sequence_curriculum, (2048, 4096, 8192, 12288, 16384, 20480))
        self.assertEqual(runtime.summary_trigger_tokens, 15_360)
        self.assertEqual(runtime.retrieval_max_tokens, 2048)
        self.assertEqual(runtime.peak_rss_limit_mib, 700)
        self.assertEqual(runtime.max_concurrent_generations, 1)

    def test_model_config_loads_and_validates(self):
        path = Path("configs/model/lafla-400m-thinking.yaml")
        config = ModelConfig.from_mapping(load_mapping(path))
        config.validate()
        self.assertEqual(config.family, "decoder-only")
        self.assertEqual(config.parameter_target, 400_000_000)
        self.assertEqual(config.display_name, "LaflaGPT 400M")
        self.assertEqual(config.creator_name, "Yavuz Selim")

    def test_lafla_1b_h200_config_loads_and_validates(self):
        model = ModelConfig.from_mapping(load_mapping("configs/model/lafla-1b-thinking.yaml"))
        training = TrainingConfig.from_mapping(load_mapping("configs/training/lightning/lightning-h200-1b-100000.yaml"))
        model.validate()
        training.validate()
        self.assertEqual(model.name, "lafla-1b-thinking")
        self.assertEqual(model.display_name, "LaflaGPT 1B")
        self.assertEqual(model.creator_name, "Yavuz Selim")
        self.assertEqual(training.max_steps, 100_000)
        self.assertEqual(training.precision, "bf16")
        self.assertEqual(training.optimizer, "adamw")
        self.assertEqual(training.save_every, 100)

    def test_lafla_380m_h200_config_loads_and_validates_for_50000_steps(self):
        model = ModelConfig.from_mapping(load_mapping("configs/model/lafla-380m-thinking.yaml"))
        training = TrainingConfig.from_mapping(load_mapping("configs/training/lightning/lightning-h200-380m-50000.yaml"))
        runtime = RuntimeConfig.from_mapping(load_mapping("configs/runtime/desktop-phone-fp16-380m.yaml"))

        model.validate()
        training.validate()
        runtime.validate()

        self.assertEqual(model.name, "lafla-380m-thinking")
        self.assertEqual(model.display_name, "LaflaGPT 380M")
        self.assertEqual(model.creator_name, "Yavuz Selim")
        self.assertIn("380M", model.identity_statement)
        self.assertEqual(training.max_steps, 50_000)
        self.assertEqual(training.precision, "bf16")
        self.assertEqual(training.optimizer, "adamw")
        self.assertEqual(training.micro_batch_size, 96)
        self.assertEqual(training.save_every, 250)
        self.assertEqual(runtime.quantization, "none")
        self.assertLessEqual(runtime.memory_budget_gb, 8.0)

    def test_colab_tpu_v5e_380m_config_is_xla_bf16_and_safer_than_t4(self):
        tpu = TrainingConfig.from_mapping(load_mapping("configs/training/colab/colab-tpu-v5e-380m-50000.yaml"))
        t4 = TrainingConfig.from_mapping(load_mapping("configs/training/colab/colab-t4-380m-fallback.yaml"))

        tpu.validate()
        t4.validate()

        self.assertEqual(tpu.accelerator, "xla")
        self.assertEqual(tpu.precision, "bf16")
        self.assertEqual(tpu.optimizer, "adamw")
        self.assertEqual(tpu.max_steps, 50_000)
        self.assertGreater(tpu.micro_batch_size, t4.micro_batch_size)
        self.assertEqual(t4.accelerator, "cuda")
        self.assertEqual(t4.precision, "fp16")

    def test_tokenizer_config_preserves_special_token_list(self):
        path = Path("configs/tokenizer/turkish-thinking-bpe.yaml")
        config = TokenizerConfig.from_mapping(load_mapping(path))
        config.validate()
        self.assertIn("<|think|>", config.required_special_tokens)

    def test_post_training_config_loads_and_keeps_thinking_private(self):
        path = Path("configs/post_training/lafla-thinking-sft.yaml")
        config = PostTrainingConfig.from_mapping(load_mapping(path))
        config.validate()
        self.assertEqual(config.stage, "thinking_sft")
        self.assertFalse(config.public_thinking_visible)

    def test_developer_runtime_allows_raw_thinking_only_in_research_mode(self):
        path = Path("configs/runtime/developer-research.yaml")
        config = RuntimeConfig.from_mapping(load_mapping(path))
        config.validate()
        self.assertTrue(config.developer_mode)
        self.assertTrue(config.raw_thinking_visible)

    def test_preflight_rejects_bad_learning_rate(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "bad.yaml"
            path.write_text(
                "\n".join(
                    [
                        "training:",
                        "  max_steps: 100",
                        "  sequence_length: 512",
                        "  micro_batch_size: 1",
                        "  gradient_accumulation_steps: 1",
                        "  precision: fp16",
                        "  optimizer: adamw",
                        "  learning_rate: 2.0",
                        "  warmup_steps: 1",
                        "  save_every: 10",
                        "  keep_last_checkpoints: 2",
                        "  require_drive_or_explicit_local_fallback: true",
                    ]
                ),
                encoding="utf-8",
            )
            report = run_preflight([path])
        self.assertFalse(report.ok)
        self.assertIn("learning_rate", report.errors[0])

    def test_preflight_rejects_sequence_longer_than_model_context(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            model_path = root / "model.yaml"
            training_path = root / "training.yaml"
            model_path.write_text(
                "\n".join(
                    [
                        "model:",
                        "  name: tiny",
                        "  family: decoder-only",
                        "  parameter_target: 100000000",
                        "  vocab_size: 128",
                        "  context_length: 512",
                        "  hidden_size: 128",
                        "  intermediate_size: 256",
                        "  num_layers: 2",
                        "  num_attention_heads: 8",
                        "  num_key_value_heads: 2",
                        "  activation: swiglu",
                        "  norm: rmsnorm",
                        "  rope: true",
                        "  qk_norm: true",
                    ]
                ),
                encoding="utf-8",
            )
            training_path.write_text(
                "\n".join(
                    [
                        "training:",
                        "  max_steps: 100",
                        "  sequence_length: 1024",
                        "  micro_batch_size: 1",
                        "  gradient_accumulation_steps: 1",
                        "  precision: fp16",
                        "  optimizer: adamw",
                        "  learning_rate: 0.001",
                        "  warmup_steps: 1",
                        "  save_every: 10",
                        "  keep_last_checkpoints: 2",
                        "  require_drive_or_explicit_local_fallback: true",
                    ]
                ),
                encoding="utf-8",
            )
            report = run_preflight([model_path, training_path])
        self.assertFalse(report.ok)
        self.assertIn("sequence_length", "\n".join(report.errors))

    def test_preflight_rejects_runtime_context_longer_than_model_context(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            model_path = root / "model.yaml"
            runtime_path = root / "runtime.yaml"
            model_path.write_text(
                "\n".join(
                    [
                        "model:",
                        "  name: tiny",
                        "  family: decoder-only",
                        "  parameter_target: 100000000",
                        "  vocab_size: 128",
                        "  context_length: 512",
                        "  hidden_size: 128",
                        "  intermediate_size: 256",
                        "  num_layers: 2",
                        "  num_attention_heads: 8",
                        "  num_key_value_heads: 2",
                        "  activation: swiglu",
                        "  norm: rmsnorm",
                        "  rope: true",
                        "  qk_norm: true",
                    ]
                ),
                encoding="utf-8",
            )
            runtime_path.write_text(
                "\n".join(
                    [
                        "runtime:",
                        "  target: desktop-cpu",
                        "  quantization: 8bit",
                        "  context_length: 1024",
                        "  max_new_tokens: 128",
                        "  temperature: 0.7",
                        "  top_p: 0.9",
                        "  repetition_penalty: 1.1",
                        "  memory_budget_gb: 1.0",
                    ]
                ),
                encoding="utf-8",
            )

            report = run_preflight([model_path, runtime_path])

        self.assertFalse(report.ok)
        self.assertIn("runtime context_length", "\n".join(report.errors))

    def test_model_config_rejects_too_small_parameter_target(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "too-small-model.yaml"
            path.write_text(
                "\n".join(
                    [
                        "model:",
                        "  name: too-small",
                        "  family: decoder-only",
                        "  parameter_target: 50000000",
                        "  vocab_size: 128",
                        "  context_length: 512",
                        "  hidden_size: 128",
                        "  intermediate_size: 256",
                        "  num_layers: 2",
                        "  num_attention_heads: 8",
                        "  num_key_value_heads: 2",
                        "  activation: swiglu",
                        "  norm: rmsnorm",
                        "  rope: true",
                        "  qk_norm: true",
                    ]
                ),
                encoding="utf-8",
            )

            report = run_preflight([path])

        self.assertFalse(report.ok)
        self.assertIn("parametre hedefi", "\n".join(report.errors))


if __name__ == "__main__":
    unittest.main()
