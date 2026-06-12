import json
import tempfile
import unittest
from pathlib import Path

from lafla_ai_core.config.loader import load_mapping
from lafla_ai_core.config.schema import ModelConfig
from lafla_ai_core.export.hf_package import write_hf_tokenizer_package


class HuggingFacePackageTest(unittest.TestCase):
    def test_100m_package_uses_model_context_and_dynamic_cache_metadata(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            tokenizer_json = root / "tokenizer.json"
            tokenizer_json.write_text(
                json.dumps(
                    {
                        "model": {
                            "vocab": {
                                "<|bos|>": 0,
                                "<|eos|>": 1,
                                "<|pad|>": 2,
                                "<|unk|>": 3,
                                "<|system|>": 4,
                                "<|user|>": 5,
                                "<|assistant|>": 6,
                                "<|think|>": 7,
                                "<|/think|>": 8,
                            }
                        }
                    }
                ),
                encoding="utf-8",
            )
            model_config = ModelConfig.from_mapping(load_mapping("configs/model/lafla-100m-thinking.yaml"))
            output = root / "hf"

            write_hf_tokenizer_package(
                tokenizer_json,
                output,
                model_name=model_config.name,
                model_config=model_config,
            )

            tokenizer_config = json.loads((output / "tokenizer_config.json").read_text(encoding="utf-8"))
            generation = json.loads((output / "generation_config.json").read_text(encoding="utf-8"))
            self.assertEqual(tokenizer_config["model_max_length"], 20_480)
            self.assertEqual(generation["cache_implementation"], "dynamic")
            self.assertTrue(generation["use_cache"])

    def test_writes_tokenizer_metadata_needed_outside_lafla_core(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            tokenizer_json = root / "tokenizer.json"
            tokenizer_json.write_text(
                json.dumps(
                    {
                        "model": {
                            "vocab": {
                                "<|bos|>": 0,
                                "<|eos|>": 1,
                                "<|pad|>": 2,
                                "<|unk|>": 3,
                                "<|system|>": 4,
                                "<|user|>": 5,
                                "<|assistant|>": 6,
                                "<|think|>": 7,
                                "<|/think|>": 8,
                            }
                        }
                    }
                ),
                encoding="utf-8",
            )
            output = root / "hf"
            write_hf_tokenizer_package(tokenizer_json, output, model_name="lafla-400m-thinking")

            tokenizer_config = json.loads((output / "tokenizer_config.json").read_text(encoding="utf-8"))
            special_map = json.loads((output / "special_tokens_map.json").read_text(encoding="utf-8"))
            generation = json.loads((output / "generation_config.json").read_text(encoding="utf-8"))

            self.assertTrue((output / "tokenizer.json").exists())
            self.assertIn("chat_template", tokenizer_config)
            self.assertEqual(special_map["eos_token"], "<|eos|>")
            self.assertEqual(generation["eos_token_id"], 1)
            self.assertIn("trust_remote_code", (output / "README.md").read_text(encoding="utf-8"))

    def test_writes_remote_code_package_when_model_config_is_given(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            tokenizer_json = root / "tokenizer.json"
            tokenizer_json.write_text(
                json.dumps(
                    {
                        "model": {
                            "vocab": {
                                "<|bos|>": 0,
                                "<|eos|>": 1,
                                "<|pad|>": 2,
                                "<|unk|>": 3,
                                "<|system|>": 4,
                                "<|user|>": 5,
                                "<|assistant|>": 6,
                                "<|think|>": 7,
                                "<|/think|>": 8,
                            }
                        }
                    }
                ),
                encoding="utf-8",
            )
            model_config = ModelConfig(
                name="lafla-400m-thinking",
                family="decoder-only",
                parameter_target=400_000_000,
                vocab_size=65536,
                context_length=2048,
                hidden_size=1536,
                intermediate_size=3840,
                num_layers=12,
                num_attention_heads=12,
                num_key_value_heads=6,
                activation="swiglu",
                norm="rmsnorm",
                rope=True,
                qk_norm=True,
                tie_word_embeddings=True,
                display_name="LaflaGPT 400M",
                creator_name="Yavuz Selim",
                identity_statement="LaflaGPT 400M, Yavuz Selim tarafindan gelistirilen Turkce oncelikli bir Lafla modelidir.",
            )

            output = root / "hf"
            write_hf_tokenizer_package(tokenizer_json, output, model_name="lafla-400m-thinking", model_config=model_config)

            config = json.loads((output / "config.json").read_text(encoding="utf-8"))
            self.assertEqual(config["model_type"], "lafla")
            self.assertEqual(config["model_name"], "lafla-400m-thinking")
            self.assertEqual(config["parameter_target"], 400_000_000)
            self.assertEqual(config["display_name"], "LaflaGPT 400M")
            self.assertEqual(config["creator_name"], "Yavuz Selim")
            self.assertEqual(config["hidden_size"], 1536)
            self.assertEqual(config["num_hidden_layers"], 12)
            self.assertEqual(config["auto_map"]["AutoModelForCausalLM"], "modeling_lafla.LaflaForCausalLM")
            self.assertTrue((output / "configuration_lafla.py").exists())
            self.assertTrue((output / "modeling_lafla.py").exists())
            self.assertIn("LaflaForCausalLM", (output / "modeling_lafla.py").read_text(encoding="utf-8"))
            self.assertIn("400,000,000", (output / "README.md").read_text(encoding="utf-8"))
            self.assertIn("Yavuz Selim", (output / "README.md").read_text(encoding="utf-8"))


if __name__ == "__main__":
    unittest.main()
