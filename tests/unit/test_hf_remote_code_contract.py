import unittest

from lafla_ai_core.config.loader import load_mapping
from lafla_ai_core.config.schema import ModelConfig
from lafla_ai_core.export.hf_remote_code import build_hf_config_payload, remote_code_files


class HuggingFaceRemoteCodeContractTest(unittest.TestCase):
    def test_100m_export_is_cache_aware_and_contains_no_400m_defaults(self):
        model = ModelConfig.from_mapping(load_mapping("configs/model/lafla-100m-thinking.yaml"))
        vocab = {"<|bos|>": 0, "<|eos|>": 1, "<|pad|>": 2}

        payload = build_hf_config_payload(model, vocab)
        files = remote_code_files()
        combined_code = "\n".join(files.values())

        self.assertTrue(payload["use_cache"])
        self.assertEqual(payload["attention_pattern"], ["local", "local", "local", "global"])
        self.assertEqual(payload["sliding_window"], 4096)
        self.assertEqual(payload["rope_scaling"]["factor"], 5.0)
        self.assertIn("past_key_values", files["modeling_lafla.py"])
        self.assertIn("cache_position", files["modeling_lafla.py"])
        self.assertIn("num_logits_to_keep", files["modeling_lafla.py"])
        self.assertIn("hidden[:, -num_logits_to_keep:, :]", files["modeling_lafla.py"])
        self.assertIn("_chunked_global_attention", files["modeling_lafla.py"])
        self.assertIn('query_len > 4096 and q.device.type == "cpu"', files["modeling_lafla.py"])
        self.assertIn("input_ids[:, -1:]", files["modeling_lafla.py"])
        self.assertNotIn("lafla-400m", combined_code.lower())
        self.assertNotIn("400000000", combined_code)


if __name__ == "__main__":
    unittest.main()
