import unittest

from lafla_ai_core.config.loader import load_mapping
from lafla_ai_core.config.schema import ModelConfig
from lafla_ai_core.runtime.checkpoint_inference import (
    DEFAULT_SYSTEM_TEXT,
    TokenizersGenerationAdapter,
    build_checkpoint_messages,
    build_model_system_text,
)


class FakeEncoding:
    def __init__(self, ids):
        self.ids = ids


class FakeTokenizer:
    def encode(self, text):
        return FakeEncoding([ord(char) for char in text])

    def decode(self, token_ids, skip_special_tokens=False):
        return "".join(chr(int(token_id)) for token_id in token_ids)


class CheckpointInferenceTest(unittest.TestCase):
    def test_adapter_accepts_tokenizers_encoding_shape(self):
        adapter = TokenizersGenerationAdapter(FakeTokenizer())

        ids = adapter.encode("4<|eos|>")
        text = adapter.decode(ids)

        self.assertEqual(text, "4<|eos|>")

    def test_build_checkpoint_messages_uses_system_and_user_roles(self):
        messages = build_checkpoint_messages("2+2 kac eder?")

        self.assertEqual(messages[0].role, "system")
        self.assertEqual(messages[0].content, DEFAULT_SYSTEM_TEXT)
        self.assertIn("Lafla", DEFAULT_SYSTEM_TEXT)
        self.assertNotIn("380M", DEFAULT_SYSTEM_TEXT)
        self.assertEqual(messages[1].role, "user")
        self.assertIn("2+2", messages[1].content)

    def test_model_system_text_comes_from_model_config_identity(self):
        config = ModelConfig.from_mapping(load_mapping("configs/model/lafla-100m-thinking.yaml"))
        text = build_model_system_text(config)
        self.assertIn(config.identity_statement, text)
        self.assertIn("sınırlarını", text)

    def test_build_checkpoint_messages_rejects_empty_prompt(self):
        with self.assertRaises(ValueError):
            build_checkpoint_messages(" ")


if __name__ == "__main__":
    unittest.main()
