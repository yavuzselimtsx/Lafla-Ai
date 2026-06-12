import unittest

from lafla_ai_core.config.loader import load_mapping
from lafla_ai_core.config.schema import ModelConfig


class ModelAttentionPatternTest(unittest.TestCase):
    def test_100m_pattern_resolves_to_nine_local_and_three_global_layers(self):
        config = ModelConfig.from_mapping(load_mapping("configs/model/lafla-100m-thinking.yaml"))

        resolved = config.resolved_attention_pattern()

        self.assertEqual(len(resolved), 12)
        self.assertEqual(resolved.count("local"), 9)
        self.assertEqual(resolved.count("global"), 3)
        self.assertEqual(resolved[3::4], ("global", "global", "global"))


if __name__ == "__main__":
    unittest.main()
