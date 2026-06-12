import unittest

from lafla_ai_core.config.loader import load_mapping
from lafla_ai_core.config.schema import ModelConfig
from lafla_ai_core.model.size import estimate_decoder_parameters


class ModelSizeTest(unittest.TestCase):
    def test_lafla_100m_config_estimates_between_95m_and_105m(self):
        config = ModelConfig.from_mapping(load_mapping("configs/model/lafla-100m-thinking.yaml"))
        estimate = estimate_decoder_parameters(config)
        self.assertEqual(config.name, "lafla-100m-thinking")
        self.assertGreaterEqual(estimate, 95_000_000)
        self.assertLessEqual(estimate, 105_000_000)

    def test_lafla_400m_config_estimates_near_target_and_names_capacity(self):
        config = ModelConfig.from_mapping(load_mapping("configs/model/lafla-400m-thinking.yaml"))
        estimate = estimate_decoder_parameters(config)
        self.assertEqual(config.name, "lafla-400m-thinking")
        self.assertEqual(config.parameter_target, 400_000_000)
        self.assertGreaterEqual(estimate, 380_000_000)
        self.assertLessEqual(estimate, 420_000_000)

    def test_lafla_380m_config_estimates_near_target_and_names_capacity(self):
        config = ModelConfig.from_mapping(load_mapping("configs/model/lafla-380m-thinking.yaml"))
        estimate = estimate_decoder_parameters(config)
        self.assertEqual(config.name, "lafla-380m-thinking")
        self.assertEqual(config.display_name, "LaflaGPT 380M")
        self.assertEqual(config.creator_name, "Yavuz Selim")
        self.assertEqual(config.parameter_target, 380_000_000)
        self.assertGreaterEqual(estimate, 360_000_000)
        self.assertLessEqual(estimate, 400_000_000)

    def test_lafla_1b_config_estimates_near_target_and_names_capacity(self):
        config = ModelConfig.from_mapping(load_mapping("configs/model/lafla-1b-thinking.yaml"))
        estimate = estimate_decoder_parameters(config)
        self.assertEqual(config.name, "lafla-1b-thinking")
        self.assertEqual(config.parameter_target, 1_000_000_000)
        self.assertGreaterEqual(estimate, 930_000_000)
        self.assertLessEqual(estimate, 1_070_000_000)


if __name__ == "__main__":
    unittest.main()
