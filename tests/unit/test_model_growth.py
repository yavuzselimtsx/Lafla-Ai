import unittest

from lafla_ai_core.config.loader import load_mapping
from lafla_ai_core.config.schema import ModelConfig
from lafla_ai_core.model.growth import build_depth_growth_plan, validate_growth_compatibility
from lafla_ai_core.model.size import estimate_decoder_parameters


class ModelGrowthTest(unittest.TestCase):
    def test_200m_config_is_compatible_and_maps_all_source_layers_in_order(self):
        source = ModelConfig.from_mapping(load_mapping("configs/model/lafla-100m-thinking.yaml"))
        target = ModelConfig.from_mapping(load_mapping("configs/model/lafla-200m-thinking.yaml"))

        validate_growth_compatibility(source, target)
        plan = build_depth_growth_plan(source, target)

        self.assertEqual(plan.source_to_target, tuple((index, index) for index in range(12)))
        self.assertEqual(plan.inserted_target_layers, tuple(range(12, 29)))
        self.assertGreaterEqual(estimate_decoder_parameters(target), 195_000_000)
        self.assertLessEqual(estimate_decoder_parameters(target), 205_000_000)
        self.assertTrue(plan.continued_pretraining_required)

    def test_growth_rejects_hidden_size_change(self):
        source = ModelConfig.from_mapping(load_mapping("configs/model/lafla-100m-thinking.yaml"))
        target_mapping = load_mapping("configs/model/lafla-200m-thinking.yaml")
        target_mapping["model"]["hidden_size"] = 1024
        target = ModelConfig.from_mapping(target_mapping)

        with self.assertRaisesRegex(ValueError, "hidden_size"):
            validate_growth_compatibility(source, target)


if __name__ == "__main__":
    unittest.main()
