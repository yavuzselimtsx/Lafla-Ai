import unittest

from lafla_ai_core.config.loader import load_mapping
from lafla_ai_core.config.schema import ModelConfig, RuntimeConfig
from lafla_ai_core.runtime.memory_budget import estimate_runtime_memory


class MemoryBudgetTest(unittest.TestCase):
    def test_100m_int8_weights_and_hybrid_20k_kv_are_derived_from_configs(self):
        model = ModelConfig.from_mapping(load_mapping("configs/model/lafla-100m-thinking.yaml"))
        runtime = RuntimeConfig.from_mapping(load_mapping("configs/runtime/desktop-i3-int8-100m.yaml"))

        estimate = estimate_runtime_memory(model, runtime)

        self.assertEqual(estimate.parameter_count, 98_324_736)
        self.assertEqual(estimate.weight_bytes, 98_324_736)
        self.assertEqual(estimate.kv_cache_bytes, 50_331_648)
        self.assertLessEqual(estimate.total_bytes, 700 * 1024 * 1024)
        self.assertTrue(estimate.within_configured_limit)


if __name__ == "__main__":
    unittest.main()
