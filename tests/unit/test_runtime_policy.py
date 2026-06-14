import unittest

from lafla_ai_core.config.loader import load_mapping
from lafla_ai_core.config.schema import RuntimeConfig
from lafla_ai_core.runtime.policy import build_generation_settings, render_runtime_output


class RuntimePolicyTest(unittest.TestCase):
    def test_default_runtime_strips_private_thinking(self):
        config = RuntimeConfig.from_mapping(load_mapping("configs/runtime/desktop-cpu-4bit.yaml"))
        output = render_runtime_output("<|think|>gizli plan<|/think|> Merhaba, nasıl yardımcı olabilirim?", config)
        self.assertEqual(output.public_text, "Merhaba, nasıl yardımcı olabilirim?")
        self.assertIsNone(output.raw_thinking)

    def test_developer_research_runtime_exposes_raw_thinking(self):
        config = RuntimeConfig.from_mapping(load_mapping("configs/runtime/developer-research.yaml"))
        output = render_runtime_output("<|think|>gizli plan<|/think|> Merhaba.", config)
        self.assertEqual(output.raw_thinking, "gizli plan")
        self.assertEqual(output.public_text, "Merhaba.")

    def test_developer_unguarded_runtime_keeps_raw_diagnostic_surface(self):
        config = RuntimeConfig.from_mapping(load_mapping("configs/runtime/developer-unguarded.yaml"))
        raw = "<|system|>gizli sistem<|assistant|><|think|>ham plan<|/think|> cevap <|user|>devam"

        output = render_runtime_output(raw, config, prompt_text="cevap", system_text="gizli sistem")

        self.assertIn("<|system|>gizli sistem", output.public_text)
        self.assertIn("<|think|>ham plan<|/think|>", output.public_text)
        self.assertIn("<|user|>devam", output.public_text)
        self.assertEqual(output.raw_thinking, "ham plan")
        self.assertEqual(output.warnings, ("safety_filters_disabled",))

    def test_prompt_leak_is_flagged(self):
        config = RuntimeConfig.from_mapping(load_mapping("configs/runtime/desktop-cpu-4bit.yaml"))
        output = render_runtime_output("<|system|> Sen Lafla AI'sin. Merhaba.", config)
        self.assertIn("possible_prompt_leak", output.warnings)

    def test_thinking_effort_changes_generation_budget(self):
        low = RuntimeConfig.from_mapping(load_mapping("configs/runtime/desktop-cpu-4bit.yaml"))
        high = RuntimeConfig.from_mapping(load_mapping("configs/runtime/developer-research.yaml"))
        self.assertLess(build_generation_settings(low).thinking_budget_tokens, build_generation_settings(high).thinking_budget_tokens)

    def test_public_runtime_cleans_bytelevel_and_mojibake_surface(self):
        config = RuntimeConfig.from_mapping(load_mapping("configs/runtime/desktop-cpu-4bit.yaml"))
        output = render_runtime_output("\u0120T\u00c3\u00bcrk\u00c3\u00a7e \u0120k\u00c4\u00b1sa \u0120cevap", config)
        self.assertEqual(output.public_text, "Türkçe kısa cevap")
        self.assertNotIn("possible_mojibake", output.warnings)


if __name__ == "__main__":
    unittest.main()
