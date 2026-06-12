import unittest

from lafla_ai_core.config.loader import load_mapping
from lafla_ai_core.config.schema import RuntimeConfig
from lafla_ai_core.runtime.output_guard import sanitize_completion
from lafla_ai_core.runtime.policy import build_generation_settings, render_runtime_output


class RuntimeOutputGuardTest(unittest.TestCase):
    def test_screenshot_like_checkpoint_output_removes_prompt_echo_and_repairs_decode(self):
        raw = (
            "\u0120Sen \u0120Laf la GPT \u0120400 M \u0120Thin king \u0120model isin. "
            "\u0120T\u00c3\u00bcrk\u00c3\u00a7e \u0120cevap \u0120ver. "
            "\u01202 + 2 \u0120ka\u00c3\u00a7 \u0120eder ? \u0120K\u00c4\u00b1sa \u0120cevap \u0120ver. "
            "\u01204 <|eos|><|user|>\nDevam etme."
        )

        result = sanitize_completion(
            raw,
            system_text="Sen LaflaGPT 400M Thinking modelisin. T\u00fcrk\u00e7e cevap ver.",
            prompt_text="2+2 ka\u00e7 eder? K\u0131sa cevap ver.",
        )

        self.assertEqual(result.text, "4")
        self.assertIn("bytelevel_surface_repaired", result.warnings)
        self.assertIn("mojibake_repaired", result.warnings)
        self.assertIn("prompt_echo_removed", result.warnings)
        self.assertIn("control_token_stop", result.warnings)

    def test_runtime_policy_uses_prompt_context_for_public_output(self):
        config = RuntimeConfig.from_mapping(load_mapping("configs/runtime/desktop-cpu-4bit.yaml"))
        output = render_runtime_output(
            "\u0120Sen \u0120LaflaGPT \u0120400M \u0120Thinking \u0120modelisin. "
            "\u0120T\u00c3\u00bcrk\u00c3\u00a7e \u0120cevap \u0120ver. "
            "\u01202 + 2 \u0120ka\u00c3\u00a7 \u0120eder ? \u0120K\u00c4\u00b1sa \u0120cevap \u0120ver. "
            "\u01204 <|assistant|> yank\u0131",
            config,
            system_text="Sen LaflaGPT 400M Thinking modelisin. T\u00fcrk\u00e7e cevap ver.",
            prompt_text="2+2 ka\u00e7 eder? K\u0131sa cevap ver.",
        )

        self.assertEqual(output.public_text, "4")
        self.assertIn("prompt_echo_removed", output.warnings)
        self.assertIn("control_token_stop", output.warnings)

    def test_generation_settings_expose_role_stop_sequences(self):
        config = RuntimeConfig.from_mapping(load_mapping("configs/runtime/desktop-cpu-4bit.yaml"))
        settings = build_generation_settings(config)

        self.assertIn("<|eos|>", settings.stop_sequences)
        self.assertIn("<|user|>", settings.stop_sequences)
        self.assertIn("<|system|>", settings.stop_sequences)

    def test_default_checkpoint_identity_names_model_and_creator(self):
        from lafla_ai_core.config.schema import ModelConfig
        from lafla_ai_core.runtime.checkpoint_inference import DEFAULT_SYSTEM_TEXT, build_model_system_text

        model = ModelConfig.from_mapping(load_mapping("configs/model/lafla-100m-thinking.yaml"))
        self.assertNotIn("380M", DEFAULT_SYSTEM_TEXT)
        self.assertIn(model.identity_statement, build_model_system_text(model))


if __name__ == "__main__":
    unittest.main()
