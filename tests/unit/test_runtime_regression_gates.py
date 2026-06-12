import unittest

from lafla_ai_core.config.loader import load_mapping
from lafla_ai_core.config.schema import RuntimeConfig
from lafla_ai_core.evaluation.runtime_regressions import RuntimeRegressionCase, evaluate_runtime_regression_case


class RuntimeRegressionGatesTest(unittest.TestCase):
    def test_prompt_echo_mojibake_and_role_boundary_case_passes_after_guard(self):
        config = RuntimeConfig.from_mapping(load_mapping("configs/runtime/desktop-cpu-4bit.yaml"))
        case = RuntimeRegressionCase(
            name="prompt_echo_guard",
            raw_text=(
                "\u0120Sen \u0120LaflaGPT \u0120400M \u0120Thinking \u0120modelisin. "
                "\u0120T\u00c3\u00bcrk\u00c3\u00a7e \u0120cevap \u0120ver. "
                "\u01202 + 2 \u0120ka\u00c3\u00a7 \u0120eder ? \u01204 <|eos|><|user|>"
            ),
            expected_public_text="4",
            system_text="Sen LaflaGPT 400M Thinking modelisin. T\u00fcrk\u00e7e cevap ver.",
            prompt_text="2+2 ka\u00e7 eder?",
            required_warnings=("prompt_echo_removed", "mojibake_repaired", "control_token_stop"),
        )

        result = evaluate_runtime_regression_case(case, config)

        self.assertTrue(result.passed, result.detail)
        self.assertEqual(result.name, "prompt_echo_guard")

    def test_runtime_regression_case_fails_when_expected_answer_is_not_recovered(self):
        config = RuntimeConfig.from_mapping(load_mapping("configs/runtime/desktop-cpu-4bit.yaml"))
        case = RuntimeRegressionCase(
            name="role_boundary_stop",
            raw_text="Yanlis cevap <|user|> devam",
            expected_public_text="4",
            required_warnings=("control_token_stop",),
        )

        result = evaluate_runtime_regression_case(case, config)

        self.assertFalse(result.passed)
        self.assertIn("expected_public_text_mismatch", result.detail)


if __name__ == "__main__":
    unittest.main()
