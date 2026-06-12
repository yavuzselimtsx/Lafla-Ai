import contextlib
import io
import json
import unittest

from lafla_ai_core.cli.render_runtime_output import main


class RuntimeRenderCliTest(unittest.TestCase):
    def test_cli_exposes_raw_thinking_with_developer_config(self):
        stdout = io.StringIO()
        with contextlib.redirect_stdout(stdout):
            exit_code = main(
                [
                    "--runtime-config",
                    "configs/runtime/developer-research.yaml",
                    "--raw-text",
                    "<|think|>plan<|/think|> Cevap.",
                ]
            )
        payload = json.loads(stdout.getvalue())
        self.assertEqual(exit_code, 0)
        self.assertEqual(payload["raw_thinking"], "plan")
        self.assertEqual(payload["public_text"], "Cevap.")

    def test_cli_accepts_prompt_context_for_echo_guard(self):
        stdout = io.StringIO()
        with contextlib.redirect_stdout(stdout):
            exit_code = main(
                [
                    "--runtime-config",
                    "configs/runtime/desktop-cpu-4bit.yaml",
                    "--raw-text",
                    "\u01202 + 2 \u0120ka\u00c3\u00a7 \u0120eder ? \u01204 <|eos|>",
                    "--prompt-text",
                    "2+2 ka\u00e7 eder?",
                ]
            )
        payload = json.loads(stdout.getvalue())
        self.assertEqual(exit_code, 0)
        self.assertEqual(payload["public_text"], "4")
        self.assertIn("prompt_echo_removed", payload["warnings"])


if __name__ == "__main__":
    unittest.main()
