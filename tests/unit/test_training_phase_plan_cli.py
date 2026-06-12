import contextlib
import io
import json
import unittest

from lafla_ai_core.cli.training_phase_plan import main


class TrainingPhasePlanCliTest(unittest.TestCase):
    def test_cli_prints_default_plan_with_runtime_gates(self):
        stdout = io.StringIO()
        with contextlib.redirect_stdout(stdout):
            exit_code = main([])

        payload = json.loads(stdout.getvalue())

        self.assertEqual(exit_code, 0)
        self.assertEqual(payload["name"], "lafla-100m-thinking-clean-room-v1")
        release = payload["phases"][-1]
        self.assertEqual(release["name"], "release_eval")
        self.assertIn("prompt_echo_guard", release["required_gates"])
        self.assertIn("role_boundary_stop", release["required_gates"])
        self.assertIn("process_tree_peak_rss", release["required_gates"])


if __name__ == "__main__":
    unittest.main()
