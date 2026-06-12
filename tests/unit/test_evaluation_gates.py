import unittest

from lafla_ai_core.config.loader import load_mapping
from lafla_ai_core.evaluation.gates import GateResult, aggregate_gates, required_gate_results


class EvaluationGatesTest(unittest.TestCase):
    def test_aggregate_requires_all_when_minimum_is_one(self):
        report = aggregate_gates(
            [
                GateResult("tokenizer", True, "ok"),
                GateResult("safety", False, "failed"),
            ]
        )
        self.assertFalse(report.ok)
        self.assertEqual(report.pass_rate, 0.5)

    def test_missing_required_gate_fails(self):
        report = required_gate_results(["tokenizer", "safety"], [GateResult("tokenizer", True, "ok")])
        self.assertFalse(report.ok)
        self.assertEqual(report.results[1].detail, "gate sonucu yok")

    def test_release_gate_config_tracks_runtime_decode_regressions(self):
        gates = load_mapping("configs/evaluation/release-gates.yaml")["evaluation"]["required_gates"]

        self.assertIn("completion_only_generation", gates)
        self.assertIn("prompt_echo_guard", gates)
        self.assertIn("mojibake_decode", gates)
        self.assertIn("role_boundary_stop", gates)


if __name__ == "__main__":
    unittest.main()
