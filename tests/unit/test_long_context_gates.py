import unittest

from lafla_ai_core.evaluation.long_context import LongContextCase, score_long_context_case


class LongContextGateTest(unittest.TestCase):
    def test_passkey_scoring_is_deterministic_and_position_aware(self):
        case = LongContextCase(
            case_id="p15k",
            kind="passkey",
            context_tokens=15_360,
            needle_position=0.73,
            expected_answer="MAVI-7319",
        )

        passed = score_long_context_case(case, "İstenen anahtar MAVI-7319.")
        failed = score_long_context_case(case, "İstenen anahtar MAVI-7139.")

        self.assertTrue(passed.passed)
        self.assertFalse(failed.passed)
        self.assertEqual(passed.context_tokens, 15_360)
        self.assertEqual(passed.needle_position, 0.73)


if __name__ == "__main__":
    unittest.main()
