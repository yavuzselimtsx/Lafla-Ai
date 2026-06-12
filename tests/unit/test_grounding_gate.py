import unittest

from lafla_ai_core.evaluation.grounding import GroundedAnswerCase, evaluate_grounded_answer


class GroundingGateTest(unittest.TestCase):
    def test_rejects_answer_without_required_evidence_marker(self):
        case = GroundedAnswerCase(
            prompt="FineWeb2 kaynağı lisanslı mı?",
            answer="Evet, bu kaynak kesinlikle uygundur.",
            evidence_ids=("manifest:fineweb2-tr",),
            require_evidence=True,
        )
        result = evaluate_grounded_answer(case)
        self.assertFalse(result.passed)
        self.assertIn("missing_evidence", result.detail)

    def test_accepts_cited_or_uncertain_answer(self):
        cited = GroundedAnswerCase(
            prompt="FineWeb2 kaynağı lisanslı mı?",
            answer="Manifest bunu izinli gösteriyor. [kaynak:manifest:fineweb2-tr]",
            evidence_ids=("manifest:fineweb2-tr",),
            require_evidence=True,
        )
        uncertain = GroundedAnswerCase(
            prompt="Bilinmeyen özel veri var mı?",
            answer="Bunu destekleyen kanıt yok, emin değilim.",
            evidence_ids=(),
            require_evidence=True,
        )
        self.assertTrue(evaluate_grounded_answer(cited).passed)
        self.assertTrue(evaluate_grounded_answer(uncertain).passed)


if __name__ == "__main__":
    unittest.main()
