import unittest

from lafla_ai_core.evaluation.uncertainty import UncertaintyCase, evaluate_uncertainty


class UncertaintyGateTest(unittest.TestCase):
    def test_unanswerable_and_stale_current_fact_require_abstention_without_sources(self):
        unanswerable = UncertaintyCase("u1", "unanswerable", reference_answer=None)
        stale = UncertaintyCase("s1", "stale_current_fact", reference_answer=None)

        self.assertTrue(evaluate_uncertainty(unanswerable, "Bu bilgi verilen bağlamda yok; bilmiyorum.").passed)
        self.assertTrue(evaluate_uncertainty(stale, "Güncel bir kaynak olmadan bunu doğrulayamıyorum.").passed)
        self.assertFalse(evaluate_uncertainty(stale, "Kesin cevap 42'dir.").passed)

    def test_answerable_case_requires_reference_answer(self):
        case = UncertaintyCase("a1", "answerable", reference_answer="Ankara")
        self.assertTrue(evaluate_uncertainty(case, "Türkiye'nin başkenti Ankara'dır.").passed)
        self.assertFalse(evaluate_uncertainty(case, "Bilmiyorum.").passed)

    def test_false_premise_requires_explicit_correction(self):
        case = UncertaintyCase(
            "f1",
            "false_premise",
            reference_answer="yanlış",
            correction_terms=("doğru değil", "yanlış"),
        )
        self.assertTrue(evaluate_uncertainty(case, "Bu öncül doğru değil; tarihsel kayıtlar farklıdır.").passed)


if __name__ == "__main__":
    unittest.main()
