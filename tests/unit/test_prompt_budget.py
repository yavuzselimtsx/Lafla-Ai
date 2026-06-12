import unittest

from lafla_ai_core.runtime.conversation_memory import MessageRecord, StructuredSummary
from lafla_ai_core.runtime.prompt_budget import PromptBudget, assemble_prompt


def count_words(text: str) -> int:
    return len(text.split())


class PromptBudgetTest(unittest.TestCase):
    def test_prompt_and_output_reservation_fit_inside_20480_tokens(self):
        budget = PromptBudget(
            total_tokens=20_480,
            output_reserve_tokens=2048,
            retrieval_max_tokens=2048,
            recent_max_tokens=4096,
        )
        summary = StructuredSummary(
            facts=("Türkiye Ankara başkent",),
            decisions=("Kısa cevap ver",),
            open_questions=(),
            user_preferences=("Türkçe",),
            source_message_ids=("m0",),
        )
        retrieval = (
            MessageRecord("r1", "document", "bilgi " * 1500),
            MessageRecord("r2", "document", "kaynak " * 900),
        )
        recent = tuple(
            MessageRecord(f"m{index}", "user", "mesaj " * 1200)
            for index in range(5)
        )

        assembled = assemble_prompt(
            system_prompt="sistem " * 1000,
            summary=summary,
            retrieval_records=retrieval,
            recent_messages=recent,
            token_counter=count_words,
            budget=budget,
        )

        self.assertLessEqual(assembled.input_tokens + assembled.output_reserve_tokens, 20_480)
        self.assertLessEqual(assembled.retrieval_tokens, 2048)
        self.assertLessEqual(assembled.recent_tokens, 4096)
        self.assertEqual(tuple(message.message_id for message in assembled.recent_messages), ("m2", "m3", "m4"))


if __name__ == "__main__":
    unittest.main()
