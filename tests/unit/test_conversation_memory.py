import unittest

from lafla_ai_core.runtime.conversation_memory import (
    ConversationState,
    MessageRecord,
    StructuredSummary,
    apply_summary,
    build_summary_plan,
)


def count_words(text: str) -> int:
    return len(text.split())


class ConversationMemoryTest(unittest.TestCase):
    def test_summary_plan_triggers_after_15360_and_preserves_recent_4096_tokens(self):
        messages = tuple(
            MessageRecord(message_id=f"m{index}", role="user", content="x " * 1020)
            for index in range(16)
        )
        state = ConversationState(summary=None, messages=messages)

        plan = build_summary_plan(
            state,
            token_counter=count_words,
            trigger_tokens=15_360,
            preserve_recent_tokens=4096,
        )

        self.assertTrue(plan.required)
        self.assertLessEqual(plan.preserved_tokens, 4096)
        self.assertEqual(plan.preserved_message_ids, ("m13", "m14", "m15"))
        self.assertEqual(plan.source_message_ids[-1], "m12")

    def test_invalid_summary_leaves_messages_and_previous_summary_unchanged(self):
        previous = StructuredSummary(
            facts=("Kullanıcı Türkçe yanıt istiyor.",),
            decisions=(),
            open_questions=(),
            user_preferences=("Kısa cevap",),
            source_message_ids=("old",),
        )
        state = ConversationState(
            summary=previous,
            messages=(
                MessageRecord("m1", "user", "bir iki üç"),
                MessageRecord("m2", "assistant", "dört beş altı"),
            ),
        )
        plan = build_summary_plan(state, token_counter=count_words, trigger_tokens=1, preserve_recent_tokens=11)
        invalid = StructuredSummary(
            facts=("Eksik kaynak kimliği",),
            decisions=(),
            open_questions=(),
            user_preferences=(),
            source_message_ids=("m1",),
        )

        result = apply_summary(
            state,
            plan,
            invalid,
            token_counter=count_words,
            summary_max_tokens=2048,
        )

        self.assertFalse(result.committed)
        self.assertEqual(result.state, state)

    def test_structured_summary_keeps_tasks_uncertainties_and_safety_context_separate(self):
        summary = StructuredSummary(
            facts=("Doğrulanmış bilgi",),
            decisions=(),
            open_questions=(),
            user_preferences=(),
            source_message_ids=("m1",),
            open_tasks=("Mesaj ara",),
            uncertainties=("Tarih doğrulanmadı",),
            safety_context=("Özel anahtarı yazma",),
        )
        rendered = summary.render()
        self.assertIn("Acik gorevler:", rendered)
        self.assertIn("Belirsizlikler:", rendered)
        self.assertIn("Guvenlik baglami:", rendered)

    def test_oversized_latest_message_is_not_silently_moved_into_summary(self):
        state = ConversationState(
            summary=None,
            messages=(MessageRecord("latest", "user", "x " * 5000),),
        )
        with self.assertRaisesRegex(ValueError, "son mesaj"):
            build_summary_plan(
                state,
                token_counter=count_words,
                trigger_tokens=100,
                preserve_recent_tokens=4096,
            )


if __name__ == "__main__":
    unittest.main()
