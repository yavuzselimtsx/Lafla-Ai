import unittest

from lafla_ai_core.runtime.message_search import (
    MessageSearchQuery,
    MessageSearchScope,
    SearchHit,
    pack_search_results,
)


def count_words(text: str) -> int:
    return len(text.split())


class MessageSearchTest(unittest.TestCase):
    def test_retrieval_never_exceeds_2048_tokens_or_splits_a_message(self):
        scope = MessageSearchScope(actor_id="u1", platform="instagram", conversation_ids=("c1",))
        query = MessageSearchQuery(text="geçmiş plan", scope=scope, limit=10)
        hits = (
            SearchHit("m1", "c1", "a " * 1200, 0.9, platform="instagram", timestamp="2026-06-12T10:00:00Z"),
            SearchHit("m2", "c1", "b " * 900, 0.8, platform="instagram", timestamp="2026-06-12T09:00:00Z"),
            SearchHit("m3", "c1", "c " * 800, 0.7, platform="instagram", timestamp="2026-06-12T08:00:00Z"),
        )

        packed = pack_search_results(query, hits, token_counter=count_words, max_tokens=2048)

        self.assertLessEqual(packed.token_count, 2048)
        self.assertEqual(tuple(item.message_id for item in packed.hits), ("m1", "m3"))
        self.assertNotIn("m2", tuple(item.message_id for item in packed.hits))
        self.assertTrue(all(item.platform == "instagram" and item.timestamp for item in packed.hits))

    def test_query_requires_an_authorized_conversation_scope(self):
        with self.assertRaisesRegex(ValueError, "conversation"):
            MessageSearchQuery(
                text="test",
                scope=MessageSearchScope(actor_id="u1", platform="discord", conversation_ids=()),
            ).validate()


if __name__ == "__main__":
    unittest.main()
