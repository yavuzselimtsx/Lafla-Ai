import unittest

from lafla_ai_core.runtime.context import ChatMessage, prepare_context_window


class RuntimeContextTest(unittest.TestCase):
    def test_context_window_preserves_system_and_latest_user_message(self):
        messages = (
            ChatMessage("system", "Gizli sistem ilkesi."),
            ChatMessage("user", "eski " * 50),
            ChatMessage("assistant", "eski cevap " * 50),
            ChatMessage("user", "Son soru burada."),
        )
        window = prepare_context_window(messages, max_chars=80)
        self.assertEqual(window.messages[0].role, "system")
        self.assertEqual(window.messages[-1].content, "Son soru burada.")
        self.assertIn("context_trimmed", window.warnings)

    def test_rejects_context_too_small_for_system_message(self):
        with self.assertRaises(ValueError):
            prepare_context_window((ChatMessage("system", "çok uzun sistem"),), max_chars=3)


if __name__ == "__main__":
    unittest.main()
