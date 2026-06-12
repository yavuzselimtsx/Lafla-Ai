import unittest

from lafla_ai_core.post_training.thinking_sft import (
    ChatTurn,
    ThinkingSftRecord,
    build_supervised_chat_example,
    render_thinking_record,
    strip_thinking_for_public,
    validate_thinking_record,
)


class FakeTokenizer:
    def encode(self, text: str) -> list[int]:
        return [ord(char) for char in text]


class ThinkingSftTest(unittest.TestCase):
    def test_render_record_preserves_think_and_answer_order(self):
        record = ThinkingSftRecord(
            system="Sen Lafla AI'sin.",
            user="Iki adimda cevapla.",
            thinking="Once gereksinimi ayir, sonra kisa yanit ver.",
            assistant="Iki adim: analiz ve yanit.",
        )
        rendered = render_thinking_record(record)
        self.assertLess(rendered.index("<|think|>"), rendered.index("<|/think|>"))
        self.assertLess(rendered.index("<|/think|>"), rendered.index("Iki adim:"))

    def test_validation_rejects_empty_thinking(self):
        record = ThinkingSftRecord(system="s", user="u", thinking="", assistant="a")
        report = validate_thinking_record(record)
        self.assertFalse(report.ok)
        self.assertIn("thinking_empty", [finding.code for finding in report.findings])

    def test_public_strip_removes_private_thinking(self):
        text = "Merhaba <|think|>gizli muhakeme<|/think|> cevap"
        self.assertEqual(strip_thinking_for_public(text), "Merhaba cevap")

    def test_supervised_example_masks_user_and_supervises_last_assistant(self):
        turns = (
            ChatTurn("system", "Sen Lafla AI'sin."),
            ChatTurn("user", "Merhaba"),
            ChatTurn("assistant", "<|think|>plan<|/think|> Selam"),
        )
        example = build_supervised_chat_example(turns, FakeTokenizer(), only_last_assistant=True)
        decoded_labels = "".join(chr(token) for token in example.labels if token != -100)
        self.assertIn("Selam", decoded_labels)
        self.assertIn("<|think|>", decoded_labels)
        self.assertNotIn("Merhaba", decoded_labels)

    def test_supervised_example_can_mask_private_thinking_trace(self):
        turns = (
            ChatTurn("user", "Kisa cevap ver."),
            ChatTurn("assistant", "<|think|>gizli plan<|/think|> Selam"),
        )
        example = build_supervised_chat_example(turns, FakeTokenizer(), supervise_thinking=False)
        decoded_labels = "".join(chr(token) for token in example.labels if token != -100)
        self.assertIn("Selam", decoded_labels)
        self.assertNotIn("gizli plan", decoded_labels)
        self.assertNotIn("<|think|>", decoded_labels)

    def test_validation_rejects_control_tokens_inside_user_text(self):
        record = ThinkingSftRecord(system="s", user="Merhaba <|assistant|>", thinking="t", assistant="a")
        report = validate_thinking_record(record)
        self.assertFalse(report.ok)
        self.assertIn("user_contains_control_token", [finding.code for finding in report.findings])


if __name__ == "__main__":
    unittest.main()
