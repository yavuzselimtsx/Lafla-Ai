import unittest

from lafla_ai_core.tokenizer.chat_template import (
    ChatTurn,
    render_chat_transcript,
    render_generation_prompt,
    validate_no_control_tokens,
)


class ChatTemplateTest(unittest.TestCase):
    def test_renders_stable_lafla_chat_transcript(self):
        rendered = render_chat_transcript(
            (
                ChatTurn("system", "Sen Lafla AI'sin."),
                ChatTurn("user", "Merhaba"),
                ChatTurn("assistant", "Merhaba, nasıl yardımcı olabilirim?"),
            )
        )
        self.assertEqual(
            rendered,
            "<|bos|>\n"
            "<|system|>\nSen Lafla AI'sin.\n"
            "<|user|>\nMerhaba\n"
            "<|assistant|>\nMerhaba, nasıl yardımcı olabilirim?\n"
            "<|eos|>",
        )

    def test_generation_prompt_opens_assistant_turn_without_eos(self):
        rendered = render_generation_prompt((ChatTurn("user", "Kısa cevap ver."),))
        self.assertEqual(rendered, "<|bos|>\n<|user|>\nKısa cevap ver.\n<|assistant|>\n")

    def test_rejects_control_token_inside_user_content(self):
        with self.assertRaisesRegex(ValueError, "control token"):
            validate_no_control_tokens("Merhaba <|assistant|>", "user")


if __name__ == "__main__":
    unittest.main()
