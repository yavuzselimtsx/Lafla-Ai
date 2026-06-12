import unittest

from lafla_ai_core.runtime.decoding import clean_decoded_text, decode_token_ids


class RuntimeDecodingTest(unittest.TestCase):
    def test_cleans_bytelevel_surface_and_repairs_display_mojibake(self):
        self.assertEqual(clean_decoded_text("ĠTÃ¼rkÃ§e ĠkÄ±sa Ġcevap"), "Türkçe kısa cevap")

    def test_preserves_thinking_tokens_when_special_tokens_are_not_stripped(self):
        text = clean_decoded_text("Ġ<|think|>planĠyap<|/think|>", strip_special_tokens=False)
        self.assertEqual(text, "<|think|>plan yap<|/think|>")

    def test_decode_token_ids_uses_tokenizer_then_cleans_output(self):
        class FakeTokenizer:
            def decode(self, token_ids, skip_special_tokens=False):
                self.seen = (list(token_ids), skip_special_tokens)
                return "ĠTÃ¼rkÃ§e"

        tokenizer = FakeTokenizer()
        self.assertEqual(decode_token_ids(tokenizer, [1, 2, 3], skip_special_tokens=True), "Türkçe")
        self.assertEqual(tokenizer.seen, ([1, 2, 3], True))


if __name__ == "__main__":
    unittest.main()
