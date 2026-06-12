import unittest

from lafla_ai_core.tokenizer.quality import analyze_tokenizer_texts, contains_turkish, has_mojibake


class TokenizerQualityTest(unittest.TestCase):
    def test_detects_single_encoded_mojibake(self):
        self.assertTrue(has_mojibake("TÃ¼rkÃ§e dÃ¼ÅŸÃ¼nÃ¼yorum"))
        self.assertTrue(has_mojibake("KÄ±sa cevap ver"))

    def test_accepts_clean_turkish_with_special_tokens(self):
        report = analyze_tokenizer_texts(
            ["Merhaba, ben Lafla AI. Türkçe düşünüyorum."],
            ["<|bos|>", "<|eos|>", "<|think|>"],
            ["<|bos|>", "<|eos|>", "<|think|>", "Mer", "haba"],
        )
        self.assertTrue(report.passed)

    def test_detects_turkish_specific_letters(self):
        self.assertTrue(contains_turkish("çğıöşü ÇĞİÖŞÜ"))
        self.assertFalse(contains_turkish("TÃ¼rkÃ§e"))

    def test_rejects_missing_special_tokens(self):
        report = analyze_tokenizer_texts(
            ["Merhaba, Türkçe örnek."],
            ["<|bos|>", "<|eos|>"],
            ["<|bos|>"],
        )
        self.assertFalse(report.passed)
        self.assertEqual(report.missing_special_tokens, ("<|eos|>",))


if __name__ == "__main__":
    unittest.main()
