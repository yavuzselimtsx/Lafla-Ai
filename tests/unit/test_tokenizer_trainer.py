import json
import tempfile
import unittest
from pathlib import Path

from lafla_ai_core.config.schema import TokenizerConfig
from lafla_ai_core.tokenizer.trainer import SamplingTextIterator, iter_jsonl_texts, train_bpe_tokenizer


class TokenizerTrainerTest(unittest.TestCase):
    def test_iter_jsonl_texts_extracts_known_fields(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "data.jsonl"
            path.write_text(
                "\n".join(
                    [
                        json.dumps({"system": "Sistem", "prompt": "Merhaba", "chosen": "Türkçe cevap"}, ensure_ascii=False),
                        json.dumps("Düz metin", ensure_ascii=False),
                    ]
                ),
                encoding="utf-8",
            )
            texts = list(iter_jsonl_texts([path]))
        self.assertEqual(texts, ["Merhaba", "Türkçe cevap", "Sistem", "Düz metin"])

    def test_iter_jsonl_texts_rejects_mojibake_before_training(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "data.jsonl"
            path.write_text(json.dumps({"text": "TÃ¼rkÃ§e bozuk veri"}, ensure_ascii=False) + "\n", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "mojibake"):
                list(iter_jsonl_texts([path]))

    def test_sampling_iterator_does_not_cache_beyond_limit(self):
        iterator = SamplingTextIterator((str(index) for index in range(10)), sample_limit=3)
        self.assertEqual(list(iterator), [str(index) for index in range(10)])
        self.assertEqual(iterator.sample, ("0", "1", "2"))
        self.assertEqual(iterator.count, 10)

    def test_trained_bytelevel_tokenizer_decodes_clean_turkish_roundtrip(self):
        try:
            from tokenizers import Tokenizer
        except ModuleNotFoundError:
            self.skipTest("tokenizers paketi kurulu degil")
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            path = root / "data.jsonl"
            rows = [
                {"text": "Merhaba, Türkçe düşünüyorum. Çalışma düzeni sağlam."},
                {"text": "LaflaGPT kısa cevap verir ve özel tokenları korur."},
                {"text": "<|system|>Türkçe cevap ver.<|user|>2+2 kaç eder?<|assistant|>4 eder."},
            ]
            path.write_text("\n".join(json.dumps(row, ensure_ascii=False) for row in rows) + "\n", encoding="utf-8")
            output = root / "tok.json"
            report = root / "report.json"
            config = TokenizerConfig(
                kind="bpe",
                vocab_size=512,
                normalization="utf8_nfc",
                required_special_tokens=(
                    "<|bos|>",
                    "<|eos|>",
                    "<|pad|>",
                    "<|system|>",
                    "<|user|>",
                    "<|assistant|>",
                    "<|think|>",
                    "<|/think|>",
                ),
                quality_gates={"turkish_roundtrip_required": True},
            )
            train_bpe_tokenizer([path], output, report, config)
            tokenizer = Tokenizer.from_file(str(output))
            decoded = tokenizer.decode(tokenizer.encode("Türkçe kısa cevap").ids)
        self.assertEqual(decoded, "Türkçe kısa cevap")
        self.assertNotIn("Ġ", decoded)
        self.assertNotIn("Ã", decoded)


if __name__ == "__main__":
    unittest.main()
