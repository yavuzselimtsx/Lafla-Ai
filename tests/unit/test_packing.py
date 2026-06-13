import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from lafla_ai_core.data.packing import iter_jsonl_texts, iter_packed_token_blocks, pack_token_sequences


class FakeTokenizer:
    def encode(self, text: str) -> list[int]:
        return [ord(ch) % 32 for ch in text]


class PackingTest(unittest.TestCase):
    def test_pack_sequences_uses_eos_and_fixed_lengths(self):
        packed = pack_token_sequences(["abcd", "ef"], FakeTokenizer(), sequence_length=3, eos_id=1)
        self.assertEqual([len(item.input_ids) for item in packed], [3, 3])
        self.assertEqual(packed[0].input_ids, packed[0].labels)

    def test_streaming_blocks(self):
        blocks = list(iter_packed_token_blocks(["abcd", "ef"], FakeTokenizer(), sequence_length=3, eos_id=1))
        self.assertEqual(len(blocks), 2)
        self.assertTrue(all(len(block) == 3 for block in blocks))

    def test_iter_jsonl_texts_rejects_mojibake(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "train.jsonl"
            path.write_text(json.dumps({"text": "TÃ¼rkÃ§e bozuk veri"}, ensure_ascii=False) + "\n", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "mojibake"):
                list(iter_jsonl_texts([path]))

    def test_prompt_response_records_use_lafla_chat_template(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "train.jsonl"
            path.write_text(
                json.dumps({"prompt": "Merhaba", "response": "Selam."}, ensure_ascii=False) + "\n",
                encoding="utf-8",
            )
            texts = list(iter_jsonl_texts([path]))
        self.assertEqual(texts, ["<|bos|>\n<|user|>\nMerhaba\n<|assistant|>\nSelam."])

    def test_iter_jsonl_texts_streams_file_without_read_text(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "train.jsonl"
            path.write_text(json.dumps({"text": "gecerli kayit"}) + "\n", encoding="utf-8")

            with patch.object(Path, "read_text", side_effect=AssertionError("tum dosya RAM'e alinmamali")):
                texts = list(iter_jsonl_texts([path]))

        self.assertEqual(texts, ["gecerli kayit"])


if __name__ == "__main__":
    unittest.main()
