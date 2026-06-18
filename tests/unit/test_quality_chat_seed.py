import hashlib
import json
import tempfile
import unittest
from pathlib import Path

from lafla_ai_core.post_training.quality_chat_seed import generate_quality_chat_seed


class QualityChatSeedTest(unittest.TestCase):
    def test_generates_balanced_unique_chat_seed_and_manifest(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            output = root / "chat.jsonl"
            manifest = root / "chat.manifest.json"

            report = generate_quality_chat_seed(output_path=output, manifest_path=manifest, count=360)

            rows = [json.loads(line) for line in output.read_text(encoding="utf-8").splitlines()]
            meta = json.loads(manifest.read_text(encoding="utf-8"))
            categories = meta["category_counts"]
            unique_pairs = {(row["user"], row["assistant"]) for row in rows}

            self.assertEqual(report.records_written, 360)
            self.assertTrue(meta["allowed_for_post_training"])
            self.assertFalse(meta["allowed_for_pretraining"])
            self.assertGreater(categories["answerable_anchor"], categories["bounded_uncertainty"])
            self.assertLessEqual(categories["safety_resilience"] / len(rows), 0.10)
            self.assertGreaterEqual(len(unique_pairs) / len(rows), 0.98)
            self.assertEqual(meta["quality_metrics"]["exact_duplicate_count"], 0)
            self.assertLessEqual(meta["quality_metrics"]["refusal_ratio"], 0.12)
            self.assertIn("Ankara", output.read_text(encoding="utf-8"))
            self.assertIn("LaflaGPT Mini", output.read_text(encoding="utf-8"))

    def test_answerable_records_do_not_refuse(self):
        with tempfile.TemporaryDirectory() as tmp:
            output = Path(tmp) / "chat.jsonl"
            manifest = Path(tmp) / "chat.manifest.json"

            generate_quality_chat_seed(output_path=output, manifest_path=manifest, count=240)

            rows = [json.loads(line) for line in output.read_text(encoding="utf-8").splitlines()]
            answerable = [row for row in rows if row["_sft_category"] == "answerable_anchor"]
            refusal_markers = ("bilmiyorum", "uydurmam", "weiß ich nicht", "weiss ich nicht")
            self.assertTrue(answerable)
            self.assertFalse(
                any(marker in row["assistant"].casefold() for row in answerable for marker in refusal_markers)
            )

    def test_records_do_not_use_visible_variant_suffixes_or_mojibake(self):
        with tempfile.TemporaryDirectory() as tmp:
            output = Path(tmp) / "chat.jsonl"
            manifest = Path(tmp) / "chat.manifest.json"

            generate_quality_chat_seed(output_path=output, manifest_path=manifest, count=240)

            text = output.read_text(encoding="utf-8")
            folded = text.casefold()
            self.assertNotIn("varyant ", folded)
            self.assertNotIn("variant ", folded)
            self.assertNotIn("Ã", text)
            self.assertNotIn("Â", text)

    def test_generation_is_deterministic(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            first = root / "first.jsonl"
            second = root / "second.jsonl"

            generate_quality_chat_seed(output_path=first, manifest_path=root / "first.json", count=180)
            generate_quality_chat_seed(output_path=second, manifest_path=root / "second.json", count=180)

            first_hash = hashlib.sha256(first.read_bytes()).hexdigest()
            second_hash = hashlib.sha256(second.read_bytes()).hexdigest()
            self.assertEqual(first_hash, second_hash)

    def test_large_generation_remains_duplicate_free(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            manifest = root / "large.manifest.json"

            generate_quality_chat_seed(output_path=root / "large.jsonl", manifest_path=manifest, count=2_000)

            meta = json.loads(manifest.read_text(encoding="utf-8"))
            self.assertEqual(meta["quality_metrics"]["exact_duplicate_count"], 0)
            self.assertGreaterEqual(meta["quality_metrics"]["unique_user_assistant_ratio"], 0.98)


if __name__ == "__main__":
    unittest.main()
