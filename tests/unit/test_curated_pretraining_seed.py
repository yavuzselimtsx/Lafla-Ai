import hashlib
import json
import tempfile
import unittest
from pathlib import Path

from lafla_ai_core.pretraining.curated_seed import generate_curated_pretraining_seed


SOURCE_TEXT = """# Lafla Ana Prompt

## 11. Kod Yazım Kuralları
Kod önce okunur, sonra hızlı olur.

### 11.1 Doğrulama
Girdi, iş mantığına ulaşmadan önce doğrulanır.

### 11.2 Hata Sözleşmesi
Hatalar kararlı kod ve açık bağlam taşır.

### 11.3 Test
Davranış birim ve entegrasyon testleriyle sabitlenir.

## 12. Güvenlik
PII loglanmaz ve gizli anahtarlar kaynak koda yazılmaz.

### 12.1 Yetki
Her hassas işlem açık yetki denetiminden geçer.

### 12.2 Gözlemlenebilirlik
Loglar tanı koydurur ancak özel veri içermez.
"""


class CuratedPretrainingSeedTest(unittest.TestCase):
    def test_generates_unique_document_style_records_from_prompt_source(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source = root / "Prompt.md"
            output = root / "pretrain.jsonl"
            manifest = root / "pretrain.manifest.json"
            source.write_text(SOURCE_TEXT, encoding="utf-8")

            report = generate_curated_pretraining_seed(
                source_path=source,
                output_path=output,
                manifest_path=manifest,
                count=12,
            )

            rows = [json.loads(line) for line in output.read_text(encoding="utf-8").splitlines()]
            meta = json.loads(manifest.read_text(encoding="utf-8"))

            self.assertEqual(report.records_written, 12)
            self.assertTrue(meta["allowed_for_pretraining"])
            self.assertFalse(meta["allowed_for_post_training"])
            self.assertEqual(meta["quality_metrics"]["exact_duplicate_count"], 0)
            self.assertEqual(meta["quality_metrics"]["unique_text_ratio"], 1.0)
            self.assertTrue(all("text" in row for row in rows))
            self.assertTrue(all("assistant" not in row and "user" not in row for row in rows))
            corpus = output.read_text(encoding="utf-8")
            self.assertIn("Kod önce okunur", corpus)
            self.assertIn("PII", corpus)

    def test_rejects_count_larger_than_unique_candidate_pool(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source = root / "Prompt.md"
            source.write_text("# Kural\n\nTek ve kısa bir teknik ilke burada açıklanır.", encoding="utf-8")

            with self.assertRaisesRegex(ValueError, "benzersiz aday"):
                generate_curated_pretraining_seed(
                    source_path=source,
                    output_path=root / "out.jsonl",
                    manifest_path=root / "manifest.json",
                    count=2,
                )

    def test_generation_is_deterministic_and_records_source_hash(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source = root / "Prompt.md"
            source.write_text(SOURCE_TEXT, encoding="utf-8")
            first = root / "first.jsonl"
            second = root / "second.jsonl"

            generate_curated_pretraining_seed(source_path=source, output_path=first, manifest_path=root / "first.json", count=10)
            generate_curated_pretraining_seed(source_path=source, output_path=second, manifest_path=root / "second.json", count=10)

            self.assertEqual(hashlib.sha256(first.read_bytes()).hexdigest(), hashlib.sha256(second.read_bytes()).hexdigest())
            manifest = json.loads((root / "first.json").read_text(encoding="utf-8"))
            self.assertEqual(manifest["source_sha256"], hashlib.sha256(source.read_bytes()).hexdigest())


if __name__ == "__main__":
    unittest.main()
