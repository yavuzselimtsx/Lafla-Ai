import json
import tempfile
import unittest
from pathlib import Path

from lafla_ai_core.data.preflight import validate_pretraining_jsonl


class PretrainingDataValidationTest(unittest.TestCase):
    def test_validates_every_jsonl_record_before_training(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "train.jsonl"
            path.write_text(
                "\n".join(
                    [
                        json.dumps({"text": "birinci"}),
                        json.dumps({"prompt": "soru", "response": "cevap"}),
                    ]
                )
                + "\n",
                encoding="utf-8",
            )

            report = validate_pretraining_jsonl((str(path),))

        self.assertTrue(report.ok)
        self.assertEqual(report.total_records, 2)
        self.assertEqual(report.files[0].records, 2)

    def test_rejects_truncated_json_with_exact_line_number(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "train.jsonl"
            path.write_text(
                json.dumps({"text": "gecerli"}) + "\n" + '{"text":"yarim\n',
                encoding="utf-8",
            )

            with self.assertRaisesRegex(ValueError, r"train\.jsonl:2 JSONL gecersiz"):
                validate_pretraining_jsonl((str(path),))


if __name__ == "__main__":
    unittest.main()
