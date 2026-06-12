import contextlib
import io
import json
import tempfile
import unittest
from pathlib import Path

from lafla_ai_core.cli.validate_thinking_sft import main as validate_thinking_main
from lafla_ai_core.post_training.thinking_dataset import validate_thinking_jsonl_file


class ThinkingDatasetTest(unittest.TestCase):
    def test_validate_thinking_jsonl_accepts_complete_record(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "thinking.jsonl"
            path.write_text(
                json.dumps(
                    {
                        "system": "Sen Lafla AI'sin.",
                        "user": "Kisa cevap ver.",
                        "thinking": "Istenen bicimi belirle.",
                        "assistant": "Tamam.",
                    },
                    ensure_ascii=False,
                )
                + "\n",
                encoding="utf-8",
            )
            report = validate_thinking_jsonl_file(path)
        self.assertTrue(report.ok)
        self.assertEqual(report.total_records, 1)

    def test_validate_thinking_jsonl_rejects_missing_field(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "bad.jsonl"
            path.write_text(json.dumps({"system": "s", "user": "u", "assistant": "a"}) + "\n", encoding="utf-8")
            report = validate_thinking_jsonl_file(path)
        self.assertFalse(report.ok)
        self.assertEqual(report.findings[0].code, "missing_field")
        self.assertEqual(report.findings[0].line, 1)

    def test_cli_writes_report_and_fails_on_bad_dataset(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            data_path = root / "bad.jsonl"
            report_path = root / "report.json"
            data_path.write_text(json.dumps({"system": "s", "user": "u", "assistant": "a"}) + "\n", encoding="utf-8")
            with contextlib.redirect_stdout(io.StringIO()):
                exit_code = validate_thinking_main(["--input", str(data_path), "--report", str(report_path)])
            report_payload = json.loads(report_path.read_text(encoding="utf-8"))
        self.assertEqual(exit_code, 2)
        self.assertFalse(report_payload["ok"])


if __name__ == "__main__":
    unittest.main()
