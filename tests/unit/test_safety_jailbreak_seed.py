import contextlib
import io
import json
import tempfile
import unittest
from pathlib import Path

from lafla_ai_core.cli.generate_safety_jailbreak_seed import main as generate_safety_main
from lafla_ai_core.post_training.safety_jailbreak_seed import (
    DEFAULT_SAFETY_COUNT,
    DEFAULT_SAFETY_MANIFEST_PATH,
    DEFAULT_SAFETY_OUTPUT_PATH,
    SafetyJailbreakSeedOptions,
    generate_safety_jailbreak_seed,
    iter_safety_jailbreak_records,
    write_safety_jailbreak_seed,
)
from lafla_ai_core.post_training.thinking_dataset import validate_thinking_jsonl_file


class SafetyJailbreakSeedTest(unittest.TestCase):
    def test_generator_produces_policy_and_jailbreak_resistance_examples(self):
        records = list(iter_safety_jailbreak_records(SafetyJailbreakSeedOptions(count=96)))
        joined_users = "\n".join(record.user for record in records)
        joined_assistant = "\n".join(record.assistant for record in records)
        joined_thinking = "\n".join(record.thinking for record in records)

        self.assertEqual(len(records), 96)
        self.assertIn("jailbreak", joined_users.lower())
        self.assertIn("önceki talimatları yok say", joined_users)
        self.assertIn("Buna yardımcı olamam", joined_assistant)
        self.assertIn("Bilmiyorum", joined_assistant)
        self.assertIn("Karar adımları:", joined_thinking)
        self.assertNotIn("<|think|>", joined_assistant)
        self.assertNotIn("adım adım saldırı kodu", joined_assistant)

    def test_writer_creates_valid_ten_thousand_record_dataset_and_manifest(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            output = Path(temp_dir) / "safety.jsonl"
            manifest = Path(temp_dir) / "safety.manifest.json"
            report = write_safety_jailbreak_seed(
                SafetyJailbreakSeedOptions(count=10_000, output_path=output, manifest_path=manifest)
            )
            validation = validate_thinking_jsonl_file(output, max_thinking_chars=1200)
            manifest_payload = json.loads(manifest.read_text(encoding="utf-8"))

        self.assertEqual(report.records_written, 10_000)
        self.assertTrue(validation.ok)
        self.assertEqual(validation.total_records, 10_000)
        self.assertEqual(manifest_payload["data_kind"], "synthetic_safety_sft_seed")
        self.assertIn("jailbreak_resistance", manifest_payload["safety_objectives"])
        self.assertFalse(manifest_payload["allowed_for_pretraining"])

    def test_default_paths_and_count_match_requested_dataset(self):
        self.assertEqual(DEFAULT_SAFETY_COUNT, 10_000)
        self.assertEqual(DEFAULT_SAFETY_OUTPUT_PATH.as_posix(), "datasets/post_training/safety/jsonl/lafla-100m-safety-jailbreak-seed-10k.jsonl")
        self.assertEqual(
            DEFAULT_SAFETY_MANIFEST_PATH.as_posix(),
            "datasets/post_training/safety/manifests/lafla-100m-safety-jailbreak-seed-10k.manifest.json",
        )

    def test_cli_writes_dataset_and_manifest(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            output = Path(temp_dir) / "cli-safety.jsonl"
            manifest = Path(temp_dir) / "cli-safety.manifest.json"
            with contextlib.redirect_stdout(io.StringIO()):
                exit_code = generate_safety_main(
                    [
                        "--count",
                        "40",
                        "--output",
                        str(output),
                        "--manifest",
                        str(manifest),
                    ]
                )
            self.assertEqual(exit_code, 0)
            self.assertTrue(output.exists())
            self.assertTrue(manifest.exists())

    def test_generate_helper_writes_ten_thousand_records_by_default(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            output = Path(temp_dir) / "seed.jsonl"
            manifest = Path(temp_dir) / "seed.manifest.json"
            report = generate_safety_jailbreak_seed(output_path=output, manifest_path=manifest)
            with output.open("r", encoding="utf-8") as handle:
                line_count = sum(1 for _ in handle)

        self.assertEqual(report.records_written, 10_000)
        self.assertEqual(line_count, 10_000)


if __name__ == "__main__":
    unittest.main()
