import json
import contextlib
import io
import tempfile
import unittest
from pathlib import Path

from lafla_ai_core.cli.generate_synthetic_chat_seed import main as generate_seed_main
from lafla_ai_core.post_training.synthetic_chat_seed import (
    DEFAULT_MANIFEST_PATH,
    DEFAULT_OUTPUT_PATH,
    SyntheticChatSeedOptions,
    generate_synthetic_chat_seed,
    iter_synthetic_chat_seed_records,
    write_synthetic_chat_seed,
)
from lafla_ai_core.post_training.thinking_dataset import validate_thinking_jsonl_file


class SyntheticChatSeedTest(unittest.TestCase):
    def test_generator_produces_requested_count_with_clean_utf8_and_policy_metadata(self):
        records = list(iter_synthetic_chat_seed_records(SyntheticChatSeedOptions(count=64)))

        self.assertEqual(len(records), 64)
        self.assertTrue(any("Türkçe" in record.system for record in records))
        self.assertTrue(any("Deutsch" in record.system for record in records))
        self.assertTrue(any("Bilmiyorum" in record.assistant for record in records))
        self.assertTrue(any("LaflaGPT Mini" in record.assistant for record in records))
        self.assertTrue(any("100 milyon" in record.assistant for record in records))
        self.assertFalse(any("GPT-5.5 gibi düşünüyorum" in record.assistant for record in records))
        self.assertFalse(any(marker in json.dumps(record.__dict__, ensure_ascii=False) for record in records for marker in ("Ã", "Ä", "Å", "�")))

    def test_writer_creates_valid_jsonl_and_synthetic_manifest(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            output = Path(temp_dir) / "seed.jsonl"
            manifest = Path(temp_dir) / "seed.manifest.json"
            report = write_synthetic_chat_seed(
                SyntheticChatSeedOptions(count=128, output_path=output, manifest_path=manifest)
            )
            validation = validate_thinking_jsonl_file(output, max_thinking_chars=900)
            manifest_payload = json.loads(manifest.read_text(encoding="utf-8"))

        self.assertEqual(report.records_written, 128)
        self.assertEqual(report.output_path, str(output))
        self.assertTrue(validation.ok)
        self.assertEqual(validation.total_records, 128)
        self.assertEqual(manifest_payload["data_kind"], "synthetic_sft_seed")
        self.assertFalse(manifest_payload["allowed_for_pretraining"])
        self.assertTrue(manifest_payload["allowed_for_post_training"])

    def test_default_paths_are_training_dataset_locations(self):
        self.assertEqual(DEFAULT_OUTPUT_PATH.as_posix(), "datasets/post_training/thinking/jsonl/lafla-100m-thinking-chat-seed-20k.jsonl")
        self.assertEqual(DEFAULT_MANIFEST_PATH.as_posix(), "datasets/post_training/thinking/manifests/lafla-100m-thinking-chat-seed-20k.manifest.json")

    def test_generate_helper_writes_twenty_thousand_records_by_default(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            output = Path(temp_dir) / "seed.jsonl"
            manifest = Path(temp_dir) / "seed.manifest.json"
            report = generate_synthetic_chat_seed(output_path=output, manifest_path=manifest)
            with output.open("r", encoding="utf-8") as handle:
                line_count = sum(1 for _ in handle)

        self.assertEqual(report.records_written, 20_000)
        self.assertEqual(line_count, 20_000)

    def test_cli_writes_dataset_and_manifest(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            output = Path(temp_dir) / "cli-seed.jsonl"
            manifest = Path(temp_dir) / "cli-seed.manifest.json"
            with contextlib.redirect_stdout(io.StringIO()):
                exit_code = generate_seed_main(
                    [
                        "--count",
                        "32",
                        "--output",
                        str(output),
                        "--manifest",
                        str(manifest),
                    ]
                )
            self.assertEqual(exit_code, 0)
            self.assertTrue(output.exists())
            self.assertTrue(manifest.exists())


if __name__ == "__main__":
    unittest.main()
