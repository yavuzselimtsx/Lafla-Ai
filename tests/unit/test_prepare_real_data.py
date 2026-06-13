import argparse
import importlib.util
import json
import sys
import tempfile
import unittest
from pathlib import Path


def _load_prepare_real_data_module():
    path = Path("scripts/data/prepare_real_data.py")
    spec = importlib.util.spec_from_file_location("prepare_real_data_for_test", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"module spec olusturulamadi: {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


class PrepareRealDataTest(unittest.TestCase):
    def test_clean_text_rejects_tokenizer_mojibake_markers_before_jsonl_write(self):
        module = _load_prepare_real_data_module()
        text = (
            "This is the article and the model writes with enough English words "
            "from this source while a broken K\u00c4\u00b1sa marker remains inside."
        )

        cleaned = module.clean_text(text, 12_000, {"domain": "english", "language": "en"})

        self.assertEqual(cleaned, "")

    def test_run_tops_up_from_successful_sources_when_another_source_fails(self):
        module = _load_prepare_real_data_module()
        records = [
            f"gercek turkce kaynak kaydi {index} ve model icin temiz egitim metni "
            + ("veri " * 380)
            for index in range(80)
        ]
        specs = (
            {
                "source_id": "working_real_source",
                "target_share": 0.25,
                "language": "tr",
                "domain": "turkish",
                "usage": "pretraining",
                "trust_tier": "primary",
                "license": "test-license",
                "source_url": "local://working",
            },
            {
                "source_id": "failed_large_source",
                "target_share": 0.75,
                "language": "en",
                "domain": "english",
                "usage": "pretraining",
                "trust_tier": "secondary",
                "license": "test-license",
                "source_url": "local://failed",
            },
        )

        def fake_iter_source_texts(spec):
            if spec["source_id"] == "failed_large_source":
                raise RuntimeError("builder config missing")
            yield from records

        module.DEFAULT_SOURCE_SPECS = specs
        module.iter_source_texts = fake_iter_source_texts
        module.clean_text = lambda raw, _max_record_chars, _spec: raw

        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            args = argparse.Namespace(
                output=str(root / "train.jsonl"),
                manifest=str(root / "veri_manifesti.json"),
                report=str(root / "data-prepare-report.json"),
                identity_jsonl="unused.jsonl",
                dataset_version="test-top-up",
                target_chars=100_000,
                min_chars=80_000,
                max_record_chars=12_000,
            )

            exit_code = module.run(args)
            report = json.loads((root / "data-prepare-report.json").read_text(encoding="utf-8"))

        self.assertEqual(exit_code, 0)
        self.assertGreaterEqual(report["total_chars"], 80_000)
        sources = {item["source_id"]: item for item in report["sources"]}
        self.assertEqual(sources["failed_large_source"]["status"], "failed")
        self.assertGreater(sources["working_real_source"]["accepted"], 30)


if __name__ == "__main__":
    unittest.main()
