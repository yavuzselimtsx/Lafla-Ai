import importlib.util
import sys
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


if __name__ == "__main__":
    unittest.main()
