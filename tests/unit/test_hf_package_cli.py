import contextlib
import io
import json
import tempfile
import unittest
from pathlib import Path

from lafla_ai_core.cli.hf_package import main


class HuggingFacePackageCliTest(unittest.TestCase):
    def test_cli_writes_package(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            tokenizer_json = root / "tokenizer.json"
            tokenizer_json.write_text(
                json.dumps(
                    {
                        "model": {
                            "vocab": {
                                "<|bos|>": 0,
                                "<|eos|>": 1,
                                "<|pad|>": 2,
                                "<|unk|>": 3,
                                "<|system|>": 4,
                                "<|user|>": 5,
                                "<|assistant|>": 6,
                                "<|think|>": 7,
                                "<|/think|>": 8,
                            }
                        }
                    }
                ),
                encoding="utf-8",
            )
            output = root / "hf"
            stdout = io.StringIO()
            with contextlib.redirect_stdout(stdout):
                exit_code = main(
                    [
                        "--tokenizer-json",
                        str(tokenizer_json),
                        "--output-dir",
                        str(output),
                        "--model-name",
                        "lafla-400m-thinking",
                    ]
                )

            self.assertEqual(exit_code, 0)
            self.assertTrue((output / "tokenizer_config.json").exists())
            self.assertIn("hf package written", stdout.getvalue())


if __name__ == "__main__":
    unittest.main()
