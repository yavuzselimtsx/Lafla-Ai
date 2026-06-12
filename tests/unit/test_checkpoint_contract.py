import json
import tempfile
import unittest
from pathlib import Path

from lafla_ai_core.model.checkpoint_contract import validate_checkpoint_directory


class CheckpointContractTest(unittest.TestCase):
    def test_rejects_ready_false(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            self._write_checkpoint(root, ready=False)
            with self.assertRaisesRegex(ValueError, "ready=true"):
                validate_checkpoint_directory(root)

    def test_rejects_missing_required_file(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            self._write_checkpoint(root)
            (root / "optimizer.pt").unlink()
            with self.assertRaisesRegex(FileNotFoundError, "optimizer.pt"):
                validate_checkpoint_directory(root)

    def test_accepts_complete_checkpoint_contract(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            self._write_checkpoint(root)
            contract = validate_checkpoint_directory(root)
            self.assertEqual(contract.format, "lafla-ai-core-checkpoint-v1")
            self.assertIn("model.pt", contract.files)

    def _write_checkpoint(self, root: Path, ready: bool = True) -> None:
        root.mkdir(parents=True, exist_ok=True)
        for name in ("config.json", "trainer_state.json", "model.pt", "optimizer.pt", "rng.pt"):
            (root / name).write_bytes(b"x")
        (root / "READY.json").write_text(
            json.dumps({"ready": ready, "format": "lafla-ai-core-checkpoint-v1"}),
            encoding="utf-8",
        )


if __name__ == "__main__":
    unittest.main()
