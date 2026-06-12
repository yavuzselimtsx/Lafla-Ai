import importlib
import sys
import tempfile
import types
import unittest
from pathlib import Path
from unittest.mock import patch


class CheckpointPublishTest(unittest.TestCase):
    def test_publish_replaces_completed_target_and_removes_backup(self):
        fake_torch = types.SimpleNamespace()
        module_name = "lafla_ai_core.model.checkpoint_io"
        with patch.dict(sys.modules, {"torch": fake_torch}):
            sys.modules.pop(module_name, None)
            checkpoint_io = importlib.import_module(module_name)

        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            target = root / "step-100"
            target.mkdir()
            (target / "state.txt").write_text("old", encoding="utf-8")
            tmp = root / ".step-100.tmp"
            tmp.mkdir()
            (tmp / "state.txt").write_text("new", encoding="utf-8")

            checkpoint_io._publish_directory(tmp, target)

            self.assertEqual((target / "state.txt").read_text(encoding="utf-8"), "new")
            self.assertFalse(tmp.exists())
            self.assertFalse((root / ".step-100.bak").exists())


if __name__ == "__main__":
    unittest.main()
