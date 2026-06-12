import contextlib
import io
import json
import tempfile
import unittest
from pathlib import Path

from lafla_ai_core.cli.artifact_manifest import main


class ArtifactManifestCliTest(unittest.TestCase):
    def test_cli_writes_manifest(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            (root / "model.pt").write_bytes(b"weights")
            output = root / "reports" / "artifact-manifest.json"
            stdout = io.StringIO()
            with contextlib.redirect_stdout(stdout):
                exit_code = main(["--root", str(root), "--output", str(output)])

            payload = json.loads(output.read_text(encoding="utf-8"))
            self.assertEqual(exit_code, 0)
            self.assertTrue(payload["files"])
            self.assertIn("artifact-manifest.json", stdout.getvalue())


if __name__ == "__main__":
    unittest.main()
