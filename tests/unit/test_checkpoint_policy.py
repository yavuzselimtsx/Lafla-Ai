import tempfile
import unittest
from pathlib import Path

from lafla_ai_core.training.checkpoint_policy import (
    CheckpointPolicy,
    archive_checkpoint_snapshot,
    apply_checkpoint_retention,
    retention_victims,
    should_save_checkpoint,
)


class CheckpointPolicyTest(unittest.TestCase):
    def test_save_every_and_final(self):
        policy = CheckpointPolicy(save_every=100, keep_last=3)
        self.assertFalse(should_save_checkpoint(0, 1000, policy))
        self.assertTrue(should_save_checkpoint(100, 1000, policy))
        self.assertTrue(should_save_checkpoint(1000, 1000, policy))

    def test_retention_keeps_last_steps(self):
        self.assertEqual(retention_victims([250, 500, 750, 1000], keep_last=3), (250,))

    def test_retention_archives_ready_checkpoint_before_deleting_it(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            checkpoints = root / "checkpoints"
            backups = root / "checkpoint-backups"
            for step in (100, 200, 300):
                checkpoint = checkpoints / f"lafla-step-{step:06d}"
                checkpoint.mkdir(parents=True)
                (checkpoint / "READY.json").write_text('{"ready": true}', encoding="utf-8")
                (checkpoint / "model.pt").write_text(f"weights {step}", encoding="utf-8")

            apply_checkpoint_retention(checkpoints, keep_last=2)

            archived = backups / "lafla-step-000100"
            self.assertFalse((checkpoints / "lafla-step-000100").exists())
            self.assertTrue((checkpoints / "lafla-step-000200").exists())
            self.assertTrue((checkpoints / "lafla-step-000300").exists())
            self.assertTrue((archived / "READY.json").exists())
            self.assertEqual((archived / "model.pt").read_text(encoding="utf-8"), "weights 100")
            self.assertTrue((archived / "CHECKPOINT_BACKUP.json").exists())

    def test_archive_snapshot_does_not_overwrite_existing_archive(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            checkpoints = root / "checkpoints"
            checkpoint = checkpoints / "lafla-step-000100"
            checkpoint.mkdir(parents=True)
            (checkpoint / "READY.json").write_text('{"ready": true}', encoding="utf-8")
            (checkpoint / "model.pt").write_text("new weights", encoding="utf-8")
            existing = root / "checkpoint-backups" / "lafla-step-000100"
            existing.mkdir(parents=True)
            (existing / "model.pt").write_text("old backup", encoding="utf-8")

            archived = archive_checkpoint_snapshot(checkpoint, checkpoints, reason="unit_test")

            self.assertEqual((existing / "model.pt").read_text(encoding="utf-8"), "old backup")
            self.assertNotEqual(archived, existing)
            self.assertEqual((archived / "model.pt").read_text(encoding="utf-8"), "new weights")

    def test_retention_keeps_existing_archive_and_writes_new_archive_before_delete(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            checkpoints = root / "checkpoints"
            backup = root / "checkpoint-backups" / "lafla-step-000100"
            backup.mkdir(parents=True)
            (backup / "model.pt").write_text("old backup", encoding="utf-8")
            for step in (100, 200):
                checkpoint = checkpoints / f"lafla-step-{step:06d}"
                checkpoint.mkdir(parents=True)
                (checkpoint / "READY.json").write_text('{"ready": true}', encoding="utf-8")
                (checkpoint / "model.pt").write_text(f"new weights {step}", encoding="utf-8")

            apply_checkpoint_retention(checkpoints, keep_last=1)

            self.assertEqual((backup / "model.pt").read_text(encoding="utf-8"), "old backup")
            self.assertFalse((checkpoints / "lafla-step-000100").exists())
            new_archives = [
                path
                for path in (root / "checkpoint-backups").iterdir()
                if path.name.startswith("lafla-step-000100-backup-")
            ]
            self.assertEqual(len(new_archives), 1)
            self.assertEqual((new_archives[0] / "model.pt").read_text(encoding="utf-8"), "new weights 100")


if __name__ == "__main__":
    unittest.main()
