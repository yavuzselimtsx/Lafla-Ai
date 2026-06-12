import unittest
from pathlib import Path
from unittest.mock import patch

from lafla_ai_core.colab.drive import inspect_drive_mount


class ColabDriveTest(unittest.TestCase):
    def test_rejects_local_folder_not_in_mount_table(self):
        with patch.object(Path, "exists", return_value=True):
            status = inspect_drive_mount("/content/drive", proc_mounts_text="")
        self.assertFalse(status.usable)

    def test_accepts_path_present_in_mount_table(self):
        mounts = "drive /content/gdrive fuse.drivefs rw,nosuid,nodev 0 0\n"
        with patch.object(Path, "exists", return_value=True):
            status = inspect_drive_mount("/content/gdrive", proc_mounts_text=mounts)
        self.assertTrue(status.usable)


if __name__ == "__main__":
    unittest.main()

