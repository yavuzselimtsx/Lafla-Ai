import unittest
from pathlib import Path


class DirectoryLayoutTest(unittest.TestCase):
    def test_dataset_tree_has_explicit_stage_and_file_type_categories(self):
        datasets = Path("datasets")

        self.assertTrue((datasets / "pretraining").is_dir())
        self.assertTrue((datasets / "post_training").is_dir())
        self.assertTrue((datasets / "evaluation").is_dir())

        expected_post_training_dirs = {
            Path("datasets/post_training/thinking/jsonl"),
            Path("datasets/post_training/thinking/manifests"),
            Path("datasets/post_training/safety/jsonl"),
            Path("datasets/post_training/safety/manifests"),
        }
        for directory in expected_post_training_dirs:
            self.assertTrue(directory.is_dir(), f"missing dataset category: {directory}")

        uncategorized = [
            path.as_posix()
            for path in (datasets / "post_training").rglob("*")
            if path.is_file() and path.name != "README.md" and path.parent.name not in {"jsonl", "manifests"}
        ]
        self.assertEqual(uncategorized, [])

    def test_config_training_and_data_are_split_by_responsibility(self):
        expected_dirs = (
            Path("configs/data/identity"),
            Path("configs/data/source-plans"),
            Path("configs/training/colab"),
            Path("configs/training/kaggle"),
            Path("configs/training/lightning"),
        )
        for directory in expected_dirs:
            self.assertTrue(directory.is_dir(), f"missing config category: {directory}")

        loose_training_files = [path.as_posix() for path in Path("configs/training").glob("*") if path.is_file()]
        self.assertEqual(loose_training_files, [])

    def test_script_and_notebook_launchers_are_platform_scoped(self):
        expected_dirs = (
            Path("scripts/colab"),
            Path("scripts/kaggle"),
            Path("scripts/lightning"),
            Path("scripts/data"),
            Path("notebooks/colab"),
        )
        for directory in expected_dirs:
            self.assertTrue(directory.is_dir(), f"missing launcher category: {directory}")

        loose_scripts = [path.as_posix() for path in Path("scripts").glob("*") if path.is_file()]
        loose_notebooks = [path.as_posix() for path in Path("notebooks").glob("*") if path.is_file()]
        self.assertEqual(loose_scripts, [])
        self.assertEqual(loose_notebooks, [])


if __name__ == "__main__":
    unittest.main()
