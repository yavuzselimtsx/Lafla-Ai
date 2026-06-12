import tempfile
import unittest
from pathlib import Path

from lafla_ai_core.quality.static_scan import StaticScanRule, collect_project_text_files, run_static_scan


class StaticQualityScanTest(unittest.TestCase):
    def test_flags_forbidden_pattern_in_production_file(self):
        report = run_static_scan(
            files={"src/example.py": "eos_id=1\n"},
            rules=(StaticScanRule("hardcoded_eos", "eos_id=1", ("src/",)),),
        )
        self.assertFalse(report.ok)
        self.assertEqual(report.findings[0].rule, "hardcoded_eos")

    def test_ignores_tests_by_default(self):
        report = run_static_scan(
            files={"tests/unit/test_example.py": "eos_id=1\n"},
            rules=(StaticScanRule("hardcoded_eos", "eos_id=1", ("src/",)),),
        )
        self.assertTrue(report.ok)

    def test_flags_raw_shell_command_in_notebook_cell(self):
        report = run_static_scan(files={"notebooks/bad.ipynb": '"tar -czf /tmp/out.tar.gz /tmp/in"\n'})
        self.assertFalse(report.ok)
        self.assertEqual(report.findings[0].rule, "python_cell_shell_tar")

    def test_flags_mojibake_in_production_text(self):
        report = run_static_scan(files={"src/example.py": 'PROMPT = "TÃ¼rkÃ§e"\n'})
        self.assertFalse(report.ok)
        self.assertEqual(report.findings[0].rule, "mojibake_marker")

    def test_allows_bang_shell_command_in_notebook_cell(self):
        report = run_static_scan(files={"notebooks/good.ipynb": '"!tar -czf /tmp/out.tar.gz /tmp/in"\n'})
        self.assertTrue(report.ok)

    def test_collection_excludes_static_scan_rule_definitions(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            quality_dir = root / "src" / "lafla_ai_core" / "quality"
            quality_dir.mkdir(parents=True)
            (quality_dir / "static_scan.py").write_text('PATTERN = "eos_id=1"\n', encoding="utf-8")
            (root / "src" / "example.py").write_text("print('ok')\n", encoding="utf-8")
            files = collect_project_text_files(root)
        self.assertNotIn("src/lafla_ai_core/quality/static_scan.py", files)
        self.assertIn("src/example.py", files)

    def test_default_rules_reject_legacy_model_default_in_active_colab_path(self):
        report = run_static_scan(
            files={"src/lafla_ai_core/colab/example.py": 'MODEL = "lafla-380m-thinking"\n'}
        )
        self.assertFalse(report.ok)
        self.assertEqual(report.findings[0].rule, "legacy_active_model_default")

    def test_default_rules_reject_disabled_cache_in_hf_export(self):
        report = run_static_scan(
            files={"src/lafla_ai_core/export/example.py": '"use_cache": False\n'}
        )
        self.assertFalse(report.ok)
        self.assertEqual(report.findings[0].rule, "hf_cache_disabled")


if __name__ == "__main__":
    unittest.main()
