import unittest

from lafla_ai_core.environment.dependencies import ModuleRequirement, check_required_modules, colab_training_requirements


class EnvironmentDependenciesTest(unittest.TestCase):
    def test_reports_missing_required_modules(self):
        report = check_required_modules(
            (
                ModuleRequirement(module="json", purpose="stdlib sanity"),
                ModuleRequirement(module="lafla_missing_dependency_for_test", purpose="missing"),
            )
        )
        self.assertFalse(report.ok)
        self.assertEqual(report.missing[0].module, "lafla_missing_dependency_for_test")

    def test_tpu_requirements_include_torch_xla(self):
        requirements = colab_training_requirements(optimizer="adamw", accelerator="xla")

        modules = {item.module for item in requirements}
        self.assertIn("torch", modules)
        self.assertIn("torch_xla", modules)
        self.assertNotIn("bitsandbytes", modules)


if __name__ == "__main__":
    unittest.main()
