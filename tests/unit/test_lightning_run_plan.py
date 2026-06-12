import unittest

from lafla_ai_core.lightning.run_plan import LightningPaths, build_lightning_run_plan


class LightningRunPlanTest(unittest.TestCase):
    def test_lightning_plan_targets_380m_h200_without_drive_mount_by_default(self):
        plan = build_lightning_run_plan(
            LightningPaths(),
            data_jsonl="/teamspace/studios/this_studio/LaflaAI380M/data/train.clean.jsonl",
            manifest_path="/teamspace/studios/this_studio/LaflaAI380M/data/veri_manifesti.json",
        )

        joined = "\n".join(plan.commands)
        self.assertIn("configs/model/lafla-380m-thinking.yaml", joined)
        self.assertIn("configs/training/lightning/lightning-h200-380m-50000.yaml", joined)
        self.assertIn("configs/data/identity/lafla-model-identity-380m.jsonl", joined)
        self.assertIn("configs/runtime/desktop-phone-fp16-380m.yaml", joined)
        self.assertIn("--model-name lafla-380m-thinking", joined)
        self.assertIn("lafla-380m-thinking-h200-50000-step-run.tar.gz", joined)
        self.assertIn("/teamspace/studios/this_studio/LaflaAI380M/data/veri_manifesti.json", joined)
        self.assertNotIn("drive.mount", joined)
        self.assertNotIn("lafla-1b-thinking", joined)
        self.assertNotIn("/teamspace/studios/this_studio/LaflaAI/data/veri_manifesti.json", joined)

    def test_lightning_plan_can_resume_from_checkpoint(self):
        plan = build_lightning_run_plan(
            LightningPaths(),
            data_jsonl="/teamspace/studios/this_studio/LaflaAI380M/data/train.clean.jsonl",
            resume_from="/teamspace/studios/this_studio/LaflaAI380M/checkpoints/lafla-step-001000",
        )

        self.assertIn("--resume-from /teamspace/studios/this_studio/LaflaAI380M/checkpoints/lafla-step-001000", "\n".join(plan.commands))

    def test_lightning_workspace_must_be_under_teamspace(self):
        with self.assertRaises(ValueError):
            build_lightning_run_plan(
                LightningPaths(workspace_root="/content/LaflaAI"),
                data_jsonl="/teamspace/studios/this_studio/LaflaAI/data/train.jsonl",
            )


if __name__ == "__main__":
    unittest.main()
