import json
import unittest
from pathlib import Path


class SeedProfileConfigTest(unittest.TestCase):
    def test_seed_generators_do_not_embed_model_identity_text(self):
        source_paths = (
            Path("src/lafla_ai_core/post_training/synthetic_chat_seed.py"),
            Path("src/lafla_ai_core/post_training/safety_jailbreak_seed.py"),
        )
        forbidden = (
            "LaflaGPT 100M",
            "LaflaGPT Mini",
            "Yavuz Selim",
            "Türkçe ve Almanca odaklı",
            "Turkish- and German",
        )

        for path in source_paths:
            text = path.read_text(encoding="utf-8")
            for marker in forbidden:
                self.assertNotIn(marker, text, f"{path} embeds model profile marker {marker!r}")

    def test_seed_profile_contains_model_specific_values_outside_source_code(self):
        profile_path = Path("configs/post_training/lafla-100m-seed-profile.json")
        payload = json.loads(profile_path.read_text(encoding="utf-8"))

        self.assertIn("model", payload)
        self.assertIn("chat_seed", payload)
        self.assertIn("safety_seed", payload)
        self.assertEqual(payload["chat_seed"]["count"], 20_000)
        self.assertEqual(payload["safety_seed"]["count"], 10_000)
        self.assertTrue(payload["model"]["identity_statement"])


if __name__ == "__main__":
    unittest.main()
