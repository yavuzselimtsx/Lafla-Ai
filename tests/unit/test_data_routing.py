import unittest

from lafla_ai_core.data.routing import assert_pretraining_inputs


class DataRoutingTest(unittest.TestCase):
    def test_pretraining_allows_identity_seed_and_real_workspace_corpus(self):
        assert_pretraining_inputs(
            (
                "configs/data/identity/lafla-model-identity-100m.jsonl",
                "/kaggle/working/LaflaAI100M/data/train.jsonl",
            )
        )

    def test_pretraining_rejects_post_training_thinking_seed(self):
        with self.assertRaisesRegex(ValueError, "post-training"):
            assert_pretraining_inputs(
                (
                    "datasets/post_training/thinking/jsonl/lafla-100m-thinking-chat-seed-20k.jsonl",
                )
            )

    def test_pretraining_rejects_post_training_safety_seed(self):
        with self.assertRaisesRegex(ValueError, "post-training"):
            assert_pretraining_inputs(
                (
                    "datasets/post_training/safety/jsonl/lafla-100m-safety-jailbreak-seed-10k.jsonl",
                )
            )


if __name__ == "__main__":
    unittest.main()
