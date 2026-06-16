import json
import tempfile
import unittest
from pathlib import Path

from lafla_ai_core.config.schema import PostTrainingConfig
from lafla_ai_core.post_training.thinking_trainer import (
    build_sft_trainer_state,
    iter_buffer_shuffled,
    iter_supervised_thinking_examples,
    pad_or_truncate_example,
)


class FakeTokenizer:
    def encode(self, text: str) -> list[int]:
        return [ord(char) for char in text]


class ThinkingSftTrainerTest(unittest.TestCase):
    def _config(self, *, label_policy: str = "assistant_with_thinking") -> PostTrainingConfig:
        return PostTrainingConfig(
            stage="thinking_sft",
            sequence_length=128,
            learning_rate=0.000025,
            epochs=1,
            label_policy=label_policy,
            only_last_assistant=True,
            public_thinking_visible=False,
            max_thinking_chars=4000,
        )

    def test_pad_or_truncate_uses_pad_id_and_masks_padding_loss(self):
        input_ids = (10, 11, 12)
        labels = (-100, 11, 12)

        padded = pad_or_truncate_example(input_ids, labels, sequence_length=6, pad_id=0)

        self.assertEqual(padded.input_ids, (10, 11, 12, 0, 0, 0))
        self.assertEqual(padded.labels, (-100, 11, 12, -100, -100, -100))

    def test_pad_or_truncate_keeps_last_token_supervision_after_truncation(self):
        input_ids = (1, 2, 3, 4, 5)
        labels = (-100, -100, 3, 4, 5)

        truncated = pad_or_truncate_example(input_ids, labels, sequence_length=4, pad_id=0)

        self.assertEqual(truncated.input_ids, (2, 3, 4, 5))
        self.assertEqual(truncated.labels, (-100, 3, 4, 5))

    def test_assistant_only_policy_masks_private_thinking_tokens(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            data_path = Path(temp_dir) / "thinking.jsonl"
            data_path.write_text(
                json.dumps(
                    {
                        "system": "Sen LaflaGPT Mini'sin.",
                        "user": "Kisa cevap ver.",
                        "thinking": "gizli plan",
                        "assistant": "Ankara.",
                    },
                    ensure_ascii=False,
                )
                + "\n",
                encoding="utf-8",
            )

            example = next(iter_supervised_thinking_examples((data_path,), FakeTokenizer(), self._config(label_policy="assistant_only")))

        decoded_labels = "".join(chr(token) for token in example.labels if token != -100)
        self.assertIn("Ankara", decoded_labels)
        self.assertNotIn("gizli plan", decoded_labels)
        self.assertNotIn("<|think|>", decoded_labels)

    def test_trainer_state_records_source_checkpoint_and_sft_step(self):
        state = build_sft_trainer_state(
            step=7,
            epoch=2,
            source_checkpoint="/tmp/source",
            total_examples=100,
            tokens_seen=2048,
            config=self._config(),
        )

        self.assertEqual(state["step"], 7)
        self.assertEqual(state["epoch"], 2)
        self.assertEqual(state["source_checkpoint"], "/tmp/source")
        self.assertEqual(state["total_examples"], 100)
        self.assertEqual(state["tokens_seen"], 2048)
        self.assertEqual(state["post_training"]["stage"], "thinking_sft")

    def test_buffer_shuffle_is_deterministic_and_keeps_all_examples(self):
        first = list(iter_buffer_shuffled(range(20), seed=42, buffer_size=5))
        second = list(iter_buffer_shuffled(range(20), seed=42, buffer_size=5))

        self.assertEqual(first, second)
        self.assertEqual(sorted(first), list(range(20)))
        self.assertNotEqual(first[:5], list(range(5)))


if __name__ == "__main__":
    unittest.main()
