import unittest

from lafla_ai_core.model.attention_policy import can_use_full_window_sdpa


class AttentionPolicyTest(unittest.TestCase):
    def test_local_attention_uses_sdpa_when_sequence_fits_sliding_window(self):
        self.assertTrue(
            can_use_full_window_sdpa(
                attention_mode="local",
                sequence_length=4096,
                sliding_window=4096,
                device_type="cuda",
            )
        )

    def test_local_attention_keeps_bounded_path_above_sliding_window(self):
        self.assertFalse(
            can_use_full_window_sdpa(
                attention_mode="local",
                sequence_length=8192,
                sliding_window=4096,
                device_type="cuda",
            )
        )

    def test_global_attention_does_not_route_through_local_fast_path(self):
        self.assertFalse(
            can_use_full_window_sdpa(
                attention_mode="global",
                sequence_length=2048,
                sliding_window=4096,
                device_type="cuda",
            )
        )


if __name__ == "__main__":
    unittest.main()
