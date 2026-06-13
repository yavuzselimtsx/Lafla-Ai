import unittest

from lafla_ai_core.training.optimizer_policy import resolve_adamw_fast_path


class OptimizerPolicyTest(unittest.TestCase):
    def test_fp16_cuda_grad_scaler_disables_fused_adamw_fast_path(self):
        policy = resolve_adamw_fast_path(
            prefer_fused_optimizer=True,
            device_type="cuda",
            precision="fp16",
        )

        self.assertFalse(policy.use_fused)
        self.assertEqual(policy.mode, "adamw_amp_safe")

    def test_bf16_cuda_can_try_fused_adamw_when_requested(self):
        policy = resolve_adamw_fast_path(
            prefer_fused_optimizer=True,
            device_type="cuda",
            precision="bf16",
        )

        self.assertTrue(policy.use_fused)
        self.assertEqual(policy.mode, "fused_adamw")

    def test_cpu_never_tries_fused_adamw(self):
        policy = resolve_adamw_fast_path(
            prefer_fused_optimizer=True,
            device_type="cpu",
            precision="fp16",
        )

        self.assertFalse(policy.use_fused)
        self.assertEqual(policy.mode, "adamw")


if __name__ == "__main__":
    unittest.main()
