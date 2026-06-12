import unittest

from lafla_ai_core.config.schema import ModelConfig

try:
    import torch
    from lafla_ai_core.model.transformer import GroupedQueryAttention, LaflaDecoderModel
except ModuleNotFoundError:
    torch = None
    GroupedQueryAttention = None
    LaflaDecoderModel = None


@unittest.skipIf(torch is None, "torch kurulu degil")
class TransformerModelTest(unittest.TestCase):
    def test_forward_shape_and_loss(self):
        assert torch is not None
        assert LaflaDecoderModel is not None
        config = ModelConfig(
            name="tiny",
            family="decoder-only",
            parameter_target=100_000_000,
            vocab_size=128,
            context_length=512,
            hidden_size=16,
            intermediate_size=32,
            num_layers=2,
            num_attention_heads=4,
            num_key_value_heads=2,
            activation="swiglu",
            norm="rmsnorm",
            rope=True,
            qk_norm=True,
            gradient_checkpointing=False,
        )
        model = LaflaDecoderModel(config)
        input_ids = torch.randint(0, config.vocab_size, (2, 8))
        output = model(input_ids, labels=input_ids)
        self.assertEqual(tuple(output.logits.shape), (2, 8, config.vocab_size))
        self.assertIsNotNone(output.loss)
        self.assertTrue(torch.isfinite(output.loss))

    def test_local_attention_cannot_see_older_tokens_outside_window(self):
        assert torch is not None
        assert GroupedQueryAttention is not None
        config = ModelConfig(
            name="tiny-local",
            family="decoder-only",
            parameter_target=100_000_000,
            vocab_size=128,
            context_length=512,
            hidden_size=4,
            intermediate_size=8,
            num_layers=1,
            num_attention_heads=1,
            num_key_value_heads=1,
            activation="swiglu",
            norm="rmsnorm",
            rope=True,
            qk_norm=False,
            gradient_checkpointing=False,
            attention_pattern=("local",),
            sliding_window=2,
        )
        attention = GroupedQueryAttention(config, attention_mode="local")
        q = torch.ones((1, 1, 4, 4))
        k = torch.ones((1, 1, 4, 4))
        v = torch.tensor([[[[1.0, 0.0, 0.0, 0.0], [2.0, 0.0, 0.0, 0.0], [4.0, 0.0, 0.0, 0.0], [8.0, 0.0, 0.0, 0.0]]]])

        output = attention._chunked_local_attention(q, k, v)

        self.assertAlmostEqual(float(output[0, 0, 3, 0]), 6.0, places=5)


if __name__ == "__main__":
    unittest.main()
