import unittest

from lafla_ai_core.data.packing import resolve_special_token_id


class FakeTokenizer:
    def token_to_id(self, token: str) -> int | None:
        return {"<|eos|>": 42}.get(token)


class TokenizerContractTest(unittest.TestCase):
    def test_resolves_eos_from_tokenizer_not_constant(self):
        self.assertEqual(resolve_special_token_id(FakeTokenizer(), "<|eos|>"), 42)

    def test_missing_special_token_fails(self):
        with self.assertRaises(ValueError):
            resolve_special_token_id(FakeTokenizer(), "<|pad|>")


if __name__ == "__main__":
    unittest.main()
