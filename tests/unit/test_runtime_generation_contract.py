import unittest

from lafla_ai_core.runtime.context import ChatMessage
from lafla_ai_core.runtime.generation_contract import (
    build_generation_request,
    decode_completion_from_ids,
    resolve_stop_token_ids,
    trim_completion_token_ids,
)


class FakeTokenizer:
    def encode(self, text: str) -> list[int]:
        return [ord(char) for char in text]

    def decode(self, token_ids, skip_special_tokens=False) -> str:
        return "".join(chr(int(token_id)) for token_id in token_ids)


class RuntimeGenerationContractTest(unittest.TestCase):
    def test_generation_request_renders_chat_prompt_and_role_stop_ids(self):
        request = build_generation_request(
            (
                ChatMessage("system", "Sen LaflaGPT modelisin."),
                ChatMessage("user", "2+2 kac eder?"),
            ),
            FakeTokenizer(),
        )

        self.assertTrue(request.prompt_text.endswith("<|assistant|>\n"))
        self.assertIn("<|user|>\n2+2 kac eder?", request.prompt_text)
        self.assertIn(tuple(FakeTokenizer().encode("<|eos|>")), request.stop_token_ids)
        self.assertIn(tuple(FakeTokenizer().encode("<|user|>")), request.stop_token_ids)

    def test_trim_completion_ids_excludes_prompt_and_stops_on_multi_token_sequence(self):
        tokenizer = FakeTokenizer()
        prompt = tokenizer.encode("<|bos|>\n<|user|>\nSoru\n<|assistant|>\n")
        generated = prompt + tokenizer.encode("4<|eos|><|user|>\nYeni soru")
        stop_ids = resolve_stop_token_ids(tokenizer, ("<|eos|>", "<|user|>"))

        completion = trim_completion_token_ids(generated, prompt_token_count=len(prompt), stop_token_ids=stop_ids)

        self.assertEqual(tokenizer.decode(completion), "4")

    def test_decode_completion_from_ids_runs_output_guard_with_prompt_context(self):
        tokenizer = FakeTokenizer()
        prompt_text = "<|bos|>\n<|user|>\n2+2 kac eder?\n<|assistant|>\n"
        prompt_ids = tokenizer.encode(prompt_text)
        generated = prompt_ids + tokenizer.encode("2 + 2 kac eder? 4 <|eos|>")

        result = decode_completion_from_ids(
            tokenizer,
            generated,
            prompt_token_count=len(prompt_ids),
            prompt_text="2+2 kac eder?",
            stop_token_ids=resolve_stop_token_ids(tokenizer, ("<|eos|>",)),
        )

        self.assertEqual(result.text, "4")
        self.assertIn("prompt_echo_removed", result.warnings)
        self.assertIn("control_token_stop", result.warnings)


if __name__ == "__main__":
    unittest.main()
