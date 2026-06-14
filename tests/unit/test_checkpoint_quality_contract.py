import contextlib
import io
import json
import unittest
from dataclasses import dataclass
from unittest.mock import patch

from lafla_ai_core.cli.test_checkpoint import main as checkpoint_cli_main
from lafla_ai_core.config.loader import load_mapping
from lafla_ai_core.runtime.checkpoint_inference import (
    BLOCKING_CHECKPOINT_WARNINGS,
    CheckpointGenerationResult,
    TokenizersGenerationAdapter,
    assess_checkpoint_generation_quality,
)
from lafla_ai_core.training.phase_plan import default_lafla_400m_thinking_plan


@dataclass(frozen=True)
class FakeCliResult:
    public_text: str
    warnings: tuple[str, ...]
    quality_ok: bool
    blocking_warnings: tuple[str, ...]

    def to_json(self) -> str:
        return json.dumps(
            {
                "public_text": self.public_text,
                "warnings": self.warnings,
                "quality_ok": self.quality_ok,
                "blocking_warnings": self.blocking_warnings,
            },
            ensure_ascii=False,
            sort_keys=True,
        )


class RecordingTokenizer:
    def __init__(self) -> None:
        self.encode_calls: list[tuple[str, bool]] = []

    def encode(self, text: str, add_special_tokens: bool = True):
        self.encode_calls.append((text, add_special_tokens))
        payload = [ord(char) for char in text]
        if add_special_tokens:
            payload = [111_111, *payload, 222_222]
        return payload

    def decode(self, token_ids, skip_special_tokens=False):
        return "".join(chr(int(token_id)) for token_id in token_ids if int(token_id) < 111_111)


class FakeEncoding:
    def __init__(self, ids):
        self.ids = ids


class EncodingObjectTokenizer(RecordingTokenizer):
    def encode(self, text: str, add_special_tokens: bool = True):
        super().encode(text, add_special_tokens=add_special_tokens)
        return FakeEncoding([ord(char) for char in text])


class CheckpointQualityContractTest(unittest.TestCase):
    def test_clean_short_math_answer_is_quality_ok(self):
        assessment = assess_checkpoint_generation_quality("4", ())

        self.assertTrue(assessment.ok)
        self.assertEqual(assessment.blocking_warnings, ())

    def test_turkish_sentence_answer_is_quality_ok(self):
        assessment = assess_checkpoint_generation_quality("Merhaba, nasıl yardımcı olabilirim?", ())

        self.assertTrue(assessment.ok)
        self.assertEqual(assessment.detail, "ok")

    def test_low_information_warning_blocks_checkpoint(self):
        assessment = assess_checkpoint_generation_quality("----", ("low_information_completion",))

        self.assertFalse(assessment.ok)
        self.assertEqual(assessment.blocking_warnings, ("low_information_completion",))

    def test_empty_after_guard_warning_blocks_checkpoint(self):
        assessment = assess_checkpoint_generation_quality("", ("empty_after_output_guard",))

        self.assertFalse(assessment.ok)
        self.assertIn("empty_after_output_guard", assessment.blocking_warnings)

    def test_empty_public_text_blocks_without_warning(self):
        assessment = assess_checkpoint_generation_quality("", ())

        self.assertFalse(assessment.ok)
        self.assertEqual(assessment.blocking_warnings, ("empty_public_text",))

    def test_whitespace_public_text_blocks_without_warning(self):
        assessment = assess_checkpoint_generation_quality("   \n\t  ", ())

        self.assertFalse(assessment.ok)
        self.assertEqual(assessment.blocking_warnings, ("empty_public_text",))

    def test_possible_prompt_leak_blocks_checkpoint(self):
        assessment = assess_checkpoint_generation_quality("system prompt: gizli", ("possible_prompt_leak",))

        self.assertFalse(assessment.ok)
        self.assertEqual(assessment.blocking_warnings, ("possible_prompt_leak",))

    def test_possible_mojibake_blocks_checkpoint(self):
        assessment = assess_checkpoint_generation_quality("TÃ¼rkÃ§e", ("possible_mojibake",))

        self.assertFalse(assessment.ok)
        self.assertEqual(assessment.blocking_warnings, ("possible_mojibake",))

    def test_control_token_stop_is_not_blocking_when_answer_survives(self):
        assessment = assess_checkpoint_generation_quality("4", ("control_token_stop",))

        self.assertTrue(assessment.ok)
        self.assertEqual(assessment.blocking_warnings, ())

    def test_prompt_echo_removed_is_not_blocking_when_answer_survives(self):
        assessment = assess_checkpoint_generation_quality("4", ("prompt_echo_removed",))

        self.assertTrue(assessment.ok)
        self.assertEqual(assessment.blocking_warnings, ())

    def test_repaired_bytelevel_warning_is_not_blocking_when_answer_survives(self):
        assessment = assess_checkpoint_generation_quality("Türkçe cevap.", ("bytelevel_surface_repaired",))

        self.assertTrue(assessment.ok)
        self.assertEqual(assessment.blocking_warnings, ())

    def test_multiple_blocking_warnings_are_deduped_in_order(self):
        assessment = assess_checkpoint_generation_quality(
            "",
            ("low_information_completion", "empty_after_output_guard", "low_information_completion"),
        )

        self.assertFalse(assessment.ok)
        self.assertEqual(assessment.blocking_warnings, ("low_information_completion", "empty_after_output_guard"))

    def test_blocking_warning_registry_contains_low_information(self):
        self.assertIn("low_information_completion", BLOCKING_CHECKPOINT_WARNINGS)
        self.assertIn("empty_after_output_guard", BLOCKING_CHECKPOINT_WARNINGS)

    def test_checkpoint_result_json_exposes_quality_status(self):
        result = CheckpointGenerationResult(
            checkpoint_dir="/tmp/checkpoint",
            prompt_text="2+2?",
            public_text="4",
            warnings=(),
            generated_token_count=1,
            device="cpu",
            quality_ok=True,
            blocking_warnings=(),
        )

        payload = json.loads(result.to_json())

        self.assertTrue(payload["quality_ok"])
        self.assertEqual(payload["blocking_warnings"], [])

    def test_checkpoint_result_json_exposes_blocking_warnings(self):
        result = CheckpointGenerationResult(
            checkpoint_dir="/tmp/checkpoint",
            prompt_text="2+2?",
            public_text="",
            warnings=("low_information_completion",),
            generated_token_count=64,
            device="cpu",
            quality_ok=False,
            blocking_warnings=("low_information_completion",),
        )

        payload = json.loads(result.to_json())

        self.assertFalse(payload["quality_ok"])
        self.assertEqual(payload["blocking_warnings"], ["low_information_completion"])

    def test_checkpoint_cli_returns_nonzero_for_low_information_generation(self):
        fake = FakeCliResult("", ("low_information_completion",), False, ("low_information_completion",))
        stdout = io.StringIO()

        with patch("lafla_ai_core.cli.test_checkpoint.generate_from_checkpoint", return_value=fake):
            with contextlib.redirect_stdout(stdout):
                exit_code = checkpoint_cli_main(["--checkpoint-dir", "ckpt", "--tokenizer-path", "tokenizer.json"])

        self.assertEqual(exit_code, 2)
        self.assertFalse(json.loads(stdout.getvalue())["quality_ok"])

    def test_checkpoint_cli_returns_nonzero_for_empty_generation(self):
        fake = FakeCliResult("", (), False, ("empty_public_text",))
        stdout = io.StringIO()

        with patch("lafla_ai_core.cli.test_checkpoint.generate_from_checkpoint", return_value=fake):
            with contextlib.redirect_stdout(stdout):
                exit_code = checkpoint_cli_main(["--checkpoint-dir", "ckpt", "--tokenizer-path", "tokenizer.json"])

        self.assertEqual(exit_code, 2)
        self.assertEqual(json.loads(stdout.getvalue())["blocking_warnings"], ["empty_public_text"])

    def test_checkpoint_cli_returns_zero_for_clean_generation(self):
        fake = FakeCliResult("4", (), True, ())
        stdout = io.StringIO()

        with patch("lafla_ai_core.cli.test_checkpoint.generate_from_checkpoint", return_value=fake):
            with contextlib.redirect_stdout(stdout):
                exit_code = checkpoint_cli_main(["--checkpoint-dir", "ckpt", "--tokenizer-path", "tokenizer.json"])

        self.assertEqual(exit_code, 0)
        self.assertTrue(json.loads(stdout.getvalue())["quality_ok"])

    def test_checkpoint_cli_passes_runtime_config_for_developer_diagnostics(self):
        fake = FakeCliResult("<|user|>ham", ("safety_filters_disabled",), True, ())
        stdout = io.StringIO()

        with patch("lafla_ai_core.cli.test_checkpoint.generate_from_checkpoint", return_value=fake) as generate:
            with contextlib.redirect_stdout(stdout):
                exit_code = checkpoint_cli_main(
                    [
                        "--checkpoint-dir",
                        "ckpt",
                        "--tokenizer-path",
                        "tokenizer.json",
                        "--runtime-config",
                        "configs/runtime/developer-unguarded.yaml",
                    ]
                )

        self.assertEqual(exit_code, 0)
        runtime_config = generate.call_args.kwargs["runtime_config"]
        self.assertEqual(runtime_config.safety_profile, "off")
        self.assertTrue(json.loads(stdout.getvalue())["quality_ok"])

    def test_tokenizers_generation_adapter_disables_special_tokens_for_prompt_encoding(self):
        tokenizer = RecordingTokenizer()
        adapter = TokenizersGenerationAdapter(tokenizer)

        ids = adapter.encode("<|assistant|>\n")

        self.assertNotIn(111_111, ids)
        self.assertEqual(tokenizer.encode_calls, [("<|assistant|>\n", False)])

    def test_tokenizers_generation_adapter_supports_encoding_object_shape_without_special_tokens(self):
        tokenizer = EncodingObjectTokenizer()
        adapter = TokenizersGenerationAdapter(tokenizer)

        ids = adapter.encode("4<|eos|>")

        self.assertEqual(ids, tuple(ord(char) for char in "4<|eos|>"))
        self.assertEqual(tokenizer.encode_calls, [("4<|eos|>", False)])

    def test_release_gate_config_requires_low_information_generation_gate(self):
        gates = load_mapping("configs/evaluation/release-gates.yaml")["evaluation"]["required_gates"]

        self.assertIn("low_information_generation", gates)

    def test_phase_plan_release_eval_requires_low_information_generation_gate(self):
        plan = default_lafla_400m_thinking_plan()
        release_phase = next(phase for phase in plan.phases if phase.name == "release_eval")

        self.assertIn("low_information_generation", release_phase.required_gates)


if __name__ == "__main__":
    unittest.main()
