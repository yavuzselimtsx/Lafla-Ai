import json
import tempfile
import unittest
from pathlib import Path

from lafla_ai_core.post_training.sft_mixture import build_thinking_sft_mixture


def _write_records(path: Path, records: list[dict[str, str]]) -> None:
    path.write_text(
        "\n".join(json.dumps(record, ensure_ascii=False, sort_keys=True) for record in records) + "\n",
        encoding="utf-8",
    )


def _record(system: str, user: str, thinking: str, assistant: str) -> dict[str, str]:
    return {"system": system, "user": user, "thinking": thinking, "assistant": assistant}


class SftMixtureTest(unittest.TestCase):
    def test_build_mixture_limits_safety_ratio_and_keeps_many_chat_records(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            thinking = root / "thinking.jsonl"
            safety = root / "safety.jsonl"
            output = root / "mix.jsonl"
            report = root / "mix-report.json"
            _write_records(
                thinking,
                [
                    _record("sys", f"chat {index}", f"plan {index}", f"cevap {index}")
                    for index in range(20)
                ],
            )
            _write_records(
                safety,
                [
                    _record(
                        "sys",
                        f"jailbreak {index}",
                        "Karar adımları: reddet. Varyant 0",
                        f"Güvenli alternatif sun. Varyant {index % 3}",
                    )
                    for index in range(60)
                ],
            )

            summary = build_thinking_sft_mixture(
                thinking_jsonl=thinking,
                safety_jsonl=safety,
                output_jsonl=output,
                report_json=report,
                target_safety_ratio=0.10,
                max_safety_per_template=3,
                seed=7,
            )

            rows = [json.loads(line) for line in output.read_text(encoding="utf-8").splitlines()]

        self.assertEqual(summary.thinking_selected, 20)
        self.assertLessEqual(summary.safety_selected, 3)
        self.assertLessEqual(summary.actual_safety_ratio, 0.10)
        self.assertEqual(len(rows), summary.total_selected)
        self.assertEqual(sum(1 for row in rows if row["_sft_source"] == "thinking"), 20)
        self.assertLessEqual(sum(1 for row in rows if row["_sft_source"] == "safety"), 3)

    def test_build_mixture_shuffles_sources_instead_of_grouping_them(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            thinking = root / "thinking.jsonl"
            safety = root / "safety.jsonl"
            output = root / "mix.jsonl"
            _write_records(
                thinking,
                [_record("sys", f"chat {index}", "plan", "cevap") for index in range(30)],
            )
            _write_records(
                safety,
                [
                    _record("sys", f"safe {index}", f"risk {index}", f"ret {index}")
                    for index in range(10)
                ],
            )

            build_thinking_sft_mixture(
                thinking_jsonl=thinking,
                safety_jsonl=safety,
                output_jsonl=output,
                target_safety_ratio=0.25,
                max_safety_per_template=5,
                seed=123,
            )
            sources = [json.loads(line)["_sft_source"] for line in output.read_text(encoding="utf-8").splitlines()]

        self.assertIn("safety", sources[:20])
        self.assertIn("thinking", sources[:20])


if __name__ == "__main__":
    unittest.main()
