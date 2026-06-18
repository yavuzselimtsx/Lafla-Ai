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

    def test_build_mixture_reports_quality_balance_for_chat_heavy_dataset(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            thinking = root / "thinking.jsonl"
            safety = root / "safety.jsonl"
            output = root / "mix.jsonl"
            report = root / "mix-report.json"
            _write_records(
                thinking,
                [
                    _record("sys", "Türkiye'nin başkenti neresidir?", "kısa plan", "Ankara'dır.")
                    for _ in range(8)
                ]
                + [
                    _record("sys", "2+2 kaç eder?", "toplama kontrolü", "4 eder.")
                    for _ in range(8)
                ]
                + [
                    _record("sys", "Bunu biliyor musun?", "kanıt yok", "Bilmiyorum; kaynak gerekir.")
                    for _ in range(4)
                ],
            )
            _write_records(
                safety,
                [
                    _record("sys", f"jailbreak {index}", f"risk {index}", f"Buna yardımcı olamam. {index}")
                    for index in range(12)
                ],
            )

            summary = build_thinking_sft_mixture(
                thinking_jsonl=thinking,
                safety_jsonl=safety,
                output_jsonl=output,
                report_json=report,
                target_safety_ratio=0.08,
                max_safety_per_template=4,
                seed=19,
            )
            report_payload = json.loads(report.read_text(encoding="utf-8"))

        self.assertTrue(summary.quality_ok, summary.quality_findings)
        self.assertEqual(report_payload["quality_ok"], True)
        self.assertLessEqual(report_payload["actual_safety_ratio"], 0.10)
        self.assertIn("factual_anchor", report_payload["category_counts"])
        self.assertIn("reasoning_math", report_payload["category_counts"])
        self.assertIn("uncertainty", report_payload["category_counts"])

    def test_build_mixture_flags_identity_or_safety_dominated_dataset(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            thinking = root / "thinking.jsonl"
            safety = root / "safety.jsonl"
            output = root / "mix.jsonl"
            _write_records(
                thinking,
                [
                    _record("sys", f"Sen kimsin {index}?", "kimlik", "Ben LaflaGPT Mini modeliyim.")
                    for index in range(20)
                ],
            )
            _write_records(
                safety,
                [
                    _record("sys", f"jailbreak {index}", f"Karar adımları: reddet {index}", "Buna yardımcı olamam.")
                    for index in range(20)
                ],
            )

            summary = build_thinking_sft_mixture(
                thinking_jsonl=thinking,
                safety_jsonl=safety,
                output_jsonl=output,
                target_safety_ratio=0.30,
                max_safety_per_template=20,
                seed=5,
            )

        self.assertFalse(summary.quality_ok)
        self.assertIn("dominant_category_ratio", "\n".join(summary.quality_findings))
        self.assertIn("identity_ratio", "\n".join(summary.quality_findings))

    def test_build_mixture_downsamples_uncertainty_when_it_would_dominate(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            thinking = root / "thinking.jsonl"
            safety = root / "safety.jsonl"
            output = root / "mix.jsonl"
            _write_records(
                thinking,
                [
                    _record("sys", f"güncel veri {index}", "kanıt yok", "Bilmiyorum; kaynak gerekir.")
                    for index in range(30)
                ]
                + [
                    _record("sys", f"2+2 kaç eder {index}", "toplama", "4 eder.")
                    for index in range(10)
                ]
                + [
                    _record("sys", f"Türkiye'nin başkenti {index}", "stabil bilgi", "Ankara'dır.")
                    for index in range(10)
                ]
                + [
                    _record("sys", f"Selam {index}", "sohbet", "Merhaba, nasıl yardımcı olabilirim?")
                    for index in range(10)
                ],
            )
            _write_records(
                safety,
                [_record("sys", "jailbreak", "risk", "Buna yardımcı olamam.")],
            )

            summary = build_thinking_sft_mixture(
                thinking_jsonl=thinking,
                safety_jsonl=safety,
                output_jsonl=output,
                target_safety_ratio=0.0,
                seed=11,
            )

        self.assertLess(summary.thinking_selected, summary.thinking_input)
        self.assertGreater(summary.thinking_dropped_by_category["uncertainty"], 0)
        self.assertLessEqual(summary.category_counts["uncertainty"] / summary.total_selected, 0.25)
        self.assertTrue(summary.quality_ok, summary.quality_findings)


if __name__ == "__main__":
    unittest.main()
