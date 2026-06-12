import unittest

from lafla_ai_core.training.phase_plan import (
    TrainingPhase,
    default_lafla_100m_thinking_plan,
    default_lafla_400m_thinking_plan,
    validate_phase_plan,
)


class TrainingPhasePlanTest(unittest.TestCase):
    def test_100m_plan_is_primary_and_requires_long_context_uncertainty_and_rss_gates(self):
        plan = default_lafla_100m_thinking_plan()
        release = next(phase for phase in plan.phases if phase.name == "release_eval")

        self.assertEqual(plan.name, "lafla-100m-thinking-clean-room-v1")
        for gate in (
            "german_quality",
            "abstention_accuracy",
            "source_faithfulness",
            "jailbreak_resistance",
            "system_prompt_exfiltration_refusal",
            "unsafe_tool_request_refusal",
            "long_context_passkey",
            "cache_equivalence",
            "process_tree_peak_rss",
        ):
            self.assertIn(gate, release.required_gates)
        self.assertTrue(validate_phase_plan(plan).ok)

    def test_default_plan_separates_pretrain_sft_and_release_eval(self):
        plan = default_lafla_400m_thinking_plan()
        names = [phase.name for phase in plan.phases]

        self.assertEqual(names[0], "tokenizer_audit")
        self.assertIn("base_pretrain", names)
        self.assertIn("anneal_midtrain", names)
        self.assertIn("instruction_sft", names)
        self.assertIn("thinking_sft", names)
        self.assertEqual(names[-1], "release_eval")
        self.assertTrue(validate_phase_plan(plan).ok)

    def test_post_training_phase_requires_chat_template_and_loss_mask(self):
        phase = TrainingPhase(
            name="bad_sft",
            order=3,
            objective="instruction_sft",
            data_usage=("dialogue_sft",),
            label_policy="all_tokens",
            required_gates=("tokenizer_roundtrip",),
            chat_template_required=False,
            checkpoint_required=True,
            export_after=False,
        )
        plan = default_lafla_400m_thinking_plan().replace(phases=(phase,))

        report = validate_phase_plan(plan)

        self.assertFalse(report.ok)
        self.assertIn("post_training_requires_chat_template", [finding.code for finding in report.findings])
        self.assertIn("post_training_requires_assistant_loss_mask", [finding.code for finding in report.findings])

    def test_release_eval_requires_prompt_echo_and_decode_gates(self):
        plan = default_lafla_400m_thinking_plan()
        release_phase = next(phase for phase in plan.phases if phase.name == "release_eval")

        self.assertIn("completion_only_generation", release_phase.required_gates)
        self.assertIn("prompt_echo_guard", release_phase.required_gates)
        self.assertIn("mojibake_decode", release_phase.required_gates)
        self.assertIn("role_boundary_stop", release_phase.required_gates)


if __name__ == "__main__":
    unittest.main()
