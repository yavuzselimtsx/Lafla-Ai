"""
@Dosya: training/phase_plan.py
@Aciklama: Lafla model aileleri icin asamali egitim ve gate sozlesmesi.
@Yazar: Lafla Gelistirme Ekibi
@Bilgi: OLMo cok asamali pretrain/anneal/SFT cizgisi ve GPT-NeoX chat-template
        loss-mask disiplini tek Lafla planinda toplanir.
@Uyari: Bu plan veri veya metrik uydurmaz; eksik asama/gate varsa release'i durdurur.
@Calisma-Semasi: phase plan -> validation findings -> training/release decision
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Iterable


POST_TRAINING_OBJECTIVES = {"instruction_sft", "thinking_sft", "dpo", "preference_tuning"}
ASSISTANT_LOSS_POLICIES = {"assistant_only", "assistant_with_thinking", "last_assistant_only"}
RELEASE_RUNTIME_GATES = (
    "completion_only_generation",
    "prompt_echo_guard",
    "mojibake_decode",
    "role_boundary_stop",
    "low_information_generation",
)


@dataclass(frozen=True)
class TrainingPhase:
    """Tek egitim/eval asamasinin sozlesmesini tasir."""

    name: str
    order: int
    objective: str
    data_usage: tuple[str, ...]
    label_policy: str
    required_gates: tuple[str, ...]
    chat_template_required: bool
    checkpoint_required: bool
    export_after: bool

    def replace(self, **changes: object) -> "TrainingPhase":
        """Immutable phase kopyasi uretir."""

        return replace(self, **changes)


@dataclass(frozen=True)
class TrainingPhasePlan:
    """Lafla egitim asamalarinin sirali planini tasir."""

    name: str
    phases: tuple[TrainingPhase, ...]

    def replace(self, **changes: object) -> "TrainingPhasePlan":
        """Immutable plan kopyasi uretir."""

        return replace(self, **changes)


@dataclass(frozen=True)
class PhasePlanFinding:
    """Phase plan dogrulama bulgusu."""

    code: str
    detail: str


@dataclass(frozen=True)
class PhasePlanReport:
    """Phase plan dogrulama raporu."""

    ok: bool
    findings: tuple[PhasePlanFinding, ...]


def default_lafla_400m_thinking_plan() -> TrainingPhasePlan:
    """Lafla 400M Thinking icin varsayilan asamali egitim planini dondurur."""

    return TrainingPhasePlan(
        name="lafla-400m-thinking-clean-room-v1",
        phases=(
            TrainingPhase(
                name="tokenizer_audit",
                order=0,
                objective="tokenizer_quality",
                data_usage=("pretraining", "dialogue_sft", "thinking_sft"),
                label_policy="none",
                required_gates=("tokenizer_roundtrip", "turkish_encoding", "special_token_contract"),
                chat_template_required=True,
                checkpoint_required=False,
                export_after=True,
            ),
            TrainingPhase(
                name="base_pretrain",
                order=1,
                objective="causal_lm_pretrain",
                data_usage=("pretraining",),
                label_policy="all_tokens",
                required_gates=("tokenizer_roundtrip", "data_license", "pii_redaction", "checkpoint_contract"),
                chat_template_required=False,
                checkpoint_required=True,
                export_after=False,
            ),
            TrainingPhase(
                name="anneal_midtrain",
                order=2,
                objective="quality_anneal",
                data_usage=("high_quality_pretraining", "turkish_quality"),
                label_policy="all_tokens",
                required_gates=("loss_stability", "turkish_identity", "checkpoint_contract"),
                chat_template_required=False,
                checkpoint_required=True,
                export_after=False,
            ),
            TrainingPhase(
                name="instruction_sft",
                order=3,
                objective="instruction_sft",
                data_usage=("dialogue_sft",),
                label_policy="last_assistant_only",
                required_gates=("chat_template_contract", "assistant_label_mask", "prompt_echo_guard"),
                chat_template_required=True,
                checkpoint_required=True,
                export_after=False,
            ),
            TrainingPhase(
                name="thinking_sft",
                order=4,
                objective="thinking_sft",
                data_usage=("thinking_sft",),
                label_policy="assistant_with_thinking",
                required_gates=("thinking_private_runtime", "assistant_label_mask", "prompt_echo_guard"),
                chat_template_required=True,
                checkpoint_required=True,
                export_after=False,
            ),
            TrainingPhase(
                name="release_eval",
                order=5,
                objective="release_eval",
                data_usage=("truthfulness_eval", "hallucination_eval", "dialogue_eval"),
                label_policy="none",
                required_gates=(
                    "tokenizer_roundtrip",
                    "turkish_identity",
                    "hallucination_refusal",
                    "safety_boundary",
                    "low_power_runtime",
                    "artifact_manifest",
                    *RELEASE_RUNTIME_GATES,
                ),
                chat_template_required=True,
                checkpoint_required=True,
                export_after=True,
            ),
        ),
    )


def default_lafla_100m_thinking_plan() -> TrainingPhasePlan:
    """100M Turkce/Almanca uzun-baglam modeli icin birincil egitim planini dondurur."""

    legacy = default_lafla_400m_thinking_plan()
    enhanced_phases: list[TrainingPhase] = []
    for phase in legacy.phases:
        gates = phase.required_gates
        data_usage = phase.data_usage
        if phase.name == "tokenizer_audit":
            gates = (*gates, "german_quality")
        elif phase.name == "anneal_midtrain":
            gates = (*gates, "long_context_passkey", "long_context_needle")
            data_usage = (*data_usage, "german_quality", "long_context")
        elif phase.name == "instruction_sft":
            gates = (*gates, "abstention_accuracy", "false_premise_correction")
            data_usage = (*data_usage, "uncertainty_sft", "retrieval_sft", "safety_jailbreak_sft")
        elif phase.name == "thinking_sft":
            data_usage = (*data_usage, "synthetic_chat_seed", "safety_jailbreak_sft")
        elif phase.name == "release_eval":
            gates = (
                *gates,
                "german_quality",
                "abstention_accuracy",
                "jailbreak_resistance",
                "system_prompt_exfiltration_refusal",
                "unsafe_tool_request_refusal",
                "contradictory_evidence_handling",
                "stale_current_fact_refusal",
                "false_premise_correction",
                "source_faithfulness",
                "long_context_passkey",
                "long_context_needle",
                "cache_equivalence",
                "process_tree_peak_rss",
            )
            data_usage = (*data_usage, "long_context_eval", "german_eval")
        enhanced_phases.append(phase.replace(required_gates=gates, data_usage=data_usage))
    return TrainingPhasePlan(
        name="lafla-100m-thinking-clean-room-v1",
        phases=tuple(enhanced_phases),
    )


def validate_phase_plan(plan: TrainingPhasePlan) -> PhasePlanReport:
    """Egitim plani sirasi, label maskesi ve release gate'lerini dogrular."""

    findings: list[PhasePlanFinding] = []
    phases = tuple(plan.phases)
    if not phases:
        findings.append(PhasePlanFinding("phase_plan_empty", "egitim plani bos olamaz"))
        return PhasePlanReport(ok=False, findings=tuple(findings))

    orders = [phase.order for phase in phases]
    if orders != sorted(orders) or len(set(orders)) != len(orders):
        findings.append(PhasePlanFinding("phase_order_invalid", "phase order degerleri tekil ve artan olmali"))

    names = {phase.name for phase in phases}
    if "tokenizer_audit" not in names:
        findings.append(PhasePlanFinding("tokenizer_audit_missing", "tokenizer audit ilk release kapisidir"))
    if "release_eval" not in names:
        findings.append(PhasePlanFinding("release_eval_missing", "release eval asamasi zorunlu"))

    for phase in phases:
        findings.extend(_validate_phase(phase))

    release_phase = next((phase for phase in phases if phase.name == "release_eval"), None)
    if release_phase is not None:
        missing = tuple(gate for gate in RELEASE_RUNTIME_GATES if gate not in release_phase.required_gates)
        if missing:
            findings.append(PhasePlanFinding("release_runtime_gates_missing", f"eksik runtime gate: {missing}"))

    return PhasePlanReport(ok=not findings, findings=tuple(findings))


def _validate_phase(phase: TrainingPhase) -> Iterable[PhasePlanFinding]:
    findings: list[PhasePlanFinding] = []
    if phase.objective in POST_TRAINING_OBJECTIVES:
        if not phase.chat_template_required:
            findings.append(PhasePlanFinding("post_training_requires_chat_template", f"{phase.name} chat template olmadan kosamaz"))
        if phase.label_policy not in ASSISTANT_LOSS_POLICIES:
            findings.append(
                PhasePlanFinding("post_training_requires_assistant_loss_mask", f"{phase.name} prompt/system tokenlarini loss'a sokmamali")
            )
    if (
        phase.checkpoint_required
        and "checkpoint_contract" not in phase.required_gates
        and phase.objective not in POST_TRAINING_OBJECTIVES | {"release_eval"}
    ):
        findings.append(PhasePlanFinding("checkpoint_gate_missing", f"{phase.name} checkpoint sozlesmesi gate'i tasimali"))
    if not phase.required_gates:
        findings.append(PhasePlanFinding("required_gates_empty", f"{phase.name} gate listesi bos olamaz"))
    return findings
