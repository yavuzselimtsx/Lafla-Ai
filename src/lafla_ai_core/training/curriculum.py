"""
@Dosya: training/curriculum.py
@Aciklama: Toplam gorulen token sayisindan aktif sequence curriculum asamasini cozer.
"""

from __future__ import annotations

from dataclasses import dataclass

from lafla_ai_core.config.schema import TrainingConfig


@dataclass(frozen=True)
class CurriculumStage:
    index: int
    sequence_length: int
    start_token: int
    next_start_token: int | None


def resolve_curriculum_stage(config: TrainingConfig, cumulative_tokens: int) -> CurriculumStage:
    if cumulative_tokens < 0:
        raise ValueError("cumulative_tokens negatif olamaz")
    if not config.sequence_curriculum:
        return CurriculumStage(0, config.sequence_length, 0, None)
    selected = 0
    for index, boundary in enumerate(config.curriculum_token_boundaries):
        if cumulative_tokens < boundary:
            break
        selected = index
    next_boundary = (
        config.curriculum_token_boundaries[selected + 1]
        if selected + 1 < len(config.curriculum_token_boundaries)
        else None
    )
    return CurriculumStage(
        index=selected,
        sequence_length=config.sequence_curriculum[selected],
        start_token=config.curriculum_token_boundaries[selected],
        next_start_token=next_boundary,
    )


def tokens_per_optimizer_step(
    config: TrainingConfig,
    stage: CurriculumStage,
    *,
    micro_batch_size: int | None = None,
    gradient_accumulation_steps: int | None = None,
) -> int:
    active_micro_batch_size = config.micro_batch_size if micro_batch_size is None else micro_batch_size
    active_accumulation_steps = (
        config.gradient_accumulation_steps
        if gradient_accumulation_steps is None
        else gradient_accumulation_steps
    )
    return stage.sequence_length * active_micro_batch_size * active_accumulation_steps
