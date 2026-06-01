"""
Lafla AI eğitim aşamalarını bütçe ve kalite kapılarına göre sıralar.

Planlayıcı checkpoint üretmez; eğitim koşusunun neyi, hangi sırayla ve hangi
asgari ölçütle çalıştıracağını belirler. Bu ayrım Colab betiğinin büyüyüp
kontrolsüz bir dosyaya dönüşmesini engeller.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class TrainingStage:
    """Tek eğitim aşamasının çalıştırılabilir tanımı."""

    batch_tokens: int
    checkpoint_every_steps: int
    name: str
    objective: str
    quality_gate: str
    steps: int

    def validate(self) -> None:
        if self.batch_tokens < 1024:
            raise ValueError(f"{self.name}: batch_tokens is too small")
        if self.steps <= 0:
            raise ValueError(f"{self.name}: steps must be positive")
        if self.checkpoint_every_steps <= 0:
            raise ValueError(f"{self.name}: checkpoint interval must be positive")
        if self.steps < self.checkpoint_every_steps:
            raise ValueError(f"{self.name}: checkpoint interval is larger than stage")
        if not self.quality_gate.strip():
            raise ValueError(f"{self.name}: quality gate is empty")


class TrainingPlan:
    """Ön eğitimden DPO'ya kadar aşamaları tutar ve doğrular."""

    def __init__(self, stages: list[TrainingStage]) -> None:
        self.stages = stages

    def validate(self) -> None:
        if len(self.stages) < 4:
            raise ValueError("training plan must include pretrain, sft, dpo, and evaluation")
        seen: set[str] = set()
        for stage in self.stages:
            if stage.name in seen:
                raise ValueError(f"duplicate training stage: {stage.name}")
            seen.add(stage.name)
            stage.validate()

    def total_steps(self) -> int:
        self.validate()
        return sum(stage.steps for stage in self.stages)


def default_training_plan() -> TrainingPlan:
    """Colab ilk sürümüne uygun ama üretime büyütülebilir aşama planı."""

    return TrainingPlan(
        stages=[
            TrainingStage(
                batch_tokens=131_072,
                checkpoint_every_steps=500,
                name="ön_eğitim",
                objective="causal_language_modeling",
                quality_gate="perplexity_tr_downward",
                steps=4_000,
            ),
            TrainingStage(
                batch_tokens=65_536,
                checkpoint_every_steps=250,
                name="talimat_uyumu",
                objective="supervised_instruction_tuning",
                quality_gate="turkish_instruction_pass_rate",
                steps=1_200,
            ),
            TrainingStage(
                batch_tokens=32_768,
                checkpoint_every_steps=200,
                name="tercih_optimizasyonu",
                objective="direct_preference_optimization",
                quality_gate="chosen_answer_win_rate",
                steps=800,
            ),
            TrainingStage(
                batch_tokens=16_384,
                checkpoint_every_steps=100,
                name="değerlendirme",
                objective="offline_eval_and_red_team",
                quality_gate="safety_and_identity_no_regression",
                steps=100,
            ),
        ]
    )
