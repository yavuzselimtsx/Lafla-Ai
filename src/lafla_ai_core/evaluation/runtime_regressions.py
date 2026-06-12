"""
@Dosya: evaluation/runtime_regressions.py
@Aciklama: Runtime cikis korumasi icin deterministik regresyon gate'leri.
@Yazar: Lafla Gelistirme Ekibi
@Bilgi: Colab/HF decode hatalari, prompt echo ve role-boundary sizintisi release
        asamasinda screenshot'a bakarak degil testle yakalanir.
@Uyari: Bu gate model kalitesini ikame etmez; sadece public cikis sozlesmesini
        fail-closed denetler.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Sequence

from lafla_ai_core.config.schema import RuntimeConfig
from lafla_ai_core.evaluation.gates import GateResult
from lafla_ai_core.runtime.output_guard import ROLE_STOP_SEQUENCES
from lafla_ai_core.runtime.policy import render_runtime_output
from lafla_ai_core.tokenizer.quality import has_mojibake


_BYTELEVEL_SURFACE_MARKERS = ("\u0120", "\u010a", "\u00c4\u00a0", "\u00c4\u0160")


@dataclass(frozen=True)
class RuntimeRegressionCase:
    """Tek runtime public-output regresyon kaydi."""

    name: str
    raw_text: str
    expected_public_text: str
    prompt_text: str | None = None
    system_text: str | None = None
    required_warnings: tuple[str, ...] = ()
    forbidden_fragments: tuple[str, ...] = ROLE_STOP_SEQUENCES


def evaluate_runtime_regression_case(case: RuntimeRegressionCase, config: RuntimeConfig) -> GateResult:
    """Runtime regression case'i GateResult formatinda degerlendirir."""

    output = render_runtime_output(case.raw_text, config, prompt_text=case.prompt_text, system_text=case.system_text)
    failures: list[str] = []
    if output.public_text != case.expected_public_text:
        failures.append(f"expected_public_text_mismatch:{output.public_text!r}")
    for warning in case.required_warnings:
        if warning not in output.warnings:
            failures.append(f"missing_warning:{warning}")
    for fragment in case.forbidden_fragments:
        if fragment and fragment in output.public_text:
            failures.append(f"forbidden_fragment:{fragment}")
    if _has_bytelevel_surface(output.public_text):
        failures.append("bytelevel_surface_visible")
    if has_mojibake(output.public_text):
        failures.append("mojibake_visible")
    if failures:
        return GateResult(name=case.name, passed=False, detail=";".join(failures))
    return GateResult(name=case.name, passed=True, detail=f"warnings:{','.join(output.warnings) or 'none'}")


def evaluate_runtime_regression_suite(
    cases: Iterable[RuntimeRegressionCase],
    config: RuntimeConfig,
) -> tuple[GateResult, ...]:
    """Birden fazla runtime regression case'ini deterministik olarak kosar."""

    return tuple(evaluate_runtime_regression_case(case, config) for case in cases)


def _has_bytelevel_surface(text: str) -> bool:
    return any(marker in text for marker in _BYTELEVEL_SURFACE_MARKERS)
