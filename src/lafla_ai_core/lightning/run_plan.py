"""
@Dosya: lightning/run_plan.py
@Aciklama: Lightning.ai H200 uzerinde LaflaGPT egitimi icin tekrar
            uretilebilir komut plani olusturur.
@Yazar: Lafla Gelistirme Ekibi
@Bilgi: Plan, repo zipi yuklendikten sonra preflight, gercek veri denetimi,
        tokenizer, egitim ve artifact adimlarini tek siraya koyar.
@Uyari: Plan sahte veri uretmez; data_jsonl gercek veri dosyasi olmali.
"""

from __future__ import annotations

import json
import shlex
from dataclasses import asdict, dataclass
from pathlib import PurePosixPath


@dataclass(frozen=True)
class LightningPaths:
    """Lightning.ai calisma yollarini tasir."""

    repo_dir: str = "/teamspace/studios/this_studio/LaflaAi-Core"
    workspace_root: str = "/teamspace/studios/this_studio/LaflaAI380M"
    artifact_name: str = "lafla-380m-thinking-h200-50000-step-run.tar.gz"
    checkpoint_dir: str = "/teamspace/studios/this_studio/LaflaAI380M/checkpoints"
    tokenizer_path: str = "/teamspace/studios/this_studio/LaflaAI380M/tokenizer/lafla-tokenizer.json"
    tokenizer_report_path: str = "/teamspace/studios/this_studio/LaflaAI380M/reports/tokenizer-quality.json"
    model_config: str = "configs/model/lafla-380m-thinking.yaml"
    training_config: str = "configs/training/lightning/lightning-h200-380m-50000.yaml"
    runtime_config: str = "configs/runtime/desktop-phone-fp16-380m.yaml"
    identity_data: str = "configs/data/identity/lafla-model-identity-380m.jsonl"
    model_name: str = "lafla-380m-thinking"

    @classmethod
    def for_workspace(
        cls,
        *,
        repo_dir: str = "/teamspace/studios/this_studio/LaflaAi-Core",
        workspace_root: str = "/teamspace/studios/this_studio/LaflaAI380M",
        artifact_name: str = "lafla-380m-thinking-h200-50000-step-run.tar.gz",
    ) -> "LightningPaths":
        """Workspace kokunden tum Lightning artifact yollarini turetir."""

        root = PurePosixPath(workspace_root)
        return cls(
            repo_dir=repo_dir,
            workspace_root=workspace_root,
            artifact_name=artifact_name,
            checkpoint_dir=str(root / "checkpoints"),
            tokenizer_path=str(root / "tokenizer" / "lafla-tokenizer.json"),
            tokenizer_report_path=str(root / "reports" / "tokenizer-quality.json"),
        )


@dataclass(frozen=True)
class LightningRunPlan:
    """Lightning komut planini tasir."""

    commands: tuple[str, ...]

    def to_json(self) -> str:
        """Komut planini JSON olarak dondurur."""

        return json.dumps(asdict(self), ensure_ascii=False, indent=2)


def build_lightning_run_plan(
    paths: LightningPaths,
    data_jsonl: str,
    manifest_path: str = "/teamspace/studios/this_studio/LaflaAI380M/data/veri_manifesti.json",
    thinking_jsonl: str | None = None,
    resume_from: str | None = None,
) -> LightningRunPlan:
    """Preflight, data audit, tokenizer, egitim ve artifact adimlari uretir."""

    workspace_root = PurePosixPath(paths.workspace_root)
    if not str(workspace_root).startswith("/teamspace/"):
        raise ValueError("Lightning workspace yolu /teamspace altinda olmali")
    repo = _q(paths.repo_dir)
    data = _q(data_jsonl)
    identity_data = _q(paths.identity_data)
    manifest = _q(manifest_path)
    reports_dir = str(workspace_root / "reports")
    hf_dir = str(workspace_root / "hf-package")
    final_dir = str(workspace_root / "final-checkpoint")
    model_config = _q(paths.model_config)
    training_config = _q(paths.training_config)
    train_command = (
        f"cd {repo} && PYTHONPATH=src python -m lafla_ai_core.cli.train_pretrain "
        f"--model-config {model_config} "
        f"--training-config {training_config} "
        f"--tokenizer-path {_q(paths.tokenizer_path)} "
        f"--checkpoint-dir {_q(paths.checkpoint_dir)} "
        f"--health-log {_q(str(workspace_root / 'reports' / 'train-health.jsonl'))} "
        f"--data-jsonl {identity_data} "
        f"--data-jsonl {data}"
    )
    if resume_from is not None:
        train_command += f" --resume-from {_q(resume_from)}"
    commands = [
        f"cd {repo} && python -m pip install -r requirements/colab.txt",
        f"cd {repo} && PYTHONPATH=src python -m lafla_ai_core.cli.check_environment --optimizer adamw",
        f"cd {repo} && PYTHONPATH=src python -m lafla_ai_core.cli.quality_scan --root .",
        (
            f"cd {repo} && PYTHONPATH=src python -m lafla_ai_core.cli.preflight "
            f"{model_config} "
            f"{training_config} "
            "configs/tokenizer/turkish-thinking-bpe.yaml "
            f"{_q(paths.runtime_config)} "
            "configs/post_training/lafla-thinking-sft.yaml"
        ),
        f"mkdir -p {_q(str(workspace_root / 'tokenizer'))} {_q(reports_dir)} {_q(hf_dir)} {_q(paths.checkpoint_dir)}",
        f"cd {repo} && PYTHONPATH=src python -m lafla_ai_core.cli.data_audit --manifest {manifest} --report {_q(str(workspace_root / 'reports' / 'data-audit.json'))}",
        f"cd {repo} && PYTHONPATH=src python -m lafla_ai_core.cli.tokenizer_train "
        f"--config configs/tokenizer/turkish-thinking-bpe.yaml "
        f"--output {_q(paths.tokenizer_path)} "
        f"--report {_q(paths.tokenizer_report_path)} "
        f"{identity_data} {data}",
        (
            f"cd {repo} && PYTHONPATH=src python -m lafla_ai_core.cli.hf_package "
            f"--tokenizer-json {_q(paths.tokenizer_path)} "
            f"--output-dir {_q(hf_dir)} "
            f"--model-config {model_config} "
            f"--model-name {_q(paths.model_name)}"
        ),
    ]
    if thinking_jsonl is not None:
        commands.append(
            f"cd {repo} && PYTHONPATH=src python -m lafla_ai_core.cli.validate_thinking_sft "
            f"--input {_q(thinking_jsonl)} --report {_q(str(workspace_root / 'reports' / 'thinking-sft-audit.json'))}"
        )
    commands.extend(
        [
            train_command,
            (
                f"cd {repo} && PYTHONPATH=src python -m lafla_ai_core.cli.artifact_manifest "
                f"--root {_q(str(workspace_root))} "
                f"--output {_q(str(workspace_root / 'reports' / 'artifact-manifest.json'))}"
            ),
            f"mkdir -p {_q(final_dir)}",
            (
                f"tar -czf {_q(str(workspace_root / 'final-checkpoint' / paths.artifact_name))} "
                f"-C {_q(paths.checkpoint_dir)} {_q('lafla-final')} "
                f"-C {_q(str(workspace_root))} {_q('tokenizer')} {_q('reports')} {_q('hf-package')}"
            ),
            f"sync && ls -lh {_q(final_dir)}",
        ]
    )
    return LightningRunPlan(commands=tuple(commands))


def _q(value: str) -> str:
    """Shell argumanini quote eder."""

    return shlex.quote(value)
