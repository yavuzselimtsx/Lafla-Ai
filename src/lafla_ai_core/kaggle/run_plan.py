"""
@Dosya: kaggle/run_plan.py
@Aciklama: Kaggle GPU uzerinde LaflaAi-Core egitimi icin tekrar uretilebilir
            komut plani olusturur.
@Yazar: Lafla Gelistirme Ekibi
@Bilgi: Birden fazla CUDA cihazi torchrun/DDP ile, tek CUDA cihazi normal
        Python runner ile kullanilir.
@Uyari: Gercek veri ve manifest olmadan egitim komutu uretilen planda durur.
@Calisma-Semasi: paths -> validate -> commands -> KaggleRunPlan
"""

from __future__ import annotations

import json
import shlex
from dataclasses import asdict, dataclass
from pathlib import PurePosixPath
from typing import Sequence

from lafla_ai_core.colab.run_plan import DEFAULT_THINKING_SFT_DATA, _merge_thinking_sft_inputs, _post_training_manifest_command


@dataclass(frozen=True)
class KagglePaths:
    """Kaggle calisma yollarini tasir."""

    repo_dir: str = "/kaggle/working/LaflaAi-Core"
    work_root: str = "/kaggle/working/LaflaAI100M"
    artifact_name: str = "lafla-100m-thinking-kaggle-gpu-run.tar.gz"
    checkpoint_dir: str = "/kaggle/working/LaflaAI100M/checkpoints"
    tokenizer_path: str = "/kaggle/working/LaflaAI100M/tokenizer/lafla-tokenizer.json"
    tokenizer_report_path: str = "/kaggle/working/LaflaAI100M/reports/tokenizer-quality.json"


@dataclass(frozen=True)
class KaggleTrainingProfile:
    """Aktif Kaggle model ailesinin config ve kimlik yollarini tasir."""

    model_config: str = "configs/model/lafla-100m-thinking.yaml"
    training_config: str = "configs/training/kaggle/kaggle-gpu-100m.yaml"
    tokenizer_config: str = "configs/tokenizer/turkish-german-thinking-bpe.yaml"
    runtime_config: str = "configs/runtime/desktop-i3-int8-100m.yaml"
    post_training_config: str = "configs/post_training/lafla-thinking-sft.yaml"
    identity_data: str = "configs/data/identity/lafla-model-identity-100m.jsonl"
    source_plan: str = "configs/data/source-plans/lafla-100m-source-plan.json"
    thinking_sft_data: tuple[str, ...] = DEFAULT_THINKING_SFT_DATA
    model_name: str = "lafla-100m-thinking"


@dataclass(frozen=True)
class KaggleRunPlan:
    """Kaggle komut planini tasir."""

    commands: tuple[str, ...]

    def to_json(self) -> str:
        """Komut planini JSON olarak dondurur."""

        return json.dumps(asdict(self), ensure_ascii=False, indent=2)


def build_kaggle_run_plan(
    paths: KagglePaths,
    data_jsonl: str,
    manifest_path: str = "/kaggle/working/LaflaAI100M/data/veri_manifesti.json",
    thinking_jsonl: str | None = None,
    thinking_jsonls: Sequence[str] | None = None,
    profile: KaggleTrainingProfile | None = None,
) -> KaggleRunPlan:
    """Kaggle GPU icin preflight, tokenizer, egitim ve artifact komutlarini uretir."""

    profile = profile or KaggleTrainingProfile()
    thinking_inputs = _merge_thinking_sft_inputs(profile.thinking_sft_data, thinking_jsonl, thinking_jsonls)
    repo = _q(paths.repo_dir)
    data = _q(data_jsonl)
    manifest = _q(manifest_path)
    identity_data = _q(profile.identity_data)
    work_root = PurePosixPath(paths.work_root)
    data_dir = str(work_root / "data")
    tokenizer_dir = str(work_root / "tokenizer")
    reports_dir = str(work_root / "reports")
    hf_package_dir = str(work_root / "hf-package")
    archives_dir = str(work_root / "archives")
    env = "PYTHONPATH=src TOKENIZERS_PARALLELISM=true"
    commands = [
        f"mkdir -p {_q(data_dir)} {_q(tokenizer_dir)} {_q(paths.checkpoint_dir)} {_q(reports_dir)} {_q(hf_package_dir)} {_q(archives_dir)}",
        f"test -d {repo} || {{ echo 'Repo bulunamadi: {paths.repo_dir}' >&2; exit 2; }}",
        (
            f"test -s {data} && test -s {manifest} && "
            f"cd {repo} && test -s {_q(profile.source_plan)} || "
            "{ echo 'real data, manifest veya source plan eksik; egitim reddedildi' >&2; exit 2; }"
        ),
        f"cd {repo} && python -m pip install --upgrade pip wheel setuptools",
        f"cd {repo} && python -m pip install -r requirements/kaggle-gpu.txt",
        (
            f"cd {repo} && {env} python - <<'PY'\n"
            "import torch\n"
            "print('cuda_available=', torch.cuda.is_available())\n"
            "print('cuda_device_count=', torch.cuda.device_count())\n"
            "for index in range(torch.cuda.device_count()):\n"
            "    print(f'cuda:{index}=', torch.cuda.get_device_name(index))\n"
            "PY"
        ),
        f"cd {repo} && {env} python -m lafla_ai_core.cli.check_environment --optimizer adamw --accelerator cuda",
        f"cd {repo} && {env} python -m lafla_ai_core.cli.quality_scan --root .",
        (
            f"cd {repo} && {env} python -m lafla_ai_core.cli.preflight "
            f"{_q(profile.model_config)} "
            f"{_q(profile.training_config)} "
            f"{_q(profile.tokenizer_config)} "
            f"{_q(profile.runtime_config)} "
            f"{_q(profile.post_training_config)}"
        ),
        f"cd {repo} && {env} python -m lafla_ai_core.cli.data_audit --manifest {manifest} --report {_q(str(work_root / 'reports/data-audit.json'))}",
        (
            f"cd {repo} && {env} python -m lafla_ai_core.cli.validate_pretraining_data "
            f"--data-jsonl {identity_data} "
            f"--data-jsonl {data} "
            f"--report {_q(str(work_root / 'reports/pretraining-data-validation.json'))}"
        ),
        f"cd {repo} && {env} python -m lafla_ai_core.cli.tokenizer_train "
        f"--config {_q(profile.tokenizer_config)} "
        f"--output {_q(paths.tokenizer_path)} "
        f"--report {_q(paths.tokenizer_report_path)} "
        f"{identity_data} {data}",
        (
            f"cd {repo} && {env} python -m lafla_ai_core.cli.hf_package "
            f"--tokenizer-json {_q(paths.tokenizer_path)} "
            f"--output-dir {_q(hf_package_dir)} "
            f"--model-config {_q(profile.model_config)} "
            f"--model-name {_q(profile.model_name)}"
        ),
    ]
    for index, sft_path in enumerate(thinking_inputs, start=1):
        commands.append(
            f"cd {repo} && test -s {_q(sft_path)} && "
            f"{env} python -m lafla_ai_core.cli.validate_thinking_sft "
            f"--input {_q(sft_path)} --report {_q(str(work_root / 'reports' / f'thinking-sft-audit-{index:03d}.json'))}"
        )
    if thinking_inputs:
        commands.append(_post_training_manifest_command(repo, env, str(work_root / "reports/post-training-sft-inputs.json"), thinking_inputs))
    commands.extend(
        [
            (
                "RESUME_ARGS=(); "
                "if [ -n \"${RESUME_FROM:-}\" ]; then RESUME_ARGS+=(--resume-from \"$RESUME_FROM\"); fi; "
                "CUDA_DEVICE_COUNT=$(python -c 'import torch; print(torch.cuda.device_count())'); "
                "TRAIN_LAUNCHER=(python); "
                "if [ \"$CUDA_DEVICE_COUNT\" -ge 2 ]; then "
                "TRAIN_LAUNCHER=(torchrun --standalone --nproc_per_node \"$CUDA_DEVICE_COUNT\"); fi; "
                f"cd {repo} && {env} \"${{TRAIN_LAUNCHER[@]}}\" -m lafla_ai_core.cli.train_pretrain "
                f"--model-config {_q(profile.model_config)} "
                f"--training-config {_q(profile.training_config)} "
                f"--tokenizer-path {_q(paths.tokenizer_path)} "
                f"--checkpoint-dir {_q(paths.checkpoint_dir)} "
                f"--health-log {_q(str(work_root / 'reports/train-health.jsonl'))} "
                f"--data-jsonl {identity_data} "
                f"--data-jsonl {data} "
                "\"${RESUME_ARGS[@]}\""
            ),
            (
                f"cd {repo} && {env} python -m lafla_ai_core.cli.artifact_manifest "
                f"--root {_q(str(work_root))} "
                f"--output {_q(str(work_root / 'reports/artifact-manifest.json'))}"
            ),
            (
                f"tar -czf {_q(str(work_root / 'archives' / paths.artifact_name))} "
                f"-C {_q(str(work_root / 'checkpoints'))} {_q('lafla-final')} "
                f"-C {_q(str(work_root))} {_q('tokenizer')} {_q('reports')} {_q('hf-package')}"
            ),
            f"sync && ls -lh {_q(archives_dir)}",
        ]
    )
    return KaggleRunPlan(commands=tuple(commands))


def _q(value: str) -> str:
    """Shell argumanini quote eder."""

    return shlex.quote(value)
