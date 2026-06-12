"""
@Dosya: colab/run_plan.py
@Aciklama: Colab uzerinde LaflaAi-Core egitimi icin tekrar uretilebilir komut
            plani olusturur.
@Yazar: Lafla Gelistirme Ekibi
@Bilgi: Notebook is mantigi tasimaz; bu modul preflight, tokenizer, egitim ve
        artifact arsiv adimlarini komut planina cevirir.
@Uyari: Gercek Drive yolu kullanilmadan final artifact komutu uretilmez.
@Calisma-Semasi: paths -> validate -> commands -> ColabRunPlan
"""

from __future__ import annotations

import json
import shlex
from dataclasses import asdict, dataclass
from pathlib import PurePosixPath
from typing import Sequence


DEFAULT_THINKING_SFT_DATA = (
    "datasets/synthetic/lafla-100m-thinking-chat-seed-20k.jsonl",
    "datasets/synthetic/lafla-100m-safety-jailbreak-seed-10k.jsonl",
)


@dataclass(frozen=True)
class ColabPaths:
    """Colab calisma yollarini tasir."""

    repo_dir: str = "/content/LaflaAi-Core"
    drive_root: str = "/content/gdrive/MyDrive/LaflaAI100M"
    work_root: str = "/content/LaflaAI100M"
    artifact_name: str = "lafla-100m-thinking-colab-tpu-v5e-run.tar.gz"
    checkpoint_dir: str = "/content/LaflaAI100M/checkpoints"
    tokenizer_path: str = "/content/LaflaAI100M/tokenizer/lafla-tokenizer.json"
    tokenizer_report_path: str = "/content/LaflaAI100M/reports/tokenizer-quality.json"


@dataclass(frozen=True)
class ColabTrainingProfile:
    """Aktif Colab model ailesinin tum config ve kimlik yollarini tasir."""

    model_config: str = "configs/model/lafla-100m-thinking.yaml"
    training_config: str = "configs/training/colab-tpu-v5e-100m.yaml"
    tokenizer_config: str = "configs/tokenizer/turkish-german-thinking-bpe.yaml"
    runtime_config: str = "configs/runtime/desktop-i3-int8-100m.yaml"
    post_training_config: str = "configs/post_training/lafla-thinking-sft.yaml"
    identity_data: str = "configs/data/lafla-model-identity-100m.jsonl"
    source_plan: str = "configs/data/lafla-100m-source-plan.json"
    thinking_sft_data: tuple[str, ...] = DEFAULT_THINKING_SFT_DATA
    model_name: str = "lafla-100m-thinking"


@dataclass(frozen=True)
class ColabRunPlan:
    """Colab komut planini tasir."""

    commands: tuple[str, ...]

    def to_json(self) -> str:
        """Komut planini JSON olarak dondurur."""

        return json.dumps(asdict(self), ensure_ascii=False, indent=2)


def build_colab_run_plan(
    paths: ColabPaths,
    data_jsonl: str,
    manifest_path: str = "/content/LaflaAI100M/data/veri_manifesti.json",
    thinking_jsonl: str | None = None,
    thinking_jsonls: Sequence[str] | None = None,
    profile: ColabTrainingProfile | None = None,
) -> ColabRunPlan:
    """Preflight, data audit, tokenizer ve artifact adimlari icin komut plani uretir."""

    drive_root = PurePosixPath(paths.drive_root)
    if not str(drive_root).startswith("/content/gdrive/MyDrive/"):
        raise ValueError("final artifact icin gercek Drive yolu /content/gdrive/MyDrive altinda olmali")
    profile = profile or ColabTrainingProfile()
    thinking_inputs = _merge_thinking_sft_inputs(profile.thinking_sft_data, thinking_jsonl, thinking_jsonls)
    repo = _q(paths.repo_dir)
    data = _q(data_jsonl)
    identity_data = _q(profile.identity_data)
    manifest = _q(manifest_path)
    work_root = PurePosixPath(paths.work_root)
    tokenizer_dir = str(work_root / "tokenizer")
    reports_dir = str(work_root / "reports")
    hf_package_dir = str(work_root / "hf-package")
    env = "PJRT_DEVICE=TPU PYTHONPATH=src"
    commands = [
        "python - <<'PY'\nfrom google.colab import drive\ndrive.mount('/content/gdrive', force_remount=True)\nPY",
        f"cd {repo} && python -m pip install torch 'torch_xla[tpu]'",
        f"cd {repo} && python -m pip install -r requirements/colab-tpu.txt",
        f"cd {repo} && {env} python - <<'PY'\nimport torch_xla\nprint('xla_device=', torch_xla.device())\nPY",
        f"cd {repo} && {env} python -m lafla_ai_core.cli.check_environment --optimizer adamw --accelerator xla",
        f"cd {repo} && {env} python -m lafla_ai_core.cli.quality_scan --root .",
        (
            f"test -s {data} && test -s {manifest} && "
            f"cd {repo} && test -s {_q(profile.source_plan)} || "
            "{ echo 'real data, manifest veya source plan eksik; egitim reddedildi' >&2; exit 2; }"
        ),
        (
            f"cd {repo} && {env} python -m lafla_ai_core.cli.preflight "
            f"{_q(profile.model_config)} "
            f"{_q(profile.training_config)} "
            f"{_q(profile.tokenizer_config)} "
            f"{_q(profile.runtime_config)} "
            f"{_q(profile.post_training_config)}"
        ),
        f"mkdir -p {_q(tokenizer_dir)} {_q(reports_dir)} {_q(hf_package_dir)} {_q(paths.checkpoint_dir)}",
        f"cd {repo} && {env} python -m lafla_ai_core.cli.data_audit --manifest {manifest} --report {_q(str(work_root / 'reports/data-audit.json'))}",
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
        report_name = f"thinking-sft-audit-{index:03d}.json"
        commands.append(
            f"cd {repo} && test -s {_q(sft_path)} && "
            f"{env} python -m lafla_ai_core.cli.validate_thinking_sft "
            f"--input {_q(sft_path)} --report {_q(str(work_root / 'reports' / report_name))}"
        )
    if thinking_inputs:
        commands.append(_post_training_manifest_command(repo, env, str(work_root / "reports/post-training-sft-inputs.json"), thinking_inputs))
    commands.extend(
        [
        (
            "RESUME_ARGS=(); "
            "if [ -n \"${RESUME_FROM:-}\" ]; then RESUME_ARGS+=(--resume-from \"$RESUME_FROM\"); fi; "
            f"cd {repo} && {env} XLA_USE_BF16=1 python -m lafla_ai_core.cli.train_pretrain "
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
        f"mkdir -p {_q(str(drive_root / 'final-checkpoint'))}",
        (
            f"tar -czf {_q(str(drive_root / 'final-checkpoint' / paths.artifact_name))} "
            f"-C {_q(str(work_root / 'checkpoints'))} {_q('lafla-final')} "
            f"-C {_q(str(work_root))} {_q('tokenizer')} {_q('reports')} {_q('hf-package')}"
        ),
        f"sync && ls -lh {_q(str(drive_root / 'final-checkpoint'))}",
        ]
    )
    return ColabRunPlan(commands=tuple(commands))


def _q(value: str) -> str:
    """Shell argumanini quote eder."""

    return shlex.quote(value)


def _merge_thinking_sft_inputs(
    defaults: Sequence[str],
    single: str | None,
    extra: Sequence[str] | None,
) -> tuple[str, ...]:
    merged: list[str] = []
    for value in (*defaults, *((single,) if single is not None else ()), *(extra or ())):
        if value and value not in merged:
            merged.append(value)
    return tuple(merged)


def _post_training_manifest_command(repo: str, env: str, report_path: str, inputs: Sequence[str]) -> str:
    payload = json.dumps(
        {
            "stage": "post_training_thinking_sft",
            "usage": "validate_then_use_after_base_pretrain_checkpoint",
            "allowed_for_pretraining": False,
            "inputs": list(inputs),
        },
        ensure_ascii=False,
        sort_keys=True,
    )
    return (
        f"cd {repo} && {env} python - <<'PY'\n"
        "import json\n"
        "from pathlib import Path\n"
        f"payload = {payload!r}\n"
        f"path = Path({_q(report_path)!r})\n"
        "path.parent.mkdir(parents=True, exist_ok=True)\n"
        "path.write_text(json.dumps(json.loads(payload), ensure_ascii=False, indent=2, sort_keys=True) + '\\n', encoding='utf-8')\n"
        "PY"
    )
