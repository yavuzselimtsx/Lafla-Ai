"""
@Dosya: quality/static_scan.py
@Aciklama: Colab'a yuklemeden once riskli metin kaliplarini tarar.
@Yazar: Lafla Gelistirme Ekibi
@Bilgi: Bu katman test, docs ve uretim kodu ayrimini koruyarak bilinen eski
        hatalari yeniden girmeden yakalar.
@Calisma-Semasi: files -> rules -> findings -> report
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable, Mapping


@dataclass(frozen=True)
class StaticScanRule:
    """Tek statik tarama kuralini tasir."""

    name: str
    pattern: str
    path_prefixes: tuple[str, ...]
    allowed_line_prefixes: tuple[str, ...] = ()


@dataclass(frozen=True)
class StaticScanFinding:
    """Tek statik tarama bulgusu."""

    rule: str
    path: str
    line: int
    excerpt: str


@dataclass(frozen=True)
class StaticScanReport:
    """Statik tarama raporu."""

    ok: bool
    findings: tuple[StaticScanFinding, ...]

    def to_json(self) -> str:
        return json.dumps(asdict(self), ensure_ascii=True, indent=2, sort_keys=True)


DEFAULT_RULES = (
    StaticScanRule("old_training_placeholder", "TRAINING_ENTRYPOINT_NOT_WIRED", ("src/", "configs/", "notebooks/")),
    StaticScanRule("wrong_colab_mount", "/content/drive/MyDrive", ("src/", "configs/", "notebooks/")),
    StaticScanRule("hardcoded_eos", "eos_id=1", ("src/",)),
    StaticScanRule("mojibake_marker", "Ã", ("src/", "configs/", "notebooks/", "README.md")),
    StaticScanRule("mojibake_marker", "Ä", ("src/", "configs/", "notebooks/", "README.md")),
    StaticScanRule("mojibake_marker", "Å", ("src/", "configs/", "notebooks/", "README.md")),
    StaticScanRule("mojibake_marker", "ï¿½", ("src/", "configs/", "notebooks/", "README.md")),
    StaticScanRule("python_cell_shell_cp", "cp -", ("notebooks/",), ("!", '"!', "'!")),
    StaticScanRule("python_cell_shell_tar", "tar -", ("notebooks/",), ("!", '"!', "'!")),
    StaticScanRule(
        "legacy_active_model_default",
        "lafla-380m",
        (
            "src/lafla_ai_core/colab/",
            "src/lafla_ai_core/cli/colab_plan.py",
            "src/lafla_ai_core/runtime/checkpoint_inference.py",
            "README.md",
            "docs/operations/next-training-plan.md",
            "docs/operations/low-power-runtime.md",
        ),
    ),
    StaticScanRule(
        "legacy_active_model_default",
        "lafla-400m",
        (
            "src/lafla_ai_core/colab/",
            "src/lafla_ai_core/cli/colab_plan.py",
            "README.md",
            "docs/operations/next-training-plan.md",
            "docs/operations/low-power-runtime.md",
        ),
    ),
    StaticScanRule("hf_cache_disabled", '"use_cache": False', ("src/lafla_ai_core/export/",)),
    StaticScanRule("hf_cache_disabled", "use_cache=False", ("src/lafla_ai_core/export/",)),
    StaticScanRule("hf_cache_disabled", "use_cache: false", ("src/lafla_ai_core/export/",)),
)


def run_static_scan(files: Mapping[str, str], rules: Iterable[StaticScanRule] = DEFAULT_RULES) -> StaticScanReport:
    """Verilen dosya iceriklerini kurallarla tarar."""

    findings: list[StaticScanFinding] = []
    for path, text in files.items():
        normalized = path.replace("\\", "/")
        for rule in rules:
            if not normalized.startswith(rule.path_prefixes):
                continue
            for line_number, line in enumerate(text.splitlines(), start=1):
                if rule.pattern in line and not _is_allowed_line(line, rule):
                    findings.append(StaticScanFinding(rule.name, path, line_number, line.strip()[:160]))
    return StaticScanReport(ok=not findings, findings=tuple(findings))


def collect_project_text_files(root: str | Path) -> dict[str, str]:
    """Proje icindeki taranabilir metin dosyalarini toplar."""

    base = Path(root)
    suffixes = {".py", ".yaml", ".yml", ".json", ".ipynb", ".md", ".txt"}
    files: dict[str, str] = {}
    for path in base.rglob("*"):
        if not path.is_file() or path.suffix.lower() not in suffixes:
            continue
        if any(part in {"__pycache__", ".git"} for part in path.parts):
            continue
        rel = path.relative_to(base).as_posix()
        if rel == "src/lafla_ai_core/quality/static_scan.py":
            continue
        files[rel] = path.read_text(encoding="utf-8")
    return files


def _is_allowed_line(line: str, rule: StaticScanRule) -> bool:
    stripped = line.strip()
    if rule.name == "mojibake_marker" and "MOJIBAKE_MARKERS" in stripped:
        return True
    return any(stripped.startswith(prefix) for prefix in rule.allowed_line_prefixes)
