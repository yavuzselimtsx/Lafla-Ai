"""
@Dosya: model/checkpoint_contract.py
@Aciklama: Checkpoint READY sozlesmesini Torch import etmeden dogrular.
@Yazar: Lafla Gelistirme Ekibi
@Bilgi: Yarim checkpointler resume veya model kabul kapisindan gecemez.
@Calisma-Semasi: checkpoint dir -> READY + dosya kontrati -> contract
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path


CHECKPOINT_FORMAT = "lafla-ai-core-checkpoint-v1"
REQUIRED_CHECKPOINT_FILES = (
    "READY.json",
    "config.json",
    "trainer_state.json",
    "model.pt",
    "optimizer.pt",
    "rng.pt",
)


@dataclass(frozen=True)
class CheckpointContract:
    """Dogrulanmis checkpoint sozlesmesi."""

    path: str
    format: str
    files: tuple[str, ...]


def validate_checkpoint_directory(checkpoint_dir: str | Path) -> CheckpointContract:
    """Checkpoint klasorunun tam ve hazir oldugunu fail-closed dogrular."""

    path = Path(checkpoint_dir)
    if not path.exists():
        raise FileNotFoundError(f"checkpoint bulunamadi: {path}")
    if not path.is_dir():
        raise NotADirectoryError(f"checkpoint klasor degil: {path}")
    for file_name in REQUIRED_CHECKPOINT_FILES:
        file_path = path / file_name
        if not file_path.exists():
            raise FileNotFoundError(f"checkpoint dosyasi eksik: {file_name}")
        if file_name != "READY.json" and file_path.stat().st_size <= 0:
            raise ValueError(f"checkpoint dosyasi bos: {file_name}")
    ready = _read_ready(path / "READY.json")
    if ready.get("ready") is not True:
        raise ValueError("checkpoint READY.json ready=true tasimali")
    if ready.get("format") != CHECKPOINT_FORMAT:
        raise ValueError(f"checkpoint format hatali: {ready.get('format')}")
    return CheckpointContract(str(path), CHECKPOINT_FORMAT, REQUIRED_CHECKPOINT_FILES)


def _read_ready(path: Path) -> dict[str, object]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ValueError(f"READY.json gecersiz JSON: {exc}") from exc
    if not isinstance(payload, dict):
        raise ValueError("READY.json kok nesnesi mapping olmali")
    return payload
