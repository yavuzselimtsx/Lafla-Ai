"""
@Dosya: model/checkpoint_io.py
@Aciklama: Model agirligi, config ve egitim state dosyalarini atomik kaydeder.
@Yazar: Lafla Gelistirme Ekibi
@Bilgi: Colab kesintilerinde yarim checkpoint bir sonraki kosuyu bozmamali.
@Uyari: Checkpoint klasoru tamamlanmadan final ada tasinmaz.
@Calisma-Semasi: tmp dir -> write files -> fsync intent -> atomic publish
"""

from __future__ import annotations

import json
import os
import random
import shutil
from dataclasses import asdict
from pathlib import Path
from typing import Any

try:
    import torch
except ModuleNotFoundError as exc:  # pragma: no cover
    raise ModuleNotFoundError("checkpoint_io icin torch kurulu olmali") from exc

from lafla_ai_core.config.schema import ModelConfig
from lafla_ai_core.model.checkpoint_contract import validate_checkpoint_directory


def save_training_checkpoint(
    target_dir: str | Path,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    model_config: ModelConfig,
    state: dict[str, Any],
) -> Path:
    """Checkpoint'i once gecici klasore yazip sonra yayinlar."""

    target = Path(target_dir)
    tmp = target.with_name(f".{target.name}.tmp")
    if tmp.exists():
        shutil.rmtree(tmp)
    tmp.mkdir(parents=True)
    _write_json(tmp / "config.json", {"model": asdict(model_config)})
    _write_json(tmp / "trainer_state.json", state)
    _save_tensor_payload(_checkpoint_state_dict(model), tmp / "model.pt")
    _save_tensor_payload(optimizer.state_dict(), tmp / "optimizer.pt")
    _save_tensor_payload(_capture_rng_state(), tmp / "rng.pt")
    _write_json(tmp / "READY.json", {"ready": True, "format": "lafla-ai-core-checkpoint-v1"})
    _publish_directory(tmp, target)
    return target


def load_training_checkpoint(
    checkpoint_dir: str | Path,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer | None = None,
    map_location: str | torch.device = "cpu",
) -> dict[str, Any]:
    """Checkpoint dosyalarini modele ve opsiyonel optimizer'a yukler."""

    path = Path(checkpoint_dir)
    validate_checkpoint_directory(path)
    _load_model_state_dict(model, torch.load(path / "model.pt", map_location=map_location))
    if optimizer is not None:
        optimizer.load_state_dict(torch.load(path / "optimizer.pt", map_location=map_location))
    rng_path = path / "rng.pt"
    if rng_path.exists():
        _restore_rng_state(torch.load(rng_path, map_location="cpu"))
    state_path = path / "trainer_state.json"
    if not state_path.exists():
        raise FileNotFoundError(f"trainer_state eksik: {state_path}")
    return json.loads(state_path.read_text(encoding="utf-8"))


def _checkpoint_state_dict(model: torch.nn.Module) -> dict[str, torch.Tensor]:
    """DataParallel/DDP sarmalini kaldirip temiz model state_dict'i dondurur."""

    return _unwrap_parallel_model(model).state_dict()


def _load_model_state_dict(model: torch.nn.Module, state_dict: dict[str, torch.Tensor]) -> None:
    """Eski DataParallel prefix'li checkpoint'leri de plain modele yukler."""

    target = _unwrap_parallel_model(model)
    if _has_module_prefix(state_dict):
        target.load_state_dict({key.removeprefix("module."): value for key, value in state_dict.items()})
        return
    target.load_state_dict(state_dict)


def _unwrap_parallel_model(model: torch.nn.Module) -> torch.nn.Module:
    module = getattr(model, "module", None)
    if isinstance(module, torch.nn.Module):
        return module
    return model


def _has_module_prefix(state_dict: dict[str, torch.Tensor]) -> bool:
    return bool(state_dict) and all(key.startswith("module.") for key in state_dict)


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True), encoding="utf-8")
    os.replace(tmp, path)


def _publish_directory(tmp: Path, target: Path) -> None:
    backup = target.with_name(f".{target.name}.bak")
    if backup.exists():
        shutil.rmtree(backup)
    if target.exists():
        os.replace(target, backup)
    try:
        os.replace(tmp, target)
    except Exception:
        if target.exists():
            shutil.rmtree(target)
        if backup.exists():
            os.replace(backup, target)
        raise
    if backup.exists():
        shutil.rmtree(backup)


def _save_tensor_payload(payload: Any, path: Path) -> None:
    if _contains_xla_tensor(payload):
        try:
            import torch_xla.core.xla_model as xm  # type: ignore

            xm.save(payload, path)
            return
        except Exception:
            pass
    torch.save(payload, path)


def _contains_xla_tensor(value: Any) -> bool:
    if torch.is_tensor(value):
        return value.device.type == "xla"
    if isinstance(value, dict):
        return any(_contains_xla_tensor(item) for item in value.values())
    if isinstance(value, (list, tuple)):
        return any(_contains_xla_tensor(item) for item in value)
    return False


def _capture_rng_state() -> dict[str, Any]:
    state: dict[str, Any] = {
        "python_random": random.getstate(),
        "torch": torch.get_rng_state(),
    }
    if torch.cuda.is_available():
        state["cuda"] = torch.cuda.get_rng_state_all()
    return state


def _restore_rng_state(state: dict[str, Any]) -> None:
    if "python_random" in state:
        random.setstate(state["python_random"])
    if "torch" in state:
        torch.set_rng_state(state["torch"])
    if torch.cuda.is_available() and "cuda" in state:
        torch.cuda.set_rng_state_all(state["cuda"])
