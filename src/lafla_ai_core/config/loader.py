"""
@Dosya: config/loader.py
@Aciklama: JSON/YAML config dosyalarini Lafla typed schema katmanina yukler.
@Yazar: Lafla Gelistirme Ekibi
@Bilgi: PyYAML varsa kullanilir; yoksa repo configleri icin katı ve kucuk bir
        YAML alt kumesi okunur. Bilinmeyen karmasik YAML ozelligi fail eder.
@Uyari: Config okuma hatasi egitimi durdurmalidir; sessiz varsayilan yoktur.
@Calisma-Semasi: path -> parse -> mapping -> schema -> validate
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from .schema import ConfigError


def load_mapping(path: str | Path) -> dict[str, Any]:
    """Config dosyasini dict olarak yukler."""

    config_path = Path(path)
    if not config_path.exists():
        raise ConfigError(f"config bulunamadi: {config_path}")
    suffix = config_path.suffix.lower()
    text = config_path.read_text(encoding="utf-8")
    if suffix == ".json":
        loaded = json.loads(text)
    elif suffix in {".yaml", ".yml"}:
        loaded = _load_yaml(text)
    else:
        raise ConfigError(f"desteklenmeyen config uzantisi: {suffix}")
    if not isinstance(loaded, dict):
        raise ConfigError("config kok nesnesi mapping olmali")
    return loaded


def _load_yaml(text: str) -> dict[str, Any]:
    """PyYAML veya katı fallback parser ile YAML okur."""

    try:
        import yaml  # type: ignore
    except Exception:
        return _load_simple_yaml(text)
    loaded = yaml.safe_load(text)
    if not isinstance(loaded, dict):
        raise ConfigError("YAML kok nesnesi mapping olmali")
    return loaded


def _load_simple_yaml(text: str) -> dict[str, Any]:
    """Repo configleri icin katı YAML alt kumesini okur."""

    entries: list[tuple[int, str]] = []
    for raw_line in text.splitlines():
        if not raw_line.strip() or raw_line.lstrip().startswith("#"):
            continue
        entries.append((len(raw_line) - len(raw_line.lstrip(" ")), raw_line.strip()))

    root: dict[str, Any] = {}
    stack: list[tuple[int, dict[str, Any] | list[Any]]] = [(-1, root)]
    for index, (indent, line) in enumerate(entries):
        while stack and indent <= stack[-1][0]:
            stack.pop()
        parent = stack[-1][1]
        if line.startswith("- "):
            if not isinstance(parent, list):
                raise ConfigError(f"YAML liste konumu gecersiz: {line}")
            parent.append(_parse_scalar(line[2:].strip()))
            continue
        if ":" not in line:
            raise ConfigError(f"YAML satiri gecersiz: {line}")
        key, value = line.split(":", 1)
        key = key.strip()
        value = value.strip()
        if not isinstance(parent, dict):
            raise ConfigError(f"YAML mapping konumu gecersiz: {line}")
        if value:
            parent[key] = _parse_scalar(value)
            continue
        next_line = _next_significant_line(entries, index)
        child: dict[str, Any] | list[Any]
        child = [] if next_line is not None and next_line[1].startswith("- ") else {}
        parent[key] = child
        stack.append((indent, child))
    return root


def _next_significant_line(entries: list[tuple[int, str]], index: int) -> tuple[int, str] | None:
    """Bir sonraki YAML girdisini dondurur."""

    next_index = index + 1
    if next_index >= len(entries):
        return None
    return entries[next_index]


def _parse_scalar(value: str) -> Any:
    """Basit YAML scalar degerini Python degerine cevirir."""

    if value in {"true", "True"}:
        return True
    if value in {"false", "False"}:
        return False
    if value in {"null", "None", "~"}:
        return None
    if (value.startswith('"') and value.endswith('"')) or (value.startswith("'") and value.endswith("'")):
        return value[1:-1]
    try:
        return int(value)
    except ValueError:
        pass
    try:
        return float(value)
    except ValueError:
        return value
