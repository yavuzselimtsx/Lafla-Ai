"""
@Dosya: observability/artifact_manifest.py
@Aciklama: Checkpoint ve export artifactleri icin hashli manifest uretir.
@Yazar: Lafla Gelistirme Ekibi
@Bilgi: Model dosyasi tek basina yeterli degildir; config, tokenizer, log ve
        state dosyalari birlikte izlenir.
@Uyari: Manifest olmadan artifact release edilemez.
@Calisma-Semasi: directory -> hash files -> manifest json
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable


@dataclass(frozen=True)
class ArtifactFile:
    """Tek artifact dosyasinin kaydini tasir."""

    path: str
    size_bytes: int
    sha256: str


@dataclass(frozen=True)
class ArtifactManifest:
    """Artifact manifest kok nesnesi."""

    root: str
    files: tuple[ArtifactFile, ...]

    def to_json(self) -> str:
        """Manifesti deterministik JSON olarak dondurur."""

        return json.dumps(asdict(self), ensure_ascii=False, indent=2, sort_keys=True)


def build_manifest(root: str | Path, include_suffixes: Iterable[str] | None = None) -> ArtifactManifest:
    """Klasor altindaki dosyalar icin hashli manifest uretir."""

    root_path = Path(root)
    suffixes = set(include_suffixes or ())
    files: list[ArtifactFile] = []
    for path in sorted(item for item in root_path.rglob("*") if item.is_file()):
        if suffixes and path.suffix not in suffixes:
            continue
        files.append(
            ArtifactFile(
                path=path.relative_to(root_path).as_posix(),
                size_bytes=path.stat().st_size,
                sha256=_sha256(path),
            )
        )
    return ArtifactManifest(root=str(root_path), files=tuple(files))


def write_manifest(root: str | Path, output_path: str | Path) -> ArtifactManifest:
    """Manifest uretir ve diske yazar."""

    manifest = build_manifest(root)
    Path(output_path).write_text(manifest.to_json() + "\n", encoding="utf-8")
    return manifest


def _sha256(path: Path) -> str:
    """Dosyanin SHA-256 ozetini hesaplar."""

    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()

