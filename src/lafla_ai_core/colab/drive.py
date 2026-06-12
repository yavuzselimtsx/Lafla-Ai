"""
@Dosya: colab/drive.py
@Aciklama: Colab Google Drive mount noktasinin gercek mi yerel fallback mi
            oldugunu ayirt eden yardimcilari tasir.
@Yazar: Lafla Gelistirme Ekibi
@Bilgi: Ilk kosuda /content/drive yerel klasor gibi doldugu icin Drive web'de
        dosya gorunmedi. Bu modul o sinifi tekrar engeller.
@Uyari: Gercek Drive dogrulanmadan final artifact basarili sayilamaz.
@Calisma-Semasi: path -> mount table -> classify -> require_real_drive
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class DriveMountStatus:
    """Drive mount denetim sonucunu tasir."""

    path: Path
    exists: bool
    has_my_drive: bool
    appears_mounted: bool
    reason: str

    @property
    def usable(self) -> bool:
        """Gercek Drive olarak kullanilabilir mi?"""

        return self.exists and self.has_my_drive and self.appears_mounted


def inspect_drive_mount(path: str | Path, proc_mounts_text: str | None = None) -> DriveMountStatus:
    """Mount noktasini Colab Drive icin denetler."""

    mount_path = Path(path)
    exists = mount_path.exists()
    has_my_drive = (mount_path / "MyDrive").exists()
    proc_text = proc_mounts_text if proc_mounts_text is not None else _read_proc_mounts()
    appears_mounted = _path_appears_in_mounts(mount_path, proc_text)
    if not exists:
        reason = "mount path yok"
    elif not has_my_drive:
        reason = "MyDrive yok"
    elif not appears_mounted:
        reason = "mount table icinde gorunmuyor; yerel klasor olabilir"
    else:
        reason = "gercek mount gibi gorunuyor"
    return DriveMountStatus(mount_path, exists, has_my_drive, appears_mounted, reason)


def require_real_drive(path: str | Path) -> DriveMountStatus:
    """Gercek Drive degilse RuntimeError firlatir."""

    status = inspect_drive_mount(path)
    if not status.usable:
        raise RuntimeError(f"Google Drive dogrulanamadi: {status.reason} ({status.path})")
    return status


def _read_proc_mounts() -> str:
    """Linux mount tablosunu okur; yoksa bos metin dondurur."""

    proc = Path("/proc/mounts")
    if not proc.exists():
        return ""
    return proc.read_text(encoding="utf-8", errors="replace")


def _path_appears_in_mounts(path: Path, proc_mounts_text: str) -> bool:
    """Verilen path mount tablosunda geciyor mu?"""

    normalized = str(path).replace("\\", "/")
    return any(normalized == line.split()[1].replace("\\040", " ") for line in proc_mounts_text.splitlines() if len(line.split()) >= 2)

