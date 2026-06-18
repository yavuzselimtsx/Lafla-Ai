"""
@Dosya: training/checkpoint_policy.py
@Aciklama: Checkpoint kayit araligi ve retention kararlarini saf fonksiyonlarla
            hesaplar.
@Yazar: Lafla Gelistirme Ekibi
@Bilgi: GPT-NeoX'teki keep_last ve final checkpoint fikri Lafla icin basit,
        test edilebilir ve Colab dostu bir politikaya cevrilir.
@Uyari: Retention yanlis olursa disk dolar veya tek guvenli checkpoint silinir.
@Calisma-Semasi: step -> save decision -> retention victims
"""

from __future__ import annotations

import json
import shutil
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class CheckpointPolicy:
    """Checkpoint politikasini tasir."""

    save_every: int
    keep_last: int
    always_save_final: bool = True

    def validate(self) -> None:
        """Politika degerlerini dogrular."""

        if self.save_every <= 0:
            raise ValueError("save_every pozitif olmali")
        if self.keep_last <= 0:
            raise ValueError("keep_last pozitif olmali")


def should_save_checkpoint(step: int, max_steps: int, policy: CheckpointPolicy) -> bool:
    """Belirli step'te checkpoint yazilip yazilmayacagini hesaplar."""

    policy.validate()
    if step <= 0:
        return False
    if policy.always_save_final and step >= max_steps:
        return True
    return step % policy.save_every == 0


def retention_victims(existing_steps: list[int], keep_last: int) -> tuple[int, ...]:
    """Retention icin silinecek checkpoint step'lerini dondurur."""

    if keep_last <= 0:
        raise ValueError("keep_last pozitif olmali")
    ordered = sorted(set(existing_steps))
    if len(ordered) <= keep_last:
        return ()
    return tuple(ordered[: len(ordered) - keep_last])


def archive_checkpoint_before_retention(target: Path, checkpoint_root: Path) -> Path:
    """Retention silmeden once READY checkpoint'i kardes backup klasorune kopyalar."""

    archive_root = checkpoint_root.parent / "checkpoint-backups"
    archive_target = archive_root / target.name
    archive_root.mkdir(parents=True, exist_ok=True)
    resolved_archive_root = archive_root.resolve()
    resolved_archive_target = archive_target.resolve()
    if resolved_archive_root not in resolved_archive_target.parents:
        raise RuntimeError(f"checkpoint archive hedefi guvenli degil: {resolved_archive_target}")
    if archive_target.exists():
        return archive_target
    shutil.copytree(target, archive_target)
    (archive_target / "ARCHIVED_BY_RETENTION.json").write_text(
        json.dumps(
            {
                "source": str(target),
                "archive": str(archive_target),
                "reason": "checkpoint_retention_backup_before_delete",
            },
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    return archive_target


def apply_checkpoint_retention(checkpoint_root: Path, keep_last: int) -> None:
    """READY checkpoint retention uygular; silmeden once backup alir."""

    steps: list[int] = []
    for child in checkpoint_root.glob("lafla-step-*"):
        if child.is_dir() and (child / "READY.json").exists():
            try:
                steps.append(int(child.name.rsplit("-", 1)[-1]))
            except ValueError:
                continue
    resolved_root = checkpoint_root.resolve()
    for victim in retention_victims(steps, keep_last):
        target = checkpoint_root / f"lafla-step-{victim:06d}"
        resolved_target = target.resolve()
        if resolved_root not in resolved_target.parents:
            raise RuntimeError(f"retention hedefi checkpoint disinda: {resolved_target}")
        archive_checkpoint_before_retention(target, checkpoint_root)
        shutil.rmtree(resolved_target)

