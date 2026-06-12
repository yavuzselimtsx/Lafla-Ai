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

from dataclasses import dataclass


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

