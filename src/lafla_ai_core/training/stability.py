"""
@Dosya: training/stability.py
@Aciklama: Loss ve gradient norm icin egitim kararliligi izleyicisi.
@Yazar: Lafla Gelistirme Ekibi
@Bilgi: OLMo tarzindaki stability monitor fikri Lafla icin saf, test edilebilir
        bir sinifa indirgenmistir.
@Uyari: NaN/Inf degerler raporlanmaz; egitim hemen durdurulur.
@Calisma-Semasi: step metrics -> finite check -> spike flags -> rolling state
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field


@dataclass(frozen=True)
class StabilityObservation:
    """Tek step kararlilik sonucunu tasir."""

    step: int
    loss_spike: bool
    grad_norm_spike: bool
    spike_score: float


@dataclass
class StabilityMonitor:
    """Loss/grad norm spike tespiti yapan kucuk monitor."""

    window_size: int = 128
    threshold_std: float = 6.0
    rolling_window: int = 1000
    _loss_history: list[float] = field(default_factory=list, init=False, repr=False)
    _grad_history: list[float] = field(default_factory=list, init=False, repr=False)
    _spike_history: list[bool] = field(default_factory=list, init=False, repr=False)

    def observe(self, step: int, loss: float, grad_norm: float) -> StabilityObservation:
        """Metrikleri izler ve spike bayraklarini dondurur."""

        if not math.isfinite(loss):
            raise ValueError(f"loss finite degil: step={step}, loss={loss}")
        if not math.isfinite(grad_norm):
            raise ValueError(f"grad_norm finite degil: step={step}, grad_norm={grad_norm}")
        loss_spike = self._is_spike(loss, self._loss_history)
        grad_spike = self._is_spike(grad_norm, self._grad_history)
        self._append(self._loss_history, loss, self.window_size)
        self._append(self._grad_history, grad_norm, self.window_size)
        any_spike = loss_spike or grad_spike
        self._append(self._spike_history, any_spike, self.rolling_window)
        spike_score = sum(self._spike_history) / len(self._spike_history)
        return StabilityObservation(
            step=step,
            loss_spike=loss_spike,
            grad_norm_spike=grad_spike,
            spike_score=spike_score,
        )

    def _is_spike(self, value: float, history: list[float]) -> bool:
        if len(history) < self.window_size:
            return False
        mean = sum(history) / len(history)
        variance = sum((item - mean) ** 2 for item in history) / len(history)
        std = math.sqrt(variance)
        if std < 1e-12:
            return False
        return value > mean + self.threshold_std * std

    @staticmethod
    def _append(history: list[float] | list[bool], value: float | bool, max_size: int) -> None:
        history.append(value)
        if len(history) > max_size:
            del history[0]
