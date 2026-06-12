"""Attention uygulama yolu secimi icin torch bagimsiz politikalar."""

from __future__ import annotations


def can_use_full_window_sdpa(
    *,
    attention_mode: str,
    sequence_length: int,
    sliding_window: int,
    device_type: str,
) -> bool:
    """Yerel pencere tum diziyi kapsiyorsa causal SDPA kullanilabilir."""

    return (
        attention_mode == "local"
        and device_type != "xla"
        and sliding_window > 0
        and sequence_length <= sliding_window
    )
