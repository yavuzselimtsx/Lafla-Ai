"""
@Dosya: runtime/rss.py
@Aciklama: Ana surec ve tum cocuklarinin peak RSS/USS kullanimini ornekler.
"""

from __future__ import annotations

import subprocess
import time
from dataclasses import dataclass
from typing import Sequence


@dataclass(frozen=True)
class ProcessMemorySample:
    rss_bytes: int
    uss_bytes: int
    process_count: int


@dataclass(frozen=True)
class ProcessMemoryPeak:
    peak_rss_bytes: int
    peak_uss_bytes: int
    peak_process_count: int
    samples: int
    duration_seconds: float
    return_code: int


def sample_process_tree(process: object) -> ProcessMemorySample:
    try:
        children = tuple(process.children(recursive=True))  # type: ignore[attr-defined]
    except Exception:
        children = ()
    processes = (process, *children)
    rss_bytes = 0
    uss_bytes = 0
    process_count = 0
    for item in processes:
        try:
            rss_bytes += int(item.memory_info().rss)  # type: ignore[attr-defined]
            try:
                uss_bytes += int(item.memory_full_info().uss)  # type: ignore[attr-defined]
            except (AttributeError, OSError):
                pass
            process_count += 1
        except Exception:
            continue
    return ProcessMemorySample(rss_bytes, uss_bytes, process_count)


def monitor_command(
    command: Sequence[str],
    *,
    sample_interval_seconds: float = 0.05,
    timeout_seconds: float | None = None,
) -> ProcessMemoryPeak:
    if not command:
        raise ValueError("command bos olamaz")
    if sample_interval_seconds <= 0:
        raise ValueError("sample interval pozitif olmali")
    try:
        import psutil
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError("process-tree RSS olcumu icin psutil kurulu olmali") from exc

    started = time.monotonic()
    child = subprocess.Popen(tuple(command), stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    process = psutil.Process(child.pid)
    peak_rss = 0
    peak_uss = 0
    peak_count = 0
    samples = 0
    while True:
        sample = sample_process_tree(process)
        peak_rss = max(peak_rss, sample.rss_bytes)
        peak_uss = max(peak_uss, sample.uss_bytes)
        peak_count = max(peak_count, sample.process_count)
        samples += 1
        return_code = child.poll()
        if return_code is not None:
            break
        if timeout_seconds is not None and time.monotonic() - started > timeout_seconds:
            child.kill()
            child.wait()
            return_code = -9
            break
        time.sleep(sample_interval_seconds)
    return ProcessMemoryPeak(
        peak_rss_bytes=peak_rss,
        peak_uss_bytes=peak_uss,
        peak_process_count=peak_count,
        samples=samples,
        duration_seconds=time.monotonic() - started,
        return_code=int(return_code),
    )
