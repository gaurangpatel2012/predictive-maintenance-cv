"""Device selection, device synchronisation and process memory probes."""
from __future__ import annotations

import os
import platform
import subprocess
import sys

import torch


def resolve_device(preference: str = "auto") -> torch.device:
    if preference != "auto":
        return torch.device(preference)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def synchronize(device: torch.device) -> None:
    """Block until queued kernels finish; without this GPU timings are meaningless."""
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    elif device.type == "mps":
        torch.mps.synchronize()


def empty_cache(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.empty_cache()
    elif device.type == "mps":
        torch.mps.empty_cache()


def device_name(device: torch.device) -> str:
    if device.type == "cuda":
        return f"{torch.cuda.get_device_name(device)} / {platform.processor() or platform.machine()}"
    if sys.platform == "darwin":
        try:
            cpu = subprocess.run(
                ["sysctl", "-n", "machdep.cpu.brand_string"], capture_output=True, text=True, check=True
            ).stdout.strip()
        except (OSError, subprocess.CalledProcessError):
            cpu = platform.processor()
        return f"{cpu} ({'GPU via MPS' if device.type == 'mps' else 'CPU'})"
    return platform.processor() or platform.machine()


def rss_mb() -> float:
    """Current resident set size of this process in MiB."""
    try:
        import psutil

        return psutil.Process(os.getpid()).memory_info().rss / 2**20
    except ImportError:  # fall back to the peak, which over-states
        return peak_rss_mb()


def peak_rss_mb() -> float:
    """Lifetime peak resident set size in MiB."""
    if sys.platform == "win32":
        import psutil  # peak_wset is the Windows equivalent of ru_maxrss

        return psutil.Process(os.getpid()).memory_info().peak_wset / 2**20
    import resource  # POSIX only

    peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss  # bytes on macOS, KiB on Linux
    return peak / 2**20 if sys.platform == "darwin" else peak / 2**10


def reset_peak_device_memory(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)


def peak_device_memory_mb(device: torch.device) -> float | None:
    if device.type == "cuda":
        return torch.cuda.max_memory_allocated(device) / 2**20
    if device.type == "mps":
        return torch.mps.driver_allocated_memory() / 2**20
    return None
