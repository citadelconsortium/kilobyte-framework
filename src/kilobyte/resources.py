from __future__ import annotations

import os
import platform
import shutil
import subprocess
from dataclasses import asdict, dataclass
from pathlib import Path

from .config import Settings


MIB = 1024 * 1024


def _meminfo() -> dict[str, int]:
    result: dict[str, int] = {}
    try:
        for line in Path("/proc/meminfo").read_text(encoding="ascii").splitlines():
            key, value = line.split(":", 1)
            result[key] = int(value.strip().split()[0]) * 1024
    except (OSError, ValueError):
        pass
    return result


def _cgroup_available() -> int | None:
    root = Path("/sys/fs/cgroup")
    try:
        limit_text = (root / "memory.max").read_text().strip()
        if limit_text == "max":
            return None
        limit = int(limit_text)
        current = int((root / "memory.current").read_text().strip())
        return max(0, limit - current)
    except (OSError, ValueError):
        return None


def _cpu_flags() -> set[str]:
    try:
        text = Path("/proc/cpuinfo").read_text(encoding="ascii", errors="ignore")
    except OSError:
        return set()
    for line in text.splitlines():
        if line.startswith("flags") or line.startswith("Features"):
            return set(line.split(":", 1)[1].split())
    return set()


def _gpu_name() -> str | None:
    if not shutil.which("lspci"):
        return None
    try:
        output = subprocess.run(
            ["lspci"], capture_output=True, text=True, timeout=3, check=False
        ).stdout
    except (OSError, subprocess.TimeoutExpired):
        return None
    matches = [line.split(": ", 1)[-1] for line in output.splitlines() if "VGA" in line or "3D controller" in line]
    return matches[0] if matches else None


@dataclass(frozen=True, slots=True)
class ResourceProfile:
    total_mb: int
    available_mb: int
    safe_available_mb: int
    model_mb: int
    context_size: int
    threads: int
    batch_size: int
    gpu: str | None
    gpu_layers: int
    cpu_arch: str
    cpu_level: str

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


class ResourceManager:
    """Chooses safe llama.cpp settings from live resources, never a fixed RAM cap."""

    def __init__(self, settings: Settings):
        self.settings = settings

    def profile(self) -> ResourceProfile:
        mem = _meminfo()
        total = mem.get("MemTotal", 0)
        available = mem.get("MemAvailable", mem.get("MemFree", 0))
        cgroup = _cgroup_available()
        if cgroup is not None:
            available = min(available, cgroup)
        reserve = max(self.settings.reserve_memory_mb * MIB, int(total * 0.18))
        safe = max(0, available - reserve)
        model_size = self.settings.model_path.stat().st_size if self.settings.model_path.exists() else 1280 * MIB

        usable_after_model = safe - model_size
        if self.settings.context_size:
            context = self.settings.context_size
        elif usable_after_model >= 1100 * MIB:
            context = 8192
        elif usable_after_model >= 500 * MIB:
            context = 4096
        else:
            context = 2048

        cores = os.cpu_count() or 1
        threads = max(1, min(cores, 8))
        batch = 256 if safe >= 2600 * MIB else 128 if safe >= 1800 * MIB else 64
        gpu = _gpu_name()
        gpu_override = os.environ.get("KILOBYTE_GPU_LAYERS")
        virtual_gpu = bool(gpu and any(marker in gpu.lower() for marker in ("vmware", "virtualbox", "bochs", "qxl", "llvmpipe")))
        render_access = any(os.access(path, os.R_OK | os.W_OK) for path in Path("/dev/dri").glob("renderD*"))
        nvidia_access = shutil.which("nvidia-smi") is not None
        if gpu_override is not None:
            gpu_layers = int(gpu_override)
        else:
            gpu_layers = 99 if gpu and not virtual_gpu and (render_access or nvidia_access) else 0
        flags = _cpu_flags()
        # Reported so the operator can see which llama.cpp CPU backend this machine gets;
        # the install is portable and the difference between these is large.
        if "avx512f" in flags:
            level = "avx512"
        elif "avx2" in flags:
            level = "avx2"
        elif "avx" in flags:
            level = "avx"
        elif "sse4_1" in flags:
            level = "sse4"
        else:
            level = "baseline"
        return ResourceProfile(
            total_mb=total // MIB,
            available_mb=available // MIB,
            safe_available_mb=safe // MIB,
            model_mb=model_size // MIB,
            context_size=context,
            threads=threads,
            batch_size=batch,
            gpu=gpu,
            gpu_layers=gpu_layers,
            cpu_arch=platform.machine(),
            cpu_level=level,
        )

    def enough_to_start(self, profile: ResourceProfile | None = None) -> tuple[bool, str]:
        p = profile or self.profile()
        required = p.model_mb + 320
        if p.safe_available_mb < required:
            return False, f"need about {required} MiB safely available; found {p.safe_available_mb} MiB"
        return True, "resources are within the safe operating envelope"

    def live_headroom(self) -> tuple[bool, int]:
        """Fast pre-step pressure check after the model is already resident."""
        available = _meminfo().get("MemAvailable", 0)
        cgroup = _cgroup_available()
        if cgroup is not None:
            available = min(available, cgroup)
        available_mb = available // MIB
        return available_mb >= 320, available_mb
