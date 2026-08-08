from __future__ import annotations

import hashlib
import os
import shutil
import sqlite3
import stat
from dataclasses import dataclass
from pathlib import Path

from .config import MODEL_SHA256, Settings
from .resources import ResourceManager


@dataclass(frozen=True, slots=True)
class Check:
    name: str
    ok: bool
    detail: str
    warning: bool = False


def sha256_file(path: Path, block_size: int = 8 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while block := handle.read(block_size):
            digest.update(block)
    return digest.hexdigest()


def run_checks(settings: Settings, verify_model: bool = False) -> list[Check]:
    checks: list[Check] = []
    binary = shutil.which(settings.llama_binary)
    checks.append(Check("llama.cpp", bool(binary), binary or f"missing {settings.llama_binary}"))
    model = settings.model_path
    checks.append(Check("model file", model.is_file(), f"{model} ({model.stat().st_size // (1024*1024)} MiB)" if model.is_file() else f"missing {model}"))
    if verify_model and model.is_file():
        actual = sha256_file(model)
        checks.append(Check("model checksum", actual == MODEL_SHA256, actual))
    resources = ResourceManager(settings)
    profile = resources.profile()
    socket_ok = settings.socket_path.exists() and stat_is_socket(settings.socket_path)
    if socket_ok:
        enough, available_mb = resources.live_headroom()
        reason = f"model resident; {available_mb} MiB live headroom (minimum 320 MiB)"
    else:
        enough, reason = resources.enough_to_start(profile)
    checks.append(Check("memory safety", enough, reason))
    # Flag the slow path explicitly: this install is portable, and a machine without
    # AVX2 falls back to llama.cpp's generic CPU backend, which is many times slower.
    fast_cpu = profile.cpu_level in {"avx2", "avx512"}
    cpu_detail = f"{profile.cpu_arch}/{profile.cpu_level}, {profile.threads} inference threads"
    if not fast_cpu:
        cpu_detail += " — no AVX2, using llama.cpp's generic backend; expect slow inference"
    # ok=False with warning=True renders as WARN and still passes the overall run.
    checks.append(Check("CPU", fast_cpu, cpu_detail, warning=not fast_cpu))
    disk = shutil.disk_usage(settings.data_dir if settings.data_dir.exists() else settings.data_dir.parent)
    checks.append(Check("disk space", disk.free > 2 * 1024**3, f"{disk.free // (1024**3)} GiB free", warning=disk.free <= 4 * 1024**3))
    for name, path in (("data directory", settings.data_dir), ("runtime directory", settings.runtime_dir), ("log directory", settings.log_dir)):
        checks.append(Check(name, path.exists() and os.access(path, os.W_OK), str(path)))
    try:
        db = sqlite3.connect(settings.database_path)
        result = db.execute("PRAGMA quick_check").fetchone()[0]
        db.close()
        checks.append(Check("memory database", result == "ok", result))
    except sqlite3.Error as exc:
        checks.append(Check("memory database", False, str(exc)))
    checks.append(Check("daemon socket", socket_ok, str(settings.socket_path), warning=not socket_ok))
    return checks


def stat_is_socket(path: Path) -> bool:
    try:
        return stat.S_ISSOCK(path.stat().st_mode)
    except OSError:
        return False
