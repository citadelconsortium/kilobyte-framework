"""Kilobyte brain lifecycle: candidate → current → previous, with rollback.

Training produces a candidate GGUF. A candidate must never overwrite the brain Kilo is
using until it has passed evaluation, and the last known-good brain must always survive so
a bad promotion can be undone. This module owns that discipline; the training pipeline and
the CLI call into it, so the rules live in one tested place.

Layout under the data directory:

    models/current/kilobyte.gguf     the brain Kilo loads
    models/candidate/kilobyte.gguf   a freshly trained brain awaiting evaluation
    models/previous/kilobyte.gguf    the brain that current replaced, kept for rollback

Promotion and rollback are atomic per file: a candidate is verified (exists, sane size,
optionally checksum) before anything moves, current is copied to previous, then the
candidate replaces current by atomic rename within the same filesystem.
"""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import time
from dataclasses import dataclass
from pathlib import Path


CANONICAL_NAME = "kilobyte.gguf"
# A GGUF small enough to be truncated or a stray text file is never a real brain.
MIN_PLAUSIBLE_BYTES = 100 * 1024 * 1024
GGUF_MAGIC = b"GGUF"


class BrainError(Exception):
    """A brain operation could not be completed safely."""


@dataclass(frozen=True, slots=True)
class BrainInfo:
    slot: str
    path: Path
    exists: bool
    size: int
    sha256: str | None


def _sha256(path: Path, block: int = 8 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(block):
            digest.update(chunk)
    return digest.hexdigest()


class BrainManager:
    def __init__(self, models_dir: Path, min_bytes: int = MIN_PLAUSIBLE_BYTES):
        self.models_dir = models_dir
        self.min_bytes = min_bytes

    def slot_path(self, slot: str) -> Path:
        return self.models_dir / slot / CANONICAL_NAME

    def info(self, slot: str, with_hash: bool = False) -> BrainInfo:
        path = self.slot_path(slot)
        if not path.is_file():
            return BrainInfo(slot, path, False, 0, None)
        size = path.stat().st_size
        return BrainInfo(slot, path, True, size, _sha256(path) if with_hash else None)

    def validate(self, path: Path, expected_sha256: str | None = None) -> None:
        """Refuse anything that is not a plausible, intact GGUF before it is promoted."""
        if not path.is_file():
            raise BrainError(f"no file at {path}")
        size = path.stat().st_size
        if size < self.min_bytes:
            raise BrainError(f"{path} is only {size} bytes; not a complete GGUF")
        with path.open("rb") as handle:
            if handle.read(4) != GGUF_MAGIC:
                raise BrainError(f"{path} is not a GGUF (bad magic)")
        if expected_sha256 is not None:
            actual = _sha256(path)
            if actual != expected_sha256:
                raise BrainError(f"checksum mismatch: expected {expected_sha256}, got {actual}")

    def stage_candidate(self, source: Path, expected_sha256: str | None = None) -> BrainInfo:
        """Place a freshly built GGUF in the candidate slot without touching current."""
        self.validate(source, expected_sha256)
        candidate = self.slot_path("candidate")
        candidate.parent.mkdir(parents=True, exist_ok=True)
        tmp = candidate.with_suffix(".gguf.staging")
        shutil.copy2(source, tmp)
        os.replace(tmp, candidate)
        self._write_meta("candidate", {"staged_at": time.time(), "source": str(source), "sha256": _sha256(candidate)})
        return self.info("candidate", with_hash=True)

    def promote(self, expected_sha256: str | None = None, brain_version: str = "unknown", framework_version: str = "unknown") -> BrainInfo:
        """Make the candidate the current brain, keeping the old current as previous.

        Nothing is destroyed: current is preserved in previous before it is replaced, so a
        failed brain can always be rolled back to the one that worked.
        """
        candidate = self.slot_path("candidate")
        self.validate(candidate, expected_sha256)
        current = self.slot_path("current")
        previous = self.slot_path("previous")
        current.parent.mkdir(parents=True, exist_ok=True)
        previous.parent.mkdir(parents=True, exist_ok=True)

        if current.is_file():
            # Keep the outgoing brain as the rollback target.
            tmp_prev = previous.with_suffix(".gguf.staging")
            shutil.copy2(current, tmp_prev)
            os.replace(tmp_prev, previous)

        # Move the candidate into place atomically within the same filesystem.
        tmp_cur = current.with_suffix(".gguf.staging")
        shutil.copy2(candidate, tmp_cur)
        os.replace(tmp_cur, current)
        digest = _sha256(current)
        self._write_meta("current", {"promoted_at": time.time(), "sha256": digest, "brain_version": brain_version})
        self.record_version("promote", brain_version, framework_version, digest)
        candidate.unlink(missing_ok=True)
        return self.info("current", with_hash=True)

    def rollback(self) -> BrainInfo:
        """Restore the previous brain as current after a failed promotion."""
        previous = self.slot_path("previous")
        if not previous.is_file():
            raise BrainError("no previous brain to roll back to")
        self.validate(previous)
        current = self.slot_path("current")
        tmp_cur = current.with_suffix(".gguf.staging")
        shutil.copy2(previous, tmp_cur)
        os.replace(tmp_cur, current)
        digest = _sha256(current)
        self._write_meta("current", {"rolled_back_at": time.time(), "sha256": digest})
        self.record_version("rollback", self.current_version() or "previous", "unknown", digest)
        return self.info("current", with_hash=True)

    def _write_meta(self, slot: str, data: dict) -> None:
        meta = self.models_dir / slot / "brain.json"
        try:
            meta.write_text(json.dumps(data, indent=2) + "\n", encoding="utf-8")
        except OSError:
            pass

    @property
    def history_path(self) -> Path:
        return self.models_dir / "versions.json"

    def record_version(self, event: str, brain_version: str, framework_version: str, sha256: str) -> None:
        """Append one immutable entry to the brain version log.

        Every promotion and rollback is recorded so a release is auditable and a specific
        brain version can be identified for rollback -- the model equivalent of a git tag.
        """
        history = self.versions()
        history.append({
            "event": event,
            "brain_version": brain_version,
            "framework_version": framework_version,
            "sha256": sha256,
            "at": time.time(),
        })
        try:
            self.history_path.parent.mkdir(parents=True, exist_ok=True)
            self.history_path.write_text(json.dumps(history, indent=2) + "\n", encoding="utf-8")
        except OSError:
            pass

    def versions(self) -> list[dict]:
        try:
            return json.loads(self.history_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return []

    def current_version(self) -> str | None:
        for entry in reversed(self.versions()):
            if entry.get("event") in {"promote", "rollback"}:
                return entry.get("brain_version")
        return None

    def status(self) -> dict[str, dict[str, object]]:
        out: dict[str, dict[str, object]] = {}
        for slot in ("current", "candidate", "previous"):
            info = self.info(slot)
            out[slot] = {"slot": info.slot, "path": info.path, "exists": info.exists, "size": info.size, "sha256": info.sha256}
        return out
