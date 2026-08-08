from __future__ import annotations

import json
import os
import shlex
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Awaitable, Callable

from .errors import PermissionDenied, SecurityError


class Risk(str, Enum):
    SAFE = "safe"
    WRITE = "write"
    ELEVATED = "elevated"
    DESTRUCTIVE = "destructive"


PermissionCallback = Callable[[str, str, Risk], Awaitable[bool]]


@dataclass(frozen=True, slots=True)
class CommandAssessment:
    argv: tuple[str, ...]
    risk: Risk
    reason: str


class PathPolicy:
    def __init__(self, roots: tuple[Path, ...]):
        self.roots = tuple(root.expanduser().resolve() for root in roots)

    def resolve(self, raw: str, cwd: Path | None = None, must_exist: bool = False) -> Path:
        candidate = Path(raw).expanduser()
        if not candidate.is_absolute():
            candidate = (cwd or Path.cwd()) / candidate
        try:
            resolved = candidate.resolve(strict=must_exist)
        except (OSError, RuntimeError) as exc:
            raise SecurityError(f"cannot safely resolve path: {raw}") from exc
        if not any(resolved == root or root in resolved.parents for root in self.roots):
            raise SecurityError(f"path is outside allowed roots: {resolved}")
        return resolved


class CommandPolicy:
    """Shell-free command gate with explicit high-risk classification."""

    NEVER_REMOTE = {
        "sudo", "su", "doas", "mount", "umount", "fdisk", "parted", "mkfs",
        "shutdown", "reboot", "poweroff", "systemctl", "iptables", "nft",
    }
    DESTRUCTIVE = {
        "rm", "rmdir", "shred", "wipefs", "dd", "mkfs", "fdisk", "parted",
        "shutdown", "reboot", "poweroff", "kill", "pkill", "killall",
        # disks / filesystems
        "cfdisk", "sgdisk", "sfdisk", "gdisk", "mkswap", "swapoff", "blkdiscard",
        "fsck", "truncate", "hdparm",
        # accounts / persistence / credentials
        "userdel", "groupdel", "crontab", "passwd", "chpasswd",
        # recursive perms / ownership can wreck a tree
        "chown", "chmod", "chattr",
    }
    ELEVATED = {"sudo", "doas", "su", "systemctl", "mount", "umount", "pacman"}
    SHELL_TOKENS = {"|", "||", "&&", ";", ">", ">>", "<", "2>", "&"}

    def assess(self, command: str | list[str], remote: bool = False) -> CommandAssessment:
        if isinstance(command, str):
            try:
                argv = shlex.split(command)
            except ValueError as exc:
                raise SecurityError(f"invalid command: {exc}") from exc
        else:
            argv = [str(item) for item in command]
        if not argv:
            raise SecurityError("empty command")
        if any(token in self.SHELL_TOKENS for token in argv):
            raise SecurityError("shell operators are not accepted; execute one program at a time")
        executable = Path(argv[0]).name
        if remote and executable in self.NEVER_REMOTE:
            raise PermissionDenied(f"{executable} is blocked over Telegram")
        if executable in self.DESTRUCTIVE or executable.split(".")[0] in {"mkfs", "fsck"}:
            risk, reason = Risk.DESTRUCTIVE, f"{executable} can destroy or interrupt data"
        elif executable in self.ELEVATED:
            risk, reason = Risk.ELEVATED, f"{executable} changes privileged system state"
        else:
            risk, reason = Risk.SAFE, "non-shell command with bounded execution"
        return CommandAssessment(tuple(argv), risk, reason)


class PermissionManager:
    def __init__(self, policy_path: Path):
        self.policy_path = policy_path
        self.rules = self._load()

    def _load(self) -> dict[str, str]:
        try:
            data = json.loads(self.policy_path.read_text(encoding="utf-8"))
            return {str(k): str(v) for k, v in data.get("permissions", {}).items()}
        except (FileNotFoundError, ValueError, OSError):
            return {}

    def save(self) -> None:
        self.policy_path.parent.mkdir(parents=True, exist_ok=True)
        temp = self.policy_path.with_suffix(".tmp")
        temp.write_text(json.dumps({"permissions": self.rules}, indent=2) + "\n", encoding="utf-8")
        os.chmod(temp, 0o600)
        temp.replace(self.policy_path)

    async def authorize(
        self,
        capability: str,
        detail: str,
        risk: Risk,
        remote: bool,
        callback: PermissionCallback | None,
    ) -> None:
        if risk is Risk.SAFE:
            return
        if remote:
            raise PermissionDenied(f"remote policy denied {capability}: {detail}")
        rule = self.rules.get(capability)
        if rule == "allow":
            return
        if rule == "deny":
            raise PermissionDenied(f"policy denied {capability}")
        if callback is None or not await callback(capability, detail, risk):
            raise PermissionDenied(f"permission not granted for {capability}")

