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

    def resolve(
        self, raw: str, cwd: Path | None = None, must_exist: bool = False
    ) -> Path:
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

    DESTRUCTIVE = {
        "rm",
        "rmdir",
        "shred",
        "wipefs",
        "dd",
        "mkfs",
        "fdisk",
        "parted",
        "shutdown",
        "reboot",
        "poweroff",
        "kill",
        "pkill",
        "killall",
        # disks / filesystems
        "cfdisk",
        "sgdisk",
        "sfdisk",
        "gdisk",
        "mkswap",
        "swapoff",
        "blkdiscard",
        "fsck",
        "truncate",
        "hdparm",
        # accounts / persistence / credentials
        "userdel",
        "groupdel",
        "crontab",
        "passwd",
        "chpasswd",
        # recursive perms / ownership can wreck a tree
        "chown",
        "chmod",
        "chattr",
    }
    ELEVATED = {"sudo", "doas", "su", "systemctl", "mount", "umount", "pacman"}
    SAFE_READ_ONLY = {
        "basename",
        "cat",
        "cmp",
        "cut",
        "date",
        "df",
        "diff",
        "dirname",
        "du",
        "echo",
        "false",
        "file",
        "free",
        "getent",
        "head",
        "hostname",
        "id",
        "journalctl",
        "jq",
        "ls",
        "lscpu",
        "lsblk",
        "md5sum",
        "ps",
        "printf",
        "pwd",
        "readlink",
        "realpath",
        "rg",
        "sha1sum",
        "sha256sum",
        "sort",
        "ss",
        "stat",
        "tail",
        "test",
        "true",
        "uname",
        "uniq",
        "uptime",
        "wc",
        "which",
        "whoami",
    }
    WRITE = {
        "bash",
        "chgrp",
        "cp",
        "curl",
        "git",
        "gh",
        "install",
        "ln",
        "mkdir",
        "mv",
        "node",
        "perl",
        "pip",
        "pip3",
        "python",
        "python3",
        "rename",
        "rsync",
        "scp",
        "sed",
        "sh",
        "ssh",
        "tar",
        "tee",
        "touch",
        "wget",
        "xargs",
        "zsh",
        # Active network/security tools touch external systems and therefore need Sir's
        # explicit approval even though they may not write a local file.
        "aircrack-ng",
        "arp-scan",
        "burpsuite",
        "ffuf",
        "gobuster",
        "hydra",
        "masscan",
        "metasploit",
        "msfconsole",
        "netcat",
        "nikto",
        "nmap",
        "nc",
        "sqlmap",
        "tcpdump",
        "wireshark",
        "zmap",
    }
    SAFE_GIT_SUBCOMMANDS = {
        "describe",
        "diff",
        "grep",
        "log",
        "ls-files",
        "ls-tree",
        "rev-parse",
        "show",
        "status",
    }
    SHELL_TOKENS = {"|", "||", "&&", ";", ">", ">>", "<", "2>", "&"}

    @staticmethod
    def _git_subcommand(argv: list[str]) -> tuple[str, list[str]]:
        index = 1
        options_with_values = {"-C", "-c", "--git-dir", "--work-tree", "--namespace"}
        while index < len(argv):
            argument = argv[index]
            if argument in options_with_values:
                index += 2
                continue
            if argument.startswith(("--git-dir=", "--work-tree=", "--namespace=")):
                index += 1
                continue
            if argument.startswith("-"):
                index += 1
                continue
            return argument, argv[index + 1 :]
        return "", []

    def assess(
        self, command: str | list[str], remote: bool = False
    ) -> CommandAssessment:
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
            raise SecurityError(
                "shell operators are not accepted; execute one program at a time"
            )
        executable = Path(argv[0]).name
        if executable == "find" and any(
            arg in {"-delete", "-exec", "-execdir", "-ok", "-okdir"} for arg in argv[1:]
        ):
            risk, reason = (
                Risk.WRITE,
                "find action can change files or launch another program",
            )
        elif executable in self.DESTRUCTIVE or executable.split(".")[0] in {
            "mkfs",
            "fsck",
        }:
            risk, reason = (
                Risk.DESTRUCTIVE,
                f"{executable} can destroy or interrupt data",
            )
        elif executable == "systemctl" and len(argv) > 1 and argv[1] in {
            "is-active",
            "is-enabled",
            "is-failed",
            "list-unit-files",
            "list-units",
            "show",
            "status",
        }:
            risk, reason = Risk.SAFE, "systemctl subcommand only inspects service state"
        elif executable == "pacman" and len(argv) > 1 and argv[1].startswith("-Q"):
            risk, reason = Risk.SAFE, "pacman query only inspects installed packages"
        elif executable in self.ELEVATED:
            risk, reason = (
                Risk.ELEVATED,
                f"{executable} changes privileged system state",
            )
        elif executable == "git":
            subcommand, subargs = self._git_subcommand(argv)
            if subcommand in self.SAFE_GIT_SUBCOMMANDS:
                risk, reason = Risk.SAFE, f"git {subcommand} only inspects repository state"
            elif subcommand == "tag" and (not subargs or subargs[0] in {"-l", "--list"}):
                risk, reason = Risk.SAFE, "git tag invocation only lists tags"
            else:
                risk, reason = Risk.WRITE, "git invocation can change local or remote state"
        elif executable in self.SAFE_READ_ONLY or executable == "find":
            risk, reason = Risk.SAFE, "read-only inspection command"
        elif executable in self.WRITE:
            risk, reason = (
                Risk.WRITE,
                f"{executable} can change local or external state",
            )
        else:
            # Unknown programs are not assumed harmless. A locally installed binary can
            # mutate anything available to the service account even without a shell.
            risk, reason = Risk.WRITE, "unclassified program may change state"
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
        temp.write_text(
            json.dumps({"permissions": self.rules}, indent=2) + "\n", encoding="utf-8"
        )
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
            if callback is None or not await callback(capability, detail, risk):
                raise PermissionDenied(f"Telegram approval not granted for {capability}")
            return
        rule = self.rules.get(capability)
        if rule == "allow":
            return
        if rule == "deny":
            raise PermissionDenied(f"policy denied {capability}")
        if callback is None or not await callback(capability, detail, risk):
            raise PermissionDenied(f"permission not granted for {capability}")
