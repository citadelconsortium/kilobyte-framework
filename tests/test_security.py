import tempfile
import unittest
from pathlib import Path

from kilobyte.errors import PermissionDenied, SecurityError
from kilobyte.security import CommandPolicy, PathPolicy, PermissionManager, Risk


class SecurityTests(unittest.IsolatedAsyncioTestCase):
    def test_paths_cannot_escape_roots(self):
        with tempfile.TemporaryDirectory() as root:
            policy = PathPolicy((Path(root),))
            self.assertEqual(
                policy.resolve("file.txt", Path(root)), Path(root) / "file.txt"
            )
            with self.assertRaises(SecurityError):
                policy.resolve("/etc/shadow")

    def test_commands_are_shell_free_and_classified(self):
        policy = CommandPolicy()
        self.assertEqual(policy.assess("ls -la").risk, Risk.SAFE)
        self.assertEqual(policy.assess("git status --short").risk, Risk.SAFE)
        self.assertEqual(policy.assess("git -C /tmp status --short").risk, Risk.SAFE)
        self.assertEqual(policy.assess("git tag release").risk, Risk.WRITE)
        self.assertEqual(policy.assess("git push origin main").risk, Risk.WRITE)
        self.assertEqual(policy.assess("python -c pass").risk, Risk.WRITE)
        self.assertEqual(policy.assess("nmap 192.0.2.1").risk, Risk.WRITE)
        self.assertEqual(policy.assess("unknown-helper --do-it").risk, Risk.WRITE)
        self.assertEqual(policy.assess("ip link set lo down").risk, Risk.WRITE)
        self.assertEqual(policy.assess("systemctl status kilobyte").risk, Risk.SAFE)
        self.assertEqual(policy.assess("sudo pacman -S x").risk, Risk.ELEVATED)
        self.assertEqual(policy.assess("rm thing").risk, Risk.DESTRUCTIVE)
        with self.assertRaises(SecurityError):
            policy.assess("ls | head")
        self.assertEqual(policy.assess("find . -delete").risk, Risk.WRITE)
        with self.assertRaises(PermissionDenied):
            policy.assess("sudo true", remote=True)

    async def test_remote_write_permission_is_always_denied(self):
        with tempfile.TemporaryDirectory() as root:
            manager = PermissionManager(Path(root) / "policy.json")
            with self.assertRaises(PermissionDenied):
                await manager.authorize("filesystem.write", "x", Risk.WRITE, True, None)


if __name__ == "__main__":
    unittest.main()
