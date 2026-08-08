import tempfile
import unittest
from pathlib import Path

from kilobyte.config import Settings
from kilobyte.resources import ResourceManager, ResourceProfile
from kilobyte.runtime import LlamaRuntime


class RuntimeTests(unittest.TestCase):
    def test_command_is_single_model_and_local_only(self):
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw)
            settings = Settings(data_dir=root, config_dir=root, runtime_dir=root, log_dir=root, home=root)
            profile = ResourceProfile(4096, 3000, 2300, 1280, 4096, 2, 128, None, 0, "x86_64", "sse4")
            command = LlamaRuntime(settings, ResourceManager(settings)).command(profile)
            self.assertEqual(command.count("--model"), 1)
            self.assertIn("127.0.0.1", command)
            self.assertIn("--jinja", command)
            self.assertIn("--parallel", command)
            self.assertEqual(command[command.index("--parallel") + 1], "1")


if __name__ == "__main__":
    unittest.main()

