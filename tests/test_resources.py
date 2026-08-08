import tempfile
import unittest
from pathlib import Path

from kilobyte.config import Settings
from kilobyte.resources import ResourceManager


class ResourceTests(unittest.TestCase):
    def test_profile_is_dynamic_and_sane(self):
        with tempfile.TemporaryDirectory() as root:
            root = Path(root)
            settings = Settings(data_dir=root, config_dir=root, runtime_dir=root, log_dir=root, home=root)
            profile = ResourceManager(settings).profile()
            self.assertGreaterEqual(profile.threads, 1)
            self.assertIn(profile.context_size, {2048, 4096, 8192})
            self.assertLessEqual(profile.safe_available_mb, profile.available_mb)
            self.assertGreater(profile.model_mb, 0)
            ok, headroom = ResourceManager(settings).live_headroom()
            self.assertIsInstance(ok, bool)
            self.assertGreaterEqual(headroom, 0)

    def test_explicit_context_override(self):
        with tempfile.TemporaryDirectory() as root:
            root = Path(root)
            settings = Settings(data_dir=root, config_dir=root, runtime_dir=root, log_dir=root, home=root, context_size=3072)
            self.assertEqual(ResourceManager(settings).profile().context_size, 3072)


if __name__ == "__main__":
    unittest.main()
