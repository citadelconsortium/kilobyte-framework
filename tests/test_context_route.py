import json
import tempfile
import unittest
from pathlib import Path

from kilobyte.providers import ProviderRegistry


class ActiveContextTests(unittest.TestCase):
    def test_explicit_cloud_route_does_not_report_default_or_local_context(self):
        with tempfile.TemporaryDirectory() as raw:
            path = Path(raw) / "providers.json"
            path.write_text(
                json.dumps(
                    {
                        "default": "second",
                        "providers": {
                            "first": {
                                "api_key": "key",
                                "model": "large",
                                "context_limit": 131072,
                            },
                            "second": {
                                "api_key": "key",
                                "model": "small",
                                "context_limit": 8192,
                            },
                        },
                    }
                )
            )
            info = ProviderRegistry(path).info("first")
            self.assertEqual(info["default"], "first")
            self.assertEqual(info["model"], "large")
            self.assertEqual(info["context_limit"], 131072)


if __name__ == "__main__":
    unittest.main()
