import json
import tempfile
import unittest
from pathlib import Path

from kilobyte.providers import ProviderError, ProviderRegistry


def _config(raw: str, payload: dict) -> Path:
    path = Path(raw) / "providers.json"
    path.write_text(json.dumps(payload))
    return path


class ProviderConfigTests(unittest.TestCase):
    def test_absent_config_means_no_cloud_path_at_all(self):
        """Cloud must be off unless it was deliberately configured."""
        with tempfile.TemporaryDirectory() as raw:
            registry = ProviderRegistry(Path(raw) / "absent.json")
            self.assertEqual(registry.providers(), {})
            self.assertIsNone(registry.default_name())
            with self.assertRaises(ProviderError):
                registry.resolve()

    def test_placeholder_or_missing_key_is_not_configured(self):
        with tempfile.TemporaryDirectory() as raw:
            path = _config(raw, {"providers": {
                "a": {"api_key": "PASTE_OPENROUTER_KEY_HERE", "model": "m"},
                "b": {"api_key": "", "model": "m"},
                "c": {"api_key": "k"},
            }})
            self.assertEqual(ProviderRegistry(path).providers(), {})

    def test_plaintext_base_url_is_refused(self):
        """The key travels in a header; http would put it on the wire in clear."""
        with tempfile.TemporaryDirectory() as raw:
            path = _config(raw, {"providers": {
                "insecure": {"api_key": "k", "model": "m", "base_url": "http://example.com/v1"},
            }})
            self.assertEqual(ProviderRegistry(path).providers(), {})

    def test_disabled_provider_is_skipped(self):
        with tempfile.TemporaryDirectory() as raw:
            path = _config(raw, {"providers": {
                "off": {"api_key": "k", "model": "m", "enabled": False},
            }})
            self.assertEqual(ProviderRegistry(path).providers(), {})

    def test_default_selection_and_unknown_name(self):
        with tempfile.TemporaryDirectory() as raw:
            path = _config(raw, {"default": "second", "providers": {
                "first": {"api_key": "k", "model": "m1"},
                "second": {"api_key": "k", "model": "m2"},
            }})
            registry = ProviderRegistry(path)
            self.assertEqual(registry.default_name(), "second")
            self.assertEqual(registry.resolve(None).model, "m2")
            self.assertEqual(registry.resolve("first").model, "m1")
            with self.assertRaises(ProviderError):
                registry.resolve("missing")

    def test_malformed_config_is_not_fatal(self):
        with tempfile.TemporaryDirectory() as raw:
            path = Path(raw) / "providers.json"
            path.write_text("{not json")
            self.assertEqual(ProviderRegistry(path).providers(), {})

    def test_label_identifies_the_brain_that_answered(self):
        with tempfile.TemporaryDirectory() as raw:
            path = _config(raw, {"providers": {"openrouter": {"api_key": "k", "model": "some/model"}}})
            self.assertEqual(ProviderRegistry(path).resolve().label, "openrouter:some/model")


if __name__ == "__main__":
    unittest.main()


class ProviderConfigureTests(unittest.TestCase):
    def test_configure_from_just_a_key_uses_catalog_and_sets_default(self):
        """A user supplies only an API key; base_url and model come from the catalog, and
        the provider becomes the default so /cloud reaches it immediately."""
        with tempfile.TemporaryDirectory() as raw:
            registry = ProviderRegistry(Path(raw) / "providers.json")
            prov = registry.configure("openrouter", "sk-or-test123")
            self.assertEqual(prov.name, "openrouter")
            self.assertTrue(prov.base_url.startswith("https://"))
            self.assertTrue(prov.model)
            self.assertEqual(registry.default_name(), "openrouter")
            self.assertIn("openrouter", registry.providers())

    def test_configure_rejects_empty_key(self):
        with tempfile.TemporaryDirectory() as raw:
            registry = ProviderRegistry(Path(raw) / "providers.json")
            with self.assertRaises(ProviderError):
                registry.configure("openrouter", "   ")

    def test_configure_writes_600_and_persists(self):
        import os, stat
        with tempfile.TemporaryDirectory() as raw:
            path = Path(raw) / "providers.json"
            ProviderRegistry(path).configure("groq", "gsk-test")
            mode = stat.S_IMODE(os.stat(path).st_mode)
            self.assertEqual(mode, 0o600)
            # A fresh registry reading the same file sees it (live, no restart).
            self.assertIn("groq", ProviderRegistry(path).providers())
