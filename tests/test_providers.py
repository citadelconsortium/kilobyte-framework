import json
import tempfile
import unittest
from unittest.mock import patch
from pathlib import Path

from kilobyte.providers import ProviderError, ProviderRegistry, _model_ids


def _config(raw: str, payload: dict) -> Path:
    path = Path(raw) / "providers.json"
    path.write_text(json.dumps(payload))
    return path


class ProviderConfigTests(unittest.TestCase):
    def test_model_catalogue_shapes_are_normalised(self):
        self.assertEqual(_model_ids({"data": [{"id": "a"}]}), ["a"])
        self.assertEqual(_model_ids({"models": [{"name": "b"}]}), ["b"])
        self.assertEqual(_model_ids({"result": ["c"]}), ["c"])
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
    def test_openrouter_zero_priced_models_are_free(self):
        class Response:
            def __enter__(self): return self
            def __exit__(self, *args): pass
        with tempfile.TemporaryDirectory() as raw:
            path = _config(raw, {"providers": {"openrouter": {"api_key": "k", "model": "x"}}})
            payload = {"data": [{"id": "free/model", "pricing": {"prompt": "0", "completion": "0"}}, {"id": "paid/model", "pricing": {"prompt": "1", "completion": "1"}}]}
            class JsonResponse(Response):
                def read(self): return json.dumps(payload).encode()
            with patch("urllib.request.urlopen", return_value=JsonResponse()):
                self.assertEqual(ProviderRegistry(path).list_models("openrouter", only_free=True), ["free/model"])
    def test_openrouter_free_picker_never_falls_back_to_paid_default(self):
        class Response:
            def __enter__(self): return self
            def __exit__(self, *args): pass
        with tempfile.TemporaryDirectory() as raw:
            path = _config(raw, {"providers": {"openrouter": {"api_key": "k", "model": "paid/default"}}})
            with patch("urllib.request.urlopen", side_effect=OSError("catalog offline")):
                self.assertEqual(ProviderRegistry(path).list_models("openrouter", only_free=True), [])
    def test_openrouter_scientific_zero_pricing_is_free(self):
        class Response:
            def __enter__(self): return self
            def __exit__(self, *args): pass
        with tempfile.TemporaryDirectory() as raw:
            path = _config(raw, {"providers": {"openrouter": {"api_key": "k", "model": "paid/default"}}})
            payload = {"data": [{"id": "free/scientific", "pricing": {"prompt": "0e-6", "completion": 0}}]}
            class JsonResponse(Response):
                def read(self): return json.dumps(payload).encode()
            with patch("urllib.request.urlopen", return_value=JsonResponse()):
                self.assertEqual(ProviderRegistry(path).list_models("openrouter", only_free=True), ["free/scientific"])
    def test_groq_catalog_uses_current_production_model(self):
        with tempfile.TemporaryDirectory() as raw:
            prov = ProviderRegistry(Path(raw) / "providers.json").configure("groq", "gsk_test")
            self.assertEqual(prov.base_url, "https://api.groq.com/openai/v1")
            self.assertEqual(prov.model, "llama-3.1-8b-instant")

    def test_groq_retired_model_is_migrated_on_read(self):
        with tempfile.TemporaryDirectory() as raw:
            path = _config(raw, {"providers": {"groq": {"api_key": "gsk_test", "model": "llama-3.3-70b-versatile"}}})
            self.assertEqual(ProviderRegistry(path).resolve().model, "llama-3.1-8b-instant")
    def test_huggingface_catalog_uses_router_api(self):
        with tempfile.TemporaryDirectory() as raw:
            registry = ProviderRegistry(Path(raw) / "providers.json")
            prov = registry.configure("huggingface", "hf_test")
            self.assertEqual(prov.base_url, "https://router.huggingface.co/v1")
            self.assertIn("Coder", prov.model)
    def test_new_openai_compatible_catalog_providers_have_correct_endpoints(self):
        with tempfile.TemporaryDirectory() as raw:
            registry = ProviderRegistry(Path(raw) / "providers.json")
            expected = {
                "nvidia": ("https://integrate.api.nvidia.com/v1", "Authorization"),
                "zai": ("https://api.z.ai/api/paas/v4", "Authorization"),
                "ainative": ("https://api.ainative.studio/v1", "X-API-Key"),
                "speka": ("https://speka.me/v1", "Authorization"),
            }
            for name, (base_url, auth_header) in expected.items():
                prov = registry.configure(name, "test-key")
                self.assertEqual(prov.base_url, base_url)
                self.assertEqual(prov.auth_header, auth_header)

    def test_cloudflare_requires_account_scoped_configuration(self):
        with tempfile.TemporaryDirectory() as raw:
            registry = ProviderRegistry(Path(raw) / "providers.json")
            with self.assertRaises(ProviderError):
                registry.configure("cloudflare", "cf_test")

    def test_cloudflare_configures_with_account_id_and_token(self):
        with tempfile.TemporaryDirectory() as raw:
            registry = ProviderRegistry(Path(raw) / "providers.json")
            prov = registry.configure("cloudflare", "cf_test", account_id="abc123")
            self.assertEqual(prov.base_url, "https://api.cloudflare.com/client/v4/accounts/abc123/ai/v1")

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
