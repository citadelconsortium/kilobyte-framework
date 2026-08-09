"""Optional cloud brains, used only when explicitly asked for.

Kilobyte's brain is the local GGUF. This module exists so a request can be escalated to a
larger model on purpose -- when the local one is too slow or too small for the job -- not
so inference quietly moves off the machine.

The rules that make that distinction real, rather than a claim:

* Local is always the default. A cloud provider is used only for a request the operator
  escalated, never as an automatic fallback and never because the local model was slow.
* Nothing is configured by default. With no providers file there is no cloud path at all.
* The answer says which brain produced it, so an escalated reply is never mistaken for a
  local one.
* Keys live in a 0600 file, are never logged, and are never passed on a command line.

Providers speak the OpenAI chat-completions shape, which OpenRouter and most hosted
services expose, so one client covers them.
"""

from __future__ import annotations

import json
import logging
import urllib.error
import urllib.request
from collections.abc import AsyncIterator
from dataclasses import dataclass
from typing import Any

import asyncio

from .errors import KilobyteError


log = logging.getLogger("kilobyte.providers")

# Sent by OpenRouter's convention so usage is attributable to this project.
_ATTRIBUTION = {
    "HTTP-Referer": "https://github.com/citadelconsortium/kilobyte",
    "X-Title": "Kilobyte",
}


class ProviderError(KilobyteError):
    """A cloud provider could not answer. Never triggers a silent local retry."""


@dataclass(frozen=True, slots=True)
class Provider:
    name: str
    base_url: str
    api_key: str
    model: str
    timeout: int = 120

    @property
    def label(self) -> str:
        return f"{self.name}:{self.model}"


# A catalog of well-known OpenAI-compatible endpoints. Because the base URL and a sensible
# default model are known ahead of time, configuring cloud escalation needs nothing from the
# user but an API key — pick the provider, paste the key. Users can still override the model.
KNOWN_PROVIDERS: dict[str, dict[str, str]] = {
    "openrouter": {"label": "OpenRouter", "base_url": "https://openrouter.ai/api/v1", "model": "anthropic/claude-sonnet-4.5"},
    "openai": {"label": "OpenAI", "base_url": "https://api.openai.com/v1", "model": "gpt-4o"},
    "anthropic": {"label": "Anthropic", "base_url": "https://api.anthropic.com/v1", "model": "claude-sonnet-4-5"},
    "groq": {"label": "Groq", "base_url": "https://api.groq.com/openai/v1", "model": "llama-3.3-70b-versatile"},
    "deepseek": {"label": "DeepSeek", "base_url": "https://api.deepseek.com/v1", "model": "deepseek-chat"},
    "together": {"label": "Together", "base_url": "https://api.together.xyz/v1", "model": "meta-llama/Llama-3.3-70B-Instruct-Turbo"},
    "mistral": {"label": "Mistral", "base_url": "https://api.mistral.ai/v1", "model": "mistral-large-latest"},
    "xai": {"label": "xAI (Grok)", "base_url": "https://api.x.ai/v1", "model": "grok-2-latest"},
    "gemini": {"label": "Google Gemini", "base_url": "https://generativelanguage.googleapis.com/v1beta/openai", "model": "gemini-2.0-flash"},
    "cerebras": {"label": "Cerebras", "base_url": "https://api.cerebras.ai/v1", "model": "llama-3.3-70b"},
    "fireworks": {"label": "Fireworks", "base_url": "https://api.fireworks.ai/inference/v1", "model": "accounts/fireworks/models/llama-v3p3-70b-instruct"},
    "perplexity": {"label": "Perplexity", "base_url": "https://api.perplexity.ai", "model": "sonar"},
    "nebius": {"label": "Nebius", "base_url": "https://api.studio.nebius.ai/v1", "model": "meta-llama/Llama-3.3-70B-Instruct"},
    "hyperbolic": {"label": "Hyperbolic", "base_url": "https://api.hyperbolic.xyz/v1", "model": "meta-llama/Llama-3.3-70B-Instruct"},
    "cohere": {"label": "Cohere", "base_url": "https://api.cohere.com/compatibility/v1", "model": "command-a-03-2025"},
    "sambanova": {"label": "SambaNova", "base_url": "https://api.sambanova.ai/v1", "model": "Meta-Llama-3.3-70B-Instruct"},
    "qwen": {"label": "Alibaba Qwen", "base_url": "https://dashscope.aliyuncs.com/compatible-mode/v1", "model": "qwen-plus"},
    "huggingface": {"label": "Hugging Face Inference Providers", "base_url": "https://router.huggingface.co/v1", "model": "Qwen/Qwen2.5-Coder-32B-Instruct"},
    # Workers AI's endpoint is account-scoped. configure() resolves the account from
    # KILOBYTE_CLOUDFLARE_ACCOUNT_ID; a raw providers.json may also set its own URL.
    "cloudflare": {"label": "Cloudflare Workers AI", "base_url": "https://api.cloudflare.com/client/v4/accounts/{account_id}/ai/v1", "model": "@cf/meta/llama-3.1-8b-instruct"},
}


class ProviderRegistry:
    """Loads provider definitions and streams completions from them on request."""

    def __init__(self, config_path):
        self.config_path = config_path

    def _raw(self) -> dict[str, Any]:
        try:
            return json.loads(self.config_path.read_text(encoding="utf-8"))
        except FileNotFoundError:
            return {}
        except (OSError, json.JSONDecodeError) as exc:
            log.warning("providers config unreadable (%s); cloud escalation stays off", exc)
            return {}

    def providers(self) -> dict[str, Provider]:
        raw = self._raw()
        found: dict[str, Provider] = {}
        for name, entry in (raw.get("providers") or {}).items():
            if not isinstance(entry, dict) or not entry.get("enabled", True):
                continue
            key = str(entry.get("api_key", "")).strip()
            model = str(entry.get("model", "")).strip()
            base_url = str(entry.get("base_url", "https://openrouter.ai/api/v1")).strip().rstrip("/")
            if not key or key.startswith("PASTE_"):
                log.warning("provider %s has no api key; skipped", name)
                continue
            if not model:
                log.warning("provider %s has no model; skipped", name)
                continue
            if not base_url.startswith("https://"):
                # A key would otherwise be sent in clear text.
                log.warning("provider %s must use https; skipped", name)
                continue
            found[str(name)] = Provider(str(name), base_url, key, model, int(entry.get("timeout", 120)))
        return found

    def configure(self, name: str, api_key: str, model: str | None = None, account_id: str | None = None) -> Provider:
        """Add or update a provider from just an API key (and optional model), using the
        catalog for the base URL and default model, then make it the default. Written 0600
        because it holds the key. Takes effect immediately: the registry reads the file live.
        """
        import os

        name = name.strip().lower()
        known = KNOWN_PROVIDERS.get(name, {})
        base_url = known.get("base_url", "https://openrouter.ai/api/v1")
        if name == "cloudflare":
            account_id = (account_id or os.environ.get("KILOBYTE_CLOUDFLARE_ACCOUNT_ID", "")).strip()
            if not account_id or not account_id.replace("-", "").isalnum():
                raise ProviderError("Cloudflare needs its account ID as well as the API token")
            base_url = base_url.replace("{account_id}", account_id)
        chosen_model = (model or known.get("model") or "").strip()
        if not api_key.strip():
            raise ProviderError("an API key is required")
        if not chosen_model:
            raise ProviderError(f"no default model known for {name}; pass a model explicitly")
        raw = self._raw()
        raw.setdefault("providers", {})[name] = {
            "base_url": base_url,
            "api_key": api_key.strip(),
            "model": chosen_model,
            "enabled": True,
        }
        raw["default"] = name
        self.config_path.parent.mkdir(parents=True, exist_ok=True)
        self.config_path.write_text(json.dumps(raw, indent=2) + "\n", encoding="utf-8")
        try:
            os.chmod(self.config_path, 0o600)
        except OSError:
            pass
        return self.resolve(name)

    def list_models(self, name: str | None = None, only_free: bool = True) -> list[str]:
        """Fetch the provider's model catalogue so the user can pick without researching.
        For OpenRouter, free models (id ending ':free') are surfaced first."""
        import urllib.request
        prov = self.resolve(name)
        if prov.name.lower() == "cloudflare":
            # Cloudflare does not expose an OpenAI-style /models route. Its account API
            # provides the searchable model catalogue at /ai/models/search.
            url = prov.base_url.replace("/ai/v1", "/ai/models/search") + "?format=openrouter"
        else:
            url = f"{prov.base_url}/models"
        req = urllib.request.Request(
            url,
            headers={"Authorization": f"Bearer {prov.api_key}", "Accept": "application/json", **_ATTRIBUTION},
        )
        with urllib.request.urlopen(req, timeout=30) as r:
            data = json.load(r)
        entries = data.get("data") or data.get("result") or []
        ids = [m.get("id") for m in entries if isinstance(m, dict) and m.get("id")]
        if only_free and "openrouter" in prov.base_url:
            free = sorted(i for i in ids if str(i).endswith(":free"))
            if free:
                return free
        return sorted(ids)

    def set_model(self, name: str, model: str) -> str:
        """Change the model for a configured provider without needing the key again."""
        import os as _os
        name = name.strip().lower()
        raw = self._raw()
        provs = raw.get("providers") or {}
        if name not in provs:
            raise ProviderError(f"provider {name} is not configured")
        provs[name]["model"] = model.strip()
        raw["providers"] = provs
        self.config_path.write_text(json.dumps(raw, indent=2) + "\n", encoding="utf-8")
        try:
            _os.chmod(self.config_path, 0o600)
        except OSError:
            pass
        return model.strip()

    def info(self) -> dict[str, Any]:
        """Which provider is default and what model it currently uses."""
        name = self.default_name()
        model = ""
        if name:
            raw = self._raw()
            model = str((raw.get("providers") or {}).get(name, {}).get("model", ""))
        return {"default": name, "model": model, "configured": sorted(self.providers())}

    def default_name(self) -> str | None:
        raw = self._raw()
        available = self.providers()
        preferred = str(raw.get("default", "")).strip()
        if preferred and preferred in available:
            return preferred
        return next(iter(available), None)

    def resolve(self, name: str | None = None) -> Provider:
        available = self.providers()
        if not available:
            raise ProviderError(
                "no cloud provider is configured; add one to the providers file to use /cloud"
            )
        chosen = name or self.default_name()
        if chosen not in available:
            raise ProviderError(f"unknown provider: {chosen}. Configured: {', '.join(sorted(available))}")
        return available[chosen]

    async def stream(self, provider: Provider, messages: list[dict[str, Any]], max_tokens: int, tools: list[dict[str, Any]] | None = None) -> AsyncIterator[dict[str, Any]]:
        """Stream an escalated completion, in the same event shape the local runtime uses.

        Tools are forwarded so an escalated (cloud) model has the same terminal/file/web
        tools the local model does — without them a cloud model is blind to the machine and
        just guesses, which reads as 'confused, no terminal access'."""
        payload = {
            "model": provider.model,
            "messages": messages,
            "max_tokens": max_tokens,
            "stream": True,
        }
        if tools:
            payload["tools"] = tools
            payload["tool_choice"] = "auto"

        def open_request():
            request = urllib.request.Request(
                f"{provider.base_url}/chat/completions",
                data=json.dumps(payload).encode(),
                # The key goes in a header, never a command line or a log.
                headers={
                    "Content-Type": "application/json",
                    "Authorization": f"Bearer {provider.api_key}",
                    "Accept": "text/event-stream",
                    **_ATTRIBUTION,
                },
            )
            return urllib.request.urlopen(request, timeout=provider.timeout)

        try:
            response = await asyncio.to_thread(open_request)
        except urllib.error.HTTPError as exc:
            detail = exc.read()[:400].decode("utf-8", "replace")
            raise ProviderError(f"{provider.label} refused the request ({exc.code}): {detail}") from exc
        except Exception as exc:
            raise ProviderError(f"{provider.label} unreachable: {exc}") from exc

        try:
            while True:
                raw = await asyncio.to_thread(response.readline)
                if not raw:
                    break
                line = raw.decode("utf-8", "replace").strip()
                if not line.startswith("data:"):
                    continue
                data = line[5:].strip()
                if data == "[DONE]":
                    break
                try:
                    event = json.loads(data)
                except json.JSONDecodeError:
                    continue
                choices = event.get("choices") or []
                if choices:
                    delta = dict(choices[0].get("delta") or {})
                    delta.pop("reasoning", None)
                    delta.pop("reasoning_content", None)
                    yield {"delta": delta, "finish_reason": choices[0].get("finish_reason")}
                if event.get("usage"):
                    yield {"usage": event["usage"]}
        finally:
            response.close()
