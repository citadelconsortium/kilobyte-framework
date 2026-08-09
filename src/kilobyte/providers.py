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
_USER_AGENT = "Kilobyte/1.0 (+https://github.com/citadelconsortium/kilobyte)"


class ProviderError(KilobyteError):
    """A cloud provider could not answer. Never triggers a silent local retry."""


def _model_ids(data: dict[str, Any]) -> list[str]:
    """Normalise common provider model-catalogue response shapes."""
    entries = data.get("data") or data.get("models") or data.get("result") or []
    if isinstance(entries, dict):
        entries = entries.get("models") or entries.get("data") or []
    found: list[str] = []
    for item in entries if isinstance(entries, list) else []:
        if isinstance(item, str):
            value = item
        elif isinstance(item, dict):
            value = item.get("id") or item.get("name") or item.get("model")
        else:
            value = None
        if value and str(value) not in found:
            found.append(str(value))
    return found


@dataclass(frozen=True, slots=True)
class Provider:
    name: str
    base_url: str
    api_key: str
    model: str
    timeout: int = 120
    auth_header: str = "Authorization"

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
    "groq": {"label": "Groq", "base_url": "https://api.groq.com/openai/v1", "model": "llama-3.1-8b-instant"},
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
    "ollama": {"label": "Ollama Cloud", "base_url": "https://ollama.com/v1", "model": "gpt-oss:120b"},
    "agnes": {"label": "Agnes AI", "base_url": "https://apihub.agnes-ai.com/v1", "model": "agnes-2.0-flash"},
    "modelscope": {"label": "ModelScope", "base_url": "https://api-inference.modelscope.cn/v1", "model": "Qwen/Qwen3-32B"},
    "llm7": {"label": "LLM7.io", "base_url": "https://api.llm7.io/v1", "model": "fast"},
    "opencode_zen": {"label": "OpenCode Zen", "base_url": "https://opencode.ai/zen/v1", "model": "big-pickle"},
    "glhf": {"label": "GLHF.chat", "base_url": "https://glhf.chat/api/openai/v1", "model": "hf:meta-llama/Llama-3.3-70B-Instruct"},
}


class ProviderRegistry:
    """Loads provider definitions and streams completions from them on request."""

    def __init__(self, config_path):
        self.config_path = config_path
        self._context_limits: dict[tuple[str, str], int] = {}

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
            if str(name).lower() == "groq" and model in {"llama-3.3-70b-versatile", "llama3-70b-8192"}:
                # These Groq IDs have been retired; keep existing configs usable.
                model = KNOWN_PROVIDERS["groq"]["model"]
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
            auth_header = str(entry.get("auth_header", "Authorization")).strip()
            if auth_header not in {"Authorization", "X-API-Key"}:
                log.warning("provider %s has unsupported auth header; skipped", name)
                continue
            found[str(name)] = Provider(str(name), base_url, key, model, int(entry.get("timeout", 120)), auth_header)
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
        auth_header = known.get("auth_header", "Authorization")
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
            "auth_header": auth_header,
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
        is_openrouter = prov.name.lower() == "openrouter" or "openrouter.ai" in prov.base_url.lower()
        if prov.name.lower() == "cloudflare":
            # Cloudflare does not expose an OpenAI-style /models route. Its account API
            # provides the searchable model catalogue at /ai/models/search.
            url = prov.base_url.replace("/ai/v1", "/ai/models/search") + "?format=openrouter"
        else:
            url = f"{prov.base_url}/models"
        req = urllib.request.Request(
            url,
            headers={prov.auth_header: f"Bearer {prov.api_key}" if prov.auth_header == "Authorization" else prov.api_key, "Accept": "application/json", "User-Agent": _USER_AGENT, **_ATTRIBUTION},
        )
        free_ids: list[str] = []
        try:
            with urllib.request.urlopen(req, timeout=30) as r:
                data = json.load(r)
            ids = _model_ids(data)
            free_ids = []
            if is_openrouter:
                for item in data.get("data") or []:
                    if not isinstance(item, dict): continue
                    ident = item.get("id")
                    pricing = item.get("pricing") or {}
                    prompt_price = str(pricing.get("prompt", "")).strip()
                    completion_price = str(pricing.get("completion", "")).strip()
                    try:
                        zero_priced = float(prompt_price) == 0.0 and float(completion_price) == 0.0
                    except (TypeError, ValueError):
                        zero_priced = False
                    if ident and (str(ident).endswith(":free") or zero_priced):
                        free_ids.append(str(ident))
            for item in (data.get("data") or data.get("models") or data.get("result") or []):
                if not isinstance(item, dict):
                    continue
                model_id = item.get("id") or item.get("name") or item.get("model")
                limits = item.get("limits") if isinstance(item.get("limits"), dict) else {}
                raw_limit = (
                    item.get("context_length")
                    or item.get("context_window")
                    or item.get("max_context_length")
                    or item.get("max_model_len")
                    or item.get("max_input_tokens")
                    or item.get("input_token_limit")
                    or limits.get("max_input_tokens")
                )
                if model_id and raw_limit:
                    try: self._context_limits[(prov.name, str(model_id))] = int(raw_limit)
                    except (TypeError, ValueError): pass
        except Exception as exc:
            # Several otherwise compatible services do not publish a catalogue route.
            # Keep /model useful and honest: expose the known configured model rather than
            # presenting an empty picker, while logging the catalogue failure for diagnosis.
            log.warning("%s model catalogue unavailable (%s); using configured default only when free-only filtering is disabled", prov.name, exc)
            ids = [prov.model]
        if only_free and is_openrouter:
            # Never put a paid configured default into a free-only picker. In
            # particular, a catalogue outage must not silently turn /model into
            # a paid-model list; the caller can retry or enter an explicit model.
            free = sorted(set(free_ids or [i for i in ids if str(i).endswith(":free")]))
            return free
        if prov.model not in ids:
            ids.insert(0, prov.model)
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

    def info(self, name: str | None = None) -> dict[str, Any]:
        """Report the requested provider, not a stale local/default route."""
        selected = name or self.default_name()
        model = ""
        if selected:
            model = self.resolve(selected).model
        return {
            "default": selected,
            "model": model,
            "context_limit": self.context_limit(selected, model),
            "configured": sorted(self.providers()),
        }

    def context_limit(self, name: str | None = None, model: str | None = None) -> int | None:
        try:
            prov = self.resolve(name)
        except ProviderError:
            return None
        chosen = model or prov.model
        value = self._context_limits.get((prov.name, chosen))
        if value:
            return value
        raw = (self._raw().get("providers") or {}).get(prov.name, {}).get("context_limit")
        try: return int(raw) if raw else None
        except (TypeError, ValueError): return None

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

        def open_request(active_payload: dict[str, Any]):
            request = urllib.request.Request(
                f"{provider.base_url}/chat/completions",
                data=json.dumps(active_payload).encode(),
                # The key goes in a header, never a command line or a log.
                headers={
                    "Content-Type": "application/json",
                    provider.auth_header: f"Bearer {provider.api_key}" if provider.auth_header == "Authorization" else provider.api_key,
                    "Accept": "text/event-stream",
                    "User-Agent": _USER_AGENT,
                    **_ATTRIBUTION,
                },
            )
            return urllib.request.urlopen(request, timeout=provider.timeout)

        try:
            response = await asyncio.to_thread(open_request, payload)
        except urllib.error.HTTPError as exc:
            detail = exc.read()[:400].decode("utf-8", "replace")
            # Some OpenAI-compatible endpoints serve models that can reason about tools
            # but reject the native `tools` request field. Retry those models once with the
            # same schemas expressed as an explicit text protocol; agent.py safely recovers
            # the resulting JSON tool envelope against the active interface allow-list.
            tool_schema_error = bool(tools) and exc.code in {400, 404, 415, 422} and any(
                word in detail.lower()
                for word in ("tool", "function", "schema", "unsupported", "unknown field")
            )
            if not tool_schema_error:
                raise ProviderError(f"{provider.label} refused the request ({exc.code}): {detail}") from exc
            compatibility = {
                "role": "system",
                "content": (
                    "This endpoint rejected native tool calling, but the framework still "
                    "provides these tools. To call one, output exactly "
                    '<tool_call>{"name":"TOOL_NAME","arguments":{...}}</tool_call> '
                    "with no surrounding prose. The framework will execute it and return "
                    "the result. Active definitions: "
                    + json.dumps(tools, ensure_ascii=False, separators=(",", ":"))
                ),
            }
            fallback_payload = dict(payload)
            fallback_payload.pop("tools", None)
            fallback_payload.pop("tool_choice", None)
            fallback_payload["messages"] = [*messages, compatibility]
            log.info("%s rejected native tools; using text-tool compatibility", provider.label)
            try:
                response = await asyncio.to_thread(open_request, fallback_payload)
            except urllib.error.HTTPError as retry:
                retry_detail = retry.read()[:400].decode("utf-8", "replace")
                raise ProviderError(
                    f"{provider.label} refused native and compatible tool requests "
                    f"({retry.code}): {retry_detail}"
                ) from retry
            except Exception as retry:
                raise ProviderError(f"{provider.label} unreachable: {retry}") from retry
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
