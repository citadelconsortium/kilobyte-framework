# Framework Builder Notes / Handoff

This is the maintainer handoff for future agents. The framework is the reusable
brain-free edition of Kilobyte.

## Product contract

The framework supplies the orchestrator, agent profiles, tools, memory, MCP,
approval/security policy, Telegram bridge, and bordered TUI. It does **not** ship
weights. Operators choose `/cloud` or deploy their own GGUF with `/gguf` or
`kilo brain deploy PATH`.

The same Sir directive, evidence-first grounding, follow-through loop, and
approval boundaries apply here. Do not silently turn cloud mode into an
automatic fallback or remove destructive-action approval.

## Provider catalog

OpenAI-compatible entries include OpenRouter, OpenAI, Anthropic, Groq, DeepSeek,
Together, Mistral, xAI, Gemini, Cerebras, Fireworks, Perplexity, Nebius,
Hyperbolic, Cohere, SambaNova, Alibaba Qwen, Hugging Face Inference Providers, and
Cloudflare Workers AI. Keys are stored in a 0600 providers file and providers require
HTTPS. Cloudflare requires an account-scoped base URL or
`KILOBYTE_CLOUDFLARE_ACCOUNT_ID`; `/model` fetches the selected provider's live model
catalog. GitHub Models was retired in July 2026 and is intentionally not advertised.
Hermes Agent is a client rather than a separate inference endpoint. Groq uses
`https://api.groq.com/openai/v1`; requests include a project user-agent to avoid
edge-signature blocking, and retired `llama-3.3-70b-versatile` configs migrate to
`llama-3.1-8b-instant` on read.

## Install / use

The one-line installer bootstraps this repository, installs the app/service, and
does not download a brain. Use `/cloud` for a configured provider or `/gguf` for
an operator-supplied local model. `/botkey` configures Telegram through the daemon
RPC; Telegram `/commands` is an alias for `/help`.
The live stats bar intentionally omits the user's request text so status indicators stay
compact; it shows phase, request count, tools, tokens, model, queue, and context instead.
Cloud context is shown only when the selected model API reports a verified limit.
The TUI retains background RPC/monitor tasks and shows their live count in the stats bar;
the daemon separately monitors and restarts a failed local runtime.

## Verification and limits

The full VM suite is 107 tests and passes. The Framework installer is intentionally
usable without `llama-server` for cloud-only operation. Local GGUF performance is
bounded by the target machine; advanced coding/security work should use capable
hardware or an explicitly selected cloud model. Keep the security profile and
permission gate intact.

## Handoff checklist

Run the full unittest discovery, shell-check both installers, verify provider
catalog HTTPS/default models, and exercise `kilo status`, `/commands`, `/botkey`,
`/cloud`, `/gguf`, and approval prompts before release. Update this file and the
wiki when contracts, providers, or brain provenance changes.
