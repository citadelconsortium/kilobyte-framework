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

The security profile has no canned playbook. It operates only on the exact target and
scope Sir supplies, derives a custom method from evidence and steering, and can recall/save
verified methods. Do not restore the removed seeded security-playbook memory.

## Provider catalog

OpenAI-compatible entries include OpenRouter, OpenAI, Anthropic, Groq, DeepSeek,
Together, Mistral, xAI, Gemini, Cerebras, Fireworks, Perplexity, Nebius,
Hyperbolic, Cohere, SambaNova, Alibaba Qwen, Hugging Face Inference Providers,
Cloudflare Workers AI, Ollama Cloud, Agnes AI, ModelScope, LLM7.io, OpenCode Zen, and
GLHF.chat. Keys are stored in a 0600 providers file and providers require
HTTPS. Cloudflare requires an account-scoped base URL or
`KILOBYTE_CLOUDFLARE_ACCOUNT_ID`; `/model` fetches the selected provider's live model
catalog. GitHub Models was retired in July 2026 and is intentionally not advertised.
Hermes Agent is a client rather than a separate inference endpoint. Groq uses
`https://api.groq.com/openai/v1`; requests include a project user-agent to avoid
edge-signature blocking, and retired `llama-3.3-70b-versatile` configs migrate to
`llama-3.1-8b-instant` on read.

The replacement integrations use these documented defaults: Ollama Cloud
`https://ollama.com/v1` (`gpt-oss:120b`), Agnes AI
`https://apihub.agnes-ai.com/v1` (`agnes-2.0-flash`), ModelScope
`https://api-inference.modelscope.cn/v1` (`Qwen/Qwen3-32B`), LLM7.io
`https://api.llm7.io/v1` (`fast`), OpenCode Zen
`https://opencode.ai/zen/v1` (`big-pickle`), and GLHF.chat
`https://glhf.chat/api/openai/v1` (`hf:meta-llama/Llama-3.3-70B-Instruct`).

## Install / use

The one-line installer bootstraps this repository, installs the app/service, and
does not download a brain. Use `/cloud` for a configured provider or `/gguf` for
an operator-supplied local model. `/botkey` configures Telegram through the daemon
RPC. Telegram publishes real command autocomplete and provides `/local`, `/cloud`,
`/switch`, `/model`, and `/agent` routing per chat; its tool boundary remains read-only.
Progress animates every 1.2 seconds and a second persistent card shows the bounded,
redacted work log and live reply preview. `agent.py` recovers XML-like tool calls emitted
inside a provider's text stream, but only for names in the already-filtered interface
schema; `telegram_render.py` renders the final Markdown as Telegram-safe HTML.
It collapses provider whitespace outside code, safely splits long formatted messages, and
reports context for the active route instead of reusing the local 8192-token value in cloud
mode. Providers that do not advertise a limit are labelled `provider-managed`.
`/cancel` and the Stop button cancel only that chat's tracked active/queued tasks.
Research-profile turns require successful `web_search` and `web_fetch` calls before an
answer is accepted. Intermediate planning and promise text is reset from clients; repeated
general non-performance is reported as failure instead of false completion.
The full-screen TUI preserves the v1.13.0 framework renderer: its original agent row, `◈`
tool activity/results, faint divider, and answer share one Kilo border. Pygments styles only
fenced-code content using the declared language. Do not split the turn into live-work/answer
boxes or add capability-manifest rows.
Cloud providers receive native tool schemas first, with a schema-bearing JSON text-tool
fallback for models/endpoints that reject the native field. XML-function and JSON envelopes
are still validated against the active interface allow-list before dispatch.
The live stats bar intentionally omits the user's request text so status indicators stay
compact; it shows phase, request count, tools, tokens, model, queue, and context instead.
The animated context meter is local-only; cloud mode omits context from the status bar.
The TUI retains background RPC/monitor tasks and shows their live count in the stats bar;
the daemon separately monitors and restarts a failed local runtime.
Past-chat selectors include local date/time. OpenRouter free-model discovery accepts both
`:free` IDs and zero-priced catalogue entries.

## Verification and limits

The full suite is 144 tests and passes. The Framework installer is intentionally
usable without `llama-server` for cloud-only operation. Local GGUF performance is
bounded by the target machine; advanced coding/security work should use capable
hardware or an explicitly selected cloud model. Keep the security profile and
permission gate intact.

## Handoff checklist

Run the full unittest discovery, shell-check both installers, verify provider
catalog HTTPS/default models, and exercise `kilo status`, `/commands`, `/botkey`,
`/cloud`, `/gguf`, and approval prompts before release. Update this file and the
wiki when contracts, providers, or brain provenance changes.
