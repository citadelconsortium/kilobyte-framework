# Kilobyte Framework

A **cloud-first, brain-free** build of the [Kilobyte](https://github.com/citadelconsortium/kilobyte)
local-AI agent framework — like a Hermes/OpenClaw-style agent harness you point at **any**
model. It ships **without a bundled brain**: bring your own GGUF, or drive it entirely from a
cloud provider.

## What it is
The full Kilobyte framework — orchestrator + specialist agents (research, coding, security,
systems, private), grounded tool use (shell, files, web, memory, an offline reference bank),
cross-session memory, an approval gate, and a boxed, colored TUI — with **no model shipped**.

## Bring your own model
- **A local GGUF**: download any GGUF (HuggingFace, etc.) into `~/` or `~/Downloads`, then in
  the TUI run **`/gguf`** to browse and select it — it is staged and loaded as the brain.
  (Or `kilo brain deploy /path/to/model.gguf`.)
- **A cloud model** (first-class here): **`/cloud`** to pick a provider and paste a key — 14
  OpenAI-compatible providers (OpenRouter, OpenAI, Anthropic, Groq, DeepSeek, Together,
  Mistral, xAI, Gemini, Cerebras, Fireworks, Perplexity, Nebius, Hyperbolic). Cloud models get
  the **same tools** the local model does, so a frontier model runs *through* the framework.

## Powerful by design
Cloud models are given the full tool set and told, emphatically, that their tools are real and
execute on the machine — so they **act** instead of describing. The agent works until the task
is done (not an arbitrary step cap); only a permission prompt pauses it.

## Install
```bash
sudo ./scripts/install.sh      # app + deps + service (no model download)
kilo                           # then /gguf to pick a GGUF, or /cloud for a hosted model
```

## Docs
See [`docs/`](docs/) — architecture and build notes carry over from Kilobyte. The only
difference in this repo is that **no brain is bundled**; everything else is the same framework.
