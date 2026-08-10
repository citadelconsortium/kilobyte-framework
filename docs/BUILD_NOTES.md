# Build notes

What Kilobyte contains, what has been added, and the reasoning behind the decisions that
are not obvious from the code.

Framework version 1.14.0 · brain-free distribution (operator-supplied GGUF or explicit
cloud provider).

## What is in it

**Brain.** There is exactly **one** Kilobyte brain, trained once by the maintainer and
shipped as a single prebuilt, checksum-pinned GGUF, served by one persistent `llama-server`.
Installing Kilo *downloads* that brain and verifies its SHA-256 — it never trains. No
adapters, no model picker, no automatic cloud fallback. Optional, explicit cloud escalation
(`/cloud`) exists for when more power is wanted, but local is always the default.

**Front ends.** An animated terminal TUI, and an optional Telegram bridge. Both talk to the
same daemon over a Unix socket and share one loaded model with separate conversations.

**Tools.** Fourteen, all verified end to end: `read_file`, `write_file`, `list_files`,
`search_files`, `run_command`, `system_info`, `web_search`, `web_fetch`, `remember`,
`recall`, `save_skill`, `list_skills`, `search_history`, `reference`. MCP servers can add more. The tool
set is deliberately **stable** (never routed per request) so the prompt prefix stays
cacheable.

**Agents.** Kilo runs in a specialist profile per task — `research`, `coding`, `security`
(the "hacking" agent; `/agent hacking` is an alias), `systems`, and `conversation` (the
default). Profiles are injected after the cached prefix, so switching one costs nothing.

**Memory.** SQLite with bounded growth: sessions, messages, long-term facts and a tool
audit trail. Retention limits are enforced on write.

**Safety.** Path sandbox, shell-free command execution with risk classification,
interactive approval for anything above `safe`, private-address blocking for web fetches,
and a stricter read-only policy for anything arriving remotely.

**Operations.** systemd unit enabled for boot, `kilo doctor` health checks, `kilo status`,
`kilo resources`, `kilo model-info`, `kilo benchmark`, `kilo logs`, and a one-line
installer that provisions dependencies, the service user, the model and the service.

## Changes and why

### 1.14.0: stronger-brain compatibility and compact one-box work output

The framework's model-building path now renders native assistant function calls and tool
results, with a fixed raw-GGUF gate plus an isolated real-framework RPC acceptance suite.
The framework repository remains brain-free; it is tested with the companion
`kilobyte-4.1-3b-q4_k_m.gguf` release but never downloads or bundles that model.

The established single Kilo TUI box remains intact. Live work is compacted into `Ran`,
`Explored`, `Wrote`, and `Used` rows with nested bounded results; leading model whitespace
is discarded after every tool event, and fenced code receives language-aware Pygments
colour without changing the box. `kilo status` refreshes detected total and available RAM
on every request while preserving the runtime's actual active context and thread settings.

### 1.13.6: Telegram machine tools with in-chat approval

Allow-listed Telegram chats now receive the full built-in tool schema, including
`run_command` and `write_file`. Safe inspection executes directly; write, outward,
elevated, and destructive actions wait for one-time Approve/Deny buttons tied to the same
chat and expire after 280 seconds. Shell operators and allowed-root enforcement remain in
the deterministic command/file layer. Missing model tool arguments now return the required
field names instead of a bare `KeyError`, giving every local or cloud model an actionable
retry. Bounded atomic writes execute directly so they cannot stall behind a constrained
VM thread pool after a command. The v1.13.5 Pygments TUI is included; no bundled GGUF is required.

### 1.13.5: language-aware code output

Fenced code in the confirmed v1.13.0 one-box TUI now uses Pygments to distinguish
keywords, functions, classes, strings, numbers, comments, operators, and punctuation.
The fence language selects the lexer and unknown languages fall back to plain code. The
box, agent/tool rows, divider, streaming path, and response placement are unchanged.
Telegram emits its supported `<pre><code class="language-…">` form for labelled fences.
The portable installer provisions Pygments alongside prompt_toolkit.

### 1.13.4: exact v1.13.0 TUI restoration

`tui_full.py` is restored byte-for-byte from the v1.13.0 Git release object. This returns
the established output exactly: `◇ orchestrator → … agent`, `◈` live tool rows, result
rows, the faint divider, and Kilo's response all inside one Kilo box. The v1.13.3 attempt
recreated the layout but changed its rows and whitespace; those changes are removed. Cloud
native/JSON/XML tool compatibility remains below the presentation layer and does not alter
the old TUI output.

### 1.13.3: restore the single-box TUI and universal cloud tool compatibility

The full-screen interface is restored to its original one-box turn layout: selected agent,
active tool manifest, live tool work, and final response all remain inside one **Kilo** box.
Thinking pauses no longer append rows; leading/repeated provider whitespace is collapsed;
discarded streamed preambles are truly removed. Cloud models use native tool schemas first,
then a schema-bearing JSON text protocol when an OpenAI-compatible endpoint rejects native
tools. The agent recovers XML-function and JSON tool envelopes against the active allow-list,
and false claims that local TUI tools or the selected agent are unavailable are retried.

### 1.13.2: deterministic Telegram research follow-through

Research-mode answers are now completion-gated by the framework: a turn must successfully
run both `web_search` and `web_fetch` before its synthesis is accepted. If a model answers
from memory, stops at a search snippet, or only announces that it will research, the agent
clears that intermediate text and continues the tool loop. General unfinished promises get
three bounded retries; repeated non-performance is surfaced as a failure instead of being
recorded as a completed task.

### 1.13.1: reliable research tools and cleaner live interfaces

**Provider tool-protocol recovery.** Some cloud chat templates return XML-like
`<tool_call>` blocks in ordinary content instead of native `delta.tool_calls`. The agent
now guards the stream, removes that protocol from user-visible output, validates recovered
names against the active interface schema, and dispatches allowed calls normally. This
kept Telegram read-only in that release even if the model asked for a disallowed tool;
1.13.6 replaces that boundary with chat-bound approvals.

**Presentation and activity.** Telegram uses a fast progress animation plus a separate
redacted work-log message, and renders common Markdown into safe HTML. This release also
temporarily split full-screen TUI work from its answer; 1.13.3 restores the established
single Kilo box while retaining complete redacted arguments/results. The security agent now
learns target-specific approaches via
skills instead of receiving a canned playbook. Active-route context reporting no longer
labels a cloud model with the local runtime's 8192-token window.
Telegram now tracks active and queued work per chat so `/cancel` and the Stop button can
terminate one chat cleanly without turning cancellation into a global daemon stop.

### 1.13.0: disconnect-safe inference and Telegram cloud control

RPC now observes peer EOF while inference is silent, closes the generator immediately,
and cancels all client handlers during daemon shutdown. Telegram uses a socket timeout
longer than its API long-poll, reloads token/allow-list changes live, rotates real sessions
for `/new`, publishes its command menu, and supports explicit per-chat local/cloud, model,
and specialist-agent selection while retaining the then-read-only remote tool boundary
(superseded by 1.13.6 approval buttons).
Command approvals now distinguish read-only inspection from local/external writes,
privileged work, and destructive operations; unknown executables are no longer assumed safe.

### 1.1.0-1.2.0: grounding, autonomy, agents, easy cloud, one-brain docs

This block of work made Kilo trustworthy and finish what it starts, and made cloud and
Telegram pleasant to use — without changing the fact that there is one brain.

**Anti-hallucination is ~80% framework, ~20% weights.** A 1.7B model cannot be trained not
to hallucinate, so the framework forces it to work from evidence: the system prompt requires
getting facts with a tool rather than recalling them, forbids inventing output/paths/results,
and sampling runs at low temperature. But grounding was *too* blunt — it made the model hedge
on things it plainly knows ("I'm not certain, but 1+1 is 2"). The prompt now separates the
two: answer known facts/arithmetic/definitions directly and confidently; reserve tool-checking
and abstention for what you would otherwise guess.

**Follow-through (the "let me calculate… <stops>" bug).** The model would sometimes reply
with only the *intent* to act and no tool call, and the loop accepted that promise as the
answer. The agent now detects an announced-but-undelivered action (`_looks_like_punt`) and
issues exactly **one** bounded nudge — call the tool now or answer now — so it either finishes
or delivers, and can never loop. Covered by deterministic tests.

**Specialist agents.** `research`, `coding`, `security` (hacking), `systems`, and a new
`conversation` agent that is the default for anything unrouted: understand the real intent,
then carry the task to a finished result. Each profile emphasises the grounding discipline for
its domain. Auto-selected from the request or forced with `/agent`; friendly aliases route
natural words (e.g. "hacking" -> security).

**Automatic cross-session recall.** `search_history` existed, but a small model would not
reliably reach for it, so relevant lines from earlier sessions are now surfaced automatically
at the start of a turn — the framework guaranteeing memory rather than hoping the model asks.

**Full-screen animated TUI.** A `prompt_toolkit` app: animated banner (with "made by
0v3r51ght"), token-by-token streaming, a stats bar with **numeric** live counters (runtime,
tools, tokens) and no step counter, an F2 runtime panel, and the active brain indicator.

**Easy cloud escalation.** `/cloud` opens a picker of known OpenAI-compatible providers; the
user supplies only an API key (base URL and default model come from a catalog), it is saved
`0600` and made default. The catalog includes Ollama Cloud, Agnes AI, ModelScope, LLM7.io,
OpenCode Zen, and GLHF.chat, each using its documented OpenAI-compatible endpoint and
Bearer authentication. `/switch` flips the active brain between that provider and local Kilo
(Kilo default), shown in the stats bar. Local is always the default; escalation is explicit.

**Telegram redesign + management.** Animated progress card (spinner, phase icon, elapsed,
tools), branded help/status/answer cards, a memory meter, `/id`, richer buttons — and a
`kilo telegram status|set-token|allow|disallow|disable` CLI so the bot is managed without
hand-editing JSON (config is polled live, no restart). Telegram was strictly read-only at
that release; 1.13.6 adds approval-gated machine tools.

**Versioning for rollback.** The framework version lives in `pyproject.toml` /
`__init__.py` and every release is a git tag (`v1.0.0`, `v1.1.0`, `v1.2.0`), so the codebase
can be reverted to a known-good point; the brain is versioned separately (`kilo brain
versions` / `rollback`).


### Replies were minutes long, or appeared to hang

Tool schemas were selected per request by keyword. Tools render into the prompt prefix, so
a varying set missed `llama-server`'s prompt cache on **every** message and reprocessed the
system prompt and schemas each turn.

Fixed by making the tool set fixed, trimming the system prompt from ~386 to ~139 tokens,
priming the cache at startup with the exact prompt and tool set real requests use, and
moving recalled memory out of the system message so the prefix stays byte-identical.

Measured after: **36 seconds** for a reply, with 940 of 956 prompt tokens served from
cache.

### One tool call could exceed the whole context

`max_tool_output` bounded results at 64 KB of characters. Dense output tokenises at about
two characters per token, so `ls -la /usr/lib` measured **33,496 tokens** against an
8192-token window — enough to displace the conversation and the system prompt entirely.

Added `context.py`, which compacts results deterministically before the model sees them.
Verified with `llama-server`'s tokeniser: the same result now measures **798 tokens**.

Conversation history is budgeted the same way, because a message count is not a bound on
context when a single turn can carry a tool result.

### Warmup cost was paid on every boot

`--slot-save-path` only enables the endpoints; nothing was saved. Each start reprocessed
the full prefix (~20 minutes on the development host) while holding the only inference
slot.

Warmup now restores a matching saved slot and otherwise processes the prefix once and
saves it. Restore measured at **0.23 seconds**. The filename hashes the prompt, tool
schemas, model path and context size, so a changed prompt or a move to a machine with
different memory re-warms correctly instead of restoring an unusable prefix. Stale slots
are pruned, as each is around 100 MB.

### Every shutdown ended in SIGKILL

Warmup, Telegram and inference perform blocking HTTP on worker threads. Cancelling their
tasks does not interrupt the threads, and `asyncio.run()` waits for the default executor
before returning, so a request in flight held shutdown open past systemd's stop timeout;
`llama-server` then crashed in `__cxa_finalize` on the way out.

`llama-server` is now stopped before the tasks are awaited, which fails those requests
immediately, and the slot save/restore timeout is well inside the stop window. Restart
went from a 60-second SIGKILL escalation to **2.5 seconds**, clean.

### A dropped client held the model hostage

Closing the agent generator did not close the inner `chat_stream` generator it was
iterating, so a disconnected client (Ctrl-C, a killed process) left the request to
`llama-server` open and its single slot held until the generation finished on its own —
observed at roughly eight minutes of wasted CPU. Fixed with `aclosing` in `agent.py` and an
explicit `aclose` in the RPC handler.

### The interface looked frozen and could not be interrupted

The spinner only redrew when an event arrived, so a slow step showed nothing for minutes,
and there was no way to stop a running generation.

The activity line is now timer-driven and reports the current action, how long that action
has been running, the running total and the model. Each tool prints its arguments when it
starts and its duration and result when it finishes, and the closing border reports total
time, time to first token, and which tools ran. SIGINT cancels the generation and returns
to the prompt without tearing down the session.

### Telegram could not be enabled, and failed silently

Writing the config after the daemon started did nothing: `run()` returned immediately and
the bridge stayed dead until a restart. Failures were swallowed by a bare `except`, leaving
the sender with silence indistinguishable from a hung bot.

The bridge now polls for its config, always replies (errors included), keeps a typing
indicator alive, edits a live progress message as work proceeds, publishes a command menu,
offers inline buttons for status/new/help, and rejects a placeholder token or non-integer
chat ids rather than treating them as configured.

### Skills and MCP

Added a skill registry: Kilo records a procedure once a multi-step task works, and
matching procedures are surfaced into context on later requests. On slow hardware this is
the cheaper side of the trade -- a few hundred tokens of known-good steps against several
planning rounds, each of which costs a full generation. Skills are keyed by name so
re-saving refines one in place, outcomes are tracked so reliable ones sort first, and
growth is bounded by dropping the least reliable and least recently used.

Added an MCP client (stdio transport, protocol 2025-06-18) so tools from external servers
can be offered alongside the built-in ones. Implemented against the published
specification: newline-delimited UTF-8 JSON-RPC with no embedded newlines, the
initialize/initialized handshake, paginated tools/list, and the documented shutdown
sequence of closing stdin then escalating. Servers are treated as untrusted -- tools are
namespaced, a tool without a usable object input schema is never shown to the model,
calling one needs permission, results go through the same compaction, a hung server hits a
request timeout rather than blocking the daemon, and a server that fails to start is
skipped. MCP tools are not offered over Telegram. Tested against a real server subprocess
over real pipes rather than a mock.

### Telegram buttons looked dead and progress froze on step 1

The poll loop answered inline. A generation takes minutes here, so every command and
button press arriving meanwhile sat unread until it finished. Replies now run on their own
task, serialised per chat, while polling continues. The progress message was also driven
only by agent events, so a step that runs for minutes without emitting anything left it
reading "step 1"; it is now rewritten on a timer with the phase, an elapsed clock and the
tools used, and the clock keeps each edit distinct, which matters because Telegram rejects
an edit that would not change the text.

### Optional cloud escalation

Added `providers.py` so a request can be sent to a hosted model deliberately, which the
original specification forbade as an automatic fallback. The distinction is kept real
rather than claimed: local remains the default, escalation is per message via `/cloud` and
never sticky, there is no automatic escalation on slowness or failure, the answering brain
is reported and displayed, and with no providers file there is no cloud path. Keys sit in a
0600 file, travel in a header over HTTPS only, and are never logged. Telegram cannot
escalate.

### Installer was not actually one line

`install.sh` required a manual `pacman` step and a pre-existing service user. Both are now
handled by the script, so the published one-liner needs nothing but running it.

## Known limits

**Hardware dominates.** Token generation is bounded by the host CPU. On a machine without
AVX2, llama.cpp falls back to its generic backend and generation runs near 0.55 tokens per
second; the development VM's host is a 2010 Core2 Duo. The cache work removes the repeated
prompt cost, not the per-token generation cost. Modern hardware loads an AVX2 or AVX-512
backend automatically, and GPU offload is detected and used when present.

**Model capability.** The brain is 1.7B parameters. It follows instructions, calls tools
correctly and completes multi-step tasks, but will not reason like a large model on long or
subtle chains. The orchestration layer is not the limiting factor.

**Model capability is the ceiling, not the framework.** Skills and MCP widen what Kilo can
reach, but a 1.7B brain still decides when to use them.

## Verification

99 automated tests covering resources, runtime, agent loop (including the follow-through
nudge), tools, context compaction, memory, skills, agent-profile routing and aliases,
security, CLI, installation, Telegram, MCP (against a real server subprocess) and cloud
provider configuration.

A static analysis pass over the source found two genuine web-tool vulnerabilities, both
confirmed by experiment before being fixed: urllib follows redirects, so validating only
the requested URL left the private-address block bypassable by a public host answering 302
with a local address; and ElementTree expands internal entities, so a hostile search
provider could return a small document that expands to gigabytes on a machine with about
2 GB to spare. Both are now closed and covered by regression tests.

Verified on the target machine: all ten tools returning real data; the full agent loop
choosing a tool, reading its result and answering correctly; `kilo doctor` passing; cache
restore across restarts; clean service restart; and boot persistence enabled.
