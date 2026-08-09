import json
import tempfile
import unittest
from pathlib import Path

from kilobyte.agent import Agent, _parse_inline_tool_calls
from kilobyte.config import Settings
from kilobyte.memory import MemoryStore
from kilobyte.security import PermissionManager
from kilobyte.tools import ToolRegistry


class FakeRuntime:
    def __init__(self):
        self.calls = 0
        self.ready_checks = 0

    async def ensure_ready(self):
        self.ready_checks += 1

    async def chat_stream(self, payload):
        self.calls += 1
        if self.calls == 1:
            yield {"delta": {"tool_calls": [{"index": 0, "id": "call-1", "function": {"name": "system_info", "arguments": "{}"}}]}}
        else:
            for token in ("Machine ", "checked."):
                yield {"delta": {"content": token}}
            yield {"usage": {"completion_tokens": 2}}


class CapturingRuntime:
    def __init__(self):
        self.payload = None

    async def ensure_ready(self):
        pass

    async def chat_stream(self, payload):
        self.payload = payload
        yield {"delta": {"content": "ready"}}


class DuplicateToolRuntime:
    def __init__(self):
        self.payloads = []

    async def ensure_ready(self):
        pass

    async def chat_stream(self, payload):
        self.payloads.append(payload)
        if len(self.payloads) <= 2:
            yield {"delta": {"tool_calls": [{"index": 0, "id": f"call-{len(self.payloads)}", "function": {"name": "system_info", "arguments": "{}"}}]}}
        else:
            yield {"delta": {"content": "Linux, 2 CPUs"}}


class PuntingRuntime:
    """First turn only announces an action; after the nudge it delivers the answer."""

    def __init__(self):
        self.payloads = []

    async def ensure_ready(self):
        pass

    async def chat_stream(self, payload):
        self.payloads.append([dict(m) for m in payload["messages"]])
        if len(self.payloads) == 1:
            yield {"delta": {"content": "Sure — let me calculate"}}
        else:
            yield {"delta": {"content": "1+1 is 2."}}


class InlineToolRuntime:
    """Simulate providers that put tool XML in content instead of delta.tool_calls."""

    def __init__(self):
        self.calls = 0

    async def ensure_ready(self):
        pass

    async def chat_stream(self, payload):
        self.calls += 1
        if self.calls == 1:
            raw = (
                "Sir, let me check the real machine first.\n\n"
                "<tool_call><function=system_info></function></tool_call>"
            )
            starts = (0, 9, 31, 48, 62)
            ends = (9, 31, 48, 62, len(raw))
            for start, end in zip(starts, ends, strict=True):
                yield {"delta": {"content": raw[start:end]}}
        else:
            yield {"delta": {"content": "Research complete."}}


class ResearchGateRuntime:
    """Ignore research twice; the framework must keep driving through both tools."""

    def __init__(self):
        self.calls = 0
        self.payloads = []

    async def ensure_ready(self):
        pass

    async def chat_stream(self, payload):
        self.calls += 1
        self.payloads.append([dict(m) for m in payload["messages"]])
        if self.calls == 1:
            yield {"delta": {"content": "I can answer from memory."}}
        elif self.calls == 2:
            yield {"delta": {"tool_calls": [{
                "index": 0,
                "id": "search-1",
                "function": {"name": "web_search", "arguments": '{"query":"Kilo"}'},
            }]}}
        elif self.calls == 3:
            yield {"delta": {"content": "The search snippet is enough."}}
        elif self.calls == 4:
            yield {"delta": {"tool_calls": [{
                "index": 0,
                "id": "fetch-1",
                "function": {"name": "web_fetch", "arguments": '{"url":"https://example.com"}'},
            }]}}
        else:
            yield {"delta": {"content": "Verified result with source: https://example.com"}}


class FakeResearchTools:
    def __init__(self):
        self.calls = []

    def schemas(self, remote=False, request=None):
        del remote, request
        return [
            {"type": "function", "function": {"name": "web_search", "parameters": {"type": "object"}}},
            {"type": "function", "function": {"name": "web_fetch", "parameters": {"type": "object"}}},
        ]

    async def execute(self, name, arguments, context):
        del context
        self.calls.append((name, arguments))
        if name == "web_search":
            return {"results": [{"url": "https://example.com", "title": "Primary"}]}
        return {"url": arguments["url"], "content": "verified source text"}


class AgentTests(unittest.IsolatedAsyncioTestCase):
    async def test_research_must_search_fetch_and_clear_unfinished_answers(self):
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw)
            settings = Settings(data_dir=root, config_dir=root, runtime_dir=root, log_dir=root, home=root)
            memory = MemoryStore(root / "memory.db")
            runtime = ResearchGateRuntime()
            tools = FakeResearchTools()
            agent = Agent(settings, runtime, memory, tools)  # type: ignore[arg-type]
            events = [event async for event in agent.run("research Kilo", remote=True)]

            visible = []
            for event in events:
                if event["type"] == "response_reset":
                    visible.clear()
                elif event["type"] == "token":
                    visible.append(event.get("text", ""))
            answer = "".join(visible)
            self.assertNotIn("answer from memory", answer)
            self.assertNotIn("snippet is enough", answer)
            self.assertIn("Verified result", answer)
            self.assertEqual([name for name, _ in tools.calls], ["web_search", "web_fetch"])
            self.assertEqual(runtime.calls, 5)
            self.assertGreaterEqual(
                sum(event["type"] == "response_reset" for event in events), 2
            )
            memory.close()

    def test_inline_parser_allows_only_current_interface_tools(self):
        raw = (
            '<tool_call><function=web_search><parameter=query>"citadel" research'
            "</parameter></function></tool_call>"
            "<tool_call><function=run_command><parameter=command>id</parameter>"
            "</function></tool_call>"
        )
        clean, calls, rejected, saw = _parse_inline_tool_calls(raw, {"web_search"})
        self.assertTrue(saw)
        self.assertEqual(clean, "")
        self.assertEqual([call["function"]["name"] for call in calls], ["web_search"])
        self.assertEqual(
            json.loads(calls[0]["function"]["arguments"]),
            {"query": '"citadel" research'},
        )
        self.assertEqual(rejected, ["run_command"])

    async def test_inline_tool_markup_is_dispatched_and_never_displayed(self):
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw)
            settings = Settings(data_dir=root, config_dir=root, runtime_dir=root, log_dir=root, home=root)
            memory = MemoryStore(root / "memory.db")
            tools = ToolRegistry(settings, memory, PermissionManager(root / "policy.json"))
            runtime = InlineToolRuntime()
            agent = Agent(settings, runtime, memory, tools)  # type: ignore[arg-type]
            events = [event async for event in agent.run("inspect this machine")]
            visible = "".join(event.get("text", "") for event in events)
            self.assertNotIn("<tool_call>", visible)
            self.assertNotIn("<function=", visible)
            self.assertTrue(any(event["type"] == "tool_start" for event in events))
            self.assertIn("Research complete.", visible)
            self.assertEqual(runtime.calls, 2)
            memory.close()

    async def test_announced_action_is_nudged_to_completion(self):
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw)
            settings = Settings(data_dir=root, config_dir=root, runtime_dir=root, log_dir=root, home=root)
            memory = MemoryStore(root / "memory.db")
            tools = ToolRegistry(settings, memory, PermissionManager(root / "policy.json"))
            runtime = PuntingRuntime()
            agent = Agent(settings, runtime, memory, tools)  # type: ignore[arg-type]
            events = [event async for event in agent.run("what is 1+1")]
            answer = "".join(e.get("text", "") for e in events)
            self.assertIn("1+1 is 2.", answer)
            # It took a second turn, and the punt was not what got saved.
            self.assertEqual(len(runtime.payloads), 2)
            self.assertEqual(memory.history(events[0]["session_id"])[-1]["content"], "Sir, 1+1 is 2., Sir.")
            # The follow-through nudge was injected as a system message before the retry.
            self.assertTrue(any(
                m["role"] == "system" and "did not do it" in m.get("content", "")
                for m in runtime.payloads[1]
            ))
            memory.close()

    async def test_repeated_punts_are_never_accepted_as_completion(self):
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw)
            settings = Settings(data_dir=root, config_dir=root, runtime_dir=root, log_dir=root, home=root)
            memory = MemoryStore(root / "memory.db")
            tools = ToolRegistry(settings, memory, PermissionManager(root / "policy.json"))

            class AlwaysPunts:
                def __init__(self):
                    self.calls = 0

                async def ensure_ready(self):
                    pass

                async def chat_stream(self, payload):
                    self.calls += 1
                    yield {"delta": {"content": "let me check"}}

            runtime = AlwaysPunts()
            agent = Agent(settings, runtime, memory, tools)  # type: ignore[arg-type]
            events = [event async for event in agent.run("do the thing")]
            # Three bounded retries, then a truthful failure — not an infinite loop and
            # never an unfinished promise presented as the result.
            self.assertEqual(runtime.calls, 4)
            self.assertTrue(any(e.get("task_failed") for e in events))
            visible = []
            for event in events:
                if event["type"] == "response_reset":
                    visible.clear()
                elif event["type"] == "token":
                    visible.append(event.get("text", ""))
            self.assertNotIn("let me check", "".join(visible).lower())
            self.assertIn("did not complete", "".join(visible).lower())
            memory.close()


    async def test_tool_loop_streams_and_persists(self):
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw)
            settings = Settings(data_dir=root, config_dir=root, runtime_dir=root, log_dir=root, home=root)
            memory = MemoryStore(root / "memory.db")
            tools = ToolRegistry(settings, memory, PermissionManager(root / "policy.json"))
            runtime = FakeRuntime()
            agent = Agent(settings, runtime, memory, tools)  # type: ignore[arg-type]
            events = [event async for event in agent.run("Check this machine")]
            self.assertEqual("".join(e.get("text", "") for e in events), "Sir, Machine checked., Sir.")
            self.assertTrue(any(e["type"] == "tool_end" and e["ok"] for e in events))
            self.assertEqual(runtime.calls, 2)
            self.assertEqual(runtime.ready_checks, 2)
            self.assertEqual(memory.stats()["tool_audit"], 1)
            memory.close()

    async def test_plain_chat_sends_the_stable_tool_schema(self):
        """A plain answer still carries the full tool list: the prefix has to stay
        identical between requests for llama-server's prompt cache to be reused."""
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw)
            settings = Settings(data_dir=root, config_dir=root, runtime_dir=root, log_dir=root, home=root)
            memory = MemoryStore(root / "memory.db")
            tools = ToolRegistry(settings, memory, PermissionManager(root / "policy.json"))
            runtime = CapturingRuntime()
            agent = Agent(settings, runtime, memory, tools)  # type: ignore[arg-type]
            events = [event async for event in agent.run("Reply with exactly: ready")]
            self.assertEqual("".join(e.get("text", "") for e in events), "Sir, ready, Sir.")
            self.assertEqual(runtime.payload["tools"], tools.schemas())
            memory.close()

    async def test_duplicate_tool_call_is_blocked_and_tools_are_disabled(self):
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw)
            settings = Settings(data_dir=root, config_dir=root, runtime_dir=root, log_dir=root, home=root)
            memory = MemoryStore(root / "memory.db")
            tools = ToolRegistry(settings, memory, PermissionManager(root / "policy.json"))
            runtime = DuplicateToolRuntime()
            agent = Agent(settings, runtime, memory, tools)  # type: ignore[arg-type]
            events = [event async for event in agent.run("Inspect this machine CPU")]
            self.assertEqual("".join(e.get("text", "") for e in events), "Sir, Linux, 2 CPUs, Sir.")
            self.assertEqual(memory.stats()["tool_audit"], 1)
            self.assertNotIn("tools", runtime.payloads[2])
            self.assertTrue(any(e["type"] == "tool_end" and not e["ok"] for e in events))
            memory.close()


if __name__ == "__main__":
    unittest.main()
