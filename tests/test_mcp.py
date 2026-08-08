"""MCP client tests.

The protocol tests run a real server subprocess over real pipes rather than a mock, so
the framing, the handshake and the shutdown are exercised as they will be in production.
"""

import json
import sys
import tempfile
import textwrap
import unittest
from pathlib import Path
from unittest import IsolatedAsyncioTestCase

from kilobyte.errors import ToolError
from kilobyte.mcp import MCPRegistry, MCPServer, MCPServerConfig

# A minimal spec-conformant server: newline-delimited JSON-RPC on stdout, logging on
# stderr, one tool, and an error path.
FAKE_SERVER = textwrap.dedent(
    """
    import json, sys
    print("starting up", file=sys.stderr, flush=True)
    for line in sys.stdin:
        line = line.strip()
        if not line:
            continue
        msg = json.loads(line)
        method, mid = msg.get("method"), msg.get("id")
        if method == "initialize":
            out = {"jsonrpc": "2.0", "id": mid, "result": {
                "protocolVersion": "2025-06-18",
                "capabilities": {"tools": {}},
                "serverInfo": {"name": "fake", "version": "1"}}}
        elif method == "notifications/initialized":
            continue
        elif method == "tools/list":
            cursor = (msg.get("params") or {}).get("cursor")
            if not cursor:
                out = {"jsonrpc": "2.0", "id": mid, "result": {"tools": [
                    {"name": "echo", "description": "Echo text back",
                     "inputSchema": {"type": "object", "properties": {"text": {"type": "string"}},
                                     "required": ["text"]}}], "nextCursor": "page2"}}
            else:
                out = {"jsonrpc": "2.0", "id": mid, "result": {"tools": [
                    {"name": "broken", "description": "no schema"},
                    {"name": "fail", "description": "always fails",
                     "inputSchema": {"type": "object", "properties": {}}}]}}
        elif method == "tools/call":
            params = msg.get("params") or {}
            if params.get("name") == "fail":
                out = {"jsonrpc": "2.0", "id": mid, "result": {
                    "content": [{"type": "text", "text": "it failed"}], "isError": True}}
            else:
                text = (params.get("arguments") or {}).get("text", "")
                out = {"jsonrpc": "2.0", "id": mid, "result": {
                    "content": [{"type": "text", "text": "echo: " + text}], "isError": False}}
        else:
            out = {"jsonrpc": "2.0", "id": mid,
                   "error": {"code": -32601, "message": "Method not found"}}
        sys.stdout.write(json.dumps(out) + "\\n")
        sys.stdout.flush()
    """
)


class ConfigTests(unittest.TestCase):
    def test_command_accepts_string_with_args_or_a_list(self):
        a = MCPServerConfig.from_dict("x", {"command": "node", "args": ["server.js"]})
        self.assertEqual(a.command, ["node", "server.js"])
        b = MCPServerConfig.from_dict("x", {"command": ["node", "server.js"]})
        self.assertEqual(b.command, ["node", "server.js"])

    def test_missing_or_empty_command_is_rejected(self):
        for entry in ({}, {"command": []}, {"command": 3}):
            with self.assertRaises(ValueError):
                MCPServerConfig.from_dict("x", entry)

    def test_absent_config_yields_no_servers(self):
        with tempfile.TemporaryDirectory() as raw:
            self.assertEqual(MCPRegistry(Path(raw) / "absent.json").configured(), [])

    def test_malformed_config_is_ignored_rather_than_fatal(self):
        with tempfile.TemporaryDirectory() as raw:
            path = Path(raw) / "mcp.json"
            path.write_text("{not json")
            self.assertEqual(MCPRegistry(path).configured(), [])

    def test_disabled_servers_are_skipped_and_both_key_names_work(self):
        with tempfile.TemporaryDirectory() as raw:
            path = Path(raw) / "mcp.json"
            path.write_text(json.dumps({"mcpServers": {
                "on": {"command": "true"},
                "off": {"command": "true", "enabled": False},
            }}))
            names = [c.name for c in MCPRegistry(path).configured()]
            self.assertEqual(names, ["on"])


class ProtocolTests(IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        script = Path(self.tmp.name) / "server.py"
        script.write_text(FAKE_SERVER)
        self.server = MCPServer(MCPServerConfig("fake", [sys.executable, str(script)]), request_timeout=15)
        await self.server.start()

    async def asyncTearDown(self):
        await self.server.stop()
        self.tmp.cleanup()

    async def test_handshake_and_paginated_tool_discovery(self):
        names = {tool["name"] for tool in self.server.tools}
        self.assertEqual(names, {"echo", "broken", "fail"})

    async def test_call_returns_text_content(self):
        result = await self.server.call_tool("echo", {"text": "hello"})
        self.assertEqual(result["content"], "echo: hello")

    async def test_tool_reported_error_becomes_an_exception(self):
        """isError is carried in a successful JSON-RPC result, so it has to be turned
        into a failure or the model would read it as a normal answer."""
        with self.assertRaises(ToolError) as caught:
            await self.server.call_tool("fail", {})
        self.assertIn("it failed", str(caught.exception))

    async def test_unknown_method_surfaces_the_servers_error(self):
        with self.assertRaises(ToolError):
            await self.server.request("does/not/exist")

    async def test_stop_is_idempotent(self):
        await self.server.stop()
        await self.server.stop()
        self.assertFalse(self.server.running)


class RegistryTests(IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        script = Path(self.tmp.name) / "server.py"
        script.write_text(FAKE_SERVER)
        config = Path(self.tmp.name) / "mcp.json"
        config.write_text(json.dumps({"servers": {"fake": {"command": [sys.executable, str(script)]}}}))
        self.registry = MCPRegistry(config, request_timeout=15)
        await self.registry.start()

    async def asyncTearDown(self):
        await self.registry.stop()
        self.tmp.cleanup()

    async def test_tools_are_namespaced_and_schemaless_ones_dropped(self):
        names = {schema["function"]["name"] for schema in self.registry.schemas()}
        self.assertIn("mcp__fake__echo", names)
        # 'broken' publishes no object input schema, so it must not reach the model.
        self.assertNotIn("mcp__fake__broken", names)

    async def test_namespaced_call_reaches_the_right_server(self):
        result = await self.registry.call("mcp__fake__echo", {"text": "hi"})
        self.assertEqual(result["content"], "echo: hi")

    async def test_unknown_names_are_rejected(self):
        self.assertIsNone(self.registry.resolve("read_file"))
        self.assertIsNone(self.registry.resolve("mcp__missing__tool"))
        with self.assertRaises(ToolError):
            await self.registry.call("mcp__missing__tool", {})

    async def test_a_failing_server_does_not_prevent_startup(self):
        with tempfile.TemporaryDirectory() as raw:
            config = Path(raw) / "mcp.json"
            config.write_text(json.dumps({"servers": {"nope": {"command": ["definitely-not-a-real-binary"]}}}))
            registry = MCPRegistry(config)
            await registry.start()
            self.assertEqual(registry.servers, {})
            await registry.stop()


if __name__ == "__main__":
    unittest.main()
