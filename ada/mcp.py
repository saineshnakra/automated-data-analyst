"""ADA as a Model Context Protocol server: ``ada mcp``.

Apps that run models and take MCP servers start ADA from one entry in their
settings, and their model then calls ADA's tools (see :mod:`ada.tools`):

    {"mcpServers": {"ada": {"command": "uvx", "args": ["--from", "automated-data-analyst", "ada", "mcp"]}}}

``ada mcp`` speaks JSON-RPC over standard input and output, the transport
desktop apps and coding assistants start; ``ada mcp --http`` serves the
Streamable HTTP transport at ``http://127.0.0.1:8000/mcp``, for apps that only
reach servers over HTTP. Both need nothing beyond Python's standard library.

The tools read files under one folder, the root: by default the folder the
server starts in, or the home folder when it starts at the top of the disk, as
desktop apps start it.
"""

from __future__ import annotations

import contextlib
import json
import sys
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any, TextIO
from urllib.parse import urlparse

from . import __version__
from .tools import NAMES, SYSTEM_PROMPT, definitions, run

#: Protocol versions the server speaks, newest first; it answers a client in the version it asks for.
PROTOCOL_VERSIONS = ("2025-06-18", "2025-03-26", "2024-11-05")
#: Hosts a browser page may call the HTTP server from without ``--allow-origin``.
LOCAL_HOSTS = frozenset({"localhost", "127.0.0.1", "::1"})


def default_root() -> Path:
    """The folder the tools read by default: where the server starts, or home when that is the disk's top."""
    here = Path.cwd()
    return Path.home() if here == Path(here.anchor) else here


class _Failure(Exception):
    def __init__(self, code: int, message: str):
        super().__init__(message)
        self.code, self.message = code, message


def _error(ident: Any, code: int, message: str) -> dict:
    return {"jsonrpc": "2.0", "id": ident, "error": {"code": code, "message": message}}


class Server:
    """Answers MCP messages for one root folder; the transport only carries them."""

    def __init__(self, root: str | Path | None = None):
        self.root = Path(root).expanduser().resolve() if root is not None else default_root()

    def handle(self, message: Any) -> dict | list | None:
        """The reply to one message or batch; None for notifications and replies, which get none."""
        if isinstance(message, list):
            replies = [reply for reply in (self.handle(item) for item in message) if reply is not None]
            return replies or None
        if not isinstance(message, dict) or message.get("jsonrpc") != "2.0":
            return _error(None, -32600, "Invalid request: not a JSON-RPC 2.0 message")
        method, ident = message.get("method"), message.get("id")
        if not isinstance(method, str):
            return None
        try:
            result = self._result(method, message.get("params") or {})
        except _Failure as failure:
            return None if "id" not in message else _error(ident, failure.code, failure.message)
        return None if "id" not in message else {"jsonrpc": "2.0", "id": ident, "result": result}

    def _result(self, method: str, params: dict) -> dict:
        if method == "initialize":
            asked = params.get("protocolVersion")
            return {
                "protocolVersion": asked if asked in PROTOCOL_VERSIONS else PROTOCOL_VERSIONS[0],
                "capabilities": {"tools": {"listChanged": False}},
                "serverInfo": {"name": "ada", "title": "ADA", "version": __version__},
                "instructions": f"{SYSTEM_PROMPT}\nThe tools read files under {self.root}; name a file by "
                                "its path in that folder or in full.",
            }
        if method in ("ping", "logging/setLevel"):
            return {}
        if method == "tools/list":
            return {"tools": definitions("mcp")}
        if method == "tools/call":
            name = params.get("name")
            if name not in NAMES:
                raise _Failure(-32602, f"Unknown tool: {name}; the tools are {', '.join(NAMES)}")
            result = run(name, params.get("arguments") or {}, root=self.root, brief=True)
            return {"content": [{"type": "text", "text": json.dumps(result, ensure_ascii=False)}],
                    "isError": result.get("status") == "error"}
        if method in ("resources/list", "resources/templates/list", "prompts/list"):
            return {method.split("/")[0] if method != "resources/templates/list" else "resourceTemplates": []}
        raise _Failure(-32601, f"Method not found: {method}")


def serve_stdio(root: str | Path | None = None, reader: TextIO | None = None,
                writer: TextIO | None = None) -> None:
    """Serve MCP over standard input and output, one JSON message per line, until input ends."""
    server = Server(root)
    reader = reader or sys.stdin
    writer = writer or sys.stdout
    while True:
        line = reader.readline()
        if not line:
            return
        if not line.strip():
            continue
        try:
            message = json.loads(line)
        except json.JSONDecodeError:
            reply = _error(None, -32700, "Parse error: not JSON")
        else:
            # Only replies may reach the output: anything printed while a tool runs goes to standard error.
            with contextlib.redirect_stdout(sys.stderr):
                reply = server.handle(message)
        if reply is not None:
            writer.write(json.dumps(reply, ensure_ascii=False) + "\n")
            writer.flush()


def http_server(root: str | Path | None = None, host: str = "127.0.0.1", port: int = 8000,
                path: str = "/mcp", allowed_origins: tuple[str, ...] = ()) -> ThreadingHTTPServer:
    """An HTTP server for MCP's Streamable HTTP transport at ``path``; call ``serve_forever()`` on it.

    Each POST carries one message or batch and gets the reply as JSON, or 202
    with no body for notifications. A request from a browser page on another
    host is refused unless its origin is in ``allowed_origins``, so a web page
    cannot reach the tools through the browser.
    """
    server = Server(root)

    class Handler(BaseHTTPRequestHandler):
        protocol_version = "HTTP/1.1"

        def log_message(self, format: str, *args: Any) -> None:  # noqa: A002
            print(f"ada mcp: {format % args}", file=sys.stderr)

        def _send(self, code: int, body: Any = None, headers: tuple[tuple[str, str], ...] = ()) -> None:
            data = b"" if body is None else json.dumps(body, ensure_ascii=False).encode("utf-8")
            self.send_response(code)
            if body is not None:
                self.send_header("Content-Type", "application/json")
            for key, value in headers:
                self.send_header(key, value)
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)

        def _refused(self) -> bool:
            if urlparse(self.path).path != path:
                self._send(404, {"error": f"MCP is served at {path}"})
                return True
            origin = self.headers.get("Origin")
            if origin and urlparse(origin).hostname not in LOCAL_HOSTS and origin not in allowed_origins:
                self._send(403, _error(None, -32600, f"Requests from {origin} are not allowed"))
                return True
            return False

        def do_POST(self) -> None:  # noqa: N802
            if self._refused():
                return
            body = self.rfile.read(int(self.headers.get("Content-Length") or 0))
            try:
                message = json.loads(body or b"null")
            except (json.JSONDecodeError, UnicodeDecodeError):
                self._send(400, _error(None, -32700, "Parse error: not JSON"))
                return
            reply = server.handle(message)
            if reply is None:
                self._send(202)
            else:
                self._send(200, reply)

        def do_GET(self) -> None:  # noqa: N802
            # The server sends nothing unasked, so it offers no stream to listen on.
            if not self._refused():
                self._send(405, headers=(("Allow", "POST"),))

        def do_DELETE(self) -> None:  # noqa: N802
            if not self._refused():
                self._send(405, headers=(("Allow", "POST"),))

    return ThreadingHTTPServer((host, port), Handler)
