import io
import json
import tempfile
import threading
import unittest
import urllib.error
import urllib.request
from pathlib import Path

import pandas as pd

from ada import tools
from ada.mcp import PROTOCOL_VERSIONS, Server, http_server, serve_stdio


def request(ident, method, params=None):
    message = {"jsonrpc": "2.0", "id": ident, "method": method}
    if params is not None:
        message["params"] = params
    return message


class ServerTests(unittest.TestCase):
    """MCP messages get the replies the protocol asks for, with ADA's tools behind them."""

    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.root = Path(self.directory.name)
        pd.DataFrame({"city": ["Oslo", "Lima"] * 5, "price": [10.0, 20.0] * 5}).to_csv(
            self.root / "sales.csv", index=False)
        self.server = Server(self.root)

    def tearDown(self):
        self.directory.cleanup()

    def test_initialize_answers_in_the_version_the_client_asks_for(self):
        reply = self.server.handle(request(1, "initialize", {"protocolVersion": "2024-11-05"}))
        self.assertEqual(reply["result"]["protocolVersion"], "2024-11-05")
        self.assertEqual(reply["result"]["serverInfo"]["name"], "ada")
        self.assertIn(str(self.root.resolve()), reply["result"]["instructions"])
        newest = self.server.handle(request(2, "initialize", {"protocolVersion": "2099-01-01"}))
        self.assertEqual(newest["result"]["protocolVersion"], PROTOCOL_VERSIONS[0])

    def test_tools_are_listed_and_called(self):
        listed = self.server.handle(request(1, "tools/list"))["result"]["tools"]
        self.assertEqual([item["name"] for item in listed], list(tools.NAMES))
        called = self.server.handle(request(2, "tools/call", {
            "name": "ada_ask", "arguments": {"path": "sales.csv", "question": "average price"}}))["result"]
        self.assertFalse(called["isError"])
        result = json.loads(called["content"][0]["text"])
        self.assertEqual(result["status"], "answered")
        self.assertIn("15", result["answer"])

    def test_a_bad_call_is_an_error_result_and_an_unknown_tool_a_protocol_error(self):
        bad = self.server.handle(request(1, "tools/call", {
            "name": "ada_ask", "arguments": {"path": "none.csv", "question": "x"}}))
        self.assertTrue(bad["result"]["isError"])
        unknown = self.server.handle(request(2, "tools/call", {"name": "ada_sum", "arguments": {}}))
        self.assertEqual(unknown["error"]["code"], -32602)
        missing = self.server.handle(request(3, "sampling/createMessage"))
        self.assertEqual(missing["error"]["code"], -32601)

    def test_notifications_get_no_reply_and_batches_one_list(self):
        self.assertIsNone(self.server.handle({"jsonrpc": "2.0", "method": "notifications/initialized"}))
        replies = self.server.handle([request(1, "ping"), {"jsonrpc": "2.0", "method": "notifications/x"},
                                      request(2, "prompts/list")])
        self.assertEqual(replies, [{"jsonrpc": "2.0", "id": 1, "result": {}},
                                   {"jsonrpc": "2.0", "id": 2, "result": {"prompts": []}}])
        self.assertEqual(self.server.handle({"id": 1})["error"]["code"], -32600)

    def test_stdio_carries_one_message_per_line(self):
        lines = [json.dumps(request(1, "initialize", {"protocolVersion": PROTOCOL_VERSIONS[0]})),
                 json.dumps({"jsonrpc": "2.0", "method": "notifications/initialized"}), "", "not json",
                 json.dumps(request(2, "tools/list"))]
        out = io.StringIO()
        serve_stdio(self.root, io.StringIO("\n".join(lines) + "\n"), out)
        replies = [json.loads(line) for line in out.getvalue().splitlines()]
        self.assertEqual([reply.get("id") for reply in replies], [1, None, 2])
        self.assertEqual(replies[1]["error"]["code"], -32700)


class HttpTests(unittest.TestCase):
    """The Streamable HTTP transport answers POSTs at /mcp and keeps other pages out."""

    @classmethod
    def setUpClass(cls):
        cls.directory = tempfile.TemporaryDirectory()
        root = Path(cls.directory.name)
        pd.DataFrame({"city": ["Oslo", "Lima"] * 5, "price": [10.0, 20.0] * 5}).to_csv(
            root / "sales.csv", index=False)
        cls.server = http_server(root, port=0)
        cls.url = f"http://127.0.0.1:{cls.server.server_address[1]}"
        cls.thread = threading.Thread(target=cls.server.serve_forever, daemon=True)
        cls.thread.start()
        cls.opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))

    @classmethod
    def tearDownClass(cls):
        cls.server.shutdown()
        cls.server.server_close()
        cls.directory.cleanup()

    def send(self, body, path="/mcp", method="POST", origin=None):
        headers = {"Content-Type": "application/json", "Accept": "application/json, text/event-stream"}
        if origin:
            headers["Origin"] = origin
        data = None if body is None else json.dumps(body).encode()
        try:
            with self.opener.open(urllib.request.Request(f"{self.url}{path}", data=data, headers=headers,
                                                         method=method), timeout=60) as response:
                text = response.read().decode()
                return response.status, json.loads(text) if text else None
        except urllib.error.HTTPError as error:
            return error.code, None

    def test_a_tool_call_over_http(self):
        status, reply = self.send(request(1, "tools/call", {
            "name": "ada_check_claim",
            "arguments": {"path": "sales.csv", "claim": "The average price is 15."}}))
        self.assertEqual(status, 200)
        self.assertEqual(json.loads(reply["result"]["content"][0]["text"])["verdict"], "true")

    def test_notifications_get_202_and_other_methods_and_pages_are_refused(self):
        self.assertEqual(self.send({"jsonrpc": "2.0", "method": "notifications/initialized"}), (202, None))
        self.assertEqual(self.send(None, method="GET")[0], 405)
        self.assertEqual(self.send(request(1, "ping"), path="/other")[0], 404)

    def test_a_web_page_elsewhere_is_refused_and_a_local_one_allowed(self):
        self.assertEqual(self.send(request(1, "ping"), origin="https://example.com")[0], 403)
        self.assertEqual(self.send(request(1, "ping"), origin="http://localhost:3000")[0], 200)


if __name__ == "__main__":
    unittest.main()
