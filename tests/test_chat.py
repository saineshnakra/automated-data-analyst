import contextlib
import io
import json
import tempfile
import threading
import unittest
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from unittest import mock

import pandas as pd

from ada.chat import Chat, ChatError, chat, default_base_url, shown
from ada.cli import main


class FakeServer:
    """An OpenAI-compatible chat server that replies with the messages it is given, in turn."""

    def __init__(self, replies, models=("qwen3:1.7b",), status=200):
        self.replies, self.requests = list(replies), []
        fake = self

        class Handler(BaseHTTPRequestHandler):
            def log_message(self, *args):
                pass

            def _send(self, code, body):
                data = json.dumps(body).encode()
                self.send_response(code)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(data)))
                self.end_headers()
                self.wfile.write(data)

            def do_GET(self):
                self._send(200, {"data": [{"id": name} for name in models]})

            def do_POST(self):
                fake.requests.append(json.loads(self.rfile.read(int(self.headers["Content-Length"]))))
                if status != 200:
                    self._send(status, fake.replies.pop(0))
                    return
                self._send(200, {"choices": [{"message": fake.replies.pop(0)}]})

        self.server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        self.url = f"http://127.0.0.1:{self.server.server_address[1]}/v1"
        threading.Thread(target=self.server.serve_forever, daemon=True).start()

    def close(self):
        self.server.shutdown()
        self.server.server_close()


def tool_call(name, arguments):
    return {"role": "assistant", "content": "", "tool_calls": [
        {"id": "call_1", "type": "function", "function": {"name": name, "arguments": json.dumps(arguments)}}]}


class ChatTests(unittest.TestCase):
    """A model's tool calls run ADA, and the model answers with what ADA returned."""

    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.root = Path(self.directory.name)
        self.path = self.root / "sales.csv"
        pd.DataFrame({"city": ["Oslo", "Lima"] * 5, "price": [10.0, 20.0] * 5}).to_csv(self.path, index=False)

    def tearDown(self):
        self.directory.cleanup()

    def test_a_tool_call_runs_ada_and_the_answer_comes_back(self):
        fake = FakeServer([tool_call("ada_ask", {"path": "sales.csv", "question": "average price"}),
                           {"role": "assistant", "content": "<think>15</think>The average price is 15."}])
        self.addCleanup(fake.close)
        seen = []
        reply = Chat(self.path, base_url=fake.url).ask("what is the average price?", on_call=seen.append)
        self.assertEqual(reply.answer, "The average price is 15.")
        self.assertEqual(reply.model, "qwen3:1.7b")
        self.assertEqual([call.name for call in reply.calls], ["ada_ask"])
        self.assertEqual(reply.calls[0].result["status"], "answered")
        self.assertEqual(seen, list(reply.calls))
        first, second = fake.requests
        self.assertIn("sales.csv (10 rows; columns: city, price)", first["messages"][0]["content"])
        self.assertEqual([tool["function"]["name"] for tool in first["tools"]][0], "ada_ask")
        self.assertEqual(second["messages"][-1]["role"], "tool")
        self.assertIn("ADA ada_ask(", shown(reply.calls[0]))

    def test_a_call_written_as_text_is_made(self):
        written = json.dumps({"name": "ada_check_claim",
                              "parameters": {"path": "sales.csv", "claim": "The average price is 15."}})
        fake = FakeServer([{"role": "assistant", "content": written},
                           {"role": "assistant", "content": "True."}])
        self.addCleanup(fake.close)
        reply = chat("is the average price 15?", self.path, base_url=fake.url)
        self.assertEqual(reply.calls[0].result["verdict"], "true")
        self.assertEqual(reply.answer, "True.")

    def test_a_wrapped_or_broken_written_call_is_read(self):
        from ada.chat import _written_call

        asked = {"path": "sales.csv", "question": "average price"}
        wrapped = json.dumps({"type": "function", "function": {"name": "ada_ask", "parameters": {
            "type": "object", "properties": asked}}})
        self.assertEqual(_written_call(wrapped), ("ada_ask", asked))
        broken = ('{"type":"function","function":{"name":"ada_ask","parameters":{"properties":'
                  '{"path":"sales.csv","question":"average price",'
                  '"answer":{"description":\\"made up\\"}}}}')
        self.assertEqual(_written_call(broken), ("ada_ask", asked))
        self.assertIsNone(_written_call("The average price is 15."))

    def test_an_empty_reply_is_asked_again_once(self):
        fake = FakeServer([{"role": "assistant", "content": ""}, {"role": "assistant", "content": "15."}])
        self.addCleanup(fake.close)
        self.assertEqual(chat("average price?", self.path, base_url=fake.url).answer, "15.")
        self.assertEqual(len(fake.requests), 2)

    def test_a_call_with_no_question_gets_the_users_words(self):
        fake = FakeServer([tool_call("ada_ask", {}), {"role": "assistant", "content": "15."}])
        self.addCleanup(fake.close)
        reply = chat("what is the average price?", self.path, base_url=fake.url)
        self.assertEqual(reply.calls[0].arguments["question"], "what is the average price?")
        self.assertEqual(reply.calls[0].result["status"], "answered")

    def test_the_conversation_is_kept_and_rounds_are_capped(self):
        fake = FakeServer([{"role": "assistant", "content": "Hello."}] + [
            tool_call("ada_scan", {"path": "."})] * 8)
        self.addCleanup(fake.close)
        session = Chat(self.path, base_url=fake.url, model="other")
        session.ask("hi")
        reply = session.ask("scan forever")
        self.assertIn("no answer after 8 rounds", reply.answer)
        # A call made again gets the same result and a nudge, and the last round offers no tools.
        self.assertIn("made this call already", reply.calls[1].result["note"])
        self.assertNotIn("tools", fake.requests[-1])
        self.assertIn("tools", fake.requests[-2])
        self.assertEqual(fake.requests[1]["messages"][1:3], [{"role": "user", "content": "hi"},
                                                            {"role": "assistant", "content": "Hello."}])
        self.assertEqual(fake.requests[0]["model"], "other")

    def test_a_long_conversation_drops_its_oldest_turns_first(self):
        fake = FakeServer([{"role": "assistant", "content": "x" * 3500}] * 4)
        self.addCleanup(fake.close)
        session = Chat(self.path, base_url=fake.url, model="m")
        for question in ("one", "two", "three", "four"):
            session.ask(question)
        sent = fake.requests[-1]["messages"]
        self.assertEqual(sent[0]["role"], "system")
        self.assertEqual([item["content"] for item in sent if item["role"] == "user"], ["three", "four"])
        self.assertEqual(len(session.messages), 9)

    def test_errors_say_what_to_do(self):
        for status, detail, words in ((404, 'model "x" not found, try pulling it first', "ollama pull x"),
                                      (400, "x does not support tools", "does not support tool calling")):
            fake = FakeServer([{"error": {"message": detail}}], status=status, models=("a", "b"))
            self.addCleanup(fake.close)
            with self.assertRaises(ChatError) as caught:
                Chat(self.path, base_url=fake.url, model="x").ask("hi")
            self.assertIn(words, str(caught.exception))
        with self.assertRaises(ChatError) as caught:
            Chat(self.path, base_url="http://127.0.0.1:9/v1", model="x").ask("hi")
        self.assertIn("Could not reach a model server", str(caught.exception))
        with self.assertRaises(ChatError):
            Chat(self.root / "none.csv")

    def test_the_default_server_and_model(self):
        with mock.patch.dict("os.environ", {"OLLAMA_HOST": "0.0.0.0:11500"}, clear=True):
            self.assertEqual(default_base_url(), "http://localhost:11500/v1")
        with mock.patch.dict("os.environ", {"ADA_BASE_URL": "http://box:1234/v1/"}, clear=True):
            self.assertEqual(default_base_url(), "http://box:1234/v1")
        fake = FakeServer([], models=("a", "b"))
        self.addCleanup(fake.close)
        self.assertEqual(Chat(self.path, base_url=fake.url).model, "qwen3:1.7b")

    def test_the_command_asks_one_question(self):
        fake = FakeServer([tool_call("ada_ask", {"path": "sales.csv", "question": "average price"}),
                           {"role": "assistant", "content": "15."}])
        self.addCleanup(fake.close)
        out, err = io.StringIO(), io.StringIO()
        with contextlib.redirect_stdout(out), contextlib.redirect_stderr(err):
            code = main(["chat", str(self.path), "average", "price?", "--base-url", fake.url, "--json"])
        self.assertEqual(code, 0)
        printed = json.loads(out.getvalue())
        self.assertEqual((printed["answer"], printed["calls"][0]["name"]), ("15.", "ada_ask"))


if __name__ == "__main__":
    unittest.main()
