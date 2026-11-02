"""Chat with a local model that uses ADA's tools: ``ada chat`` and :class:`Chat`.

    ollama pull qwen3:1.7b
    ada chat data/penguins.csv
    > which species has the longest bills?

The model reads the question and calls ADA's tools; ADA computes every number
with pandas, SciPy and statsmodels, and the model answers with them. Each
tool call and what ADA returned is shown, so every number can be traced.

Any server with an OpenAI-compatible chat API works: Ollama, the default, at
``http://localhost:11434/v1``; LM Studio at ``http://localhost:1234/v1``;
llama.cpp's server at ``http://localhost:8080/v1``; vLLM at
``http://localhost:8000/v1``. The model has to support tool calling (on
Ollama's site, the models tagged "tools"). Only Python's standard library is
used to reach the server.
"""

from __future__ import annotations

import json
import os
import re
import urllib.error
import urllib.request
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

from .tools import LISTED_FILES, NAMES, SYSTEM_PROMPT, _data_file_paths, definitions, run

#: The model used when none is named, the environment's ADA_MODEL is unset and the server has several.
DEFAULT_MODEL = "qwen3:1.7b"
#: Rounds of tool calls one question may take before the model must answer.
ROUNDS = 8
#: Files whose columns the instructions list; past this, only the files' paths.
DESCRIBED_FILES = 10
#: Characters of earlier questions and answers sent with a new question. Ollama gives a model 4,096 tokens
#: by default, and the instructions, the tools and a thinking model's reasoning need most of them.
HISTORY_CHARS = 6000


class ChatError(Exception):
    """The model's server could not be reached or refused the request; the message says what to do."""


@dataclass(frozen=True)
class Call:
    """One tool call the model made and what ADA returned."""

    name: str
    arguments: dict
    result: dict

    def to_dict(self) -> dict:
        return {"name": self.name, "arguments": self.arguments, "result": self.result}


@dataclass(frozen=True)
class Reply:
    """The model's answer to one question and the tool calls it made on the way."""

    answer: str
    calls: tuple[Call, ...] = ()
    model: str = ""

    def to_dict(self) -> dict:
        return {"answer": self.answer, "model": self.model, "calls": [call.to_dict() for call in self.calls]}


def default_base_url() -> str:
    """ADA_BASE_URL, else Ollama's OpenAI-compatible API at OLLAMA_HOST or localhost:11434."""
    if os.environ.get("ADA_BASE_URL"):
        return os.environ["ADA_BASE_URL"].rstrip("/")
    host = os.environ.get("OLLAMA_HOST", "").strip() or "localhost:11434"
    host = host if "://" in host else f"http://{host}"
    host = host.replace("://0.0.0.0", "://localhost")
    return f"{host.rstrip('/')}/v1"


def _strip_thinking(text: str | None) -> str:
    """The answer without a model's reasoning, or the tool results some models go on to make up."""
    text = re.sub(r"<think>.*?</think>", "", text or "", flags=re.DOTALL)
    return re.sub(r"<tool_response>.*?(?:</tool_response>|$)", "", text, flags=re.DOTALL).strip()


def _call_in(value: Any) -> tuple[str, dict] | None:
    """The tool call a decoded JSON value stands for, looking inside wrappers such as
    {"type": "function", "function": {...}} and arguments written as a schema's "properties"."""
    if isinstance(value, dict):
        named = value.get("name")
        arguments = value.get("arguments", value.get("parameters"))
        if isinstance(arguments, str):
            try:
                arguments = json.loads(arguments)
            except json.JSONDecodeError:
                arguments = None
        if isinstance(arguments, dict) and isinstance(arguments.get("properties"), dict):
            arguments = {key: item for key, item in arguments["properties"].items() if isinstance(item, str)}
        if named in NAMES and isinstance(arguments, dict):
            return named, arguments
        for item in value.values():
            if (found := _call_in(item)) is not None:
                return found
    return None


#: The arguments a call written as broken JSON can still be read for.
WRITTEN_FIELDS = re.compile(r'"(path|question|claim|expression)"\s*:\s*"((?:[^"\\]|\\.)*)"')


def _written_call(text: str) -> tuple[str, dict] | None:
    """A tool call a small model wrote as text instead of making it:
    {"name": "ada_ask", "parameters": {...}}, wrapped or not, or broken JSON that still names a tool and
    gives its arguments as quoted strings."""
    for found in re.finditer(r"\{", text):
        try:
            value, _ = json.JSONDecoder().raw_decode(text[found.start():])
        except json.JSONDecodeError:
            continue
        if (call := _call_in(value)) is not None:
            return call
    named = re.search(r'"(ada_(?:ask|check_claim|scan|compute))"', text)
    fields = dict(WRITTEN_FIELDS.findall(text))
    if named and fields:
        return named.group(1), fields
    return None


def describe_data(root: Path, data: Path | None = None) -> str:
    """The lines that tell the model which data files it has, with their columns when there are few."""
    from .folder import scan  # noqa: PLC0415

    if data is not None and data.is_file():
        paths, cut = [data], False
    else:
        paths, cut = _data_file_paths(root)
    if not paths:
        return f"There are no data files under {root}."
    lines = []
    for path in paths[:LISTED_FILES]:
        shown = path.relative_to(root).as_posix() if path.is_relative_to(root) else str(path)
        columns = ""
        if len(paths) <= DESCRIBED_FILES:
            try:
                found = scan(path).files
            except Exception:  # noqa: BLE001 - a file that can't be read is still listed, without columns
                found = ()
            if found and found[0].columns:
                columns = f" ({found[0].rows:,} rows; columns: {', '.join(map(str, found[0].columns))})"
        lines.append(f"- {shown}{columns}")
    more = "\n- and more" if len(paths) > LISTED_FILES or cut else ""
    return "The data files, with paths to pass to the tools:\n" + "\n".join(lines) + more


class Chat:
    """A conversation with a model that calls ADA's tools on the files under ``root``.

    ``data`` is a file or folder to work on; its folder is the root unless
    ``root`` is given. ``model`` defaults to ADA_MODEL, else the server's only
    model, else :data:`DEFAULT_MODEL`. ``base_url`` defaults to
    :func:`default_base_url`; ``api_key`` to ADA_API_KEY, for servers that ask
    for one. Each :meth:`ask` keeps the conversation so far.
    """

    def __init__(self, data: str | Path = ".", *, model: str | None = None, base_url: str | None = None,
                 api_key: str | None = None, root: str | Path | None = None, timeout: float = 900):
        target = Path(data).expanduser().resolve()
        if not target.exists():
            raise ChatError(f"{data} does not exist.")
        self.data = target
        self.root = Path(root).expanduser().resolve() if root is not None else (
            target if target.is_dir() else target.parent)
        self.base_url = (base_url or default_base_url()).rstrip("/")
        self.api_key = api_key if api_key is not None else os.environ.get("ADA_API_KEY", "")
        self.timeout = timeout
        self.model = model or os.environ.get("ADA_MODEL") or self._default_model()
        self.messages: list[dict] = [{"role": "system", "content": (
            f"{SYSTEM_PROMPT}\n{describe_data(self.root, target if target.is_file() else None)}")}]

    def _opener(self) -> urllib.request.OpenerDirector:
        # A server on this computer is reached directly, never through a proxy the environment names.
        local = urlparse(self.base_url).hostname in ("localhost", "127.0.0.1", "::1")
        return urllib.request.build_opener(urllib.request.ProxyHandler({})) if local else \
            urllib.request.build_opener()

    def _request(self, path: str, body: dict | None = None) -> dict:
        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        data = None if body is None else json.dumps(body).encode("utf-8")
        request = urllib.request.Request(f"{self.base_url}{path}", data=data, headers=headers)
        try:
            with self._opener().open(request, timeout=self.timeout) as response:
                return json.loads(response.read().decode("utf-8"))
        except urllib.error.HTTPError as error:
            text = error.read().decode("utf-8", "replace")
            try:
                detail = json.loads(text).get("error", text)
                detail = detail.get("message", detail) if isinstance(detail, dict) else detail
            except (json.JSONDecodeError, AttributeError):
                detail = text
            raise ChatError(self._explained(error.code, str(detail))) from None
        except (urllib.error.URLError, TimeoutError, ConnectionError) as error:
            raise ChatError(
                f"Could not reach a model server at {self.base_url} ({getattr(error, 'reason', error)}). "
                "Start Ollama (ollama serve), or name the server with --base-url, such as "
                "http://localhost:1234/v1 for LM Studio."
            ) from None

    def _explained(self, code: int, detail: str) -> str:
        if "support" in detail.lower() and "tool" in detail.lower():
            return (f"{self.model} does not support tool calling, so it cannot use ADA: {detail}. Pick a "
                    f"model that does, such as {DEFAULT_MODEL} (ollama pull {DEFAULT_MODEL}).")
        if code == 404 or "not found" in detail.lower():
            return (f"The server has no model called {self.model}: {detail}. Pull it (ollama pull "
                    f"{self.model}) or name another with --model{self._installed()}.")
        return f"The model server refused the request ({code}): {detail}"

    def _models(self) -> list[str]:
        try:
            listed = self._request("/models")
        except ChatError:
            return []
        return sorted(str(item.get("id")) for item in listed.get("data", []) if item.get("id"))

    def _installed(self) -> str:
        models = self._models()
        return f"; the server has {', '.join(models)}" if models else ""

    def _default_model(self) -> str:
        models = self._models()
        if len(models) == 1:
            return models[0]
        return DEFAULT_MODEL

    def _window(self, start: int) -> list[dict]:
        """The instructions, the latest earlier turns that fit :data:`HISTORY_CHARS`, and this question's
        messages from ``start`` on: a long conversation drops its oldest turns first."""
        kept: list[dict] = []
        size = 0
        for message in reversed(self.messages[1:start]):
            size += len(json.dumps(message, ensure_ascii=False))
            if size > HISTORY_CHARS:
                break
            kept.insert(0, message)
        while kept and kept[0].get("role") != "user":
            kept.pop(0)
        return [self.messages[0], *kept, *self.messages[start:]]

    def ask(self, question: str, on_call: Callable[[Call], None] | None = None) -> Reply:
        """The model's answer to ``question``, after the tool calls it makes; ``on_call`` sees each one."""
        start = len(self.messages)
        self.messages.append({"role": "user", "content": question})
        calls: list[Call] = []
        made: dict[str, dict] = {}
        retried = False
        for round_ in range(ROUNDS):
            body = {"model": self.model, "messages": self._window(start), "stream": False}
            if round_ < ROUNDS - 1:
                # The last round offers no tools, so the model answers with what it has.
                body["tools"] = definitions()
            reply = self._request("/chat/completions", body)
            choices = reply.get("choices") or [{}]
            message = choices[0].get("message") or {}
            content = _strip_thinking(message.get("content"))
            requested = message.get("tool_calls") or []
            written = None if requested else _written_call(content)
            if not requested and written is None and not content and not retried:
                # Small models now and then reply with nothing at all: asked again, once.
                retried = True
                continue
            if not requested and written is None:
                self.messages.append({"role": "assistant", "content": content})
                return Reply(content, tuple(calls), self.model)
            if written is not None:
                # A call written as text is made, and recorded as the call it stands for.
                requested = [{"id": f"call_{len(calls)}", "type": "function",
                              "function": {"name": written[0], "arguments": json.dumps(written[1])}}]
            self.messages.append({"role": "assistant", "content": content if not written else "",
                                  "tool_calls": requested})
            for item in requested:
                function = item.get("function") or {}
                name, arguments = function.get("name", ""), function.get("arguments") or {}
                if isinstance(arguments, str):
                    try:
                        arguments = json.loads(arguments) if arguments.strip() else {}
                    except json.JSONDecodeError:
                        arguments = {"text": arguments}
                if isinstance(arguments, dict):
                    # A call that leaves out the question or the claim gets the user's own words.
                    field = {"ada_ask": "question", "ada_check_claim": "claim"}.get(name)
                    if field and not arguments.get(field):
                        arguments = {**arguments, field: question}
                key = json.dumps([name, arguments], sort_keys=True, default=str)
                if key in made:
                    # A call made before gets its result again, with a nudge to answer: small models repeat.
                    result = {**made[key], "note": "You made this call already; answer the user with this "
                                                   "result."}
                else:
                    result = made[key] = run(name, arguments, root=self.root, brief=True)
                call = Call(name, arguments if isinstance(arguments, dict) else {}, result)
                calls.append(call)
                if on_call is not None:
                    on_call(call)
                self.messages.append({"role": "tool", "tool_call_id": item.get("id", ""), "name": name,
                                      "content": json.dumps(result, ensure_ascii=False)})
        answer = f"(no answer after {ROUNDS} rounds of tool calls)"
        self.messages.append({"role": "assistant", "content": answer})
        return Reply(answer, tuple(calls), self.model)


def chat(question: str, data: str | Path = ".", **options: Any) -> Reply:
    """One question to a model that uses ADA's tools on ``data``: ``chat("average price by city",
    "sales.csv", model="qwen3:1.7b")``. The options are :class:`Chat`'s."""
    return Chat(data, **options).ask(question)


def shown(call: Call) -> str:
    """How ``ada chat`` prints a tool call: the call, then ADA's result in one line."""
    arguments = ", ".join(f"{key}={json.dumps(value, ensure_ascii=False)}"
                          for key, value in call.arguments.items())
    result = call.result
    said = result.get("answer") or result.get("message") or ""
    if result.get("verdict"):
        said = f"{result['verdict']}: {said}"
    if "files" in result and not said:
        said = ", ".join(f"{item['path']} ({item['rows']} rows)" for item in result["files"][:5])
    status = result.get("status") or ("scanned" if "files" in result else "")
    return f"  ADA {call.name}({arguments})\n    -> {status}: {said}"
