"""ADA's tools as LangChain tools, for LangChain agents and LangGraph."""

import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from ada import tools

try:
    import langchain_core
except ImportError:  # the [langchain] extra is not installed
    langchain_core = None


@unittest.skipIf(langchain_core is None, "langchain-core is not installed")
class LangChainTests(unittest.TestCase):
    def test_the_tools_carry_the_same_definitions(self):
        from langchain_core.utils.function_calling import convert_to_openai_tool

        made = tools.langchain()
        self.assertEqual([tool.name for tool in made], list(tools.NAMES))
        self.assertEqual([convert_to_openai_tool(tool) for tool in made], tools.definitions())

    def test_a_tool_call_made_the_way_an_agent_makes_it(self):
        compute = {tool.name: tool for tool in tools.langchain()}["ada_compute"]
        message = compute.invoke({"name": "ada_compute", "args": {"expression": "1/3 + 1/6"}, "id": "call-1",
                                  "type": "tool_call"})
        self.assertEqual(message.tool_call_id, "call-1")
        result = json.loads(message.content)
        self.assertEqual((result["status"], result["value"]), ("computed", "0.5"))

    def test_root_keeps_the_model_inside_a_folder(self):
        with tempfile.TemporaryDirectory() as folder, tempfile.TemporaryDirectory() as elsewhere:
            Path(folder, "orders.csv").write_text("region,revenue\nNorth,10\nSouth,30\nNorth,20\n",
                                                  encoding="utf-8")
            secret = Path(elsewhere, "salaries.csv")
            secret.write_text("name,salary\nA,1\n", encoding="utf-8")
            ask = {tool.name: tool for tool in tools.langchain(root=folder)}["ada_ask"]
            inside = json.loads(ask.invoke({"path": "orders.csv", "question": "total revenue by region"}))
            self.assertNotEqual(inside["status"], "error", inside)
            outside = json.loads(ask.invoke({"path": str(secret), "question": "total salary"}))
            self.assertEqual(outside["status"], "error")


class WithoutLangChainTests(unittest.TestCase):
    def test_the_message_says_what_to_install(self):
        with mock.patch.dict(sys.modules, {"langchain_core.tools": None}):
            with self.assertRaises(ImportError) as caught:
                tools.langchain()
        self.assertIn('pip install "automated-data-analyst[langchain]"', str(caught.exception))


if __name__ == "__main__":
    unittest.main()
