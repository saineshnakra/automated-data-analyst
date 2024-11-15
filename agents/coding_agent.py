# agents/coding_agent.py

import asyncio
import json
import openai
from autogen.agentchat.agents import CodingAssistantAgent
from autogen.core.components.models import OpenAIChatCompletionClient

class CodingAgent:
    def __init__(self):
        self.model_client = OpenAIChatCompletionClient(model="gpt-4", api_key=openai.api_key)
        self.coding_assistant = CodingAssistantAgent("coding_assistant", model_client=self.model_client)

    async def generate_code(self, experiment_plan):
        prompt = f"Generate Python code for the following experiment plan:\n{json.dumps(experiment_plan, indent=2)}\n\nInclude necessary imports and ensure the code is ready to run."
        response = await self.coding_assistant.run(task=prompt)
        code = self._extract_code_from_response(response)
        return code

    def _extract_code_from_response(self, response):
        code_blocks = response.split("```")
        if len(code_blocks) >= 3:
            return code_blocks[1].strip()
        return response.strip()

    async def process_experiment(self, experiment_plan):
        generated_code = await self.generate_code(experiment_plan)
        return {
            "generated_code": generated_code
        }