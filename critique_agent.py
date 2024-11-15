# critique_agent.py
from pydantic import BaseModel
import openai
import os
from dotenv import load_dotenv
from openai import OpenAI

class Critique(BaseModel):
    feedback: str

def critique_outputs(graph_paths: list, inference_text: str, context: dict) -> Critique:
    load_dotenv()
    client = OpenAI()
    
    graph_descriptions = "Graphs generated:\n" + "\n".join(graph_paths)
    prompt = f"""
You are an expert data analyst reviewing the following analysis and visualizations.

Data description:
{context['data_description']}

Inferences:
{inference_text}

{graph_descriptions}

Please provide a critique that includes:
- Assessment of the accuracy and relevance of the inferences.
- Suggestions for additional analyses or visualizations.
- Recommendations for improving the presentation of findings.

Provide your feedback in a constructive manner.
"""
    response = client.chat.completions.create(
        model='gpt-4o-mini',
        messages=[{'role': 'user', 'content': prompt}],
        max_tokens=500,
        temperature=0.3
    )
    critique_feedback = response.choices[0].message.content.strip()
    return Critique(feedback=critique_feedback)