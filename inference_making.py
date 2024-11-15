# inference_making.py
from pydantic import BaseModel
import openai
import os
import base64
import pandas as pd
from dotenv import load_dotenv
from openai import OpenAI

class Inferences(BaseModel):
    text: str

def encode_image(image_path):
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode('utf-8')

def make_inferences_with_graphs(df: pd.DataFrame, graph_paths: list, context: dict) -> Inferences:
    load_dotenv()
    client = OpenAI()
    
    # Prepare data summary
    data_summary = df.describe(include='all').to_string()
    
    # Prepare messages with images
    messages = [
        {
            "role": "user",
            "content": [
                {
                    "type": "text",
                    "text": f"You are a senior data analyst presenting insights to business executives."
                             f" Based on the following data summary and visualizations, provide key insights.\n\n"
                             f"Data description:\n{context['data_description']}\n\n"
                             f"Data Summary:\n{data_summary}\n\n"
                             f"Please provide:\n"
                             f"1. A clear, executive-level summary of the main findings.\n"
                             f"2. Key trends and patterns identified.\n"
                             f"3. Business-relevant recommendations.\n\n"
                             f"Format the response in clear, non-technical language with bullet points where appropriate."
                    }
                ]
            }
        ]
    
    # Add images to the messages
    for graph_path in graph_paths:
        base64_image = encode_image(graph_path)
        messages[0]["content"].append(
            {
                "type": "image_url",
                "image_url": {
                    "url": f"data:image/png;base64,{base64_image}"
                }
            }
        )
    
    # Get main insights
    response = client.chat.completions.create(
        model='gpt-4o-mini',
        messages=messages,
        max_tokens=1000,
        temperature=0.3
    )
    main_insights = response.choices[0].message.content.strip()
    
    return Inferences(text=main_insights)