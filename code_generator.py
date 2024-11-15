# code_generator.py
import openai
import pandas as pd
from pydantic import BaseModel
import os
from dotenv import load_dotenv
from openai import OpenAI

from models import GeneratedCode

load_dotenv()
client = OpenAI()

def generate_ml_code(df: pd.DataFrame, error_message: str = None, current_code: str = None) -> GeneratedCode:
    # Prepare the data summary
    data_summary = df.describe(include='all').to_string()
    prompt = f"""
You are a data scientist. Write Python code to analyze the following dataset.
Data summary:
{data_summary}

Requirements:
- Perform necessary data preprocessing.
- Explore the data and identify potential predictive models.
- Implement a suitable machine learning algorithm to make predictions.
- Generate relevant graphs to visualize the data and the model's performance.
- Ensure the code is runnable and includes necessary imports.
- Use pandas, numpy, scikit-learn, matplotlib, seaborn.

Output:
- The complete Python code as a string.

"""

    if error_message and current_code:
        # If there's an error, include it in the prompt
        prompt += f"""
Previous code had the following error:
{error_message}

Current code:
{current_code}

Please correct the code to fix the error.
"""
    # Generate the code using OpenAI ChatCompletion
    response = client.chat.completions.create(
        model='gpt-4.o-mini',  # Replace with your accessible model
        messages=[{'role': 'user', 'content': prompt}],
        max_tokens=1500,
        temperature=0
    )
    code = response.choices[0].message.content.strip()
    return GeneratedCode(code=code)
