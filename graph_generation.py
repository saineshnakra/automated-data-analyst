import os
from pydantic import BaseModel
from openai import OpenAI
import json
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from typing_extensions import override
from openai import AssistantEventHandler
from dotenv import load_dotenv
import logging
from typing import List, Dict, Any, Optional

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class Graphs(BaseModel):
    graph_paths: List[str]

class GraphAssistantHandler(AssistantEventHandler):
    def __init__(self):
        self.response_content = ""
        self._stream = None
        
    @override
    def _init(self, stream):
        self._stream = stream
        
    @override
    def on_text_created(self, text) -> None:
        pass
          
    @override
    def on_text_delta(self, delta, snapshot):
        self.response_content += delta.value

class GraphGenerationError(Exception):
    """Custom exception for graph generation errors"""
    pass

def parse_assistant_response(response_content: str) -> Dict[str, List[Dict[str, str]]]:
    """
    Parse the assistant's response and ensure it has the correct structure.
    """
    try:
        # First try to parse the entire response
        graphs_info = json.loads(response_content)
        
        # If parsed successfully, validate the structure
        if not isinstance(graphs_info, dict):
            graphs_info = {"graphs": graphs_info if isinstance(graphs_info, list) else []}
        
        if "graphs" not in graphs_info:
            graphs_info = {"graphs": []}
            
        return graphs_info
        
    except json.JSONDecodeError:
        # Try to extract JSON from a larger text block
        try:
            start_idx = response_content.find('{')
            end_idx = response_content.rfind('}') + 1
            if start_idx >= 0 and end_idx > start_idx:
                json_str = response_content[start_idx:end_idx]
                graphs_info = json.loads(json_str)
                
                if not isinstance(graphs_info, dict):
                    graphs_info = {"graphs": graphs_info if isinstance(graphs_info, list) else []}
                
                if "graphs" not in graphs_info:
                    graphs_info = {"graphs": []}
                    
                return graphs_info
        except Exception as e:
            logger.error(f"Failed to extract JSON from response: {e}")
            
        return {"graphs": []}

def create_graph(df: pd.DataFrame, graph_info: Dict[str, str], idx: int) -> Optional[str]:
    """
    Create a single graph based on the provided information.
    Returns the path to the saved graph or None if creation fails.
    """
    try:
        title = graph_info.get('title', f'Graph {idx + 1}')
        graph_type = graph_info.get('type', '').lower()
        x = graph_info.get('x')
        y = graph_info.get('y')
        hue = graph_info.get('hue')

        if not x or not y:
            raise GraphGenerationError("Missing x or y axis specification")

        if x not in df.columns or y not in df.columns:
            raise GraphGenerationError(f"Columns '{x}' or '{y}' not found in DataFrame")

        if hue and hue not in df.columns:
            logger.warning(f"Hue column '{hue}' not found, ignoring hue parameter")
            hue = None

        plt.figure(figsize=(10, 6))
        
        if graph_type == 'bar':
            sns.barplot(data=df, x=x, y=y, hue=hue)
            plt.xticks(rotation=90)
        elif graph_type == 'line':
            sns.lineplot(data=df, x=x, y=y, hue=hue)
            plt.xticks(rotation=90)
        elif graph_type == 'scatter':
            sns.scatterplot(data=df, x=x, y=y, hue=hue)
            plt.xticks(rotation=90)
        elif graph_type == 'heatmap':
            pivot_table = df.pivot_table(index=y, columns=x, aggfunc='size', fill_value=0)
            sns.heatmap(pivot_table, annot=True, fmt='g')
            plt.xticks(rotation=90)
        else:
            raise GraphGenerationError(f"Unsupported graph type: {graph_type}")

        plt.title(title)
        plt.xlabel(x)
        plt.ylabel(y)
        plt.tight_layout()

        graph_folder = 'graphs'
        graph_path = os.path.join(graph_folder, f"graph_{idx + 1}.png")
        plt.savefig(graph_path)
        plt.close()
        
        return graph_path

    except Exception as e:
        logger.error(f"Error generating graph '{title}': {str(e)}")
        plt.close()
        return None

def generate_graphs(df, context):
    """
    Generate graphs based on the data and context provided.
    """
    if not isinstance(df, pd.DataFrame):
        raise ValueError("Input 'df' must be a pandas DataFrame")
        
    if not isinstance(context, dict):
        raise ValueError("Input 'context' must be a dictionary")

    # Create graphs directory if it doesn't exist
    os.makedirs('graphs', exist_ok=True)

    try:
        load_dotenv()
        client = OpenAI()
        
        # Prepare a sample of the data
        sample_data = df.head(5).to_csv(index=False)
        
        # Create an Assistant
        assistant = client.beta.assistants.create(
            name="Data Visualization Expert",
            instructions="""You are a data visualization expert. Analyze datasets and suggest insightful graphs 
            that reveal key trends and patterns. Provide responses in JSON format with the following structure:
            {
                "graphs": [
                    {
                        "title": "Graph Title",
                        "description": "What the graph shows",
                        "type": "bar/line/scatter/heatmap",
                        "x": "x_column_name",
                        "y": "y_column_name",
                        "hue": "grouping_column_name (optional)"
                    }
                ]
            }""",
            model="gpt-4",
        )

        thread = None
        graph_paths = []

        try:
            # Create a Thread
            thread = client.beta.threads.create()
            
            # Add a Message to the Thread
            message = client.beta.threads.messages.create(
                thread_id=thread.id,
                role="user",
                content=f"""
    You are a senior data analyst. Please analyze the following dataset and suggest multiple insightful graphs across features that reveal key trends, make sense and patterns in the data.
    Data description:
    {context.get('data_description', 'No description provided')}
    Data sample (CSV format):
    {sample_data}
    Available features:
    {', '.join(df.columns.tolist())}
    Requirements:
    - Suggest final graphs such as bar charts, line charts, scatter plots, heatmaps, etc.
    - For each graph, provide all required JSON fields
    - Do not include any code in your response.
    """
            )
            
            # Create and monitor run
            run = client.beta.threads.runs.create(
                thread_id=thread.id,
                assistant_id=assistant.id
            )
            
            max_retries = 3
            retry_count = 0
            
            while retry_count < max_retries:
                run_status = client.beta.threads.runs.retrieve(
                    thread_id=thread.id,
                    run_id=run.id
                )
                
                if run_status.status == 'completed':
                    break
                elif run_status.status == 'failed':
                    retry_count += 1
                    if retry_count == max_retries:
                        raise GraphGenerationError("Assistant run failed after maximum retries")
                    logger.warning(f"Assistant run failed, attempt {retry_count} of {max_retries}")
                    continue
                
            # Get the messages
            messages = client.beta.threads.messages.list(thread_id=thread.id)
            response_content = None
            
            for message in messages:
                if message.role == "assistant":
                    response_content = message.content[0].text.value
                    break
                    
            if not response_content:
                raise GraphGenerationError("No response received from assistant")

            # Parse the response and generate graphs
            graphs_info = parse_assistant_response(response_content)
            
            for idx, graph_info in enumerate(graphs_info.get('graphs', [])):
                graph_path = create_graph(df, graph_info, idx)
                if graph_path:
                    graph_paths.append(graph_path)

        finally:
            # Clean up resources
            if thread:
                try:
                    client.beta.threads.delete(thread.id)
                except Exception as e:
                    logger.error(f"Error deleting thread: {e}")

    except Exception as e:
        logger.error(f"Error in graph generation process: {e}")
        return Graphs(graph_paths=[])
        
    finally:
        # Clean up the assistant
        try:
            client.beta.assistants.delete(assistant.id)
        except Exception as e:
            logger.error(f"Error deleting assistant: {e}")

    return Graphs(graph_paths=graph_paths)