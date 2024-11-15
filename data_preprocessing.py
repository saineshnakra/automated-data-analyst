# data_preprocessing.py
import pandas as pd
from pydantic import BaseModel, ConfigDict

class PreprocessedData(BaseModel):
    data: pd.DataFrame
    context: dict
    model_config = ConfigDict(arbitrary_types_allowed=True)

def preprocess_data(file, data_description: str) -> PreprocessedData:
    df = pd.read_csv(file)
    df.dropna(inplace=True)
    df = df.head(100)
    
    features = df.columns.tolist()
    
    context = {
        'data_description': data_description,
        'features': features
    }
    return PreprocessedData(data=df, context=context)