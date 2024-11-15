# data_analysis.py
import pandas as pd
from pydantic import BaseModel
import numpy as np

class AnalysisResults(BaseModel):
    description: str

def format_number(number):
    return f"{number:,.2f}"

def analyze_data(df: pd.DataFrame, context: dict) -> AnalysisResults:
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    
    insights = []
    
    # Basic statistics with formatted numbers
    insights.append("Dataset Overview:")
    insights.append(f"- Total records: {format_number(len(df))}")
    insights.append(f"- Number of features: {len(df.columns)}")
    
    # General analysis of numeric columns
    if len(numeric_cols) > 0:
        insights.append("\nNumeric Features Analysis:")
        for col in numeric_cols:
            insights.append(f"{col}:")
            insights.append(f"- Mean: {format_number(df[col].mean())}")
            insights.append(f"- Median: {format_number(df[col].median())}")
            insights.append(f"- Standard deviation: {format_number(df[col].std())}")
    
    # Correlation analysis
    if len(numeric_cols) > 1:
        insights.append("\nCorrelation Analysis:")
        correlations = df[numeric_cols].corr()
        insights.append(correlations.to_string())
    
    # Missing values
    missing = df.isnull().sum()
    if missing.any():
        insights.append("\nMissing Values:")
        for col, count in missing[missing > 0].items():
            insights.append(f"- {col}: {count} missing values")
    
    # Outliers detection for numeric columns
    insights.append("\nOutlier Detection:")
    for col in numeric_cols:
        Q1 = df[col].quantile(0.25)
        Q3 = df[col].quantile(0.75)
        IQR = Q3 - Q1
        outliers = df[(df[col] < Q1 - 1.5 * IQR) | (df[col] > Q3 + 1.5 * IQR)][col]
        if len(outliers) > 0:
            insights.append(f"- {col}: {len(outliers)} potential outliers detected")
    
    return AnalysisResults(description="\n".join(insights))