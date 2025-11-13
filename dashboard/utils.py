"""
Utility functions for data loading and processing
"""
import pandas as pd
from pathlib import Path
import streamlit as st


@st.cache_data
def load_priority_data():
    """Load the pre-computed priority list data"""
    script_dir = Path(__file__).parent
    data_path = script_dir / "data" / "analyst_priority_list.csv"
    
    if not data_path.exists():
        raise FileNotFoundError(f"Data file not found at {data_path}")
    
    df = pd.read_csv(data_path)
    
    # Ensure numeric columns are properly typed
    df['Churn_Probability'] = pd.to_numeric(df['Churn_Probability'], errors='coerce')
    df['CLTV'] = pd.to_numeric(df['CLTV'], errors='coerce')
    df['Priority_Score'] = pd.to_numeric(df['Priority_Score'], errors='coerce')
    
    return df


def calculate_impact_metrics(df):
    """Calculate key business impact metrics"""
    total_at_risk_cltv = df['CLTV'].sum()
    high_risk_count = len(df[df['ML_Risk_Level'] == 'High'])
    medium_risk_count = len(df[df['ML_Risk_Level'] == 'Medium'])
    low_risk_count = len(df[df['ML_Risk_Level'] == 'Low'])
    
    high_risk_cltv = df[df['ML_Risk_Level'] == 'High']['CLTV'].sum()
    avg_churn_prob = df['Churn_Probability'].mean()
    
    return {
        'total_at_risk_cltv': total_at_risk_cltv,
        'high_risk_count': high_risk_count,
        'medium_risk_count': medium_risk_count,
        'low_risk_count': low_risk_count,
        'high_risk_cltv': high_risk_cltv,
        'avg_churn_prob': avg_churn_prob,
        'total_customers': len(df)
    }


def format_currency(value):
    """Format number as currency"""
    return f"${value:,.0f}"


def get_customer_details(df, customer_id):
    """Get full details for a specific customer"""
    customer = df[df['CustomerID'] == customer_id]
    if customer.empty:
        return None
    return customer.iloc[0].to_dict()
