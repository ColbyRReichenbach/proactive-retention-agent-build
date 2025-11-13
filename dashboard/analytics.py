"""
Analytics and visualization functions
"""
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots


def create_risk_distribution_chart(df):
    """Create risk level distribution chart"""
    risk_counts = df['ML_Risk_Level'].value_counts().sort_index()
    
    colors = {'High': '#dc3545', 'Medium': '#ffc107', 'Low': '#28a745'}
    color_list = [colors.get(level, '#6c757d') for level in risk_counts.index]
    
    fig = px.bar(
        x=risk_counts.index,
        y=risk_counts.values,
        labels={'x': 'Risk Level', 'y': 'Number of Customers'},
        title='Customer Risk Distribution',
        color=risk_counts.index,
        color_discrete_map=colors
    )
    fig.update_layout(
        showlegend=False, 
        height=300,
        plot_bgcolor='#f5f5f5',
        paper_bgcolor='#f5f5f5'
    )
    return fig


def create_theme_breakdown_chart(df):
    """Create complaint theme breakdown chart"""
    theme_counts = df['LLM_Theme'].value_counts()
    
    fig = px.pie(
        values=theme_counts.values,
        names=theme_counts.index,
        title='Complaint Themes Distribution',
        hole=0.4
    )
    fig.update_layout(
        height=350,
        plot_bgcolor='#f5f5f5',
        paper_bgcolor='#f5f5f5'
    )
    return fig


def create_sentiment_chart(df):
    """Create sentiment distribution chart"""
    sentiment_counts = df['LLM_Sentiment'].value_counts()
    
    colors = {'Negative': '#dc3545', 'Positive': '#28a745', 'Neutral': '#6c757d'}
    
    fig = px.bar(
        x=sentiment_counts.index,
        y=sentiment_counts.values,
        labels={'x': 'Sentiment', 'y': 'Number of Reviews'},
        title='Review Sentiment Distribution',
        color=sentiment_counts.index,
        color_discrete_map=colors
    )
    fig.update_layout(
        showlegend=False, 
        height=300,
        plot_bgcolor='#f5f5f5',
        paper_bgcolor='#f5f5f5'
    )
    return fig


def create_cltv_vs_churn_scatter(df):
    """Create CLTV vs Churn Probability scatter plot"""
    fig = px.scatter(
        df,
        x='Churn_Probability',
        y='CLTV',
        size='Priority_Score',
        color='ML_Risk_Level',
        hover_data=['CustomerID', 'LLM_Theme'],
        labels={
            'Churn_Probability': 'Churn Probability',
            'CLTV': 'Customer Lifetime Value ($)',
            'ML_Risk_Level': 'Risk Level'
        },
        title='CLTV vs Churn Probability (Bubble size = Priority Score)',
        color_discrete_map={'High': '#dc3545', 'Medium': '#ffc107', 'Low': '#28a745'}
    )
    fig.update_layout(
        height=400,
        plot_bgcolor='#f5f5f5',
        paper_bgcolor='#f5f5f5'
    )
    return fig


def create_priority_score_distribution(df):
    """Create priority score distribution histogram"""
    fig = px.histogram(
        df,
        x='Priority_Score',
        nbins=30,
        labels={'Priority_Score': 'Priority Score', 'count': 'Number of Customers'},
        title='Priority Score Distribution',
        color='ML_Risk_Level',
        color_discrete_map={'High': '#dc3545', 'Medium': '#ffc107', 'Low': '#28a745'}
    )
    fig.update_layout(
        height=350,
        plot_bgcolor='#f5f5f5',
        paper_bgcolor='#f5f5f5'
    )
    return fig


def create_theme_sentiment_heatmap(df):
    """Create theme vs sentiment heatmap"""
    pivot = pd.crosstab(df['LLM_Theme'], df['LLM_Sentiment'])
    
    fig = px.imshow(
        pivot,
        labels=dict(x="Sentiment", y="Theme", color="Count"),
        title="Complaint Theme vs Sentiment Heatmap",
        color_continuous_scale='RdYlGn_r'
    )
    fig.update_layout(
        height=400,
        plot_bgcolor='#f5f5f5',
        paper_bgcolor='#f5f5f5'
    )
    return fig


def create_avg_churn_by_theme(df):
    """Create average churn probability by theme"""
    theme_avg = df.groupby('LLM_Theme')['Churn_Probability'].mean().sort_values(ascending=False)
    
    fig = px.bar(
        x=theme_avg.index,
        y=theme_avg.values,
        labels={'x': 'Complaint Theme', 'y': 'Average Churn Probability'},
        title='Average Churn Probability by Complaint Theme',
        color=theme_avg.values,
        color_continuous_scale='Reds'
    )
    fig.update_layout(
        showlegend=False, 
        height=350,
        plot_bgcolor='#f5f5f5',
        paper_bgcolor='#f5f5f5'
    )
    return fig


def create_top_customers_table(df, n=10):
    """Create a formatted table of top N customers"""
    top_customers = df.head(n)[['CustomerID', 'ML_Risk_Level', 'Churn_Probability', 
                                'LLM_Theme', 'LLM_Sentiment', 'CLTV', 'Priority_Score']].copy()
    
    # Format columns for display
    top_customers['Churn_Probability'] = top_customers['Churn_Probability'].apply(lambda x: f"{x:.1%}")
    top_customers['CLTV'] = top_customers['CLTV'].apply(lambda x: f"${x:,.0f}")
    top_customers['Priority_Score'] = top_customers['Priority_Score'].apply(lambda x: f"{x:,.0f}")
    
    top_customers.columns = ['Customer ID', 'Risk Level', 'Churn Prob.', 'Theme', 
                             'Sentiment', 'CLTV', 'Priority Score']
    
    return top_customers

