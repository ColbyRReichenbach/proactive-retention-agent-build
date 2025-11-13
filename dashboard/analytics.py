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
        paper_bgcolor='#f5f5f5',
        legend=dict(
            orientation="v",
            yanchor="top",
            y=1,
            xanchor="left",
            x=1.02,
            bgcolor="rgba(255,255,255,0.8)"
        ),
        margin=dict(r=120)  # Extra margin for legend
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
        paper_bgcolor='#f5f5f5',
        legend=dict(
            orientation="v",
            yanchor="top",
            y=1,
            xanchor="left",
            x=1.02,
            bgcolor="rgba(255,255,255,0.8)"
        ),
        margin=dict(r=120)  # Extra margin for legend
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
        paper_bgcolor='#f5f5f5',
        margin=dict(b=80)  # Extra bottom margin for x-axis labels
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


# ============================================================================
# LIVE PIPELINE SPECIFIC ANALYTICS (More in-depth than demo)
# ============================================================================

def create_risk_value_quadrant(df):
    """Create a quadrant analysis: Risk vs Value (CLTV)"""
    # Create risk score from risk level
    risk_map = {'High': 3, 'Medium': 2, 'Low': 1}
    df_plot = df.copy()
    df_plot['Risk_Score'] = df_plot['ML_Risk_Level'].map(risk_map)
    
    # Calculate thresholds for quadrants
    cltv_median = df_plot['CLTV'].median()
    churn_median = df_plot['Churn_Probability'].median()
    
    fig = px.scatter(
        df_plot,
        x='Churn_Probability',
        y='CLTV',
        color='ML_Risk_Level',
        size='Priority_Score',
        hover_data=['CustomerID', 'LLM_Theme', 'LLM_Sentiment'],
        labels={
            'Churn_Probability': 'Churn Probability',
            'CLTV': 'Customer Lifetime Value ($)',
            'ML_Risk_Level': 'Risk Level'
        },
        title='Risk vs Value Quadrant Analysis',
        color_discrete_map={'High': '#dc3545', 'Medium': '#ffc107', 'Low': '#28a745'}
    )
    
    # Add quadrant lines
    fig.add_hline(y=cltv_median, line_dash="dash", line_color="gray", 
                  annotation_text=f"Median CLTV: ${cltv_median:,.0f}")
    fig.add_vline(x=churn_median, line_dash="dash", line_color="gray",
                  annotation_text=f"Median Churn: {churn_median:.1%}")
    
    fig.update_layout(
        height=500,
        plot_bgcolor='#f5f5f5',
        paper_bgcolor='#f5f5f5',
        legend=dict(
            orientation="v",
            yanchor="top",
            y=1,
            xanchor="left",
            x=1.02,
            bgcolor="rgba(255,255,255,0.8)"
        ),
        margin=dict(r=120)  # Extra margin for legend
    )
    return fig


def create_theme_impact_analysis(df):
    """Analyze which themes have the highest business impact (Priority Score)"""
    theme_impact = df.groupby('LLM_Theme').agg({
        'Priority_Score': ['sum', 'mean', 'count'],
        'CLTV': 'sum',
        'Churn_Probability': 'mean'
    }).round(2)
    
    theme_impact.columns = ['Total_Priority', 'Avg_Priority', 'Count', 'Total_CLTV', 'Avg_Churn']
    theme_impact = theme_impact.sort_values('Total_Priority', ascending=False)
    
    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=('Total Priority Score by Theme', 'Average Priority Score',
                       'Total CLTV at Risk by Theme', 'Average Churn Probability'),
        specs=[[{"secondary_y": False}, {"secondary_y": False}],
               [{"secondary_y": False}, {"secondary_y": False}]]
    )
    
    # Total Priority Score
    fig.add_trace(
        go.Bar(x=theme_impact.index, y=theme_impact['Total_Priority'],
               name='Total Priority', marker_color='#dc3545'),
        row=1, col=1
    )
    
    # Average Priority Score
    fig.add_trace(
        go.Bar(x=theme_impact.index, y=theme_impact['Avg_Priority'],
               name='Avg Priority', marker_color='#ffc107'),
        row=1, col=2
    )
    
    # Total CLTV at Risk
    fig.add_trace(
        go.Bar(x=theme_impact.index, y=theme_impact['Total_CLTV'],
               name='Total CLTV', marker_color='#2196f3'),
        row=2, col=1
    )
    
    # Average Churn Probability
    fig.add_trace(
        go.Bar(x=theme_impact.index, y=theme_impact['Avg_Churn'],
               name='Avg Churn', marker_color='#9c27b0'),
        row=2, col=2
    )
    
    fig.update_xaxes(title_text="Theme", row=2, col=1)
    fig.update_xaxes(title_text="Theme", row=2, col=2)
    fig.update_yaxes(title_text="Priority Score", row=1, col=1)
    fig.update_yaxes(title_text="Priority Score", row=1, col=2)
    fig.update_yaxes(title_text="CLTV ($)", row=2, col=1)
    fig.update_yaxes(title_text="Churn Probability", row=2, col=2)
    
    fig.update_layout(
        height=700,
        showlegend=False,
        plot_bgcolor='#f5f5f5',
        paper_bgcolor='#f5f5f5',
        margin=dict(b=100, t=50)  # Extra bottom margin for x-axis labels
    )
    return fig


def create_sentiment_severity_analysis(df):
    """Analyze sentiment severity by theme and risk level"""
    pivot = df.groupby(['LLM_Theme', 'LLM_Sentiment', 'ML_Risk_Level']).agg({
        'Priority_Score': 'sum',
        'CustomerID': 'count'
    }).reset_index()
    
    # Create a stacked bar chart
    fig = go.Figure()
    
    sentiments = ['Negative', 'Neutral', 'Positive']
    colors_sentiment = {'Negative': '#dc3545', 'Neutral': '#ffc107', 'Positive': '#28a745'}
    
    for sentiment in sentiments:
        sentiment_data = pivot[pivot['LLM_Sentiment'] == sentiment]
        fig.add_trace(go.Bar(
            name=sentiment,
            x=sentiment_data['LLM_Theme'],
            y=sentiment_data['Priority_Score'],
            marker_color=colors_sentiment[sentiment],
            text=sentiment_data['CustomerID'],
            textposition='inside',
            hovertemplate='<b>%{x}</b><br>%{fullData.name}<br>Priority Score: %{y:,.0f}<br>Customers: %{text}<extra></extra>'
        ))
    
    fig.update_layout(
        barmode='stack',
        title='Priority Score by Theme and Sentiment (Stacked)',
        xaxis_title='Complaint Theme',
        yaxis_title='Total Priority Score',
        height=450,
        plot_bgcolor='#f5f5f5',
        paper_bgcolor='#f5f5f5',
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1
        ),
        margin=dict(b=100, t=80)  # Extra margins
    )
    return fig


def create_action_urgency_matrix(df):
    """Create a matrix showing action urgency based on risk and priority"""
    # Categorize customers into urgency buckets
    df_plot = df.copy()
    
    # Define urgency based on risk level and priority score
    priority_75th = df_plot['Priority_Score'].quantile(0.75)
    priority_50th = df_plot['Priority_Score'].quantile(0.50)
    
    def get_urgency(row):
        if row['ML_Risk_Level'] == 'High' and row['Priority_Score'] >= priority_75th:
            return 'Immediate Action'
        elif row['ML_Risk_Level'] == 'High' or row['Priority_Score'] >= priority_75th:
            return 'High Priority'
        elif row['Priority_Score'] >= priority_50th:
            return 'Medium Priority'
        else:
            return 'Monitor'
    
    df_plot['Urgency'] = df_plot.apply(get_urgency, axis=1)
    
    urgency_counts = df_plot['Urgency'].value_counts()
    urgency_order = ['Immediate Action', 'High Priority', 'Medium Priority', 'Monitor']
    urgency_counts = urgency_counts.reindex([u for u in urgency_order if u in urgency_counts.index])
    
    colors_urgency = {
        'Immediate Action': '#8b0000',
        'High Priority': '#dc3545',
        'Medium Priority': '#ffc107',
        'Monitor': '#6c757d'
    }
    
    fig = px.bar(
        x=urgency_counts.index,
        y=urgency_counts.values,
        labels={'x': 'Action Urgency', 'y': 'Number of Customers'},
        title='Action Urgency Distribution',
        color=urgency_counts.index,
        color_discrete_map=colors_urgency,
        text=urgency_counts.values
    )
    fig.update_traces(textposition='outside')
    fig.update_layout(
        showlegend=False,
        height=400,
        plot_bgcolor='#f5f5f5',
        paper_bgcolor='#f5f5f5',
        margin=dict(b=80)
    )
    return fig


def create_cltv_at_risk_by_theme(df):
    """Show total CLTV at risk grouped by theme"""
    theme_cltv = df.groupby('LLM_Theme').agg({
        'CLTV': 'sum',
        'Churn_Probability': lambda x: (df.loc[x.index, 'CLTV'] * x).sum() / df.loc[x.index, 'CLTV'].sum(),
        'CustomerID': 'count'
    }).reset_index()
    theme_cltv.columns = ['Theme', 'Total_CLTV', 'Weighted_Avg_Churn', 'Count']
    theme_cltv['Expected_Loss'] = theme_cltv['Total_CLTV'] * theme_cltv['Weighted_Avg_Churn']
    theme_cltv = theme_cltv.sort_values('Expected_Loss', ascending=False)
    
    fig = make_subplots(
        rows=1, cols=2,
        subplot_titles=('Total CLTV at Risk by Theme', 'Expected Revenue Loss by Theme'),
        specs=[[{"secondary_y": False}, {"secondary_y": False}]]
    )
    
    fig.add_trace(
        go.Bar(x=theme_cltv['Theme'], y=theme_cltv['Total_CLTV'],
               name='Total CLTV', marker_color='#2196f3',
               text=[f'${x:,.0f}' for x in theme_cltv['Total_CLTV']],
               textposition='outside'),
        row=1, col=1
    )
    
    fig.add_trace(
        go.Bar(x=theme_cltv['Theme'], y=theme_cltv['Expected_Loss'],
               name='Expected Loss', marker_color='#dc3545',
               text=[f'${x:,.0f}' for x in theme_cltv['Expected_Loss']],
               textposition='outside'),
        row=1, col=2
    )
    
    fig.update_xaxes(title_text="Theme", row=1, col=1)
    fig.update_xaxes(title_text="Theme", row=1, col=2)
    fig.update_yaxes(title_text="CLTV ($)", row=1, col=1)
    fig.update_yaxes(title_text="Expected Loss ($)", row=1, col=2)
    
    fig.update_layout(
        height=450,
        showlegend=False,
        plot_bgcolor='#f5f5f5',
        paper_bgcolor='#f5f5f5',
        margin=dict(b=100, t=50)
    )
    return fig


def create_risk_concentration_analysis(df):
    """Analyze how risk is concentrated across different dimensions"""
    # Risk concentration by theme
    theme_risk = df.groupby(['LLM_Theme', 'ML_Risk_Level']).size().unstack(fill_value=0)
    
    fig = px.bar(
        theme_risk,
        barmode='group',
        labels={'value': 'Number of Customers', 'LLM_Theme': 'Complaint Theme'},
        title='Risk Level Distribution by Theme',
        color_discrete_map={'High': '#dc3545', 'Medium': '#ffc107', 'Low': '#28a745'}
    )
    
    fig.update_layout(
        height=450,
        plot_bgcolor='#f5f5f5',
        paper_bgcolor='#f5f5f5',
        legend=dict(
            orientation="v",
            yanchor="top",
            y=1,
            xanchor="left",
            x=1.02,
            bgcolor="rgba(255,255,255,0.8)"
        ),
        margin=dict(r=120, b=100)  # Extra margins for legend and x-axis
    )
    return fig

