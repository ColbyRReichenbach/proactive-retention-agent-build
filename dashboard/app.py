"""
Proactive Retention Agent - Interactive Dashboard
A hybrid approach showcasing the end-to-end pipeline with analytics
"""
import streamlit as st
import pandas as pd
import plotly.express as px
import os
from pathlib import Path

# Import utility functions
from utils import load_priority_data, calculate_impact_metrics, format_currency, get_customer_details
from analytics import (
    create_risk_distribution_chart,
    create_theme_breakdown_chart,
    create_sentiment_chart,
    create_cltv_vs_churn_scatter,
    create_priority_score_distribution,
    create_theme_sentiment_heatmap,
    create_avg_churn_by_theme,
    create_top_customers_table
)
from live_pipeline import run_live_pipeline

# Page configuration
st.set_page_config(
    page_title="Proactive Retention Agent",
    page_icon=None,
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for professional styling
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: 700;
        color: #1a1a1a;
        text-align: center;
        margin-bottom: 0.5rem;
        letter-spacing: -0.5px;
    }
    .sub-header {
        font-size: 1.1rem;
        color: #666;
        text-align: center;
        margin-bottom: 2.5rem;
        font-weight: 400;
    }
    .step-container {
        background: linear-gradient(135deg, #f8f9fa 0%, #ffffff 100%);
        border: 1px solid #e0e0e0;
        border-radius: 12px;
        padding: 2rem;
        margin: 1.5rem 0;
        box-shadow: 0 2px 8px rgba(0, 0, 0, 0.08);
        transition: box-shadow 0.3s ease;
    }
    .step-container:hover {
        box-shadow: 0 4px 12px rgba(0, 0, 0, 0.12);
    }
    .step-number {
        display: inline-block;
        background: linear-gradient(135deg, #1f77b4 0%, #2c8fd4 100%);
        color: white;
        width: 36px;
        height: 36px;
        border-radius: 50%;
        text-align: center;
        line-height: 36px;
        font-weight: 700;
        font-size: 1.1rem;
        margin-right: 1rem;
        box-shadow: 0 2px 4px rgba(31, 119, 180, 0.3);
    }
    .step-title {
        font-size: 1.5rem;
        font-weight: 600;
        color: #1a1a1a;
        margin-bottom: 1rem;
        display: inline-block;
    }
    .step-content {
        color: #333;
        line-height: 1.7;
        font-size: 1rem;
    }
    .step-content strong {
        color: #1a1a1a;
        font-weight: 600;
    }
    .metric-container {
        background: linear-gradient(135deg, #f8f9fa 0%, #ffffff 100%);
        border: 1px solid #e0e0e0;
        border-radius: 12px;
        padding: 1.5rem;
        box-shadow: 0 2px 6px rgba(0, 0, 0, 0.06);
        transition: transform 0.2s ease, box-shadow 0.2s ease;
        height: 100%;
        text-align: center;
        display: flex;
        flex-direction: column;
        justify-content: center;
    }
    .metric-container:hover {
        transform: translateY(-2px);
        box-shadow: 0 4px 10px rgba(0, 0, 0, 0.1);
    }
    .metric-label {
        font-size: 0.875rem;
        color: #666;
        font-weight: 500;
        margin-bottom: 0.5rem;
        text-transform: uppercase;
        letter-spacing: 0.5px;
        text-align: center;
    }
    .metric-value {
        font-size: 2rem;
        font-weight: 700;
        color: #1a1a1a;
        line-height: 1.2;
        margin: 0;
        text-align: center;
    }
    .pipeline-overview-card {
        background: linear-gradient(135deg, #f8f9fa 0%, #ffffff 100%);
        border: 1px solid #e0e0e0;
        border-radius: 12px;
        padding: 1.5rem;
        text-align: center;
        box-shadow: 0 2px 6px rgba(0, 0, 0, 0.06);
        height: 100%;
        transition: transform 0.2s ease, box-shadow 0.2s ease;
    }
    .pipeline-overview-card:hover {
        transform: translateY(-2px);
        box-shadow: 0 4px 10px rgba(0, 0, 0, 0.1);
    }
    .pipeline-overview-card h3 {
        font-size: 1.2rem;
        font-weight: 600;
        color: #1a1a1a;
        margin-bottom: 0.75rem;
    }
    .pipeline-overview-card p {
        color: #666;
        font-size: 0.95rem;
        margin: 0;
    }
    .customer-journey-step {
        background: linear-gradient(135deg, #f8f9fa 0%, #ffffff 100%);
        border: 1px solid #e0e0e0;
        border-radius: 12px;
        padding: 1.75rem;
        margin: 1.25rem 0;
        box-shadow: 0 2px 8px rgba(0, 0, 0, 0.08);
    }
    .risk-indicator {
        display: inline-block;
        padding: 0.25rem 0.75rem;
        border-radius: 20px;
        font-size: 0.85rem;
        font-weight: 600;
    }
    .risk-high {
        background-color: #fee;
        color: #c33;
        border: 1px solid #fcc;
    }
    .risk-medium {
        background-color: #fff8e1;
        color: #f57c00;
        border: 1px solid #ffe082;
    }
    .risk-low {
        background-color: #e8f5e9;
        color: #2e7d32;
        border: 1px solid #a5d6a7;
    }
    div[data-testid="stPlotlyChart"] {
        background-color: #f5f5f5 !important;
        padding: 1rem !important;
        border-radius: 8px !important;
        margin-bottom: 1.5rem !important;
    }
    .stExpander {
        border: 1px solid #e0e0e0;
        border-radius: 8px;
    }
</style>
""", unsafe_allow_html=True)


def render_metric_card(label, value, help_text=""):
    """Render a styled metric card"""
    return f"""
    <div class="metric-container" title="{help_text}">
        <div class="metric-label">{label}</div>
        <div class="metric-value">{value}</div>
    </div>
    """


def render_hero_section(metrics):
    """Render the hero/landing section with impact metrics"""
    st.markdown('<div class="main-header">Proactive Retention Agent</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-header">Stop Customer Churn Before It Happens</div>', unsafe_allow_html=True)
    
    # Impact metrics in columns
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.markdown(
            render_metric_card(
                "Total Revenue at Risk",
                format_currency(metrics['total_at_risk_cltv']),
                "Sum of CLTV for all at-risk customers"
            ),
            unsafe_allow_html=True
        )
    
    with col2:
        st.markdown(
            render_metric_card(
                "High Risk Customers",
                f"{metrics['high_risk_count']}",
                "Customers with >75% churn probability"
            ),
            unsafe_allow_html=True
        )
    
    with col3:
        st.markdown(
            render_metric_card(
                "Total Customers Analyzed",
                f"{metrics['total_customers']}",
                "Total customers in the priority list"
            ),
            unsafe_allow_html=True
        )
    
    with col4:
        st.markdown(
            render_metric_card(
                "Avg Churn Probability",
                f"{metrics['avg_churn_prob']:.1%}",
                "Average churn probability across all customers"
            ),
            unsafe_allow_html=True
        )
    
    st.markdown("---")


def render_pipeline_overview():
    """Render the quick pipeline overview"""
    st.header("How It Works")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.markdown("""
        <div class="pipeline-overview-card">
            <h3>Step 1: Data Sources</h3>
            <p>Customer profiles + Live reviews</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
        <div class="pipeline-overview-card">
            <h3>Step 2: ML Prediction</h3>
            <p>XGBoost model predicts churn risk</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        st.markdown("""
        <div class="pipeline-overview-card">
            <h3>Step 3: LLM Analysis</h3>
            <p>AI classifies complaint theme & sentiment</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col4:
        st.markdown("""
        <div class="pipeline-overview-card">
            <h3>Step 4: Priority Score</h3>
            <p>Risk × Value = Action priority</p>
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown("---")


def render_step_container_header(step_num, title):
    """Render the header of a step container"""
    return f"""
    <div class="step-container">
        <div style="display: flex; align-items: center; margin-bottom: 1rem;">
            <span class="step-number">{step_num}</span>
            <span class="step-title">{title}</span>
        </div>
    """


def render_detailed_walkthrough(df):
    """Render the detailed 'How it Works' walkthrough"""
    with st.expander("**Learn More: Detailed Pipeline Walkthrough**", expanded=False):
        st.markdown("### Step-by-Step Process")
        
        # Step 1: Data Sources
        st.markdown(render_step_container_header("1", "Data Sources"), unsafe_allow_html=True)
        st.markdown("""
        **What happens:** The system loads two data sources:
        - **Customer Data Warehouse:** Contains customer profiles with 19 features (tenure, charges, contract type, etc.)
        - **Live Customer Reviews:** Real-time feed of customer complaints and feedback
        
        **Why it matters:** We need both structured data (for ML predictions) and unstructured text (for understanding WHY customers are unhappy).
        """)
        st.markdown("</div>", unsafe_allow_html=True)
        
        col1, col2 = st.columns(2)
        with col1:
            st.info("**Example Customer Data:**\n- Tenure: 12 months\n- Monthly Charges: $89.50\n- Contract: Month-to-month\n- Internet: Fiber optic")
        with col2:
            st.info("**Example Review:**\n'The download speeds I was getting are too slow. I found another provider who offers much faster internet.'")
        
        st.markdown("<br>", unsafe_allow_html=True)
        
        # Step 2: ML Prediction
        example_customer = df.iloc[0]
        st.markdown(render_step_container_header("2", "Machine Learning Prediction"), unsafe_allow_html=True)
        st.markdown("""
        **What happens:** An XGBoost model analyzes 19 customer features to predict churn probability.
        
        **How it works:** The model was trained on historical data from thousands of customers. It learned patterns like:
        - Month-to-month contracts = higher risk
        - Short tenure + high charges = higher risk
        - Customers without tech support = higher risk
        
        **Output:** Churn probability (0-100%) and risk level (High/Medium/Low)
        """)
        st.markdown("</div>", unsafe_allow_html=True)
        
        col1, col2 = st.columns(2)
        with col1:
            st.metric("Example: Churn Probability", f"{example_customer['Churn_Probability']:.1%}")
        with col2:
            st.metric("Example: Risk Level", example_customer['ML_Risk_Level'])
        
        st.info("**Business Decision:** The model was tuned for high Recall (69%) to catch as many at-risk customers as possible, even if it means some false positives.")
        
        st.markdown("<br>", unsafe_allow_html=True)
        
        # Step 3: LLM Analysis
        st.markdown(render_step_container_header("3", "LLM Analysis (AI Text Classification)"), unsafe_allow_html=True)
        st.markdown("""
        **What happens:** Google Gemini AI reads each customer review and classifies:
        - **Theme:** What is the complaint about? (Price, Product/Service, Customer Support, Competitor, Other)
        - **Sentiment:** How negative is the review? (Positive, Negative, Neutral)
        
        **Why it matters:** This tells retention teams WHAT to offer when they call. A price complaint needs a discount, while a service complaint needs a technical solution.
        """)
        st.markdown("</div>", unsafe_allow_html=True)
        
        col1, col2 = st.columns(2)
        with col1:
            st.metric("Example: Theme", example_customer['LLM_Theme'])
        with col2:
            st.metric("Example: Sentiment", example_customer['LLM_Sentiment'])
        
        st.text_area("Example Review:", example_customer['Review_Text'], height=100, disabled=True)
        
        st.markdown("<br>", unsafe_allow_html=True)
        
        # Step 4: Priority Calculation
        st.markdown(render_step_container_header("4", "Priority Score Calculation"), unsafe_allow_html=True)
        st.markdown("""
        **Formula:** Priority Score = Churn Probability × Customer Lifetime Value (CLTV)
        
        **Why this formula:** Not all customers are equal. A high-value customer at moderate risk is more important than a low-value customer at high risk.
        
        **Example:**
        - Customer A: 85% risk × $5,869 CLTV = **Priority Score: 4,990**
        - Customer B: 90% risk × $2,000 CLTV = **Priority Score: 1,800**
        
        Customer A gets called first, even though Customer B has higher risk, because the potential revenue loss is greater.
        """)
        st.markdown("</div>", unsafe_allow_html=True)
        
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Churn Probability", f"{example_customer['Churn_Probability']:.1%}")
        with col2:
            st.metric("CLTV", format_currency(example_customer['CLTV']))
        with col3:
            st.metric("Priority Score", f"{example_customer['Priority_Score']:,.0f}")
        
        st.success(f"**Calculation:** {example_customer['Churn_Probability']:.1%} × {format_currency(example_customer['CLTV'])} = {example_customer['Priority_Score']:,.0f}")
        
        st.markdown("<br>", unsafe_allow_html=True)
        
        # Step 5: Final Output
        st.markdown(render_step_container_header("5", "Prioritized Action List"), unsafe_allow_html=True)
        st.markdown("""
        **Output:** A sorted list of customers ranked by Priority Score, ready for retention teams to call.
        
        **Business Impact:** Analysts know exactly who to call first, saving time and maximizing revenue retention.
        """)
        st.markdown("</div>", unsafe_allow_html=True)


def render_interactive_demo(df):
    """Render the 'Try it Yourself' interactive demo"""
    st.header("Interactive Demo: Follow a Customer's Journey")
    
    # Customer selector
    customer_ids = df['CustomerID'].tolist()
    selected_customer = st.selectbox(
        "Select a customer to see their journey through the pipeline:",
        customer_ids,
        index=0
    )
    
    if selected_customer:
        customer = get_customer_details(df, selected_customer)
        
        if customer:
            st.markdown("### Customer Journey")
            
            # Step 1: Customer Data
            step1_html = f"""
            <div class="customer-journey-step">
                <h4 style="margin-top: 0; color: #1a1a1a; font-size: 1.2rem; font-weight: 600;">Step 1: Customer Profile</h4>
            </div>
            """
            st.markdown(step1_html, unsafe_allow_html=True)
            
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Customer ID", customer['CustomerID'])
            with col2:
                st.metric("CLTV", format_currency(customer['CLTV']))
            with col3:
                risk_class = customer['ML_Risk_Level'].lower()
                risk_html = f'<span class="risk-indicator risk-{risk_class}">{customer["ML_Risk_Level"]}</span>'
                st.markdown(f"**Risk Level:** {risk_html}", unsafe_allow_html=True)
            
            # Step 2: ML Prediction
            step2_html = f"""
            <div class="customer-journey-step">
                <h4 style="margin-top: 0; color: #1a1a1a; font-size: 1.2rem; font-weight: 600;">Step 2: ML Prediction</h4>
            </div>
            """
            st.markdown(step2_html, unsafe_allow_html=True)
            
            col1, col2 = st.columns(2)
            with col1:
                st.metric("Churn Probability", f"{customer['Churn_Probability']:.1%}")
                st.progress(customer['Churn_Probability'])
            with col2:
                risk_class = customer['ML_Risk_Level'].lower()
                risk_html = f'<span class="risk-indicator risk-{risk_class}">{customer["ML_Risk_Level"]} Risk</span>'
                st.markdown(f"**Risk Level:** {risk_html}", unsafe_allow_html=True)
            
            # Step 3: LLM Analysis
            step3_html = f"""
            <div class="customer-journey-step">
                <h4 style="margin-top: 0; color: #1a1a1a; font-size: 1.2rem; font-weight: 600;">Step 3: LLM Analysis</h4>
            </div>
            """
            st.markdown(step3_html, unsafe_allow_html=True)
            
            st.text_area("Customer Review:", customer['Review_Text'], height=100, disabled=True)
            col1, col2 = st.columns(2)
            with col1:
                st.metric("Complaint Theme", customer['LLM_Theme'])
            with col2:
                st.metric("Sentiment", customer['LLM_Sentiment'])
            
            # Step 4: Priority Calculation
            step4_html = f"""
            <div class="customer-journey-step">
                <h4 style="margin-top: 0; color: #1a1a1a; font-size: 1.2rem; font-weight: 600;">Step 4: Priority Score Calculation</h4>
            </div>
            """
            st.markdown(step4_html, unsafe_allow_html=True)
            
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Churn Probability", f"{customer['Churn_Probability']:.1%}")
            with col2:
                st.markdown("<div style='text-align: center; padding-top: 1.5rem; font-size: 1.5rem; color: #666;'>×</div>", unsafe_allow_html=True)
            with col3:
                st.metric("CLTV", format_currency(customer['CLTV']))
            
            st.success(f"**Priority Score = {customer['Priority_Score']:,.0f}**")
            
            # Step 5: Ranking
            step5_html = f"""
            <div class="customer-journey-step">
                <h4 style="margin-top: 0; color: #1a1a1a; font-size: 1.2rem; font-weight: 600;">Step 5: Priority Ranking</h4>
            </div>
            """
            st.markdown(step5_html, unsafe_allow_html=True)
            
            rank = df[df['Priority_Score'] >= customer['Priority_Score']].shape[0]
            total = len(df)
            percentile = (1 - rank / total) * 100
            
            col1, col2 = st.columns(2)
            with col1:
                st.metric("Rank in Priority List", f"#{rank} of {total}")
            with col2:
                st.metric("Top Percentile", f"{percentile:.1f}%")
            
            st.info(f"This customer should be called **#{rank}** in priority order. They're in the top {percentile:.1f}% of at-risk customers.")


def render_analytics_dashboard(df):
    """Render the full analytics dashboard"""
    st.header("Analytics Dashboard")
    
    # Filters
    st.subheader("Filters")
    col1, col2, col3 = st.columns(3)
    
    with col1:
        risk_filter = st.multiselect(
            "Risk Level",
            options=['High', 'Medium', 'Low'],
            default=['High', 'Medium', 'Low']
        )
    
    with col2:
        theme_filter = st.multiselect(
            "Complaint Theme",
            options=df['LLM_Theme'].unique().tolist(),
            default=df['LLM_Theme'].unique().tolist()
        )
    
    with col3:
        sentiment_filter = st.multiselect(
            "Sentiment",
            options=df['LLM_Sentiment'].unique().tolist(),
            default=df['LLM_Sentiment'].unique().tolist()
        )
    
    # Apply filters
    filtered_df = df[
        (df['ML_Risk_Level'].isin(risk_filter)) &
        (df['LLM_Theme'].isin(theme_filter)) &
        (df['LLM_Sentiment'].isin(sentiment_filter))
    ]
    
    st.info(f"Showing {len(filtered_df)} of {len(df)} customers")
    
    # Top customers table
    st.subheader("Top Priority Customers")
    top_customers = create_top_customers_table(filtered_df, n=20)
    st.dataframe(top_customers, use_container_width=True, hide_index=True)
    
    # Charts - Row 1
    col1, col2 = st.columns(2)
    with col1:
        st.plotly_chart(create_risk_distribution_chart(filtered_df), use_container_width=True)
    with col2:
        st.plotly_chart(create_theme_breakdown_chart(filtered_df), use_container_width=True)
    
    # Charts - Row 2
    col1, col2 = st.columns(2)
    with col1:
        st.plotly_chart(create_sentiment_chart(filtered_df), use_container_width=True)
    with col2:
        st.plotly_chart(create_priority_score_distribution(filtered_df), use_container_width=True)
    
    # Charts - Row 3
    st.plotly_chart(create_cltv_vs_churn_scatter(filtered_df), use_container_width=True)
    
    # Charts - Row 4
    col1, col2 = st.columns(2)
    with col1:
        st.plotly_chart(create_theme_sentiment_heatmap(filtered_df), use_container_width=True)
    with col2:
        st.plotly_chart(create_avg_churn_by_theme(filtered_df), use_container_width=True)


def render_live_pipeline():
    """Render the live pipeline execution page"""
    st.header("Live Pipeline Execution")
    
    # Initialize session state for caching results
    if 'live_pipeline_results' not in st.session_state:
        st.session_state.live_pipeline_results = None
    if 'live_pipeline_processed_count' not in st.session_state:
        st.session_state.live_pipeline_processed_count = 0
    
    # Check environment variables
    ml_api_url = os.environ.get("ML_API_URL")
    google_api_key = os.environ.get("GOOGLE_API_KEY")
    
    col1, col2 = st.columns(2)
    with col1:
        if ml_api_url:
            st.success(f"ML API URL: {ml_api_url}")
        else:
            st.error("ML_API_URL not set. Add it to Streamlit Cloud secrets.")
    
    with col2:
        if google_api_key:
            st.success("Google API Key: Configured")
        else:
            st.error("GOOGLE_API_KEY not set. Add it to Streamlit Cloud secrets.")
    
    if not ml_api_url or not google_api_key:
        st.warning("""
        **Setup Required:**
        
        To run the live pipeline, you need to set these environment variables in Streamlit Cloud:
        1. Go to your app settings → Secrets
        2. Add `ML_API_URL` with your Render API URL (e.g., `https://churn-api-xxxx.onrender.com`)
        3. Add `GOOGLE_API_KEY` with your Google Gemini API key
        
        See `DEPLOYMENT.md` for detailed instructions.
        """)
        return
    
    st.markdown("---")
    
    # Show cached results indicator
    if st.session_state.live_pipeline_results is not None:
        st.info(f"Showing cached results from previous run ({st.session_state.live_pipeline_processed_count} customers processed). Click 'Run Live Pipeline' to process new data.")
        clear_cache = st.button("Clear Cached Results", type="secondary")
        if clear_cache:
            st.session_state.live_pipeline_results = None
            st.session_state.live_pipeline_processed_count = 0
            st.rerun()
        st.markdown("---")
    
    # Pipeline controls
    st.subheader("Run Pipeline")
    
    col1, col2 = st.columns([2, 1])
    with col1:
        limit = st.number_input(
            "Number of reviews to process (leave empty for all)",
            min_value=1,
            max_value=100,
            value=10,
            help="Limit the number of reviews for faster testing. Full pipeline processes all reviews."
        )
    
    with col2:
        st.markdown("<br>", unsafe_allow_html=True)
        run_button = st.button("Run Live Pipeline", type="primary", use_container_width=True)
    
    # Get results (either from cache or new run)
    df_results = st.session_state.live_pipeline_results
    
    if run_button:
        with st.container():
            st.markdown("### Processing Pipeline")
            
            # Run pipeline
            df_results = run_live_pipeline(limit=limit if limit else None)
            
            # Cache results
            if df_results is not None and len(df_results) > 0:
                st.session_state.live_pipeline_results = df_results
                st.session_state.live_pipeline_processed_count = len(df_results)
    
    # Display results (from cache or new run)
    if df_results is not None and len(df_results) > 0:
        st.success(f"Pipeline completed! Processed {len(df_results)} customers.")
        
        # Show metrics
        st.subheader("Results Summary")
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Total Processed", len(df_results))
        with col2:
            high_risk = len(df_results[df_results['ML_Risk_Level'] == 'High'])
            st.metric("High Risk", high_risk)
        with col3:
            total_cltv = df_results['CLTV'].sum()
            st.metric("Total CLTV at Risk", format_currency(total_cltv))
        with col4:
            avg_prob = df_results['Churn_Probability'].mean()
            st.metric("Avg Churn Prob", f"{avg_prob:.1%}")
        
        # Show top customers
        st.subheader("Prioritized Customer List")
        st.markdown("""
        The pipeline has processed all reviews and created a prioritized list. Customers are sorted by Priority Score 
        (Churn Probability × CLTV), ensuring the most valuable at-risk customers are addressed first.
        """)
        top_customers = create_top_customers_table(df_results, n=20)
        st.dataframe(top_customers, use_container_width=True, hide_index=True)
        
        st.info("""
        **Note:** For detailed analytics and visualizations of these results, visit the **Analytics Dashboard** page in the sidebar.
        The Live Pipeline page focuses on demonstrating the technical process (ML + LLM integration) rather than business analysis.
        """)
        
        # Download results
        st.markdown("---")
        csv = df_results.to_csv(index=False)
        st.download_button(
            label="Download Results as CSV",
            data=csv,
            file_name=f"live_pipeline_results_{pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')}.csv",
            mime="text/csv"
        )
    elif df_results is None and run_button:
        st.error("Pipeline failed. Check the error messages above.")
    
    # What-If Scenario Tester Section (at bottom)
    st.markdown("---")
    st.subheader("What-If Scenario Tester")
    st.markdown("""
    Test how the ML model and LLM respond to different inputs. This demonstrates the models' behavior and sensitivity.
    """)
    
    what_if_tab1, what_if_tab2 = st.tabs(["ML Feature Tester", "LLM Review Text Tester"])
    
    with what_if_tab1:
        st.markdown("**Test how changing key customer features affects the ML prediction:**")
        st.caption("These 5 features are among the most important predictors of churn in the model.")
        
        col1, col2 = st.columns(2)
        with col1:
            test_contract = st.selectbox(
                "Contract Type", 
                ["Month-to-month", "One year", "Two year"],
                key="whatif_contract",
                help="Longer contracts typically indicate lower churn risk"
            )
            test_tenure = st.number_input(
                "Tenure Months", 
                min_value=0, 
                max_value=100, 
                value=12,
                key="whatif_tenure",
                help="How long the customer has been with the company"
            )
            test_monthly = st.number_input(
                "Monthly Charges ($)", 
                min_value=0.0, 
                max_value=200.0, 
                value=70.0,
                key="whatif_monthly",
                help="Monthly service charges"
            )
        
        with col2:
            test_total = st.number_input(
                "Total Charges ($)", 
                min_value=0.0, 
                max_value=10000.0, 
                value=840.0,
                key="whatif_total",
                help="Total amount charged over customer lifetime"
            )
            test_payment = st.selectbox(
                "Payment Method", 
                ["Electronic check", "Mailed check", "Bank transfer (automatic)", "Credit card (automatic)"],
                key="whatif_payment",
                help="Payment method - electronic check often correlates with higher churn"
            )
        
        if st.button("Test ML Prediction", type="primary", key="test_ml_button"):
            if ml_api_url:
                # Use default values for other features (typical customer profile)
                test_features = {
                    "Tenure Months": test_tenure,
                    "Monthly Charges": test_monthly,
                    "Total Charges": test_total,
                    "Gender": "Male",  # Default
                    "Senior Citizen": "No",  # Default
                    "Partner": "Yes",  # Default
                    "Dependents": "No",  # Default
                    "Phone Service": "Yes",  # Default
                    "Multiple Lines": "No",  # Default
                    "Internet Service": "Fiber optic",  # Default
                    "Online Security": "No",  # Default
                    "Online Backup": "No",  # Default
                    "Device Protection": "No",  # Default
                    "Tech Support": "No",  # Default
                    "Streaming TV": "Yes",  # Default
                    "Streaming Movies": "Yes",  # Default
                    "Contract": test_contract,
                    "Paperless Billing": "Yes",  # Default
                    "Payment Method": test_payment
                }
                
                with st.spinner("Calling ML API..."):
                    from live_pipeline import get_ml_prediction
                    result = get_ml_prediction("TEST_CUSTOMER", test_features, ml_api_url)
                    
                    if result:
                        st.success("ML Prediction Generated!")
                        col1, col2, col3 = st.columns(3)
                        with col1:
                            st.metric("Churn Probability", f"{result['churn_probability']:.1%}")
                        with col2:
                            st.metric("Risk Level", result['risk_level'])
                        with col3:
                            # Calculate what CLTV would need to be for high priority
                            if result['churn_probability'] > 0:
                                high_priority_cltv = 10000 / result['churn_probability']
                                st.metric("CLTV for $10K Priority", f"${high_priority_cltv:,.0f}")
                        
                        st.info("""
                        **Interpretation:** 
                        - **Churn Probability** is the raw model output (0-100%)
                        - **Risk Level** is categorized: High (≥75%), Medium (≥40%), Low (<40%)
                        - Try adjusting features like Contract, Tenure, or Monthly Charges to see how predictions change
                        - Note: Other features are set to default values for this test
                        """)
                    else:
                        st.error("Failed to get prediction. Check ML API connection.")
            else:
                st.error("ML API URL not configured. Cannot test.")
    
    with what_if_tab2:
        st.markdown("**Test how the LLM classifies different review texts:**")
        
        test_review = st.text_area(
            "Enter a customer review to classify:",
            value="I'm really frustrated with the slow internet speeds. I've been a customer for 2 years and the service keeps getting worse. I'm considering switching to a competitor.",
            height=100,
            key="whatif_review"
        )
        
        if st.button("Test LLM Classification", type="primary", key="test_llm_button"):
            if google_api_key:
                from live_pipeline import setup_gemini_client, get_llm_analysis
                primary_model, fallback_model = setup_gemini_client()
                
                if primary_model:
                    with st.spinner("Analyzing review with LLM..."):
                        result = get_llm_analysis(primary_model, fallback_model, test_review)
                        
                        if result:
                            st.success("LLM Classification Complete!")
                            
                            col1, col2 = st.columns(2)
                            with col1:
                                theme_color = {
                                    'Price': '#ff9800',
                                    'Product/Service': '#2196f3',
                                    'Customer Support': '#f44336',
                                    'Competitor': '#9c27b0',
                                    'Other': '#607d8b'
                                }.get(result.get('theme', 'N/A'), '#666')
                                
                                sentiment_color = {
                                    'Negative': '#f44336',
                                    'Positive': '#4caf50',
                                    'Neutral': '#ff9800'
                                }.get(result.get('sentiment', 'N/A'), '#666')
                                
                                st.markdown("**Classification Results:**")
                                st.markdown(f"**Theme:** <span style='color: {theme_color}; font-weight: 600; font-size: 1.2em;'>{result.get('theme', 'N/A')}</span>", unsafe_allow_html=True)
                                st.markdown(f"**Sentiment:** <span style='color: {sentiment_color}; font-weight: 600; font-size: 1.2em;'>{result.get('sentiment', 'N/A')}</span>", unsafe_allow_html=True)
                            
                            with col2:
                                st.markdown("**Technical Details:**")
                                if result.get('model_used'):
                                    st.caption(f"Model Used: {result.get('model_used')}")
                                with st.expander("View Prompt"):
                                    st.code(result.get('prompt', ''), language='text')
                                with st.expander("View Raw Response"):
                                    st.code(result.get('raw_response', ''), language='json')
                            
                            st.info("""
                            **Tips for Testing:**
                            - Try reviews with different themes (price complaints, service issues, competitor mentions)
                            - Test positive, negative, and neutral sentiments
                            - Try ambiguous reviews to see how the LLM handles uncertainty
                            """)
                        else:
                            st.error("Failed to get LLM classification.")
                else:
                    st.error("Failed to setup Gemini client.")
            else:
                st.error("Google API Key not configured. Cannot test.")


def render_technical_details():
    """Render technical details section"""
    with st.expander("**Technical Details**", expanded=False):
        st.markdown("### Architecture")
        
        st.markdown("""
        **Tech Stack:**
        - **ML Model:** XGBoost (scikit-learn pipeline)
        - **ML Serving:** FastAPI (Docker containerized)
        - **LLM:** Google Gemini 2.0 Flash Lite API
        - **Pipeline:** Python with pandas, requests
        - **Dashboard:** Streamlit
        
        **Model Performance:**
        - **Recall:** 69% (catches 69% of all churners)
        - **Precision:** 54% (54% of predicted churners actually churn)
        - **Strategy:** Tuned for high recall to minimize false negatives
        
        **Pipeline Flow:**
        1. Load customer data warehouse (CSV)
        2. Load live customer reviews (CSV)
        3. For each review:
           - Lookup customer in warehouse
           - Extract 19 features
           - Call ML API (deployed on Render) for prediction
           - Call Gemini API for text classification
           - Calculate Priority Score
        4. Sort by Priority Score
        5. Output prioritized list
        """)
        
        st.markdown("### Key Features")
        st.markdown("""
        - **MLOps:** Model containerized with Docker, served via REST API
        - **LLMOps:** Automated LLM integration for text classification
        - **Business Logic:** Risk-adjusted prioritization (CLTV × Churn Probability)
        - **Analytics:** Comprehensive BI dashboard with multiple visualizations
        """)


def main():
    """Main application"""
    try:
        # Load data
        df = load_priority_data()
        metrics = calculate_impact_metrics(df)
        
        # Sidebar navigation
        st.sidebar.title("Navigation")
        page = st.sidebar.radio(
            "Go to:",
            ["Overview", "Analytics Dashboard", "Interactive Demo", "Live Pipeline"]
        )
        
        # Render based on selection
        if page == "Overview":
            render_hero_section(metrics)
            render_pipeline_overview()
            render_detailed_walkthrough(df)
            render_technical_details()
        
        elif page == "Analytics Dashboard":
            render_hero_section(metrics)
            render_analytics_dashboard(df)
        
        elif page == "Interactive Demo":
            render_hero_section(metrics)
            render_interactive_demo(df)
        
        elif page == "Live Pipeline":
            render_hero_section(metrics)
            render_live_pipeline()
        
        # Footer
        st.markdown("---")
        st.markdown("""
        <div style='text-align: center; color: #666; padding: 1rem;'>
            <p>Proactive Retention Agent | Built with Streamlit</p>
            <p><small>Demo Mode - Using pre-computed results | Live Pipeline available in navigation</small></p>
        </div>
        """, unsafe_allow_html=True)
    
    except Exception as e:
        st.error(f"Error loading application: {str(e)}")
        st.exception(e)


if __name__ == "__main__":
    main()
