# Proactive Retention Agent - Interactive Dashboard

An interactive web-based dashboard showcasing an end-to-end customer retention pipeline. Demonstrates data science, data engineering, and analytical skills through an accessible interface for both technical and non-technical users.

## Quick Start

The dashboard is deployed and accessible via Streamlit Cloud. No installation required - everything runs in your browser.

## Features

**Demo Mode (Default)**
- Pre-computed results for instant exploration
- No API keys or Docker required
- Works immediately for anyone

**Live Pipeline Mode**
- Real-time ML and LLM API integration
- Live processing of customer reviews
- Technical transparency (features, prompts, raw responses)
- Interactive model testing

## Pages

### Overview
Landing page with impact metrics, visual pipeline overview, step-by-step walkthrough, and technical architecture details.

### Analytics Dashboard
Comprehensive BI dashboard featuring:
- Interactive filters (risk level, theme, sentiment)
- Top priority customers table
- 8 interactive visualizations: risk distribution, theme breakdown, sentiment analysis, priority score distribution, CLTV vs churn scatter, theme-sentiment heatmap, average churn by theme

### Interactive Demo
Follow individual customers through the pipeline. Select any customer to see their complete profile, ML prediction, LLM classification, priority score calculation, and rank in the priority list.

### Live Pipeline
Real-time technical demonstration with:
- **Real-Time LLM Analysis Feed**: Watch reviews being classified as they're processed, with technical details tabs showing ML features, LLM prompts, and raw responses
- **Results Summary**: Metrics and prioritized customer list
- **What-If Scenario Tester**: 
  - Test ML predictions by adjusting 5 key features (Contract, Tenure, Monthly Charges, Total Charges, Payment Method)
  - Test LLM classification with custom review text
- **Result Caching**: Results persist when navigating away

## Data

**Demo Mode**: Uses pre-computed results from `data/analyst_priority_list.csv`

**Live Pipeline**: Loads from `data/Telco_customer_churn.csv` (customer warehouse) and `pipeline/customer_reviews.csv` (live reviews)

## For Developers

To run locally for development:

1. Install dependencies: `pip install -r requirements.txt`
2. Run: `streamlit run app.py`
3. For Live Pipeline: Set `GOOGLE_API_KEY` and `ML_API_URL` environment variables
4. If using local ML API: Start Docker container (see main README)

## Deployment

Deployed on Streamlit Cloud. For deployment details, see `DEPLOYMENT.md` in the project root.

## Technical Stack

- Frontend: Streamlit
- Visualizations: Plotly Express
- Data Processing: Pandas
- ML API Integration: Requests
- LLM Integration: Google Generative AI (Gemini)
