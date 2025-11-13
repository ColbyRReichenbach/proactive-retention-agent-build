# Proactive Retention Agent - Interactive Dashboard

An interactive web-based dashboard showcasing the end-to-end customer retention pipeline. This dashboard demonstrates data science, data engineering, and analytical skills through an accessible interface.

## Features

- **Hero Section**: Impact metrics and business value at a glance
- **Pipeline Overview**: Visual explanation of the 4-step process
- **Detailed Walkthrough**: Step-by-step explanation of how the system works
- **Interactive Demo**: Follow individual customers through the pipeline
- **Analytics Dashboard**: Comprehensive BI visualizations with filters
- **Technical Details**: Architecture and implementation information

## Quick Start

### Prerequisites
- Python 3.8 or higher
- pip

### Installation

1. Navigate to the dashboard directory:
```bash
cd dashboard
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Run the dashboard:
```bash
streamlit run app.py
```

4. Open your browser to the URL shown (typically `http://localhost:8501`)

## Demo Mode

This dashboard runs in **Demo Mode** by default, which means:
- ✅ No Docker required
- ✅ No API keys needed
- ✅ Uses pre-computed results
- ✅ Works immediately for anyone

The dashboard loads data from `data/analyst_priority_list.csv`, which contains pre-computed results from the full pipeline.

## Navigation

The dashboard has 4 main sections accessible via the sidebar:

1. **🏠 Overview**: Landing page with impact metrics, pipeline overview, and detailed walkthrough
2. **📊 Analytics Dashboard**: Full BI dashboard with filters and visualizations
3. **🎮 Interactive Demo**: Follow a customer's journey through the pipeline
4. **⚙️ Technical Details**: Architecture and technical implementation details

## Data

The dashboard uses pre-computed results from the pipeline stored in:
- `data/analyst_priority_list.csv` - Final prioritized customer list

This file contains:
- Customer IDs
- ML predictions (risk level, churn probability)
- LLM analysis (theme, sentiment)
- Customer Lifetime Value (CLTV)
- Priority scores
- Review text

## Customization

To update the dashboard with new data:
1. Run the full pipeline (see main README)
2. Copy the output CSV to `dashboard/data/analyst_priority_list.csv`
3. Restart the Streamlit app

## Deployment

This dashboard can be easily deployed to:
- **Streamlit Cloud**: Connect to GitHub repo, auto-deploys
- **Heroku**: Add Procfile and deploy
- **Render**: Connect repo, set build command
- **Railway**: Connect repo, auto-detects Streamlit

## For Technical Users

If you want to run the full pipeline with live API calls:
1. Set up Docker and run the ML API (see main README)
2. Add `.env` file with `GOOGLE_API_KEY`
3. Run the pipeline to generate new results
4. Update the dashboard data file

The dashboard itself doesn't require these - it works perfectly with pre-computed data.

