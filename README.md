# Proactive Retention Agent

A full-stack data science project demonstrating MLOps, LLMOps, and automated data analysis. This system proactively identifies high-risk customers, analyzes their complaints using an LLM, and creates a prioritized list for retention teams.

## The Problem

A telecom company has thousands of customer interactions daily. By the time an analyst finds an at-risk customer, that customer has already churned. We need a system to **proactively** identify high-risk customers *in real-time* and understand *why* they are unhappy, allowing a human team to intervene *before* they leave.

## The Solution

An end-to-end automated pipeline that:

1. **Predicts Risk (MLOps):** High-recall XGBoost churn model containerized with Docker and served via FastAPI
2. **Analyzes Complaints (LLMOps):** Live feed of customer reviews classified by Google Gemini API for theme and sentiment
3. **Prioritizes Action (Business Logic):** Calculates Priority_Score (CLTV × Churn_Probability) to create risk-adjusted value, ensuring analysts call the most valuable and most at-risk customers first

## Model Performance

The ML model was tuned to prioritize high recall, strategically accepting more false positives to minimize false negatives:

| Model | Recall (Catches Churn) | Precision (Avoids False Positives) |
| :--- |:-----------------------|:-----------------------------------|
| **Baseline** | 55% | 62% |
| **Final (Tuned)** | **69%** | **54%** |

## Interactive Dashboard

The project is accessible via a deployed Streamlit dashboard with two modes:

**Demo Mode (Default)**
- Pre-computed results for instant exploration
- No setup required
- Comprehensive analytics and visualizations

**Live Pipeline Mode**
- Real-time ML and LLM API integration
- Technical transparency (features, prompts, raw responses)
- Interactive model testing

See `dashboard/README.md` for detailed feature descriptions.

## Project Structure

```
proactive-retention-agent/
├── dashboard/              # Streamlit interactive dashboard
├── ml_model_api/          # FastAPI service (Docker)
├── pipeline/              # Main processing pipeline
├── notebooks/             # Data exploration & model training
└── data/                  # Source data files
```

## For Developers

To run the pipeline locally:

1. Set up API key: Create `.env` file with `GOOGLE_API_KEY`
2. Build & Run ML API:
   ```bash
   cd ml_model_api
   docker build -t churn-api .
   docker run -d -p 8000:8000 --name churn_api_container churn-api
   ```
3. Run Pipeline:
   ```bash
   cd pipeline
   pip install -r requirements.txt
   python main.py
   ```

Output is saved to `pipeline/analyst_priority_list.csv`.

## Technical Stack

- **ML Model**: XGBoost (scikit-learn pipeline)
- **ML Serving**: FastAPI (Docker containerized)
- **LLM**: Google Gemini 2.0 Flash Lite API
- **Dashboard**: Streamlit
- **Visualizations**: Plotly Express
