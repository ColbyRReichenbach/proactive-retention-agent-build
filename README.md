# Proactive Retention Agent

A full-stack data science project that demonstrates MLOps, LLMOps, and automated data analysis. This pipeline automatically identifies high-risk customers, analyzes their complaints using an LLM, and creates a prioritized list for retention teams.

## The Problem
A telecom company has thousands of customer interactions daily. By the time an analyst finds a customer who is "at-risk," that customer has already churned. We need a system to **proactively** identify high-risk customers *in real-time* and understand *why* they are unhappy, allowing a human team to intervene *before* they leave.

## The Solution
This project is an end-to-end automated pipeline that:
1.  **Predicts Risk (MLOps):** A high-recall XGBoost churn model is containerized with Docker and served via a FastAPI.
2.  **Analyzes Complaints (LLMOps):** A "live" feed of customer reviews is fed to the Google Gemini API to classify the *theme* and *sentiment* of each complaint.
3.  **Prioritizes Action (Business Logic):** The system calculates a **Priority_Score (CLTV * Churn_Probability)** to create a "Risk-Adjusted Value" for every at-risk customer, ensuring analysts call the most valuable and most at-risk customers first.

## Model Iteration: Aligning with Business Goals
A key part of this project was tuning the ML model. A baseline model had poor Recall (55%), meaning it missed half of all churners.

| Model | Recall (Catches Churn) | Precision (Avoids False Positives) |
| :--- |:-----------------------|:-----------------------------------|
| **Baseline** | 55%                    | 62%                                |
| **Final (Tuned)** | **69%**                | **54%**                            |

The final model was tuned using `scale_pos_weight` to prioritize **high Recall**. We strategically accepted more False Positives (a small cost) to minimize False Negatives (a high cost).

## How to Run This Project

### Option 1: Interactive Dashboard (Recommended for Demo)

The easiest way to explore this project is through the interactive Streamlit dashboard:

```bash
cd dashboard
pip install -r requirements.txt
streamlit run app.py
```

The dashboard includes:
- **Demo Mode**: Pre-computed results (no Docker or API keys needed)
- **Analytics Dashboard**: Comprehensive BI visualizations
- **Interactive Demo**: Follow individual customers through the pipeline
- **Step-by-Step Walkthrough**: Learn how the system works

See `dashboard/README.md` for more details.

### Option 2: Full Pipeline (Local Development)

1.  **Clone the repo:** `git clone ...`
2.  **Set up API key:** Create a `.env` file in the root with your `GOOGLE_API_KEY`.
3.  **Build & Run the ML API:**
    ```bash
    cd ml_model_api
    docker build -t churn-api .
    docker run -d -p 8000:8000 --name churn_api_container churn-api
    ```
4.  **Run the Pipeline:**
    ```bash
    cd pipeline
    pip install -r requirements.txt
    python main.py
    ```
5.  **View the results:** The output is saved to `pipeline/analyst_priority_list.csv`.

### Option 3: Deploy to Production

Deploy the full system to the cloud for a live technical demo:

- **ML API**: Deploy to Render (free tier)
- **Dashboard**: Deploy to Streamlit Cloud (free)

See `DEPLOYMENT.md` for detailed deployment instructions.

## Project Structure

```
proactive-retention-agent/
├── dashboard/              # Streamlit interactive dashboard
├── ml_model_api/          # FastAPI service (Docker)
├── pipeline/              # Main processing pipeline
├── notebooks/             # Data exploration & model training
├── data/                 # Source data files
└── DEPLOYMENT.md         # Deployment guide
```