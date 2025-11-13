# Deployment Checklist

Use this checklist to ensure everything is set up correctly for deployment.

## Pre-Deployment

- [ ] All code changes committed to GitHub
- [ ] Model file (`churn_model_v2.pkl`) is in the repository
- [ ] `.env` file is NOT committed (check with `git status`)
- [ ] `.env.example` is committed (template for others)
- [ ] `.gitignore` properly excludes sensitive files
- [ ] Tested locally with Docker

## Render (ML API) Setup

- [ ] Created Render account
- [ ] Connected GitHub repository
- [ ] Created new Web Service
- [ ] Set Root Directory to `ml_model_api`
- [ ] Set Environment to `Docker`
- [ ] Added environment variable: `GOOGLE_API_KEY`
- [ ] Service deployed successfully
- [ ] Tested API endpoint: `https://your-api.onrender.com/`
- [ ] Copied API URL for Streamlit Cloud

## Streamlit Cloud (Dashboard) Setup

- [ ] Created Streamlit Cloud account
- [ ] Connected GitHub repository
- [ ] Set Main file path to `dashboard/app.py`
- [ ] Added secret: `ML_API_URL` (your Render API URL)
- [ ] Added secret: `GOOGLE_API_KEY` (your Google API key)
- [ ] Dashboard deployed successfully
- [ ] Tested dashboard loads correctly
- [ ] Verified demo mode works (pre-computed data)

## Post-Deployment Testing

- [ ] ML API responds at root endpoint (`/`)
- [ ] ML API prediction endpoint works (`/predict`)
- [ ] Dashboard loads without errors
- [ ] Demo mode shows pre-computed data
- [ ] All visualizations render correctly
- [ ] No console errors in browser

## Troubleshooting Reference

### ML API Issues
- Check Render logs for errors
- Verify model file is in repository
- Confirm `GOOGLE_API_KEY` is set (if needed by API)

### Dashboard Issues
- Check Streamlit Cloud logs
- Verify secrets are set correctly
- Test with `ML_API_URL` pointing to Render service

## URLs to Save

- Render ML API: `https://________________.onrender.com`
- Streamlit Dashboard: `https://________________.streamlit.app`

