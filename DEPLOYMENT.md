# Deployment Guide

This guide explains how to deploy the Proactive Retention Agent to production.

## Architecture

- **ML API**: FastAPI service (deployed on Render)
- **Dashboard**: Streamlit app (deployed on Streamlit Cloud)
- **Pipeline**: Can run locally or be integrated into the dashboard

## Prerequisites

1. GitHub account with this repository
2. Render account (free tier)
3. Streamlit Cloud account (free)
4. Google Gemini API key

## Step 1: Deploy ML API to Render

### Option A: Using Render Blueprint (Recommended)

1. Go to [render.com](https://render.com) and sign up/login
2. Click "New +" → "Blueprint"
3. Connect your GitHub repository
4. Render will detect `render.yaml` and create the service automatically
5. Add environment variable in Render dashboard:
   - Key: `GOOGLE_API_KEY`
   - Value: Your Google Gemini API key
6. Click "Apply" to deploy

### Option B: Manual Setup

1. Go to [render.com](https://render.com) and sign up/login
2. Click "New +" → "Web Service"
3. Connect your GitHub repository
4. Configure:
   - **Name**: `churn-ml-api`
   - **Environment**: `Docker`
   - **Root Directory**: `ml_model_api`
   - **Dockerfile Path**: `ml_model_api/Dockerfile`
   - **Docker Context**: `ml_model_api`
5. Add environment variable:
   - Key: `GOOGLE_API_KEY`
   - Value: Your Google Gemini API key
6. Click "Create Web Service"
7. Wait for deployment (first deploy takes ~5-10 minutes)
8. Copy the service URL (e.g., `https://churn-ml-api-xxxx.onrender.com`)

### Verify ML API Deployment

Once deployed, test the API:
```bash
curl https://your-api-url.onrender.com/
```

Should return: `{"status":"ok","message":"Churn Model API is running."}`

## Step 2: Deploy Dashboard to Streamlit Cloud

1. Go to [share.streamlit.io](https://share.streamlit.io) and sign up/login with GitHub
2. Click "New app"
3. Select your repository: `proactive-retention-agent-build`
4. Configure:
   - **Branch**: `main`
   - **Main file path**: `dashboard/app.py`
   - **App URL**: (auto-generated, e.g., `your-app-name.streamlit.app`)
5. Add secrets (click "Advanced settings" → "Secrets"):
   ```toml
   ML_API_URL = "https://your-render-api-url.onrender.com"
   GOOGLE_API_KEY = "your_google_api_key"
   ```
6. Click "Deploy"
7. Wait for deployment (~2-3 minutes)

### Streamlit Secrets

In Streamlit Cloud, secrets are managed via the dashboard UI. The secrets will be available as environment variables in your app.

## Step 3: Update Environment Variables

### For Local Development

Create a `.env` file in the project root:
```env
GOOGLE_API_KEY=your_google_api_key_here
ML_API_URL=http://localhost:8000
```

### For Production (Render)

Add in Render dashboard:
- `GOOGLE_API_KEY`: Your Google Gemini API key

### For Production (Streamlit Cloud)

Add in Streamlit Cloud secrets:
- `ML_API_URL`: Your Render API URL (e.g., `https://churn-ml-api-xxxx.onrender.com`)
- `GOOGLE_API_KEY`: Your Google Gemini API key

## Testing the Deployment

1. **Test ML API**: Visit `https://your-api-url.onrender.com/`
2. **Test Dashboard**: Visit your Streamlit Cloud URL
3. **Test Live Pipeline**: Use the "Live Pipeline" mode in the dashboard (if implemented)

## Troubleshooting

### ML API Issues

- **Service spins down**: Render free tier spins down after 15 min inactivity. First request takes ~30 seconds.
- **Model not found**: Ensure `churn_model_v2.pkl` is committed to the repo
- **Port issues**: Render auto-assigns port, but Dockerfile should expose 8000

### Dashboard Issues

- **Can't connect to ML API**: Check `ML_API_URL` secret in Streamlit Cloud
- **API key errors**: Verify `GOOGLE_API_KEY` is set in Streamlit secrets
- **Import errors**: Ensure all dependencies are in `dashboard/requirements.txt`

### Common Fixes

1. **Check logs**: Both Render and Streamlit Cloud provide logs
2. **Verify environment variables**: Double-check all secrets are set correctly
3. **Check API URL format**: Should be `https://...` (not `http://`) for production
4. **Wait for cold start**: Render free tier takes ~30 seconds on first request

## Cost

- **Render**: Free tier (spins down after inactivity)
- **Streamlit Cloud**: Free (always on)
- **Total**: $0/month

## Next Steps

- Add "Live Pipeline" mode to dashboard for real-time processing
- Set up monitoring/alerting (optional)
- Consider upgrading to paid tiers for always-on ML API (if needed)

