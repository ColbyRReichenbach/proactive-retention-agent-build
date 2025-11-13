#!/bin/bash
# Quick start script for the dashboard

echo "🎯 Starting Proactive Retention Agent Dashboard..."
echo ""

# Check if streamlit is installed
if ! python -c "import streamlit" 2>/dev/null; then
    echo "📦 Installing dependencies..."
    pip install -r requirements.txt
    echo ""
fi

echo "🚀 Launching dashboard..."
echo "The dashboard will open in your browser automatically."
echo ""

streamlit run app.py


