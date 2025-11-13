"""
Live Pipeline Module - Runs the full pipeline with deployed APIs
"""
import pandas as pd
import requests
import google.generativeai as genai
import os
import streamlit as st
from pathlib import Path
import time
import re


def setup_gemini_client():
    """Setup Gemini client from environment variable"""
    api_key = os.environ.get("GOOGLE_API_KEY")
    if not api_key:
        return None
    
    try:
        genai.configure(api_key=api_key)
        return genai.GenerativeModel('gemini-2.0-flash-lite')
    except Exception as e:
        st.error(f"Error configuring Gemini: {e}")
        return None


def get_ml_prediction(customer_id, features, ml_api_url):
    """Call the deployed ML API"""
    url = f"{ml_api_url}/predict?customer_id={customer_id}"
    
    try:
        response = requests.post(url, json=features, timeout=30)
        response.raise_for_status()
        return response.json()
    except requests.exceptions.ConnectionError:
        st.error(f"Could not connect to ML API at {ml_api_url}")
        return None
    except requests.exceptions.Timeout:
        st.error(f"ML API request timed out for {customer_id}")
        return None
    except Exception as e:
        st.error(f"ML API call failed: {e}")
        return None


def get_llm_analysis(gemini_model, review_text):
    """Call Gemini API to classify review - returns detailed response"""
    if gemini_model is None:
        return {
            "theme": "Error", 
            "sentiment": "Error",
            "raw_response": "Model not configured",
            "prompt": "",
            "confidence": None
        }
    
    prompt = f"""
    Analyze the following customer complaint:
    "{review_text}"

    Classify it into one theme and one sentiment.
    - Valid Themes: [Competitor, Price, Product/Service, Customer Support, Other]
    - Valid Sentiments: [Positive, Negative, Neutral]

    Return your answer in a simple JSON format, like:
    {{"theme": "Price", "sentiment": "Negative"}}
    """
    
    try:
        response = gemini_model.generate_content(prompt)
        raw_response = response.text.strip()
        
        # Try to extract JSON from response
        json_output = raw_response.replace("```json", "").replace("```", "").strip()
        
        # Try to parse JSON
        try:
            parsed = pd.read_json(json_output, typ='series').to_dict()
        except:
            # If JSON parsing fails, try to extract theme and sentiment from text
            parsed = {"theme": "Parse Error", "sentiment": "Parse Error"}
        
        # Try to extract confidence from response if available
        confidence = None
        if "confidence" in raw_response.lower() or "certainty" in raw_response.lower():
            # Look for confidence indicators in the raw response
            conf_match = re.search(r'(?:confidence|certainty)[:\s]+(\d+\.?\d*)%?', raw_response.lower())
            if conf_match:
                try:
                    confidence = float(conf_match.group(1))
                    if confidence > 1:
                        confidence = confidence / 100  # Convert percentage to decimal
                except:
                    pass
        
        return {
            "theme": parsed.get('theme', 'N/A'),
            "sentiment": parsed.get('sentiment', 'N/A'),
            "raw_response": raw_response,
            "prompt": prompt,
            "confidence": confidence
        }
    except Exception as e:
        st.warning(f"LLM analysis failed: {e}")
        return {
            "theme": "LLM Error", 
            "sentiment": "Error",
            "raw_response": str(e),
            "prompt": prompt,
            "confidence": None
        }


def load_data_sources():
    """Load customer data warehouse and reviews"""
    script_dir = Path(__file__).parent
    project_root = script_dir.parent
    
    # Load telco data warehouse
    telco_path = project_root / "data" / "Telco_customer_churn.csv"
    if not telco_path.exists():
        st.error(f"Telco data not found at {telco_path}")
        return None, None
    
    df_telco = pd.read_csv(telco_path)
    df_telco['Total Charges'] = pd.to_numeric(df_telco['Total Charges'], errors='coerce')
    df_telco['Total Charges'] = df_telco['Total Charges'].fillna(0)
    df_telco = df_telco.set_index('CustomerID')
    
    # Load customer reviews
    reviews_path = project_root / "pipeline" / "customer_reviews.csv"
    if not reviews_path.exists():
        st.error(f"Customer reviews not found at {reviews_path}")
        return df_telco, None
    
    df_reviews = pd.read_csv(reviews_path)
    
    return df_telco, df_reviews


def run_live_pipeline(limit=None):
    """Run the live pipeline with progress tracking"""
    # Check environment variables
    ml_api_url = os.environ.get("ML_API_URL")
    if not ml_api_url:
        st.error("ML_API_URL environment variable not set. Please set it in Streamlit Cloud secrets.")
        return None
    
    google_api_key = os.environ.get("GOOGLE_API_KEY")
    if not google_api_key:
        st.error("GOOGLE_API_KEY environment variable not set. Please set it in Streamlit Cloud secrets.")
        return None
    
    # Setup
    st.info(f"Connecting to ML API at: {ml_api_url}")
    gemini_model = setup_gemini_client()
    if gemini_model is None:
        return None
    
    # Load data
    with st.spinner("Loading data sources..."):
        df_warehouse, df_reviews = load_data_sources()
        if df_warehouse is None or df_reviews is None:
            return None
    
    # Limit reviews if specified
    if limit:
        df_reviews = df_reviews.head(limit)
        st.info(f"Processing first {limit} reviews for demo purposes.")
    
    st.success(f"Loaded {len(df_reviews)} reviews to process")
    
    # ML features list
    ml_features_list = [
        'Tenure Months', 'Monthly Charges', 'Total Charges', 'Gender',
        'Senior Citizen', 'Partner', 'Dependents', 'Phone Service',
        'Multiple Lines', 'Internet Service', 'Online Security',
        'Online Backup', 'Device Protection', 'Tech Support', 'Streaming TV',
        'Streaming Movies', 'Contract', 'Paperless Billing', 'Payment Method'
    ]
    
    # Processing loop with progress and real-time LLM feed
    results = []
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    # Create container for real-time LLM analysis feed
    st.markdown("### Real-Time LLM Analysis Feed")
    llm_feed_container = st.container()
    
    with llm_feed_container:
        # Create a placeholder that will be updated
        feed_placeholder = st.empty()
        llm_analyses = []  # Store analyses to display
    
    for idx, (_, review) in enumerate(df_reviews.iterrows()):
        customer_id = review['CustomerID']
        review_text = review['Generated_Review']
        
        # Update progress
        progress = (idx + 1) / len(df_reviews)
        progress_bar.progress(progress)
        status_text.text(f"Processing {idx + 1}/{len(df_reviews)}: {customer_id}")
        
        # Look up customer
        try:
            customer_data = df_warehouse.loc[customer_id]
        except KeyError:
            st.warning(f"CustomerID {customer_id} not found in warehouse. Skipping.")
            continue
        
        # Call ML API
        features_for_api = customer_data[ml_features_list].to_dict()
        ml_result = get_ml_prediction(customer_id, features_for_api, ml_api_url)
        
        if ml_result is None:
            continue
        
        # Call LLM API with visual feedback
        with st.spinner(f"Analyzing review {idx + 1} with LLM..."):
            llm_result = get_llm_analysis(gemini_model, review_text)
        
        # Store detailed analysis for display
        llm_analysis_item = {
            "customer_id": customer_id,
            "review_text": review_text,
            "theme": llm_result.get('theme', 'N/A'),
            "sentiment": llm_result.get('sentiment', 'N/A'),
            "ml_risk": ml_result['risk_level'],
            "churn_prob": ml_result['churn_probability'],
            "ml_features": features_for_api,  # Store features sent to ML API
            "llm_prompt": llm_result.get('prompt', ''),
            "llm_raw_response": llm_result.get('raw_response', ''),
            "llm_confidence": llm_result.get('confidence', None)
        }
        llm_analyses.append(llm_analysis_item)
        
        # Update the feed in real-time
        with feed_placeholder.container():
            # Show most recent analyses (last 10)
            display_count = min(10, len(llm_analyses))
            recent_analyses = llm_analyses[-display_count:]
            
            st.markdown(f"**Showing {len(recent_analyses)} most recent analyses (newest first):**")
            
            for i, analysis in enumerate(reversed(recent_analyses)):  # Show newest first
                with st.expander(
                    f"Review {len(llm_analyses) - i} | Customer: {analysis['customer_id']} | Theme: {analysis['theme']} | Sentiment: {analysis['sentiment']}",
                    expanded=(i == 0)  # Expand most recent only
                ):
                    # Review text section
                    st.markdown("**Customer Review:**")
                    st.info(analysis['review_text'])
                    
                    # Classification results
                    col1, col2 = st.columns(2)
                    with col1:
                        st.markdown("**LLM Analysis:**")
                        theme_color = {
                            'Price': '#ff9800',
                            'Product/Service': '#2196f3',
                            'Customer Support': '#f44336',
                            'Competitor': '#9c27b0',
                            'Other': '#607d8b'
                        }.get(analysis['theme'], '#666')
                        
                        sentiment_color = {
                            'Negative': '#f44336',
                            'Positive': '#4caf50',
                            'Neutral': '#ff9800'
                        }.get(analysis['sentiment'], '#666')
                        
                        st.markdown(f"**Theme:** <span style='color: {theme_color}; font-weight: 600;'>{analysis['theme']}</span>", unsafe_allow_html=True)
                        st.markdown(f"**Sentiment:** <span style='color: {sentiment_color}; font-weight: 600;'>{analysis['sentiment']}</span>", unsafe_allow_html=True)
                        
                        # Show confidence if available
                        if analysis.get('llm_confidence') is not None:
                            st.metric("LLM Confidence", f"{analysis['llm_confidence']:.1%}")
                    
                    with col2:
                        st.markdown("**ML Prediction:**")
                        st.metric("Risk Level", analysis['ml_risk'])
                        st.metric("Churn Probability", f"{analysis['churn_prob']:.1%}")
                    
                    # Technical Details Tabs
                    tab1, tab2, tab3 = st.tabs(["ML Features", "LLM Prompt", "Raw LLM Response"])
                    
                    with tab1:
                        st.markdown("**19 Features Sent to ML API:**")
                        features_df = pd.DataFrame([analysis.get('ml_features', {})]).T
                        features_df.columns = ['Value']
                        st.dataframe(features_df, use_container_width=True, height=400)
                    
                    with tab2:
                        st.markdown("**Prompt Sent to Gemini API:**")
                        st.code(analysis.get('llm_prompt', ''), language='text')
                    
                    with tab3:
                        st.markdown("**Raw LLM Response (Before Parsing):**")
                        st.code(analysis.get('llm_raw_response', ''), language='json')
                        if analysis.get('llm_confidence') is None:
                            st.caption("Note: Confidence score not detected in response. Gemini may not provide explicit confidence scores.")
                    
                    st.markdown("---")
        
        # Combine results
        combined_data = {
            "CustomerID": customer_id,
            "ML_Risk_Level": ml_result['risk_level'],
            "Churn_Probability": ml_result['churn_probability'],
            "LLM_Theme": llm_result.get('theme', 'N/A'),
            "LLM_Sentiment": llm_result.get('sentiment', 'N/A'),
            "CLTV": customer_data['CLTV'],
            "Review_Text": review_text
        }
        results.append(combined_data)
        
        # Rate limiting for LLM API (adjust as needed)
        time.sleep(1)  # Reduced from 6.1 for faster demo
    
    progress_bar.empty()
    status_text.empty()
    
    if not results:
        st.error("No results generated. Check API connections.")
        return None
    
    # Create final dataframe
    df_final = pd.DataFrame(results)
    df_final['Priority_Score'] = df_final['Churn_Probability'] * df_final['CLTV']
    df_final = df_final.sort_values(by='Priority_Score', ascending=False)
    
    return df_final

