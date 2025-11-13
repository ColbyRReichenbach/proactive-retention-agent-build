import pandas as pd
import requests  # To call our ML Model API
import google.generativeai as genai  # To call the Gemini API
import os
from dotenv import load_dotenv  # To securely load our API key
from tqdm import tqdm  # For a nice progress bar
import time

# Load API keys and clients
def setup_clients():
   """
   Loads API keys from .env and configures the Gemini client.
   """
   # Find the .env file in the project root
   script_dir = os.path.dirname(os.path.abspath(__file__))
   project_root = os.path.dirname(script_dir)
   dotenv_path = os.path.join(project_root, '.env')


   if os.path.exists(dotenv_path):
       load_dotenv(dotenv_path)
       print(".env file loaded successfully.")
   else:
       print("Warning: .env file not found. Make sure it's in the project root.")


   # Configure Gemini
   try:
       api_key = os.environ.get("GOOGLE_API_KEY")
       if api_key is None:
           raise ValueError("GOOGLE_API_KEY not found in environment.")
       genai.configure(api_key=api_key)
       print("Google Gemini client configured successfully.")
   except Exception as e:
       print(f"ERROR: Could not configure Gemini. {e}")
       return None


   return genai.GenerativeModel('gemini-2.0-flash-lite')

# Load and prep data
def load_data_warehouse(path_to_telco_csv):
   """
   Loads the main Telco data and prepares it for fast lookups.
   """
   try:
       df_telco = pd.read_csv(path_to_telco_csv)
   except FileNotFoundError:
       print(f"ERROR: Main Telco data file not found at {path_to_telco_csv}")
       return None


   # Clean data to match 1st notebook
   df_telco['Total Charges'] = pd.to_numeric(df_telco['Total Charges'], errors='coerce')
   # Fill any new NaNs for the model
   df_telco['Total Charges'] = df_telco['Total Charges'].fillna(0)


   # Set CustomerID as the index for fast lookups
   df_telco = df_telco.set_index('CustomerID')


   print(f"Data Warehouse loaded. {len(df_telco)} customers found.")
   return df_telco

# Define api caller fxns
def get_ml_prediction(customer_id, features):
   """
   Calls the ML API (local or deployed) to get a churn prediction.
   Uses ML_API_URL environment variable if set, otherwise defaults to localhost.
   """
   # Get ML API URL from environment variable, default to localhost for local development
   ml_api_url = os.environ.get("ML_API_URL", "http://localhost:8000")
   url = f"{ml_api_url}/predict?customer_id={customer_id}"

   # The API expects two parts: a customer_id and a 'features' JSON object
   payload = features

   try:
       response = requests.post(url, json=payload, timeout=30)
       response.raise_for_status()  # Raises an error for 4xx/5xx responses
       return response.json()

   except requests.exceptions.ConnectionError:
       print(f"\nERROR: Could not connect to ML API at {url}.")
       if ml_api_url == "http://localhost:8000":
           print("Is the Docker container running? `docker run -d -p 8000:8000 ...`")
       else:
           print(f"Check that the ML API is deployed and accessible at {ml_api_url}")
       return None
   except requests.exceptions.Timeout:
       print(f"\nERROR: ML API request timed out for {customer_id}.")
       return None
   except Exception as e:
       print(f"\nERROR: ML API call failed for {customer_id}: {e}")
       return None

def get_llm_analysis(gemini_model, review_text):
   """
   Calls the Gemini API to classify the review text.
   """
   if gemini_model is None:
       return {"theme": "Error", "sentiment": "Error"}

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
       # Clean the output
       json_output = response.text.strip().replace("```json", "").replace("```", "")
       # Convert string JSON to Python dictionary
       return pd.read_json(json_output, typ='series').to_dict()
   except Exception as e:
       print(f"\nERROR: Gemini API call failed: {e}")
       return {"theme": "LLM Error", "sentiment": "Error"}

# Main
def main():
  print("Starting Proactive Retention Pipeline")

  # Setup
  gemini_model = setup_clients()
  if gemini_model is None:
      print("Exiting due to API configuration error.")
      return

  # Define file paths (relative to this script in the /pipeline folder)
  script_dir = os.path.dirname(os.path.abspath(__file__))
  project_root = os.path.dirname(script_dir)

  review_file_path = os.path.join(script_dir, "customer_reviews.csv")
  telco_file_path = os.path.join(project_root, "data", "Telco_customer_churn.csv")
  output_file_path = os.path.join(script_dir, "analyst_priority_list.csv")

  # Load data
  df_warehouse = load_data_warehouse(telco_file_path)
  if df_warehouse is None:
      return

  try:
      df_reviews = pd.read_csv(review_file_path)
  except FileNotFoundError:
      print(f"ERROR: Live review file not found at {review_file_path}")
      return

  print(f"'Live' review file loaded. Found {len(df_reviews)} reviews to process.")

  # These are the *exact* 19 features our ML model expects
  ml_features_list = [
      'Tenure Months', 'Monthly Charges', 'Total Charges', 'Gender',
      'Senior Citizen', 'Partner', 'Dependents', 'Phone Service',
      'Multiple Lines', 'Internet Service', 'Online Security',
      'Online Backup', 'Device Protection', 'Tech Support', 'Streaming TV',
      'Streaming Movies', 'Contract', 'Paperless Billing', 'Payment Method'
  ]

  # Main loop
  results = []

  # Use tqdm for a progress bar
  for _, review in tqdm(df_reviews.iterrows(), total=len(df_reviews), desc="Processing Reviews"):

      customer_id = review['CustomerID']
      review_text = review['Generated_Review']

      # Look up customer
      try:
          customer_data = df_warehouse.loc[customer_id]
      except KeyError:
          print(f"Warning: CustomerID {customer_id} not found in data warehouse. Skipping.")
          continue

      # Call ML Model API
      features_for_api = customer_data[ml_features_list].to_dict()
      ml_result = get_ml_prediction(customer_id, features_for_api)

      if ml_result is None:
          continue  # Error was already printed in the function

      # Call LLM (Gemini) API
      llm_result = get_llm_analysis(gemini_model, review_text)

      # Combine all results
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
      time.sleep(6.1)

  # Finish + save
  if not results:
      print("No results to save. Did all API calls fail?")
      return

  print("\nPipeline complete! Generating priority dashboard...")

  # Convert list of results into our final DataFrame
  df_final = pd.DataFrame(results)

  # Risk-Adjusted value
  df_final['Priority_Score'] = df_final['Churn_Probability'] * df_final['CLTV']

  df_final = df_final.sort_values(by='Priority_Score', ascending=False)

  # Save to CSV
  df_final.to_csv(output_file_path, index=False)

  print(f"--- Success! ---")
  print(f"Priority dashboard saved to {output_file_path}")
  print("\n--- Top 5 Customers for Analysts to Call ---")
  print(df_final.head())

if __name__ == "__main__":
  main()