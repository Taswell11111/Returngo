import google.generativeai as genai
import os
import pathlib
import textwrap

# --- Configuration ---
# 1. Get your API key from environment variables or another secret store.
#    It's recommended to use Streamlit secrets if running in that context.
#    For local scripts, environment variables are a good choice.
GOOGLE_API_KEY = os.getenv('GOOGLE_API_KEY')
if not GOOGLE_API_KEY:
    raise ValueError("GOOGLE_API_KEY environment variable not set.")

genai.configure(api_key=GOOGLE_API_KEY)

# 2. Set up the model - in this case, Gemini Flash
generation_config = {
  "temperature": 1,
  "top_p": 0.95,
  "top_k": 64,
  "max_output_tokens": 8192,
}
model = genai.GenerativeModel(model_name="gemini-1.5-flash-latest", generation_config=generation_config)

try:
    # Example: Read the content of README.md and ask the model to summarize it
    readme_content = pathlib.Path(r"C:\Users\Taswell\OneDrive\Documents\GitHub\Returngo\README.md").read_text()
    prompt = f"Please summarize the following README file:\n\n{readme_content}"
    response = model.generate_content(prompt)
    print(textwrap.fill(response.text, width=80))
except Exception as e:
    print(f"An error occurred: {e}")
