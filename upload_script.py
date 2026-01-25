import anthropic
import os

# Your API key (I've removed it for safety, please re-insert yours)
os.environ["ANTHROPIC_API_KEY"] = "your-key-here"

client = anthropic.Anthropic()

# The path to your file
file_path = r"C:\Users\Taswell\OneDrive\Documents\GitHub\Returngo\README.md"

try:
    with open(file_path, "rb") as f:
        response = client.beta.files.upload(
            file=("README.md", f, "text/markdown"),
        )
    print(f"Success! File uploaded. File ID: {response.id}")
except Exception as e:
    print(f"An error occurred: {e}")