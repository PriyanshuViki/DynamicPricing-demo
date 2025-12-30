import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

print("Testing chatbot agent...")
print(f"AWS Region: {os.getenv('AWS_DEFAULT_REGION')}")
print(f"Model ID: {os.getenv('MODEL_ID')}")

from chatbot_agent import process_query

# Test query
response = process_query("What is the total revenue opportunity?")
print("\n=== Response ===")
print(response)
