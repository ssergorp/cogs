import requests
import os
from dotenv import load_dotenv

load_dotenv()

# Get API key from environment
api_key = os.getenv("MISTRAL_API_KEY")
if not api_key:
    print("Error: MISTRAL_API_KEY not found in environment variables")
    exit(1)

client = requests
resp = client.post(
    "https://api.mistral.ai/v1/embeddings",
    json={"model": "mistral-embed", "input": ["Hello world"]},
    headers={"Authorization": f"Bearer {api_key}"}
)

if resp.status_code == 200:
    print("✅ API call successful!")
    data = resp.json()
    print(f"Embedding dimensions: {len(data['data'][0]['embedding'])}")
    print("First few embedding values:", data['data'][0]['embedding'][:5])
else:
    print(f"❌ API call failed with status {resp.status_code}")
    print("Response:", resp.text)
