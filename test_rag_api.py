import requests

url = "http://localhost:5000/api/rag"
payload = {
    "query": "What are the best practices for sugarcane pest control?",
    "language": "english"
}

print("Testing RAG API endpoint...")
print(f"URL: {url}")
print(f"Payload: {payload}")
print()

try:
    response = requests.post(url, json=payload)
    print(f"Status Code: {response.status_code}")
    print(f"Response: {response.json()}")
except Exception as e:
    print(f"Error: {e}")
