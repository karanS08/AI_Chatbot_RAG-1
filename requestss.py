import requests

url = "http://localhost:5000/api/rag"
payload = {
    "query": "What are the best practices for sugarcane pest control?",
    "language": "english"
}
response = requests.post(url, json=payload)
print(response.json())