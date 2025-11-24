import requests
import os

url = "http://localhost:5000/api/rag"

# Example 1: Simple text query (JSON)
print("=" * 60)
print("Test 1: Simple text query (JSON)")
print("=" * 60)
payload = {
    "query": "What are the best practices for sugarcane pest control?",
    "language": "english"
}

try:
    response = requests.post(url, json=payload)
    print(f"Status Code: {response.status_code}")
    print(f"Response: {response.json()}")
except Exception as e:
    print(f"Error: {e}")

print("\n" + "=" * 60)
print("Test 2: Query with media attachment (multipart/form-data)")
print("=" * 60)

# Example 2: Query with media files
# Note: Create a sample text file for testing
test_file_path = "test_document.txt"
with open(test_file_path, "w") as f:
    f.write("Sugarcane pest control best practices:\n")
    f.write("1. Regular monitoring\n")
    f.write("2. Use of biological control\n")
    f.write("3. Proper irrigation management\n")

data = {
    "query": "Based on the uploaded document, what are the key recommendations?",
    "language": "english"
}

try:
    with open(test_file_path, "rb") as f:
        files = [("media", (test_file_path, f, "text/plain"))]
        response = requests.post(url, data=data, files=files)
        print(f"Status Code: {response.status_code}")
        print(f"Response: {response.json()}")
except Exception as e:
    print(f"Error: {e}")
finally:
    # Cleanup
    if os.path.exists(test_file_path):
        os.remove(test_file_path)

print("\n" + "=" * 60)
