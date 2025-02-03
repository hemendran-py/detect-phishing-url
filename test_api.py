import requests

url = "http://127.0.0.1:3000/predict"
data = {"url": "https://google.com"}

response = requests.post(url, json=data)
print("Response:", response.json())
