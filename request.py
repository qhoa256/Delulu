import requests

url = "http://localhost:8000/chat/"
headers = {
    "Content-Type": "application/json"
}
data = {
    "body": {
        "user_id": "user123",
        "session_id": "session123",
        "text": "hello, tôi là Hùng"
    }
}

response = requests.post(url, json=data, headers=headers)
print(response.json())