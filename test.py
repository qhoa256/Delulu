import requests

# Define the URL of the FastAPI server
url = "http://localhost:8000/chat"  # Adjust if the API is running on a different server
url = "https://aitools.ptit.edu.vn/nho/chat"
# Prepare the data to be sent in the POST request
data = {
    "user_id": "user1234",  # Replace with the actual user ID
    "session_id": "session1234",  # Replace with the actual session ID
    "text": "giải thích chi tiết về nó",  # Replace with the actual text message
}

# You can also upload an image (optional), if needed:
with open("dongho.jpg", "rb") as image_file:
    files = {"image": image_file}
    response = requests.post(url, data=data, files=files)

# If there is no image, just post the data
response = requests.post(url, data=data)

# Check the response
if response.status_code == 200:
    print("Response from API:", response.json())
else:
    print("Failed to get response. Status code:", response.status_code)