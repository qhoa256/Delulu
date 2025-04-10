import requests

# Define the URL of the FastAPI server
url = "https://aitools.ptit.edu.vn/nho/analyze-image"  # Update with your actual server URL

# Prepare the form data
form_data = {
    "user_id": "user1234",      # Replace with actual user ID
    "session_id": "session1234", # Replace with actual session ID
    "text": "Describe this image"  # Your prompt or question about the image
}

try:
    # Test with an image
    with open("dongho.jpg", "rb") as image_file:
        files = {
            "image": ("dongho.jpg", image_file, "image/jpeg")  # (filename, file object, content type)
        }
        # Send POST request with both form data and image
        response = requests.post(url, data=form_data, files=files)

    # Check the response
    if response.status_code == 200:
        print("Success! Response from API:", response.json())
    else:
        print(f"Error: Status code {response.status_code}")
        print("Response:", response.text)

except FileNotFoundError:
    print("Error: Image file 'dongho.jpg' not found")
except requests.exceptions.RequestException as e:
    print(f"Error making request: {e}")
except Exception as e:
    print(f"Unexpected error: {e}")