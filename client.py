import requests
import base64

# Read the image file in binary mode
with open("landscape_sketch.jpeg", "rb") as image_file:
    # Encode the image file into Base64
    encoded_string = base64.b64encode(image_file.read())

    # Convert the Base64 bytes to a string
    encoded_image = encoded_string.decode('utf-8')

payload = {
    'prompt': "a fairytale landscape in a hot desert",
    'encoded_image': encoded_image,
    'color': '#fcba03'
}

# Send the POST request
response = requests.post("http://localhost:8000/generation", 
                         json=payload)

# save html response to file
file = open("processed.html", "w")
file.write(response.text)
file.close()