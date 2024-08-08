import requests

with open('landscape_sketch.jpeg', 'rb') as image_file:
    files = {
        'img': ('image.jpg', image_file, 'image/jpeg')
    }

    data = {
        'prompt': "a fairytale landscape in a hot desert"
    }

    # Send the POST request
    response = requests.post("http://localhost:8000/items", 
                             files=files, 
                             data=data)


file = open("processed.html", "w")
file.write(response.text)
file.close()