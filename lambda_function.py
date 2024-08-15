import io
import ast
import base64

from PIL import Image
import torch
from transformers import DetrImageProcessor, DetrForObjectDetection

# you can specify the revision tag if you don't want the timm dependency
processor = DetrImageProcessor.from_pretrained("facebook/detr-resnet-50", 
                                               revision="no_timm", 
                                               cache_dir='./hf_cache')
model = DetrForObjectDetection.from_pretrained("facebook/detr-resnet-50", 
                                               revision="no_timm",
                                               cache_dir='./hf_cache')                                             

def handler(event, context):
    body = ast.literal_eval(event['body'])
    
    encoded_image = body['encoded_image']
    color = body['color']

    # decode image string to byte
    image_bytes = base64.b64decode(encoded_image)

    # Convert bytes to a BytesIO object
    init_image = Image.open(io.BytesIO(image_bytes))

    # run object detection model
    inputs = processor(images=init_image, return_tensors="pt")
    outputs = model(**inputs)

    # only keep detections with score > 0.9
    target_sizes = torch.tensor([init_image.size[::-1]])
    results = processor.post_process_object_detection(outputs, target_sizes=target_sizes, threshold=0.9)[0]

    # report detected objects in html file
    detected_labels = []
    prompt = ""
    for score, label in zip(results["scores"], results["labels"]):
        if model.config.id2label[label.item()] not in detected_labels:
            prompt += f"Detected {model.config.id2label[label.item()]} with confidence {round(score.item(), 3)}. <br>"
            detected_labels.append(model.config.id2label[label.item()])

    html_string = f"""
        <html>
            <head>
                <meta charset="UTF-8">
                <meta name="viewport" content="width=device-width, initial-scale=1.0">
                <style>
                    .container {{
                        text-align: center;
                    }}

                    h1 {{
                        color: {color};
                    }}

                    .button-container {{
                        margin-top: 20px;
                        text-align: center;
                    }}
                    
                    .my-button {{
                        background-color: {color};
                        width: 150px;
                        height: 50px;
                        border: none;
                        border-radius: 5px;
                        cursor: pointer;
                        align-items: center;
                    }}
                    
                    .top-line {{
                        height: 10px; 
                        width: 50%; 
                        background-color: {color}; 
                        margin: 0 auto; 
                    }}
                    
                    .bottom-line {{
                        height: 10px; 
                        width: 50%; 
                        background-color: {color}; 
                        position: fixed;
                        bottom: 0;
                        left: 0;
                        right: 0;
                        margin: 0 auto;
                    }}           
                </style>
            </head>

            <body>
                <div class="top-line"></div>
                <div class="container">
                <h1>{prompt}</h1>
                <img src="data:image/jpeg;base64,{encoded_image}" alt="Base64 Image" />
            </div>
            <div class="button-container">
                <button class="my-button">Like</button>
            </div>
            <div class="bottom-line"></div>
            </body>
        </html>
    """

    return {
      "isBase64Encoded": False,
      "statusCode": 200,
      "body": html_string,
      "headers": {
        "content-type": "text/html"
      }
    }
