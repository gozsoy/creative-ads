import os
import io
import base64

from fastapi import FastAPI, Form, File, UploadFile
from fastapi.responses import HTMLResponse
from typing import Annotated
from PIL import Image
import torch

from diffusers import StableDiffusionImg2ImgPipeline

# edit StableDiffusionSafetyChecker class so it just returns the images and True values
from diffusers.pipelines.stable_diffusion import safety_checker

def sc(self, clip_input, images) :
    return images, [False for i in images]

safety_checker.StableDiffusionSafetyChecker.forward = sc

pipe = StableDiffusionImg2ImgPipeline.from_pretrained("runwayml/stable-diffusion-v1-5",
                                                      cache_dir="./hf_cache")

app = FastAPI()

@app.post("/generation", response_class=HTMLResponse)
async def read_items(prompt: Annotated[str, Form()],
                     img: Annotated[UploadFile, File()],
                     color: Annotated[str, Form()]):
    
    contents = await img.read()
    
    init_image = Image.open(io.BytesIO(contents)).convert("RGB").resize((256, 256))

    generated_image = pipe(prompt=prompt, 
                           image=init_image, 
                           strength=0.95,
                           guidance_scale=8.5
                           ).images[0]

    buffered = io.BytesIO()
    generated_image.save(buffered, format="JPEG")
    base64_image = base64.b64encode(buffered.getvalue()).decode('utf-8')

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
                <img src="data:image/jpeg;base64,{base64_image}" alt="Base64 Image" />
            </div>
            <div class="button-container">
                <button class="my-button">Like</button>
            </div>
            <div class="bottom-line"></div>
            </body>
        </html>
    """

    return html_string
