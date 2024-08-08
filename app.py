import os
import io
import base64
os.environ['HF_HOME'] = './hf_cache'

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

pipe = StableDiffusionImg2ImgPipeline.from_pretrained("runwayml/stable-diffusion-v1-5")

app = FastAPI()

@app.post("/generation", response_class=HTMLResponse)
async def read_items(prompt: Annotated[str, Form()],
                     img: Annotated[UploadFile, File()]):
    
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

    return f"""
        <html>
        <body>
            <h1>Here is your image:</h1>
            <img src="data:image/jpeg;base64,{base64_image}" alt="Base64 Image" />
        </body>
        </html>
    """
