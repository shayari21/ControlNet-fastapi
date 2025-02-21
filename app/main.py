from fastapi import FastAPI, HTTPException, File, UploadFile, Form
from pydantic import BaseModel
from model import ControlNetModel
from io import BytesIO
from fastapi.responses import StreamingResponse
from PIL import Image
import numpy as np
import uvicorn

#initial Fastapi 
app = FastAPI()

# Initialize ControlNetModel from model.py
model = ControlNetModel()

@app.post("/generate_image/")
async def generate_image(file: UploadFile = File(...), prompt: str = Form(...)):
    try:
        # Read input image and convert the RGB image to numpy array
        input_image = Image.open(file.file).convert("RGB")
        input_image= np.array(input_image)
        
        # Generate image using the model
        combined_image = model.generate_synthetic_image(input_image,prompt)

        # Convert combined image to PNG format in memory
        img_io = BytesIO()
        combined_image.save(img_io, "PNG")
        img_io.seek(0)
        # Return image as response
        return StreamingResponse(img_io, media_type="image/png")
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Image generation failed: {e}")
    
#automatically launches the API after adding command python main.py on the cmd.
if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8000)