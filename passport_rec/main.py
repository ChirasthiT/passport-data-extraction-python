from fastapi import FastAPI, File, UploadFile, HTTPException
from passport_data_extracter import PassportDataExtractor  # Adjust the import path based on your file structure
import cv2
import numpy as np
import uvicorn
import os

app = FastAPI()

@app.post("/extract_passport_data/")
async def extract_passport_data(file_path: str):
    try:
        if not os.path.isfile(file_path):
            raise HTTPException(status_code=400, detail="File does not exist")

        # Read the image from the file path
        image = cv2.imread(file_path)
        
        if image is None:
            raise HTTPException(status_code=400, detail="Invalid image file")


        extractor = PassportDataExtractor(image)
        
        data = extractor.process_image()

        return data

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"An error occurred: {str(e)}")

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)