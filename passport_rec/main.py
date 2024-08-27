from fastapi import FastAPI, File, UploadFile, HTTPException, Form
from passport_data_extracter import PassportDataExtractor  # Adjust the import path based on your file structure
import cv2
import uvicorn
import os
import base64
app = FastAPI()

@app.post("/extract_passport_data/")
async def extract_passport_data(file_path: str = Form(...)):
    try:
        if not os.path.isfile(file_path):
            raise HTTPException(status_code=400, detail="File does not exist")

        image = cv2.imread(file_path)
        
        if image is None:
            raise HTTPException(status_code=400, detail="Invalid image file")

        extractor = PassportDataExtractor()
        face, data = extractor.process_image(image)
        is_success, buffer = cv2.imencode(".jpg", face)
        if not is_success:
            raise HTTPException(status_code=500, detail="Failed to encode face image")

        # Convert to Base64
        encoded_face = base64.b64encode(buffer).decode('utf-8')

        # Return as JSON
        return {"encoded_face": encoded_face, "data": data}

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"An error occurred: {str(e)}")

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)