from fastapi import FastAPI, File, UploadFile, HTTPException, Form
from passport_data_extracter import PassportDataExtractor 
import cv2
import uvicorn
import numpy as np
import base64


app = FastAPI()

@app.post("/extract_passport_data/")
async def extract_passport_data(file: UploadFile = File(...), file_2: UploadFile = File(...)):
    try:
        ### This part is for when the an image link is has to be sent only
        # if not os.path.isfile(file_path):
        #     raise HTTPException(status_code=400, detail="File does not exist")

        # image = cv2.imread(file_path)
        
        file_bytes = await file.read()
        file_bytes_2 = await file_2.read()
        
        # Convert the byte data to a NumPy array
        np_array = np.frombuffer(file_bytes, np.uint8)
        np_array_2 = np.frombuffer(file_bytes_2, np.uint8)
        
        # Decode the image using OpenCV
        image = cv2.imdecode(np_array, cv2.IMREAD_COLOR)
        image_2 = cv2.imdecode(np_array_2, cv2.IMREAD_COLOR)
        
        if image is None:
            raise HTTPException(status_code=400, detail="Invalid image file")

        extractor = PassportDataExtractor()
        face, data, fc_bool_result = extractor.process_image(image, image_2)
        is_success, buffer = cv2.imencode(".jpg", face)
        if not is_success:
            raise HTTPException(status_code=500, detail="Failed to encode face image")

        # Convert to Base64
        encoded_face = base64.b64encode(buffer).decode('utf-8')

        # Return as JSON
        return {"encoded_face": encoded_face, "data": data, "face_compare":fc_bool_result}

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"An error occurred: {str(e)}")

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)