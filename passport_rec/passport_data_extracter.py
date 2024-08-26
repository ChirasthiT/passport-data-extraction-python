import numpy as np
import pytesseract
import cv2
from mtcnn import MTCNN
import os
from datetime import datetime
import spacy
import json

class PassportDataExtractor:
    def __init__(self, image) -> None:
        self.image = image
        self.detector = MTCNN()
        self.nlp = spacy.load('en_core_web_sm')
        self.face = None
        self.details = None

    def preprocess_image(self, image):
        # Check if the image is grayscale
        if len(image.shape) == 2 or image.shape[2] == 1:  # If grayscale
            # Convert grayscale image to 3-channel grayscale (to mimic RGB)
            image_gray = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
        else:
            # If already a 3-channel image, just use it
            image_gray = image

        # Apply CLAHE to enhance contrast
        lab = cv2.cvtColor(image_gray, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        cl = clahe.apply(l)
        enhanced_image = cv2.merge((cl, a, b))
        enhanced_image = cv2.cvtColor(enhanced_image, cv2.COLOR_LAB2BGR)

        # Optionally, increase the brightness
        beta = 30  # Brightness control (0-100)
        brightened_image = cv2.convertScaleAbs(enhanced_image, alpha=1, beta=beta)

        return brightened_image
    
    def detect_face(self, image):
        faces = self.detector.detect_faces(image)
        return faces
    
    def extract_face(self, image, im2):
        faces = self.detect_face(image)
        if len(faces) > 0:
            # Use the first detected face
            face = faces[0]
            x, y, width, height = face['box']
            cropped_face = im2[y:y+height, x:x+width]
            return cropped_face
        else:
            return None  
    
    def extract_info_ner(self, text):
        doc = self.nlp(text)
        info = {}

        for ent in doc.ents:
            if ent.label_ == 'PERSON':
                info['name'] = ent.text
            elif ent.label_ == 'DATE':
                info['dob'] = ent.text
            elif ent.label_ == 'CARDINAL':  # For things like passport numbers
                info['passport_number'] = ent.text
        
        return info
    
    def text_extraction(self, image):
        text = pytesseract.image_to_string(image)
        extracted_text = self.extract_info_ner(text)
        return extracted_text
        
    def process_image(self):
        image_copy = self.image.copy()
        preprocessed_image = self.preprocess_image(self.image)
        face = self.extract_face(preprocessed_image, image_copy)
        text = self.text_extraction(image_copy)
        text['face'] = face
        
        json_string = json.dumps(text)
        #print(json_string)
        return json_string
