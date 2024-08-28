import cv2
import spacy
import json
import sys
import re

class PassportDataExtractor:
    def __init__(self) -> None:
        try:
            self.nlp = spacy.load('en_core_web_sm')
        except OSError:
            spacy.cli.download("en_core_web_sm")
            self.nlp = spacy.load('en_core_web_sm')

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
    
    def clean_text(text):
        cleaned_text = re.sub(r'\s+', ' ', text).strip()
        return cleaned_text
    
    def detect_face(self, image):
        from mtcnn import MTCNN
        detector = MTCNN()
        faces = detector.detect_faces(image)
        del(detector)
        del sys.modules['mtcnn']
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
        text = self.clean_text(text)
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
        import pytesseract
        text = pytesseract.image_to_string(image)
        del sys.modules['pytesseract']
        extracted_text = self.extract_info_ner(text)
        return extracted_text
        
    def process_image(self, image):
        image_copy = image.copy()
        preprocessed_image = self.preprocess_image(image)
        face = self.extract_face(preprocessed_image, image_copy)
        text = self.text_extraction(image_copy)
        
        json_string = json.dumps(text)
        #print(json_string)
        return face, json_string
