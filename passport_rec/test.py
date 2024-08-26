from passport_data_extracter import PassportDataExtractor  # Adjust the import path based on your file structure
import cv2
import numpy as np
import os


filepath = 'passport_rec/test_images/p26.jpeg'
image = cv2.imread(filepath)
extractor = PassportDataExtractor(image)
extractor.process_image()