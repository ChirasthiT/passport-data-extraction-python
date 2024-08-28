import matplotlib.pyplot as plt
import cv2
from passport_data_extracter import PassportDataExtractor


# Load the image
filepath = 'Set a filepath'
image = cv2.imread(filepath)

if image is not None:
    extractor = PassportDataExtractor()
    face, data = extractor.process_image(image)

    if face is not None:
        plt.imshow(cv2.cvtColor(face, cv2.COLOR_BGR2RGB))
        plt.axis('off')
        plt.show()
    else:
        print("No face detected.")
    
    print(data)
else:
    print("Image not found. Please check the file path.")