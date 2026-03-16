import cv2
import numpy as np
import joblib

# Load model
model = joblib.load("../saved_model/deepfake_model.pkl")

IMG_SIZE = 128

# Give image path here
image_path ="../external_test/R6-F.jpg"   # change this to test image

img = cv2.imread(image_path)
img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
img = img.flatten()
img = img.reshape(1, -1)

prediction = model.predict(img)

if prediction[0] == 0:
    print("Result: This image is Real")
else:
    print("Result: This image is AI Generated")