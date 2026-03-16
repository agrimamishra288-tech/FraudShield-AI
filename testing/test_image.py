import cv2
import numpy as np
import joblib

# Load model
model = joblib.load("../saved_model/deepfake_model.pkl")

IMG_SIZE = 128

# Give image path here
image_path ="../external_test/real_045505.jpg" # Change this to your test image path

img = cv2.imread(image_path)

if img is None:
    print("Error: Image not found.")
    exit()

# Same preprocessing used during training
img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
img = cv2.Canny(img, 100, 200)

img = img.flatten()
img = img.reshape(1, -1)

prediction = model.predict(img)

if prediction[0] == 0:
    print("Result: This image is Real")
else:
    print("Result: This image is AI Generated")