import os
import cv2
import numpy as np

# Paths
real_path = "../dataset/real"
fake_path = "../dataset/fake"

data = []
labels = []

IMG_SIZE = 128

# Load real images
for img_name in os.listdir(real_path):
    img_path = os.path.join(real_path, img_name)
    img = cv2.imread(img_path)

    if img is not None:
        img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
        data.append(img)
        labels.append(0)  # Real = 0

# Load fake images
for img_name in os.listdir(fake_path):
    img_path = os.path.join(fake_path, img_name)
    img = cv2.imread(img_path)

    if img is not None:
        img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
        data.append(img)
        labels.append(1)  # Fake = 1

# Convert to numpy arrays
data = np.array(data)
labels = np.array(labels)

print("Total images processed:", len(data))