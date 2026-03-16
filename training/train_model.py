import numpy as np
import cv2
import os
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
import joblib

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
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        img = cv2.Canny(img, 100, 200)

        img = img.flatten()  # Convert image to 1D array
        data.append(img)
        labels.append(0)

# Load fake images
for img_name in os.listdir(fake_path):
    img_path = os.path.join(fake_path, img_name)
    img = cv2.imread(img_path)
    if img is not None:
        img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        img = cv2.Canny(img, 100, 200)
        img = img.flatten()
        data.append(img)
        labels.append(1)

data = np.array(data)
labels = np.array(labels)

# Split data
X_train, X_test, y_train, y_test = train_test_split(
    data, labels, test_size=0.2, random_state=42
)

# Train model
model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)

# Predict
y_pred = model.predict(X_test)

# Accuracy
accuracy = accuracy_score(y_test, y_pred)
print("Model Accuracy:", accuracy)

# Save model
os.makedirs("../saved_model", exist_ok=True)
joblib.dump(model, "../saved_model/deepfake_model.pkl")

print("Model saved successfully!")