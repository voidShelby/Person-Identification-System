import os
import cv2
import face_recognition
import matplotlib.pyplot as plt

DATASET_DIR = "dataset"

def variance_of_laplacian(image):
    return cv2.Laplacian(image, cv2.CV_64F).var()

person_ids = []
image_counts = []
blurry_counts = []
undetected_counts = []

for person_id in os.listdir(DATASET_DIR):
    person_path = os.path.join(DATASET_DIR, person_id)
    if not os.path.isdir(person_path):
        continue

    total_images = 0
    blurry = 0
    undetected = 0

    for img_name in os.listdir(person_path):
        img_path = os.path.join(person_path, img_name)
        image = cv2.imread(img_path)

        if image is None:
            continue

        total_images += 1
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        # Check blur
        if variance_of_laplacian(gray) < 70:
            blurry += 1

        # Check if face is detectable
        rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        faces = face_recognition.face_locations(rgb)
        if len(faces) == 0:
            undetected += 1

    person_ids.append(person_id)
    image_counts.append(total_images)
    blurry_counts.append(blurry)
    undetected_counts.append(undetected)

# 📊 Summary Plot
plt.figure(figsize=(12, 6))
plt.bar(person_ids, image_counts, label="Total Images")
plt.bar(person_ids, blurry_counts, label="Blurry Images")
plt.bar(person_ids, undetected_counts, label="No Face Detected")
plt.title("EDA Summary for Face Dataset")
plt.xlabel("Person ID")
plt.ylabel("Image Count")
plt.legend()
plt.tight_layout()
plt.xticks(rotation=45)
plt.grid(True)
plt.show()
