import os
import numpy as np
import cv2
import face_recognition
from sklearn.neighbors import KNeighborsClassifier
import joblib
from tqdm import tqdm

DATASET_PATH = "dataset"
MODEL_DIR = "models"
EMBEDDINGS_FILE = os.path.join(MODEL_DIR, "embeddings.npy")
LABELS_FILE = os.path.join(MODEL_DIR, "labels.txt")
CLASSIFIER_FILE = os.path.join(MODEL_DIR, "knn_classifier.pkl")

def is_clear(image):
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    score = cv2.Laplacian(gray, cv2.CV_64F).var()
    return score > 100  # edge clarity

def extract_embedding(image_path):
    try:
        image = face_recognition.load_image_file(image_path)
        if not is_clear(image):
            return None
        encodings = face_recognition.face_encodings(image)
        return encodings[0] if encodings else None
    except:
        return None

all_embeddings = []
all_labels = []

print("🔁 Extracting averaged embeddings...")

for label in tqdm(os.listdir(DATASET_PATH)):
    person_dir = os.path.join(DATASET_PATH, label)
    if not os.path.isdir(person_dir):
        continue

    person_embeddings = []
    for img_file in os.listdir(person_dir):
        path = os.path.join(person_dir, img_file)
        embedding = extract_embedding(path)
        if embedding is not None:
            person_embeddings.append(embedding)

    # ✅ Accept persons with even 1 valid image
    if len(person_embeddings) >= 1:
        avg_embedding = np.mean(person_embeddings, axis=0)
        all_embeddings.append(avg_embedding)
        all_labels.append(label)

os.makedirs(MODEL_DIR, exist_ok=True)
np.save(EMBEDDINGS_FILE, np.array(all_embeddings))
with open(LABELS_FILE, "w") as f:
    f.write("\n".join(all_labels))

print("🎯 Training KNN classifier...")
knn = KNeighborsClassifier(n_neighbors=3, weights='distance')
knn.fit(all_embeddings, all_labels)
joblib.dump(knn, CLASSIFIER_FILE)

print("✅ Model saved in 'models/' folder.")
