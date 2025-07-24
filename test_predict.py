import face_recognition
import numpy as np
from utils.recognizer import recognize_face

def load_labels():
    with open("models/labels.txt", "r") as f:
        return f.read().splitlines()

def identify(image_path):
    known_embeddings = np.load("models/embeddings.npy")
    labels = load_labels()

    image = face_recognition.load_image_file(image_path)
    face_locations = face_recognition.face_locations(image)

    if not face_locations:
        return "❌ No face found", None

    test_embedding = face_recognition.face_encodings(image, known_face_locations=face_locations)[0]
    predictions = recognize_face(test_embedding, known_embeddings, labels, top_k=3)
    return predictions

# EXAMPLE RUN:
if __name__ == "__main__":
    test_img = "images_to_test/test1.jpg"
    result = identify(test_img)

    print("\n✅ Top Matches:")
    for name, score in result:
        print(f"{name} (similarity: {score:.4f})")
