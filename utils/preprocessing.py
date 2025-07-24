import os
import face_recognition
import numpy as np

def build_embeddings(dataset_path='dataset'):
    embeddings = []
    labels = []

    for person in os.listdir(dataset_path):
        person_path = os.path.join(dataset_path, person)
        for img_name in os.listdir(person_path):
            img_path = os.path.join(person_path, img_name)
            image = face_recognition.load_image_file(img_path)

            face_locations = face_recognition.face_locations(image)
            if len(face_locations) != 1:
                continue  # skip unclear faces

            encoding = face_recognition.face_encodings(image, known_face_locations=face_locations)[0]
            embeddings.append(encoding)
            labels.append(person)

    return np.array(embeddings), labels
