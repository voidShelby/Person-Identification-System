from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

def recognize_face(test_embedding, known_embeddings, labels, top_k=3):
    similarities = cosine_similarity([test_embedding], known_embeddings)[0]
    top_k_indices = np.argsort(similarities)[-top_k:][::-1]
    predictions = [(labels[i], similarities[i]) for i in top_k_indices]
    return predictions
