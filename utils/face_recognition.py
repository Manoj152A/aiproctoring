import cv2
import numpy as np
from deepface import DeepFace

class FaceRecognition:
    def __init__(self, reference_image_path):
        self.reference_image_path = reference_image_path
        self.reference_embeddings = []
        self.similarity_threshold = 1.0
        self.recent_results = []
        self.max_recent_results = 5

        # Generate multiple reference embeddings
        for _ in range(3):  # Capture 3 different angles
            embedding = self.get_face_embedding(reference_image_path)
            if embedding is not None:
                self.reference_embeddings.append(embedding)

    def get_face_embedding(self, image_path):
        try:
            # DeepFace.represent() returns a list of dictionaries with the embedding under the 'embedding' key
            embeddings = DeepFace.represent(img_path=image_path, model_name="ArcFace", enforce_detection=False)
            if embeddings and isinstance(embeddings, list) and 'embedding' in embeddings[0]:
                embedding = embeddings[0]['embedding']  # Extract the embedding from the first result
                return np.array(embedding)
            else:
                print("No embedding found.")
                return None
        except Exception as e:
            print(f"Error getting face embedding: {str(e)}")
            return None

    def verify_face(self, current_image_path):
        current_embedding = self.get_face_embedding(current_image_path)
        if current_embedding is None:
            return False, None, "Failed to get embedding for current image"

        # Calculate the distance between the current embedding and each reference embedding
        distances = [np.linalg.norm(np.array(ref_emb) - current_embedding) for ref_emb in self.reference_embeddings]
        min_distance = min(distances)
        is_same = min_distance < self.similarity_threshold

        # Track the most recent results
        self.recent_results.append(is_same)
        if len(self.recent_results) > self.max_recent_results:
            self.recent_results.pop(0)
        is_same_average = sum(self.recent_results) / len(self.recent_results) > 0.6

        return is_same_average, min_distance, None
