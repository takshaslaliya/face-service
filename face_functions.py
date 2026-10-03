import os
import cv2
import numpy as np
from insightface.app import FaceAnalysis

_face_app = None

def get_face_app():
    """
    Lazy loader for InsightFace model to prevent Passenger WSGI startup timeouts.
    """
    global _face_app
    if _face_app is None:
        model_name = os.getenv("INSIGHTFACE_MODEL", "buffalo_sc")
        model_root = os.getenv("INSIGHTFACE_ROOT", os.path.expanduser("~/.insightface"))
        
        print(f"[INFO] Initializing InsightFace ({model_name}) from root: {model_root}...")
        _face_app = FaceAnalysis(
            name=model_name,
            root=model_root,
            allowed_modules=['detection', 'recognition'],
            providers=["CPUExecutionProvider"]
        )
        _face_app.prepare(
            ctx_id=0,
            det_size=(640, 640)
        )
        print(f"[INFO] InsightFace {model_name} Model Ready.")
    return _face_app


def get_face_embedding(image_input):
    """
    Read an image and generate a 512-D face embedding vector.
    
    Args:
        image_input: File path (str), Path object, or pre-decoded numpy array (cv2 image)
    
    Returns:
        embedding -> numpy array of 512 values
        
    Raises:
        ValueError -> if image cannot be read, 0 faces found, or multiple faces found
    """
    if isinstance(image_input, (str, bytes, os.PathLike)):
        image = cv2.imread(str(image_input))
    elif isinstance(image_input, np.ndarray):
        image = image_input
    else:
        raise ValueError("Invalid image input format.")

    if image is None:
        raise ValueError("Unable to read image")

    app_instance = get_face_app()
    faces = app_instance.get(image)

    if len(faces) == 0:
        raise ValueError("No face detected")

    if len(faces) > 1:
        raise ValueError(f"Multiple faces detected ({len(faces)} found). Only single face is allowed")

    return faces[0].embedding


def calculate_similarity(embedding_a, embedding_b):
    """
    Calculates cosine similarity between two face embeddings.
    """
    emb_a = np.array(embedding_a, dtype=np.float32)
    emb_b = np.array(embedding_b, dtype=np.float32)

    norm_a = np.linalg.norm(emb_a)
    norm_b = np.linalg.norm(emb_b)

    if norm_a == 0 or norm_b == 0:
        return 0.0

    return float(np.dot(emb_a, emb_b) / (norm_a * norm_b))


def compare_faces(embedding_a, embedding_b, threshold=0.5):
    """
    Compares two embeddings and checks against threshold.
    """
    similarity = calculate_similarity(embedding_a, embedding_b)
    is_same = similarity >= threshold
    return is_same, similarity