import os
import sys
import json
from typing import List, Optional

import cv2
import numpy as np
import httpx
from fastapi import FastAPI, UploadFile, File, Form, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import RedirectResponse
from pydantic import BaseModel
from dotenv import load_dotenv

from face_functions import get_face_embedding, calculate_similarity

load_dotenv()

# ==============================================================================
# 1. FastAPI App Initialization
# ==============================================================================
app = FastAPI(
    title="Face AI Microservice",
    description="Stateless Face Recognition & Duplicate Verification Microservice for Node.js Backend",
    version="2.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.on_event("startup")
def startup_event():
    try:
        from face_functions import get_face_app
        print("[INFO] Preloading face model on startup...")
        get_face_app()
        print("[INFO] Face model loaded successfully.")
    except Exception as e:
        print(f"[WARN] Startup model preload failed: {e}")


# Configuration from Environment Variables
SIMILARITY_THRESHOLD = float(os.getenv("SIMILARITY_THRESHOLD", "0.50"))
NODE_API_BASE_URL = os.getenv("NODE_API_BASE_URL", "https://attendentsnews.hpys.in").rstrip("/")



# ==============================================================================
# 2. Schemas
# ==============================================================================
class CompareEmbeddingsRequest(BaseModel):
    target_embedding: List[float]
    done_faces: List[dict]  # list of {"student_id": int, "embedding": list[float]}
    threshold: Optional[float] = None


# ==============================================================================
# 3. Health & Status Endpoints
# ==============================================================================
@app.get("/", include_in_schema=False)
def root():
    return RedirectResponse(url="/docs")


@app.get("/api/health", summary="Health Check")
@app.get("/health", include_in_schema=False)
def health_check():
    return {
        "success": True,
        "service": "Face AI Microservice",
        "status": "healthy",
        "docs": "/docs"
    }


# ==============================================================================
# 4. Primary Endpoint: Detect Face + Extract Vector + Compare with Done Faces
# ==============================================================================
@app.post(
    "/api/face/detect-and-compare",
    summary="Extract face embedding and compare against done session faces passed by Node.js"
)
@app.post(
    "/face/detect-and-compare",
    include_in_schema=False
)
async def detect_and_compare(
    image: UploadFile = File(..., description="Student face image (JPEG/PNG)"),
    existing_faces: Optional[str] = Form(
        "[]",
        description="JSON array of done faces: [{'student_id': 101, 'embedding': [...]}]"
    ),
    threshold: Optional[float] = Form(
        None,
        description="Optional similarity threshold (defaults to SIMILARITY_THRESHOLD)"
    )
):
    """
    Called by Node.js API.
    1. Reads and decodes image.
    2. Runs InsightFace Buffalo model to detect single face & extract 512-D vector.
    3. Compares cosine similarity against array of existing faces passed from MySQL.
    4. Returns duplicate status and 512-D embedding.
    """
    # 1. Read and decode image
    contents = await image.read()
    nparr = np.frombuffer(contents, np.uint8)
    img_cv = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

    if img_cv is None:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail={"success": False, "message": "Invalid image file or unable to decode image"}
        )

    # 2. Extract 512-D embedding
    try:
        new_embedding = get_face_embedding(img_cv)
    except ValueError as err:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail={"success": False, "message": str(err)}
        )

    embedding_list = new_embedding.tolist()

    # 3. Parse existing DONE faces passed from Node.js
    done_records = []
    if existing_faces:
        try:
            done_records = json.loads(existing_faces)
        except Exception:
            done_records = []

    # 4. Check similarity against done faces
    applied_threshold = threshold if threshold is not None else SIMILARITY_THRESHOLD
    is_duplicate = False
    matched_student_id = None
    max_similarity = None

    highest_sim = -1.0
    best_match_id = None

    for rec in done_records:
        stored_emb_raw = rec.get("embedding")
        if not stored_emb_raw:
            continue

        if isinstance(stored_emb_raw, str):
            try:
                stored_emb = np.array(json.loads(stored_emb_raw), dtype=np.float32)
            except Exception:
                continue
        elif isinstance(stored_emb_raw, (list, tuple)):
            stored_emb = np.array(stored_emb_raw, dtype=np.float32)
        else:
            continue

        sim = calculate_similarity(new_embedding, stored_emb)
        if sim > highest_sim:
            highest_sim = sim
            best_match_id = rec.get("student_id")

    if highest_sim >= applied_threshold and best_match_id is not None:
        is_duplicate = True
        matched_student_id = best_match_id
        max_similarity = round(highest_sim, 7)

    return {
        "success": True,
        "duplicate": is_duplicate,
        "matched_student_id": matched_student_id,
        "similarity": max_similarity,
        "embedding": embedding_list,
        "message": "Duplicate face detected" if is_duplicate else "Face is not duplicated"
    }


# ==============================================================================
# 5. Direct Vector Comparison Endpoint (JSON Body)
# ==============================================================================
@app.post(
    "/api/face/compare-embeddings",
    summary="Compare a target embedding against a list of embeddings"
)
def compare_embeddings(req: CompareEmbeddingsRequest):
    applied_threshold = req.threshold if req.threshold is not None else SIMILARITY_THRESHOLD
    target_emb = np.array(req.target_embedding, dtype=np.float32)

    is_duplicate = False
    matched_student_id = None
    max_similarity = None

    highest_sim = -1.0
    best_match_id = None

    for rec in req.done_faces:
        stored_emb_raw = rec.get("embedding")
        if not stored_emb_raw:
            continue

        if isinstance(stored_emb_raw, str):
            try:
                stored_emb = np.array(json.loads(stored_emb_raw), dtype=np.float32)
            except Exception:
                continue
        elif isinstance(stored_emb_raw, (list, tuple)):
            stored_emb = np.array(stored_emb_raw, dtype=np.float32)
        else:
            continue

        sim = calculate_similarity(target_emb, stored_emb)
        if sim > highest_sim:
            highest_sim = sim
            best_match_id = rec.get("student_id")

    if highest_sim >= applied_threshold and best_match_id is not None:
        is_duplicate = True
        matched_student_id = best_match_id
        max_similarity = round(highest_sim, 7)

    return {
        "success": True,
        "duplicate": is_duplicate,
        "matched_student_id": matched_student_id,
        "similarity": max_similarity,
        "message": "Duplicate face detected" if is_duplicate else "Face is not duplicated"
    }


if __name__ == "__main__":
    import uvicorn
    port = int(os.getenv("PORT", 8000))
    uvicorn.run("app:app", host="0.0.0.0", port=port, reload=True)
