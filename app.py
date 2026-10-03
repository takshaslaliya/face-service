import os
import sys
import json
import uuid
from typing import List, Optional
from pathlib import Path

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
    title="Face AI Attendance Microservice",
    description="Face Verification, Vector Extraction, and Duplicate Attendance Detection API",
    version="2.1.0",
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

# Configuration from Environment Variables
SIMILARITY_THRESHOLD = float(os.getenv("SIMILARITY_THRESHOLD", "0.50"))
NODE_API_BASE_URL = os.getenv("NODE_API_BASE_URL", "https://attendentsnews.hpys.in").rstrip("/")
UPLOAD_BASE_DIR = Path(os.getenv("UPLOAD_DIR", "uploads/faces"))
UPLOAD_BASE_DIR.mkdir(parents=True, exist_ok=True)

# Preload Face Model on Startup
@app.on_event("startup")
def startup_event():
    try:
        from face_functions import get_face_app
        print("[INFO] Preloading face model on startup...")
        get_face_app()
        print("[INFO] Face model loaded successfully.")
    except Exception as e:
        print(f"[WARN] Startup model preload failed: {e}")


# ==============================================================================
# 2. Schemas
# ==============================================================================
class CompareEmbeddingsRequest(BaseModel):
    target_embedding: List[float]
    done_faces: List[dict]  # list of {"student_id": int, "embedding": list[float]}
    threshold: Optional[float] = None

class UpdateStatusRequest(BaseModel):
    status: str


# ==============================================================================
# 3. Database & Node API Sync Helpers
# ==============================================================================
async def fetch_session_done_faces(session_id: int) -> List[dict]:
    """
    Fetches all DONE faces for the active attendance session from Node.js production backend or MySQL.
    """
    done_faces = []
    # 1. Try querying Node.js backend API
    try:
        async with httpx.AsyncClient(timeout=10.0) as client:
            resp = await client.get(f"{NODE_API_BASE_URL}/api/face/attendance/session/{session_id}")
            if resp.status_code == 200:
                data = resp.json()
                done_faces = data.get("data", data.get("faces", []))
                return done_faces
    except Exception as e:
        print(f"[NOTE] Could not fetch done faces from Node API ({e}). Checking local DB fallback...")

    # 2. Try querying MySQL if DB credentials configured
    db_host = os.getenv("DB_HOST")
    db_user = os.getenv("DB_USER")
    db_name = os.getenv("DB_NAME")
    if db_host and db_user and db_name:
        try:
            import mysql.connector
            conn = mysql.connector.connect(
                host=db_host,
                port=int(os.getenv("DB_PORT", 3306)),
                user=db_user,
                password=os.getenv("DB_PASSWORD", ""),
                database=db_name
            )
            cursor = conn.cursor(dictionary=True)
            cursor.execute(
                "SELECT id, student_id, embedding FROM attendance_faces WHERE attendance_session_id = %s AND status = 'done'",
                (session_id,)
            )
            done_faces = cursor.fetchall()
            cursor.close()
            conn.close()
        except Exception as err:
            print(f"[NOTE] Local DB query skipped: {err}")

    return done_faces


async def save_attendance_record_to_db(
    attendance_session_id: int,
    student_id: int,
    image_path: str,
    embedding: list,
    is_duplicate: bool,
    matched_student_id: Optional[int],
    similarity: Optional[float],
    status: str = "pending"
) -> Optional[int]:
    """
    Saves the attendance record into the production database table `attendance_faces`.
    """
    record_id = None
    payload = {
        "attendance_session_id": attendance_session_id,
        "student_id": student_id,
        "image_path": image_path,
        "embedding": embedding,
        "duplicated": "yes" if is_duplicate else "no",
        "dupl_std_id": matched_student_id,
        "similarity": similarity,
        "status": status
    }

    # 1. Forward to Node.js Production API
    try:
        async with httpx.AsyncClient(timeout=15.0) as client:
            resp = await client.post(
                f"{NODE_API_BASE_URL}/api/face/attendance",
                json=payload
            )
            if resp.status_code in (200, 201):
                data = resp.json()
                record_id = data.get("record_id", data.get("data", {}).get("id"))
                if record_id:
                    return int(record_id)
    except Exception as e:
        print(f"[NOTE] Node API save call failed: {e}. Trying direct DB insert...")

    # 2. Fallback: Direct MySQL insert if DB credentials exist
    db_host = os.getenv("DB_HOST")
    db_user = os.getenv("DB_USER")
    db_name = os.getenv("DB_NAME")
    if db_host and db_user and db_name:
        try:
            import mysql.connector
            conn = mysql.connector.connect(
                host=db_host,
                port=int(os.getenv("DB_PORT", 3306)),
                user=db_user,
                password=os.getenv("DB_PASSWORD", ""),
                database=db_name
            )
            cursor = conn.cursor()
            insert_query = """
                INSERT INTO attendance_faces 
                (attendance_session_id, student_id, image_path, embedding, status, duplicated, dupl_std_id, similarity)
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
            """
            cursor.execute(
                insert_query,
                (
                    attendance_session_id,
                    student_id,
                    image_path,
                    json.dumps(embedding),
                    status,
                    "yes" if is_duplicate else "no",
                    matched_student_id,
                    similarity
                )
            )
            conn.commit()
            record_id = cursor.lastrowid
            cursor.close()
            conn.close()
        except Exception as err:
            print(f"[WARN] Direct MySQL insert failed: {err}")

    return record_id


# ==============================================================================
# 4. Health & Status Endpoints
# ==============================================================================
@app.get("/", include_in_schema=False)
def root():
    return RedirectResponse(url="/docs")


@app.get("/api/health", summary="Health Check")
@app.get("/health", include_in_schema=False)
def health_check():
    return {
        "success": True,
        "service": "Face AI Attendance Microservice",
        "status": "healthy",
        "docs": "/docs",
        "node_api_url": NODE_API_BASE_URL
    }


# ==============================================================================
# 5. Main Attendance Endpoint: Direct Call with Auto-Save + Match with Done Faces
# ==============================================================================
@app.post(
    "/api/face/attendance",
    summary="Check student face, detect duplicates against DONE faces, and save to database"
)
async def process_face_attendance(
    attendance_session_id: int = Form(..., description="ID of the active attendance session"),
    student_id: int = Form(..., description="ID of the student submitting attendance"),
    image: UploadFile = File(..., description="Student face image (JPEG/PNG)"),
    threshold: Optional[float] = Form(None, description="Optional similarity threshold (defaults to 0.50)")
):
    """
    Direct API for Attendance:
    1. Reads and decodes image.
    2. Validates single face rule & extracts 512-D vector via InsightFace.
    3. Saves image file to disk (`uploads/faces/<session_id>/<student_id>_<uuid>.jpg`).
    4. Fetches all DONE faces in the session and calculates Cosine Similarity.
    5. Flags duplicate = True if similarity >= threshold and matches existing student_id.
    6. Automatically stores the record into the `attendance_faces` database table.
    """
    # 1. Read & decode image
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

    # 3. Save image to disk
    session_dir = UPLOAD_BASE_DIR / str(attendance_session_id)
    session_dir.mkdir(parents=True, exist_ok=True)
    filename = f"{student_id}_{uuid.uuid4().hex[:8]}.jpg"
    image_rel_path = f"uploads/faces/{attendance_session_id}/{filename}"
    image_full_path = session_dir / filename
    try:
        with open(image_full_path, "wb") as f:
            f.write(contents)
    except Exception as e:
        print(f"[WARN] Failed saving image file to disk: {e}")

    # 4. Fetch all DONE faces in this attendance session
    done_records = await fetch_session_done_faces(attendance_session_id)

    # 5. Perform Cosine Similarity against all done faces
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

    # 6. Auto-save record into database table `attendance_faces`
    record_id = await save_attendance_record_to_db(
        attendance_session_id=attendance_session_id,
        student_id=student_id,
        image_path=image_rel_path,
        embedding=embedding_list,
        is_duplicate=is_duplicate,
        matched_student_id=matched_student_id,
        similarity=max_similarity,
        status="pending"
    )

    return {
        "success": True,
        "record_id": record_id,
        "attendance_session_id": attendance_session_id,
        "student_id": student_id,
        "duplicate": is_duplicate,
        "matched_student_id": matched_student_id,
        "similarity": max_similarity,
        "status": "pending",
        "image_path": image_rel_path,
        "message": "Duplicate face detected" if is_duplicate else "Face is not duplicated",
        "embedding": embedding_list
    }


# ==============================================================================
# 6. Detect and Compare (with Explicit existing_faces passed)
# ==============================================================================
@app.post(
    "/api/face/detect-and-compare",
    summary="Extract face embedding and compare against passed existing_faces"
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
    contents = await image.read()
    nparr = np.frombuffer(contents, np.uint8)
    img_cv = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

    if img_cv is None:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail={"success": False, "message": "Invalid image file or unable to decode image"}
        )

    try:
        new_embedding = get_face_embedding(img_cv)
    except ValueError as err:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail={"success": False, "message": str(err)}
        )

    embedding_list = new_embedding.tolist()

    done_records = []
    if existing_faces:
        try:
            done_records = json.loads(existing_faces)
        except Exception:
            done_records = []

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
# 7. Compare Embeddings (JSON Body)
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
