import os
import io
import json
import uuid
from typing import Optional
from pathlib import Path

import cv2
import numpy as np
import mysql.connector
from mysql.connector import pooling
from fastapi import FastAPI, File, UploadFile, Form, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from insightface.app import FaceAnalysis

# ==============================================================================
# 1. Configuration & Database Connection Pool
# ==============================================================================
DB_HOST = os.getenv("DB_HOST", "localhost")
DB_USER = os.getenv("DB_USER", "root")
DB_PASSWORD = os.getenv("DB_PASSWORD", "")
DB_NAME = os.getenv("DB_NAME", "esp32_attendent")
DB_PORT = int(os.getenv("DB_PORT", 3306))

SIMILARITY_THRESHOLD = float(os.getenv("SIMILARITY_THRESHOLD", 0.50))
UPLOAD_DIR = Path("uploads/attendance")
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)

# Database Connection Helper
db_pool = None

def get_db_pool():
    global db_pool
    if db_pool is None:
        try:
            db_pool = mysql.connector.pooling.MySQLConnectionPool(
                pool_name="face_att_pool",
                pool_size=10,
                pool_reset_session=True,
                host=DB_HOST,
                user=DB_USER,
                password=DB_PASSWORD,
                database=DB_NAME,
                port=DB_PORT
            )
        except Exception as e:
            print(f"[WARNING] Could not create MySQL connection pool: {e}")
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail={"success": False, "message": f"Database connection unavailable: {str(e)}"}
            )
    return db_pool

def get_db_connection():
    pool = get_db_pool()
    return pool.get_connection()

# Initialize Table
def init_db():
    try:
        conn = get_db_connection()
        cursor = conn.cursor()
        create_table_query = """
        CREATE TABLE IF NOT EXISTS attendance_faces (
            id BIGINT UNSIGNED AUTO_INCREMENT PRIMARY KEY,
            attendance_session_id BIGINT UNSIGNED NOT NULL,
            student_id BIGINT UNSIGNED NOT NULL,
            image_path VARCHAR(500) NULL,
            embedding JSON NOT NULL,
            status ENUM('pending', 'done') NOT NULL DEFAULT 'pending',
            duplicated ENUM('yes', 'no') NOT NULL DEFAULT 'no',
            dupl_std_id BIGINT UNSIGNED NULL,
            similarity DECIMAL(10,7) NULL,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP,
            INDEX idx_session_status (attendance_session_id, status),
            INDEX idx_student_id (student_id),
            INDEX idx_duplicate_student (dupl_std_id)
        ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;
        """
        cursor.execute(create_table_query)
        conn.commit()
        cursor.close()
        conn.close()
        print("[INFO] Database table `attendance_faces` verified/initialized successfully.")
    except Exception as e:
        print(f"[NOTE] Database auto-init skipped (will connect on request): {e}")

try:
    init_db()
except Exception:
    pass

# ==============================================================================
# 2. InsightFace Model Preparation
# ==============================================================================
print("[INFO] Initializing InsightFace Buffalo_L Model...")
face_app = FaceAnalysis(name="buffalo_l")
face_app.prepare(ctx_id=0, det_size=(640, 640))
print("[INFO] InsightFace Model Ready.")

# ==============================================================================
# 3. FastAPI Application Initialization
# ==============================================================================
app = FastAPI(
    title="HAMS Face Attendance & Duplicate Verification API",
    version="1.0.0",
    description="Facial embedding extraction, session-based duplicate detection, and attendance verification service."
)

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ==============================================================================
# 4. Helper Functions
# ==============================================================================
def calculate_cosine_similarity(emb_a: np.ndarray, emb_b: np.ndarray) -> float:
    norm_a = np.linalg.norm(emb_a)
    norm_b = np.linalg.norm(emb_b)
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return float(np.dot(emb_a, emb_b) / (norm_a * norm_b))

class UpdateStatusRequest(BaseModel):
    status: str
    duplicated: Optional[str] = None
    dupl_std_id: Optional[int] = None
    similarity: Optional[float] = None

# ==============================================================================
# 5. API Endpoints
# ==============================================================================

# ------------------------------------------------------------------------------
# API 1: POST /api/face/attendance
# Main Endpoint: Extract embedding, check duplicates against DONE records in session, store record
# ------------------------------------------------------------------------------
@app.post("/api/face/attendance", summary="Check face + store attendance result")
async def process_face_attendance(
    attendance_session_id: int = Form(..., description="ID of the attendance session"),
    student_id: int = Form(..., description="ID of the student submitting attendance"),
    image: UploadFile = File(..., description="Image file containing student face (multipart/form-data)")
):
    # 1. Read and decode uploaded image
    contents = await image.read()
    nparr = np.frombuffer(contents, np.uint8)
    img_cv = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

    if img_cv is None:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail={"success": False, "message": "Invalid image file or unable to decode image"}
        )

    # 2. Detect face and validate single face rule
    faces = face_app.get(img_cv)
    if len(faces) == 0:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail={"success": False, "message": "No face detected in the provided image"}
        )
    if len(faces) > 1:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail={"success": False, "message": f"Multiple faces detected ({len(faces)} found). Only single face is allowed"}
        )

    # 3. Extract 512-D embedding
    new_embedding = faces[0].embedding
    embedding_list = new_embedding.tolist()

    # 4. Save image to disk
    session_upload_dir = UPLOAD_DIR / str(attendance_session_id)
    session_upload_dir.mkdir(parents=True, exist_ok=True)
    filename = f"{student_id}_{uuid.uuid4().hex[:8]}.jpg"
    image_rel_path = f"attendance/{attendance_session_id}/{filename}"
    image_full_path = session_upload_dir / filename
    with open(image_full_path, "wb") as f:
        f.write(contents)

    # 5. Retrieve all DONE faces in this attendance session
    conn = get_db_connection()
    cursor = conn.cursor(dictionary=True)
    try:
        select_done_query = """
            SELECT id, student_id, embedding
            FROM attendance_faces
            WHERE attendance_session_id = %s
              AND status = 'done'
        """
        cursor.execute(select_done_query, (attendance_session_id,))
        done_records = cursor.fetchall()

        # 6. Compare with all DONE faces for duplicate detection
        is_duplicate = False
        matched_student_id = None
        max_similarity = None

        highest_sim = -1.0
        best_match_std = None

        for rec in done_records:
            stored_emb_raw = rec["embedding"]
            if isinstance(stored_emb_raw, str):
                stored_emb = np.array(json.loads(stored_emb_raw), dtype=np.float32)
            elif isinstance(stored_emb_raw, (list, tuple)):
                stored_emb = np.array(stored_emb_raw, dtype=np.float32)
            else:
                continue

            sim = calculate_cosine_similarity(new_embedding, stored_emb)
            if sim > highest_sim:
                highest_sim = sim
                best_match_std = rec["student_id"]

        if highest_sim >= SIMILARITY_THRESHOLD and best_match_std is not None:
            is_duplicate = True
            matched_student_id = best_match_std
            max_similarity = round(highest_sim, 7)

        # 7. Insert new record with status = 'pending'
        insert_query = """
            INSERT INTO attendance_faces (
                attendance_session_id,
                student_id,
                image_path,
                embedding,
                status,
                duplicated,
                dupl_std_id,
                similarity
            ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
        """
        duplicated_str = "yes" if is_duplicate else "no"
        cursor.execute(
            insert_query,
            (
                attendance_session_id,
                student_id,
                image_rel_path,
                json.dumps(embedding_list),
                "pending",
                duplicated_str,
                matched_student_id,
                max_similarity
            )
        )
        conn.commit()
        record_id = cursor.lastrowid

        # 8. Return response
        return {
            "success": True,
            "record_id": record_id,
            "student_id": student_id,
            "duplicate": is_duplicate,
            "matched_student_id": matched_student_id,
            "similarity": max_similarity,
            "status": "pending",
            "message": "Duplicate face detected" if is_duplicate else "Face is not duplicated"
        }
    finally:
        cursor.close()
        conn.close()


# ------------------------------------------------------------------------------
# API 2: GET /api/face/attendance/{id}
# Retrieve a specific attendance face record by ID
# ------------------------------------------------------------------------------
@app.get("/api/face/attendance/{record_id}", summary="Get attendance face record by ID")
async def get_face_attendance_record(record_id: int):
    conn = get_db_connection()
    cursor = conn.cursor(dictionary=True)
    try:
        query = """
            SELECT id, attendance_session_id, student_id, image_path,
                   status, duplicated, dupl_std_id, similarity, created_at, updated_at
            FROM attendance_faces
            WHERE id = %s
        """
        cursor.execute(query, (record_id,))
        record = cursor.fetchone()

        if not record:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail={"success": False, "message": f"Attendance face record {record_id} not found"}
            )

        if record.get("similarity") is not None:
            record["similarity"] = float(record["similarity"])
        if record.get("created_at") is not None:
            record["created_at"] = str(record["created_at"])
        if record.get("updated_at") is not None:
            record["updated_at"] = str(record["updated_at"])

        return {
            "success": True,
            "data": record
        }
    finally:
        cursor.close()
        conn.close()


# ------------------------------------------------------------------------------
# API 3: PUT /api/face/attendance/{id}
# Update status (e.g. pending -> done) of attendance face record
# ------------------------------------------------------------------------------
@app.put("/api/face/attendance/{record_id}", summary="Update status / edit attendance face record")
async def update_face_attendance_status(record_id: int, payload: UpdateStatusRequest):
    if payload.status not in ["pending", "done"]:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail={"success": False, "message": "Status must be either 'pending' or 'done'"}
        )

    conn = get_db_connection()
    cursor = conn.cursor(dictionary=True)
    try:
        # Verify record exists
        cursor.execute("SELECT id FROM attendance_faces WHERE id = %s", (record_id,))
        if not cursor.fetchone():
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail={"success": False, "message": f"Attendance face record {record_id} not found"}
            )

        # Build update query dynamically
        update_fields = ["status = %s"]
        params = [payload.status]

        if payload.duplicated is not None:
            update_fields.append("duplicated = %s")
            params.append(payload.duplicated)
        if payload.dupl_std_id is not None:
            update_fields.append("dupl_std_id = %s")
            params.append(payload.dupl_std_id)
        if payload.similarity is not None:
            update_fields.append("similarity = %s")
            params.append(payload.similarity)

        params.append(record_id)
        update_query = f"UPDATE attendance_faces SET {', '.join(update_fields)} WHERE id = %s"
        cursor.execute(update_query, tuple(params))
        conn.commit()

        return {
            "success": True,
            "message": f"Attendance face record {record_id} updated successfully",
            "record_id": record_id,
            "status": payload.status
        }
    finally:
        cursor.close()
        conn.close()


# ------------------------------------------------------------------------------
# Health Check Endpoint
# ------------------------------------------------------------------------------
@app.get("/health", summary="Health check")
def health_check():
    return {"status": "ok", "service": "hams-face-attendance-api"}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("api_server:app", host="0.0.0.0", port=8000, reload=True)
