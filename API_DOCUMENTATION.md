# 🚀 Face Attendance & Duplicate Detection API Documentation

This service provides facial verification, embedding extraction, and session-scoped duplicate detection for attendance sessions using InsightFace (`buffalo_l`) and MySQL.

---

## 🗄️ Database Table

```sql
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
```

---

## 📡 API Overview

| Method | Endpoint | Description | Content-Type |
|---|---|---|---|
| `POST` | `/api/face/attendance` | Detect single face, check session duplicates against `done` faces, store as `pending` | `multipart/form-data` |
| `GET` | `/api/face/attendance/{id}` | Get attendance face record by ID | `application/json` |
| `PUT` | `/api/face/attendance/{id}` | Update record status (e.g., `pending` ➔ `done`) | `application/json` |

---

## 1. POST — Check Face + Store Attendance Result

### **Endpoint:**
`POST /api/face/attendance`

### **Content-Type:**
`multipart/form-data`

### **Request Parameters (Form Data):**

| Field | Type | Required | Description | Example |
|---|---|---|---|---|
| `attendance_session_id` | `integer` | Yes | Active attendance session ID | `55` |
| `student_id` | `integer` | Yes | Student ID submitting attendance | `105` |
| `image` | `file (binary)` | Yes | JPEG / PNG photo of student | `student.jpg` |

---

### **cURL Example:**
```bash
curl -X POST "http://localhost:8000/api/face/attendance" \
  -F "attendance_session_id=55" \
  -F "student_id=105" \
  -F "image=@/path/to/student.jpg"
```

---

### **Responses:**

#### 🟢 Case A: No Duplicate Face Detected (New Unique Student)
**Status Code:** `200 OK`
```json
{
  "success": true,
  "record_id": 10,
  "student_id": 105,
  "duplicate": false,
  "matched_student_id": null,
  "similarity": null,
  "status": "pending",
  "message": "Face is not duplicated"
}
```

#### 🔴 Case B: Duplicate Face Detected (Another student used the same face)
**Status Code:** `200 OK`
```json
{
  "success": true,
  "record_id": 11,
  "student_id": 105,
  "duplicate": true,
  "matched_student_id": 101,
  "similarity": 0.8213965,
  "status": "pending",
  "message": "Duplicate face detected"
}
```

#### ⚠️ Case C: No Face Found in Image
**Status Code:** `400 Bad Request`
```json
{
  "detail": {
    "success": false,
    "message": "No face detected in the provided image"
  }
}
```

#### ⚠️ Case D: Multiple Faces Found in Image
**Status Code:** `400 Bad Request`
```json
{
  "detail": {
    "success": false,
    "message": "Multiple faces detected (2 found). Only single face is allowed"
  }
}
```

---

## 2. GET — Get Attendance Face Record

### **Endpoint:**
`GET /api/face/attendance/{id}`

### **cURL Example:**
```bash
curl -X GET "http://localhost:8000/api/face/attendance/11" \
  -H "Accept: application/json"
```

---

### **Response:**
**Status Code:** `200 OK`
```json
{
  "success": true,
  "data": {
    "id": 11,
    "attendance_session_id": 55,
    "student_id": 105,
    "image_path": "attendance/55/105_3a8b9f12.jpg",
    "status": "pending",
    "duplicated": "yes",
    "dupl_std_id": 101,
    "similarity": 0.8213965,
    "created_at": "2026-10-03 01:25:00",
    "updated_at": "2026-10-03 01:25:00"
  }
}
```

---

## 3. PUT — Change Status (Pending ➔ Done)

### **Endpoint:**
`PUT /api/face/attendance/{id}`

### **Content-Type:**
`application/json`

### **Request Body (JSON):**
```json
{
  "status": "done"
}
```

---

### **cURL Example:**
```bash
curl -X PUT "http://localhost:8000/api/face/attendance/11" \
  -H "Content-Type: application/json" \
  -d '{"status": "done"}'
```

---

### **Response:**
**Status Code:** `200 OK`
```json
{
  "success": true,
  "message": "Attendance face record 11 updated successfully",
  "record_id": 11,
  "status": "done"
}
```

---

## 🚀 How to Run the Server

From the `python face detection` directory, execute:

```bash
# Activate virtual environment
venv\Scripts\activate

# Run FastAPI server with Uvicorn
python api_server.py
```

The interactive Swagger UI documentation will be available at:
👉 **`http://localhost:8000/docs`**
