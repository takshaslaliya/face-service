-- ==============================================================================
-- 🏢 HAMS Face Attendance Table Schema
-- ==============================================================================

CREATE TABLE IF NOT EXISTS attendance_faces (
    id BIGINT UNSIGNED AUTO_INCREMENT PRIMARY KEY,

    attendance_session_id BIGINT UNSIGNED NOT NULL,

    student_id BIGINT UNSIGNED NOT NULL,

    image_path VARCHAR(500) NULL,

    embedding JSON NOT NULL,

    status ENUM('pending', 'done')
        NOT NULL DEFAULT 'pending',

    duplicated ENUM('yes', 'no')
        NOT NULL DEFAULT 'no',

    dupl_std_id BIGINT UNSIGNED NULL,

    similarity DECIMAL(10,7) NULL,

    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,

    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        ON UPDATE CURRENT_TIMESTAMP,

    INDEX idx_session_status (
        attendance_session_id,
        status
    ),

    INDEX idx_student_id (student_id),

    INDEX idx_duplicate_student (dupl_std_id)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci;
