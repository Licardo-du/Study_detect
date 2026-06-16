import json
import sqlite3
from contextlib import contextmanager
from datetime import datetime
from pathlib import Path


BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "project_data"
DEFAULT_DB_PATH = DATA_DIR / "study_monitor.db"


def now_text():
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


class Database:
    def __init__(self, db_path=DEFAULT_DB_PATH):
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self.init_tables()

    @contextmanager
    def connect(self):
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        try:
            yield conn
            conn.commit()
        except Exception:
            conn.rollback()
            raise
        finally:
            conn.close()

    def init_tables(self):
        with self.connect() as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS users (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    username TEXT NOT NULL UNIQUE,
                    password_hash TEXT NOT NULL,
                    salt TEXT NOT NULL,
                    role TEXT NOT NULL CHECK(role IN ('admin', 'user')),
                    created_at TEXT NOT NULL
                )
            """)
            conn.execute("""
                CREATE TABLE IF NOT EXISTS operation_logs (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    user_id INTEGER,
                    action TEXT NOT NULL,
                    detail TEXT,
                    created_at TEXT NOT NULL,
                    FOREIGN KEY(user_id) REFERENCES users(id)
                )
            """)
            conn.execute("""
                CREATE TABLE IF NOT EXISTS detection_records (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    user_id INTEGER,
                    source TEXT NOT NULL,
                    summary_json TEXT NOT NULL,
                    alerts_json TEXT NOT NULL,
                    output_path TEXT,
                    created_at TEXT NOT NULL,
                    FOREIGN KEY(user_id) REFERENCES users(id)
                )
            """)
            conn.execute("""
                CREATE TABLE IF NOT EXISTS model_resources (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    name TEXT NOT NULL UNIQUE,
                    url TEXT,
                    local_path TEXT,
                    status TEXT NOT NULL,
                    updated_at TEXT NOT NULL
                )
            """)
            conn.execute("""
                CREATE TABLE IF NOT EXISTS misclassified_samples (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    image_path TEXT NOT NULL,
                    original_predictions TEXT,
                    corrected_annotations TEXT,
                    user_id INTEGER,
                    is_reviewed INTEGER DEFAULT 0,
                    relabel_image_path TEXT,
                    relabel_label_path TEXT,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL,
                    FOREIGN KEY(user_id) REFERENCES users(id)
                )
            """)
            conn.execute("""
                CREATE TABLE IF NOT EXISTS training_cycles (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    version TEXT UNIQUE NOT NULL,
                    base_dataset TEXT,
                    additional_sample_count INTEGER DEFAULT 0,
                    mAP REAL,
                    status TEXT DEFAULT 'pending',
                    started_at TEXT,
                    finished_at TEXT,
                    model_path TEXT,
                    created_by INTEGER,
                    created_at TEXT NOT NULL,
                    FOREIGN KEY(created_by) REFERENCES users(id)
                )
            """)
            conn.execute("""
                CREATE TABLE IF NOT EXISTS data_reflux_log (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    sample_id INTEGER,
                    source_image_path TEXT,
                    target_image_path TEXT,
                    target_label_path TEXT,
                    operation TEXT DEFAULT 'relabel',
                    user_id INTEGER,
                    created_at TEXT NOT NULL,
                    FOREIGN KEY(sample_id) REFERENCES misclassified_samples(id),
                    FOREIGN KEY(user_id) REFERENCES users(id)
                )
            """)
            conn.execute("""
                CREATE TABLE IF NOT EXISTS study_sessions (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    user_id INTEGER NOT NULL,
                    session_name TEXT,
                    start_time TEXT NOT NULL,
                    end_time TEXT,
                    total_duration_seconds INTEGER,
                    effective_study_seconds INTEGER,
                    focus_score INTEGER CHECK(focus_score BETWEEN 0 AND 100),
                    total_alerts INTEGER DEFAULT 0,
                    phone_alerts INTEGER DEFAULT 0,
                    sleep_alerts INTEGER DEFAULT 0,
                    eat_alerts INTEGER DEFAULT 0,
                    notes TEXT,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL,
                    FOREIGN KEY(user_id) REFERENCES users(id) ON DELETE CASCADE
                )
            """)
            conn.execute("""
                CREATE TABLE IF NOT EXISTS study_behavior_records (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    user_id INTEGER NOT NULL,
                    session_id INTEGER,
                    timestamp TEXT NOT NULL,
                    behavior_type TEXT NOT NULL CHECK(behavior_type IN ('normal', 'phone', 'sleep', 'eat', 'other')),
                    confidence REAL NOT NULL,
                    is_alert INTEGER NOT NULL DEFAULT 0,
                    alert_reason TEXT,
                    source_type TEXT NOT NULL DEFAULT 'image' CHECK(source_type IN ('image', 'camera', 'video')),
                    source_path TEXT,
                    image_path TEXT,
                    output_image_path TEXT,
                    extra_info TEXT,
                    created_at TEXT NOT NULL,
                    FOREIGN KEY(user_id) REFERENCES users(id) ON DELETE CASCADE,
                    FOREIGN KEY(session_id) REFERENCES study_sessions(id) ON DELETE SET NULL
                )
            """)
            conn.execute("CREATE INDEX IF NOT EXISTS idx_behavior_user_time ON study_behavior_records(user_id, timestamp)")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_behavior_type ON study_behavior_records(behavior_type)")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_sessions_user_time ON study_sessions(user_id, start_time)")

    def create_user(self, username, password_hash, salt, role="user"):
        with self.connect() as conn:
            cursor = conn.execute(
                "INSERT INTO users(username, password_hash, salt, role, created_at) VALUES (?, ?, ?, ?, ?)",
                (username, password_hash, salt, role, now_text())
            )
            return cursor.lastrowid

    def get_user_by_username(self, username):
        with self.connect() as conn:
            row = conn.execute("SELECT * FROM users WHERE username = ?", (username,)).fetchone()
            return dict(row) if row else None

    def get_user_by_id(self, user_id):
        with self.connect() as conn:
            row = conn.execute("SELECT * FROM users WHERE id = ?", (user_id,)).fetchone()
            return dict(row) if row else None

    def list_users(self):
        with self.connect() as conn:
            rows = conn.execute(
                "SELECT id, username, role, created_at FROM users ORDER BY id"
            ).fetchall()
            return [dict(row) for row in rows]

    def update_user_password(self, username, password_hash, salt):
        with self.connect() as conn:
            cursor = conn.execute(
                "UPDATE users SET password_hash = ?, salt = ? WHERE username = ?",
                (password_hash, salt, username)
            )
            return cursor.rowcount

    def update_user_role(self, username, new_role):
        with self.connect() as conn:
            cursor = conn.execute(
                "UPDATE users SET role = ? WHERE username = ?",
                (new_role, username)
            )
            return cursor.rowcount

    def delete_user(self, username):
        with self.connect() as conn:
            cursor = conn.execute("DELETE FROM users WHERE username = ?", (username,))
            return cursor.rowcount

    def user_exists(self, username):
        with self.connect() as conn:
            row = conn.execute("SELECT 1 FROM users WHERE username = ?", (username,)).fetchone()
            return row is not None

    def log_operation(self, user_id, action, detail=""):
        with self.connect() as conn:
            cursor = conn.execute(
                "INSERT INTO operation_logs(user_id, action, detail, created_at) VALUES (?, ?, ?, ?)",
                (user_id, action, detail, now_text())
            )
            return cursor.lastrowid

    def list_operation_logs(self, limit=100):
        with self.connect() as conn:
            rows = conn.execute(
                """
                SELECT operation_logs.id, users.username, operation_logs.action,
                       operation_logs.detail, operation_logs.created_at
                FROM operation_logs
                LEFT JOIN users ON users.id = operation_logs.user_id
                ORDER BY operation_logs.id DESC
                LIMIT ?
                """,
                (limit,)
            ).fetchall()
            return [dict(row) for row in rows]

    def record_detection(self, user_id, source, summary, alerts, output_path=None):
        with self.connect() as conn:
            cursor = conn.execute(
                """
                INSERT INTO detection_records(user_id, source, summary_json, alerts_json, output_path, created_at)
                VALUES (?, ?, ?, ?, ?, ?)
                """,
                (
                    user_id,
                    source,
                    json.dumps(summary, ensure_ascii=False),
                    json.dumps(alerts, ensure_ascii=False),
                    str(output_path) if output_path else None,
                    now_text()
                )
            )
            return cursor.lastrowid

    def list_detection_records(self, limit=100):
        with self.connect() as conn:
            rows = conn.execute(
                """
                SELECT detection_records.id, users.username, detection_records.source,
                       detection_records.summary_json, detection_records.alerts_json,
                       detection_records.output_path, detection_records.created_at
                FROM detection_records
                LEFT JOIN users ON users.id = detection_records.user_id
                ORDER BY detection_records.id DESC
                LIMIT ?
                """,
                (limit,)
            ).fetchall()
            return [dict(row) for row in rows]

    def upsert_model_resource(self, name, url, local_path, status):
        with self.connect() as conn:
            conn.execute(
                """
                INSERT INTO model_resources(name, url, local_path, status, updated_at)
                VALUES (?, ?, ?, ?, ?)
                ON CONFLICT(name) DO UPDATE SET
                    url = excluded.url,
                    local_path = excluded.local_path,
                    status = excluded.status,
                    updated_at = excluded.updated_at
                """,
                (name, url, str(local_path) if local_path else None, status, now_text())
            )

    def list_model_resources(self):
        with self.connect() as conn:
            rows = conn.execute(
                "SELECT id, name, url, local_path, status, updated_at FROM model_resources ORDER BY id DESC"
            ).fetchall()
            return [dict(row) for row in rows]

    def delete_model_resource(self, name):
        with self.connect() as conn:
            cursor = conn.execute("DELETE FROM model_resources WHERE name = ?", (name,))
            return cursor.rowcount

    def insert_misclassified_sample(self, image_path, original_predictions, user_id):
        with self.connect() as conn:
            cursor = conn.execute(
                """
                INSERT INTO misclassified_samples (image_path, original_predictions, user_id, created_at, updated_at)
                VALUES (?, ?, ?, ?, ?)
                """,
                (image_path, json.dumps(original_predictions, ensure_ascii=False), user_id, now_text(), now_text())
            )
            return cursor.lastrowid

    def update_misclassified_corrected(self, sample_id, corrected_annotations):
        with self.connect() as conn:
            if not isinstance(corrected_annotations, str):
                corrected_annotations = json.dumps(corrected_annotations, ensure_ascii=False)
            cursor = conn.execute(
                "UPDATE misclassified_samples SET corrected_annotations = ?, updated_at = ? WHERE id = ?",
                (corrected_annotations, now_text(), sample_id)
            )
            return cursor.rowcount

    def get_misclassified_sample(self, sample_id):
        with self.connect() as conn:
            row = conn.execute("SELECT * FROM misclassified_samples WHERE id = ?", (sample_id,)).fetchone()
            return dict(row) if row else None

    def list_misclassified_samples(self, reviewed_only=False):
        with self.connect() as conn:
            if reviewed_only:
                rows = conn.execute(
                    "SELECT * FROM misclassified_samples WHERE is_reviewed = 1 ORDER BY created_at DESC"
                ).fetchall()
            else:
                rows = conn.execute("SELECT * FROM misclassified_samples ORDER BY created_at DESC").fetchall()
            return [dict(row) for row in rows]

    def update_misclassified_relabel_paths(self, sample_id, relabel_image_path, relabel_label_path):
        with self.connect() as conn:
            cursor = conn.execute(
                """
                UPDATE misclassified_samples
                SET relabel_image_path = ?, relabel_label_path = ?, is_reviewed = 1, updated_at = ?
                WHERE id = ?
                """,
                (relabel_image_path, relabel_label_path, now_text(), sample_id)
            )
            return cursor.rowcount

    def delete_misclassified_sample(self, sample_id):
        with self.connect() as conn:
            cursor = conn.execute("DELETE FROM misclassified_samples WHERE id = ?", (sample_id,))
            return cursor.rowcount

    def create_training_cycle(self, version, base_dataset, created_by):
        with self.connect() as conn:
            cursor = conn.execute(
                "INSERT INTO training_cycles (version, base_dataset, created_by, created_at) VALUES (?, ?, ?, ?)",
                (version, base_dataset, created_by, now_text())
            )
            return cursor.lastrowid

    def update_training_cycle_status(self, cycle_id, status, mAP=None, model_path=None):
        with self.connect() as conn:
            updates = ["status = ?"]
            params = [status]
            if mAP is not None:
                updates.append("mAP = ?")
                params.append(mAP)
            if model_path is not None:
                updates.append("model_path = ?")
                params.append(model_path)
            if status == "completed":
                updates.append("finished_at = ?")
                params.append(now_text())
            elif status == "training":
                updates.append("started_at = ?")
                params.append(now_text())
            params.append(cycle_id)
            sql = f"UPDATE training_cycles SET {', '.join(updates)} WHERE id = ?"
            cursor = conn.execute(sql, params)
            return cursor.rowcount

    def get_training_cycles(self):
        with self.connect() as conn:
            rows = conn.execute("SELECT * FROM training_cycles ORDER BY created_at DESC").fetchall()
            return [dict(row) for row in rows]

    def log_data_reflux(self, sample_id, source_image_path, target_image_path, target_label_path, user_id):
        with self.connect() as conn:
            cursor = conn.execute(
                """
                INSERT INTO data_reflux_log (sample_id, source_image_path, target_image_path, target_label_path, user_id, created_at)
                VALUES (?, ?, ?, ?, ?, ?)
                """,
                (sample_id, source_image_path, target_image_path, target_label_path, user_id, now_text())
            )
            return cursor.lastrowid

    def start_study_session(self, user_id, session_name=""):
        now = now_text()
        with self.connect() as conn:
            cursor = conn.execute(
                "INSERT INTO study_sessions (user_id, session_name, start_time, created_at, updated_at) VALUES (?, ?, ?, ?, ?)",
                (user_id, session_name, now, now, now)
            )
            return cursor.lastrowid

    def end_study_session(self, session_id, focus_score=None, effective_seconds=None, notes=None, alert_stats=None):
        with self.connect() as conn:
            row = conn.execute("SELECT start_time FROM study_sessions WHERE id = ?", (session_id,)).fetchone()
            if not row:
                return False
            start = datetime.strptime(row["start_time"], "%Y-%m-%d %H:%M:%S")
            duration = int((datetime.now() - start).total_seconds())
            updates = ["end_time = ?", "total_duration_seconds = ?", "updated_at = ?"]
            params = [now_text(), duration, now_text()]
            if focus_score is not None:
                updates.append("focus_score = ?")
                params.append(focus_score)
            if effective_seconds is not None:
                updates.append("effective_study_seconds = ?")
                params.append(effective_seconds)
            if notes is not None:
                updates.append("notes = ?")
                params.append(notes)
            if alert_stats:
                if "total" in alert_stats:
                    updates.append("total_alerts = ?")
                    params.append(alert_stats["total"])
                if "phone" in alert_stats:
                    updates.append("phone_alerts = ?")
                    params.append(alert_stats["phone"])
                if "sleep" in alert_stats:
                    updates.append("sleep_alerts = ?")
                    params.append(alert_stats["sleep"])
                if "eat" in alert_stats:
                    updates.append("eat_alerts = ?")
                    params.append(alert_stats["eat"])
            params.append(session_id)
            sql = f"UPDATE study_sessions SET {', '.join(updates)} WHERE id = ?"
            conn.execute(sql, params)
            return True

    def list_study_sessions(self, user_id=None, limit=50, offset=0):
        with self.connect() as conn:
            if user_id:
                rows = conn.execute(
                    "SELECT * FROM study_sessions WHERE user_id = ? ORDER BY start_time DESC LIMIT ? OFFSET ?",
                    (user_id, limit, offset)
                ).fetchall()
            else:
                rows = conn.execute(
                    "SELECT * FROM study_sessions ORDER BY start_time DESC LIMIT ? OFFSET ?",
                    (limit, offset)
                ).fetchall()
            return [dict(row) for row in rows]

    def add_behavior_record(self, user_id, behavior_type, confidence, session_id=None,
                            is_alert=False, alert_reason="", source_type="image",
                            source_path="", image_path="", output_image_path="", extra_info=None):
        timestamp = now_text()
        extra_json = json.dumps(extra_info, ensure_ascii=False) if extra_info else None
        with self.connect() as conn:
            cursor = conn.execute(
                """
                INSERT INTO study_behavior_records
                (user_id, session_id, timestamp, behavior_type, confidence, is_alert, alert_reason,
                 source_type, source_path, image_path, output_image_path, extra_info, created_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (user_id, session_id, timestamp, behavior_type, confidence,
                 1 if is_alert else 0, alert_reason,
                 source_type, source_path, image_path, output_image_path, extra_json, now_text())
            )
            return cursor.lastrowid

    def get_behavior_records(self, user_id=None, session_id=None, limit=100, offset=0,
                             behavior_type=None, start_time=None, end_time=None):
        sql = "SELECT * FROM study_behavior_records WHERE 1=1"
        params = []
        if user_id:
            sql += " AND user_id = ?"
            params.append(user_id)
        if session_id:
            sql += " AND session_id = ?"
            params.append(session_id)
        if behavior_type:
            sql += " AND behavior_type = ?"
            params.append(behavior_type)
        if start_time:
            sql += " AND timestamp >= ?"
            params.append(start_time)
        if end_time:
            sql += " AND timestamp <= ?"
            params.append(end_time)
        sql += " ORDER BY timestamp DESC LIMIT ? OFFSET ?"
        params.extend([limit, offset])
        with self.connect() as conn:
            rows = conn.execute(sql, params).fetchall()
            return [dict(row) for row in rows]

    def get_behavior_statistics(self, user_id, start_time, end_time):
        with self.connect() as conn:
            rows = conn.execute(
                """
                SELECT behavior_type, COUNT(*) as total, SUM(is_alert) as alerts
                FROM study_behavior_records
                WHERE user_id = ? AND timestamp BETWEEN ? AND ?
                GROUP BY behavior_type
                """,
                (user_id, start_time, end_time)
            ).fetchall()
            stats = {}
            for row in rows:
                stats[row["behavior_type"]] = {"total": row["total"], "alerts": row["alerts"] or 0}
            return stats
