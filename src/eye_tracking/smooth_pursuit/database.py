"""
SQLite database for smooth pursuit eye tracking data and model tracking.

Schema:
- sessions: Recording sessions with distance, timestamps, device info
- frames: Individual frames with gaze angles and image paths
- models: Trained models with distance, file path, training metadata
- training_runs: Links models to sessions used for training
"""

import sqlite3
from datetime import datetime
from pathlib import Path
from typing import List, Optional, Tuple


class SmoothPursuitDB:
    """Database interface for smooth pursuit data and models."""

    def __init__(self, db_path: Optional[Path] = None):
        """
        Initialize database connection.
        
        Args:
            db_path: Path to SQLite database file. Defaults to Dataset/smooth_pursuit_data.db
        """
        if db_path is None:
            base_dir = Path(__file__).parent / "Dataset"
            base_dir.mkdir(parents=True, exist_ok=True)
            db_path = base_dir / "smooth_pursuit_data.db"
        
        self.db_path = Path(db_path)
        self.conn = sqlite3.connect(str(self.db_path), check_same_thread=False)
        self.conn.row_factory = sqlite3.Row  # Return rows as dict-like objects
        self._init_schema()

    def _init_schema(self):
        """Create database tables if they don't exist."""
        cursor = self.conn.cursor()
        
        # Sessions table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS sessions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                session_id TEXT UNIQUE NOT NULL,
                distance_cm INTEGER NOT NULL,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                screen_width_cm REAL,
                screen_height_cm REAL,
                screen_width_px INTEGER,
                screen_height_px INTEGER,
                camera_x_px INTEGER,
                camera_y_px INTEGER,
                frame_count INTEGER DEFAULT 0,
                notes TEXT
            )
        """)
        
        # Frames table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS frames (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                session_id INTEGER NOT NULL,
                frame_number INTEGER NOT NULL,
                frame_filename TEXT NOT NULL,
                theta_h REAL NOT NULL,
                theta_v REAL NOT NULL,
                distance_cm INTEGER NOT NULL,
                timestamp REAL NOT NULL,
                FOREIGN KEY (session_id) REFERENCES sessions(id) ON DELETE CASCADE,
                UNIQUE(session_id, frame_number)
            )
        """)
        
        # Models table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS models (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                model_name TEXT UNIQUE NOT NULL,
                distance_cm INTEGER NOT NULL,
                model_path TEXT NOT NULL,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                best_val_loss REAL,
                total_epochs INTEGER,
                batch_size INTEGER,
                learning_rate REAL,
                notes TEXT
            )
        """)
        
        # Training runs table - links models to sessions used for training
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS training_runs (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                model_id INTEGER NOT NULL,
                session_id INTEGER NOT NULL,
                split_type TEXT NOT NULL,  -- 'train' or 'val'
                FOREIGN KEY (model_id) REFERENCES models(id) ON DELETE CASCADE,
                FOREIGN KEY (session_id) REFERENCES sessions(id) ON DELETE CASCADE,
                UNIQUE(model_id, session_id)
            )
        """)
        
        # Create indexes for performance
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_frames_session ON frames(session_id)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_frames_distance ON frames(distance_cm)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_sessions_distance ON sessions(distance_cm)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_models_distance ON models(distance_cm)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_training_runs_model ON training_runs(model_id)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_training_runs_session ON training_runs(session_id)")
        
        self.conn.commit()

    def create_session(
        self,
        distance_cm: int,
        screen_width_cm: Optional[float] = None,
        screen_height_cm: Optional[float] = None,
        screen_width_px: Optional[int] = None,
        screen_height_px: Optional[int] = None,
        camera_x_px: Optional[int] = None,
        camera_y_px: Optional[int] = None,
        notes: Optional[str] = None,
    ) -> Tuple[int, str]:
        """
        Create a new recording session.
        
        Returns:
            (session_db_id, session_id_string) tuple
        """
        session_id = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        cursor = self.conn.cursor()
        cursor.execute("""
            INSERT INTO sessions (
                session_id, distance_cm, screen_width_cm, screen_height_cm,
                screen_width_px, screen_height_px, camera_x_px, camera_y_px, notes
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            session_id, distance_cm, screen_width_cm, screen_height_cm,
            screen_width_px, screen_height_px, camera_x_px, camera_y_px, notes
        ))
        
        self.conn.commit()
        return cursor.lastrowid, session_id

    def add_frame(
        self,
        session_db_id: int,
        frame_number: int,
        frame_filename: str,
        theta_h: float,
        theta_v: float,
        distance_cm: int,
        timestamp: float,
    ):
        """Add a frame to a session."""
        cursor = self.conn.cursor()
        cursor.execute("""
            INSERT INTO frames (
                session_id, frame_number, frame_filename, theta_h, theta_v,
                distance_cm, timestamp
            ) VALUES (?, ?, ?, ?, ?, ?, ?)
        """, (session_db_id, frame_number, frame_filename, theta_h, theta_v, distance_cm, timestamp))
        
        # Update session frame count
        cursor.execute("""
            UPDATE sessions SET frame_count = frame_count + 1 WHERE id = ?
        """, (session_db_id,))
        
        self.conn.commit()

    def get_sessions_by_distance(self, distance_cm: int) -> List[sqlite3.Row]:
        """Get all sessions for a specific distance."""
        cursor = self.conn.cursor()
        cursor.execute("""
            SELECT * FROM sessions WHERE distance_cm = ? ORDER BY created_at DESC
        """, (distance_cm,))
        return cursor.fetchall()

    def get_frames_for_sessions(self, session_db_ids: List[int]) -> List[sqlite3.Row]:
        """Get all frames for given session IDs."""
        if not session_db_ids:
            return []
        
        placeholders = ",".join("?" * len(session_db_ids))
        cursor = self.conn.cursor()
        cursor.execute(f"""
            SELECT f.*, s.session_id, s.distance_cm
            FROM frames f
            JOIN sessions s ON f.session_id = s.id
            WHERE f.session_id IN ({placeholders})
            ORDER BY f.session_id, f.frame_number
        """, session_db_ids)
        return cursor.fetchall()

    def create_model(
        self,
        model_name: str,
        distance_cm: int,
        model_path: str,
        best_val_loss: Optional[float] = None,
        total_epochs: Optional[int] = None,
        batch_size: Optional[int] = None,
        learning_rate: Optional[float] = None,
        notes: Optional[str] = None,
    ) -> int:
        """Create a new model record."""
        cursor = self.conn.cursor()
        cursor.execute("""
            INSERT INTO models (
                model_name, distance_cm, model_path, best_val_loss,
                total_epochs, batch_size, learning_rate, notes
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        """, (model_name, distance_cm, model_path, best_val_loss, total_epochs, batch_size, learning_rate, notes))
        
        self.conn.commit()
        return cursor.lastrowid

    def link_training_sessions(
        self,
        model_id: int,
        train_session_ids: List[int],
        val_session_ids: List[int],
    ):
        """Link sessions used for training a model."""
        cursor = self.conn.cursor()
        
        # Add training sessions
        for session_id in train_session_ids:
            cursor.execute("""
                INSERT OR IGNORE INTO training_runs (model_id, session_id, split_type)
                VALUES (?, ?, 'train')
            """, (model_id, session_id))
        
        # Add validation sessions
        for session_id in val_session_ids:
            cursor.execute("""
                INSERT OR IGNORE INTO training_runs (model_id, session_id, split_type)
                VALUES (?, ?, 'val')
            """, (model_id, session_id))
        
        self.conn.commit()

    def get_model_training_sessions(self, model_id: int) -> Tuple[List[int], List[int]]:
        """Get train and validation session IDs for a model."""
        cursor = self.conn.cursor()
        cursor.execute("""
            SELECT session_id, split_type FROM training_runs WHERE model_id = ?
        """, (model_id,))
        
        train_ids = []
        val_ids = []
        for row in cursor.fetchall():
            if row["split_type"] == "train":
                train_ids.append(row["session_id"])
            else:
                val_ids.append(row["session_id"])
        
        return train_ids, val_ids

    def get_model_by_name(self, model_name: str) -> Optional[sqlite3.Row]:
        """Get model by name."""
        cursor = self.conn.cursor()
        cursor.execute("SELECT * FROM models WHERE model_name = ?", (model_name,))
        return cursor.fetchone()

    def get_all_models(self) -> List[sqlite3.Row]:
        """Get all models."""
        cursor = self.conn.cursor()
        cursor.execute("SELECT * FROM models ORDER BY created_at DESC")
        return cursor.fetchall()

    def close(self):
        """Close database connection."""
        self.conn.close()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()


