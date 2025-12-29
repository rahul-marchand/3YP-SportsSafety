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
        # Note: distance_cm represents the total distance (eye to virtual image) for beam splitter
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
        
        # Inference sessions table - tracks inference runs
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS inference_sessions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                session_id TEXT UNIQUE NOT NULL,
                model_id INTEGER NOT NULL,
                distance_cm INTEGER NOT NULL,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                frame_count INTEGER DEFAULT 0,
                mean_error_horizontal REAL,
                mean_error_vertical REAL,
                rms_error REAL,
                mean_velocity_error_horizontal REAL,
                mean_velocity_error_vertical REAL,
                smooth_pursuit_gain_horizontal REAL,
                smooth_pursuit_gain_vertical REAL,
                lag_horizontal_ms REAL,
                lag_vertical_ms REAL,
                lag_horizontal_correlation REAL,
                lag_vertical_correlation REAL,
                notes TEXT,
                FOREIGN KEY (model_id) REFERENCES models(id) ON DELETE SET NULL
            )
        """)
        
        # Inference results table - stores individual frame predictions
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS inference_results (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                inference_session_id INTEGER NOT NULL,
                frame_number INTEGER NOT NULL,
                timestamp REAL NOT NULL,
                theta_h_predicted REAL NOT NULL,
                theta_v_predicted REAL NOT NULL,
                theta_h_target REAL NOT NULL,
                theta_v_target REAL NOT NULL,
                error_horizontal REAL NOT NULL,
                error_vertical REAL NOT NULL,
                expected_vel_h REAL,
                expected_vel_v REAL,
                actual_vel_h REAL,
                actual_vel_v REAL,
                vel_error_h REAL,
                vel_error_v REAL,
                FOREIGN KEY (inference_session_id) REFERENCES inference_sessions(id) ON DELETE CASCADE,
                UNIQUE(inference_session_id, frame_number)
            )
        """)
        
        # Create indexes for performance
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_frames_session ON frames(session_id)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_frames_distance ON frames(distance_cm)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_sessions_distance ON sessions(distance_cm)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_models_distance ON models(distance_cm)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_training_runs_model ON training_runs(model_id)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_training_runs_session ON training_runs(session_id)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_inference_results_session ON inference_results(inference_session_id)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_inference_sessions_model ON inference_sessions(model_id)")
        
        self.conn.commit()

    def create_session(
        self,
        distance_cm: int,
        total_distance_cm: Optional[int] = None,
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
        
        Args:
            distance_cm: Total distance from eye to virtual image (for beam splitter)
            total_distance_cm: Alias for distance_cm (kept for backward compatibility)
        
        Returns:
            (session_db_id, session_id_string) tuple
        """
        session_id = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Use total_distance_cm if provided, otherwise use distance_cm
        # Both represent the total distance (eye to virtual image) for beam splitter
        final_distance = total_distance_cm if total_distance_cm is not None else distance_cm
        
        cursor = self.conn.cursor()
        cursor.execute("""
            INSERT INTO sessions (
                session_id, distance_cm, screen_width_cm, screen_height_cm,
                screen_width_px, screen_height_px, camera_x_px, camera_y_px, notes
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            session_id, final_distance, screen_width_cm, screen_height_cm,
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
        """
        Get all sessions for a specific distance.
        
        Args:
            distance_cm: Total distance from eye to virtual image
        """
        cursor = self.conn.cursor()
        cursor.execute("""
            SELECT * FROM sessions 
            WHERE distance_cm = ?
            ORDER BY created_at DESC
        """, (distance_cm,))
        return cursor.fetchall()

    def get_frames_for_sessions(self, session_db_ids: List[int]) -> List[sqlite3.Row]:
        """Get all frames for given session IDs."""
        if not session_db_ids:
            return []
        
        placeholders = ",".join("?" * len(session_db_ids))
        cursor = self.conn.cursor()
        cursor.execute(f"""
            SELECT f.*, s.session_id as session_id_string, s.distance_cm as session_distance_cm
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

    def create_inference_session(
        self,
        model_id: int,
        distance_cm: int,
        notes: Optional[str] = None,
    ) -> Tuple[int, str]:
        """
        Create a new inference session.
        
        Returns:
            (inference_session_db_id, session_id_string) tuple
        """
        session_id = datetime.now().strftime("inf_%Y%m%d_%H%M%S")
        
        cursor = self.conn.cursor()
        cursor.execute("""
            INSERT INTO inference_sessions (
                session_id, model_id, distance_cm, notes
            ) VALUES (?, ?, ?, ?)
        """, (session_id, model_id, distance_cm, notes))
        
        self.conn.commit()
        return cursor.lastrowid, session_id

    def add_inference_result(
        self,
        inference_session_id: int,
        frame_number: int,
        timestamp: float,
        theta_h_predicted: float,
        theta_v_predicted: float,
        theta_h_target: float,
        theta_v_target: float,
        error_horizontal: float,
        error_vertical: float,
        expected_vel_h: Optional[float] = None,
        expected_vel_v: Optional[float] = None,
        actual_vel_h: Optional[float] = None,
        actual_vel_v: Optional[float] = None,
        vel_error_h: Optional[float] = None,
        vel_error_v: Optional[float] = None,
    ):
        """Add an inference result (predicted vs target angles for one frame)."""
        cursor = self.conn.cursor()
        cursor.execute("""
            INSERT OR REPLACE INTO inference_results (
                inference_session_id, frame_number, timestamp,
                theta_h_predicted, theta_v_predicted,
                theta_h_target, theta_v_target,
                error_horizontal, error_vertical,
                expected_vel_h, expected_vel_v,
                actual_vel_h, actual_vel_v,
                vel_error_h, vel_error_v
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            inference_session_id, frame_number, timestamp,
            theta_h_predicted, theta_v_predicted,
            theta_h_target, theta_v_target,
            error_horizontal, error_vertical,
            expected_vel_h, expected_vel_v,
            actual_vel_h, actual_vel_v,
            vel_error_h, vel_error_v,
        ))
        self.conn.commit()

    def update_inference_session_metrics(
        self,
        inference_session_id: int,
        frame_count: int,
        mean_error_horizontal: Optional[float] = None,
        mean_error_vertical: Optional[float] = None,
        rms_error: Optional[float] = None,
        mean_velocity_error_horizontal: Optional[float] = None,
        mean_velocity_error_vertical: Optional[float] = None,
        smooth_pursuit_gain_horizontal: Optional[float] = None,
        smooth_pursuit_gain_vertical: Optional[float] = None,
        lag_horizontal_ms: Optional[float] = None,
        lag_vertical_ms: Optional[float] = None,
        lag_horizontal_correlation: Optional[float] = None,
        lag_vertical_correlation: Optional[float] = None,
    ):
        """Update inference session with final metrics."""
        cursor = self.conn.cursor()
        cursor.execute("""
            UPDATE inference_sessions SET
                frame_count = ?,
                mean_error_horizontal = ?,
                mean_error_vertical = ?,
                rms_error = ?,
                mean_velocity_error_horizontal = ?,
                mean_velocity_error_vertical = ?,
                smooth_pursuit_gain_horizontal = ?,
                smooth_pursuit_gain_vertical = ?,
                lag_horizontal_ms = ?,
                lag_vertical_ms = ?,
                lag_horizontal_correlation = ?,
                lag_vertical_correlation = ?
            WHERE id = ?
        """, (
            frame_count,
            mean_error_horizontal,
            mean_error_vertical,
            rms_error,
            mean_velocity_error_horizontal,
            mean_velocity_error_vertical,
            smooth_pursuit_gain_horizontal,
            smooth_pursuit_gain_vertical,
            lag_horizontal_ms,
            lag_vertical_ms,
            lag_horizontal_correlation,
            lag_vertical_correlation,
            inference_session_id,
        ))
        self.conn.commit()

    def clear_all_data(self):
        """Clear all data from database (sessions, frames, models, training runs)."""
        cursor = self.conn.cursor()
        cursor.execute("DELETE FROM training_runs")
        cursor.execute("DELETE FROM frames")
        cursor.execute("DELETE FROM sessions")
        cursor.execute("DELETE FROM models")
        self.conn.commit()
        print("All data cleared from database")

    def clear_sessions_by_distance(self, distance_cm: int):
        """Clear all sessions and frames for a specific distance."""
        cursor = self.conn.cursor()
        # Get session IDs for this distance
        cursor.execute("SELECT id FROM sessions WHERE distance_cm = ?", (distance_cm,))
        session_ids = [row[0] for row in cursor.fetchall()]
        
        if session_ids:
            # Delete training runs linked to these sessions
            placeholders = ",".join("?" * len(session_ids))
            cursor.execute(f"DELETE FROM training_runs WHERE session_id IN ({placeholders})", session_ids)
            
            # Delete frames for these sessions
            cursor.execute(f"DELETE FROM frames WHERE session_id IN ({placeholders})", session_ids)
            
            # Delete sessions
            cursor.execute(f"DELETE FROM sessions WHERE id IN ({placeholders})", session_ids)
            
            self.conn.commit()
            print(f"Cleared {len(session_ids)} sessions and associated frames for distance {distance_cm}cm")
        else:
            print(f"No sessions found for distance {distance_cm}cm")

    def get_statistics(self) -> dict:
        """Get database statistics."""
        cursor = self.conn.cursor()
        
        cursor.execute("SELECT COUNT(*) FROM sessions")
        total_sessions = cursor.fetchone()[0]
        
        cursor.execute("SELECT COUNT(*) FROM frames")
        total_frames = cursor.fetchone()[0]
        
        cursor.execute("SELECT COUNT(*) FROM models")
        total_models = cursor.fetchone()[0]
        
        cursor.execute("SELECT COUNT(DISTINCT distance_cm) FROM sessions")
        unique_distances = cursor.fetchone()[0]
        
        cursor.execute("SELECT distance_cm, COUNT(*) as count FROM sessions GROUP BY distance_cm")
        sessions_by_distance = {row[0]: row[1] for row in cursor.fetchall()}
        
        return {
            "total_sessions": total_sessions,
            "total_frames": total_frames,
            "total_models": total_models,
            "unique_distances": unique_distances,
            "sessions_by_distance": sessions_by_distance,
        }

    def close(self):
        """Close database connection."""
        self.conn.close()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()



