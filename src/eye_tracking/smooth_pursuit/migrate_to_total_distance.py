#!/usr/bin/env python3
"""
Migration script to consolidate distance fields to total_distance_cm only.

This script:
1. Updates existing sessions: copies total_distance_cm to distance_cm if total_distance_cm is NULL
2. Updates the schema to remove distance_cm and keep only total_distance_cm
3. Updates frames and models tables similarly
"""

import sqlite3
from pathlib import Path
import sys

def migrate_database(db_path: Path):
    """Migrate database to use only total_distance_cm."""
    
    if not db_path.exists():
        print(f"Database not found: {db_path}")
        return False
    
    conn = sqlite3.connect(str(db_path))
    cursor = conn.cursor()
    
    try:
        print("Starting migration...")
        
        # Check if total_distance_cm column exists
        cursor.execute("PRAGMA table_info(sessions)")
        columns = [row[1] for row in cursor.fetchall()]
        has_total_distance = "total_distance_cm" in columns
        
        # Step 1: Update sessions where total_distance_cm is NULL (if column exists)
        if has_total_distance:
            cursor.execute("""
                UPDATE sessions 
                SET total_distance_cm = distance_cm 
                WHERE total_distance_cm IS NULL
            """)
            updated_sessions = cursor.rowcount
            print(f"  Updated {updated_sessions} sessions (copied distance_cm -> total_distance_cm)")
        else:
            print("  No total_distance_cm column found - database already uses single distance field")
            updated_sessions = 0
        
        # Step 2: Remove total_distance_cm column if it exists
        # SQLite doesn't support ALTER COLUMN, so we need to recreate the table
        if has_total_distance:
            print("  Migrating sessions table (removing total_distance_cm column)...")
            cursor.execute("""
                CREATE TABLE sessions_new (
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
            
            cursor.execute("""
                INSERT INTO sessions_new 
                (id, session_id, distance_cm, created_at, screen_width_cm, screen_height_cm,
                 screen_width_px, screen_height_px, camera_x_px, camera_y_px, frame_count, notes)
                SELECT id, session_id, COALESCE(total_distance_cm, distance_cm), created_at,
                       screen_width_cm, screen_height_cm, screen_width_px, screen_height_px,
                       camera_x_px, camera_y_px, frame_count, notes
                FROM sessions
            """)
            
            cursor.execute("DROP TABLE sessions")
            cursor.execute("ALTER TABLE sessions_new RENAME TO sessions")
            print("  ✓ Sessions table migrated")
        else:
            print("  ✓ Sessions table already uses single distance_cm field")
        
        # Step 3: Update frames table - distance_cm should match session's distance
        print("  Updating frames table...")
        cursor.execute("""
            UPDATE frames
            SET distance_cm = (
                SELECT s.distance_cm 
                FROM sessions s 
                WHERE s.id = frames.session_id
            )
        """)
        updated_frames = cursor.rowcount
        print(f"  Updated {updated_frames} frames to match session distance")
        
        # Step 4: Update models table - distance_cm is already correct (it's the total distance)
        print("  Models table already uses distance_cm correctly")
        
        # Step 5: Recreate indexes
        print("  Recreating indexes...")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_frames_distance ON frames(distance_cm)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_sessions_distance ON sessions(distance_cm)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_models_distance ON models(distance_cm)")
        
        conn.commit()
        print("\n✓ Migration completed successfully!")
        print(f"  - Sessions: {updated_sessions} updated")
        print(f"  - Frames: {updated_frames} updated")
        print("  - Schema: Simplified to use only distance_cm (total distance)")
        
        return True
        
    except Exception as e:
        conn.rollback()
        print(f"\n✗ Migration failed: {e}")
        import traceback
        traceback.print_exc()
        return False
    finally:
        conn.close()

def main():
    db_path = Path(__file__).parent / "Dataset" / "smooth_pursuit_data.db"
    
    print("=" * 60)
    print("DATABASE MIGRATION: Consolidate to total_distance_cm only")
    print("=" * 60)
    print(f"\nDatabase: {db_path}")
    
    if not db_path.exists():
        print("\n✗ Database not found. Nothing to migrate.")
        sys.exit(1)
    
    # Backup suggestion
    backup_path = db_path.parent / f"{db_path.stem}_backup_{Path(db_path).stat().st_mtime}.db"
    print(f"\n⚠ RECOMMENDED: Backup database first:")
    print(f"   cp {db_path} {backup_path}")
    
    # Allow non-interactive mode with --yes flag
    auto_yes = "--yes" in sys.argv or "-y" in sys.argv
    
    if not auto_yes:
        response = input("\nProceed with migration? (yes/no): ").strip().lower()
        if response != "yes":
            print("Migration cancelled.")
            sys.exit(0)
    else:
        print("\nProceeding with migration (--yes flag)...")
    
    success = migrate_database(db_path)
    
    if success:
        print("\n" + "=" * 60)
        print("Next steps:")
        print("1. Restart the backend server")
        print("2. Test data collection and training")
        print("3. Verify that 'python control.py data' shows correct distances")
        print("=" * 60)
    else:
        print("\n✗ Migration failed. Check error messages above.")
        sys.exit(1)

if __name__ == "__main__":
    main()

