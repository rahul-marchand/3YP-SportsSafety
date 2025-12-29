#!/usr/bin/env python3
"""Check frame capture rate and timestamp intervals."""

from database import SmoothPursuitDB

db = SmoothPursuitDB()
cursor = db.conn.cursor()

# Get all frames ordered by timestamp
cursor.execute("""
    SELECT f.frame_number, f.timestamp, s.session_id, s.distance_cm
    FROM frames f
    JOIN sessions s ON f.session_id = s.id
    ORDER BY f.timestamp
""")
rows = cursor.fetchall()

if not rows:
    print("No frames found in database")
    db.close()
    exit(0)

print(f"Total frames in database: {len(rows)}")
print(f"Total sessions: {len(set(r[2] for r in rows))}")
print()

# Group by session
sessions = {}
for row in rows:
    session_id = row[2]
    if session_id not in sessions:
        sessions[session_id] = []
    sessions[session_id].append((row[0], row[1], row[3]))

# Analyze each session
for session_id, frames in sessions.items():
    print(f"Session: {session_id}")
    print(f"  Frames: {len(frames)}")
    print(f"  Distance: {frames[0][2]}cm")
    
    if len(frames) > 1:
        # Calculate intervals
        intervals = []
        for i in range(1, len(frames)):
            interval = frames[i][1] - frames[i-1][1]
            intervals.append(interval)
        
        avg_interval = sum(intervals) / len(intervals)
        min_interval = min(intervals)
        max_interval = max(intervals)
        
        print(f"  Frame intervals:")
        print(f"    Average: {avg_interval*1000:.1f}ms (expected: ~100ms for 10Hz)")
        print(f"    Min: {min_interval*1000:.1f}ms")
        print(f"    Max: {max_interval*1000:.1f}ms")
        print(f"  First 10 intervals:")
        for i, interval in enumerate(intervals[:10]):
            print(f"    Frame {frames[i+1][0]} - {frames[i][0]}: {interval*1000:.1f}ms")
        
        # Check for gaps
        gaps = [i for i in intervals if i > 0.5]  # Gaps > 500ms
        if gaps:
            print(f"  WARNING: Found {len(gaps)} large gaps (>500ms)")
    else:
        print(f"  Only 1 frame - cannot calculate intervals")
    print()

db.close()



