#!/usr/bin/env python3
"""
Database management script for smooth pursuit eye tracking data.

Usage:
    python manage_db.py stats              # Show database statistics
    python manage_db.py clear              # Clear ALL data (sessions, frames, models)
    python manage_db.py clear-distance 30  # Clear data for specific distance
"""

import sys
from database import SmoothPursuitDB


def show_stats():
    """Show database statistics."""
    db = SmoothPursuitDB()
    stats = db.get_statistics()
    
    print("=== Database Statistics ===")
    print(f"Total sessions:     {stats['total_sessions']}")
    print(f"Total frames:       {stats['total_frames']}")
    print(f"Total models:       {stats['total_models']}")
    print(f"Unique distances:   {stats['unique_distances']}")
    print()
    
    if stats['sessions_by_distance']:
        print("Sessions by distance:")
        for distance, count in sorted(stats['sessions_by_distance'].items()):
            print(f"  {distance}cm: {count} sessions")
    else:
        print("No sessions found")
    
    print("========================")
    db.close()


def clear_all():
    """Clear all data from database."""
    print("WARNING: This will delete ALL data (sessions, frames, models, training runs)")
    response = input("Are you sure? Type 'yes' to confirm: ")
    
    if response.lower() == 'yes':
        db = SmoothPursuitDB()
        db.clear_all_data()
        db.close()
        print("Database cleared successfully")
    else:
        print("Operation cancelled")


def clear_distance(distance_cm: int):
    """Clear data for a specific distance."""
    print(f"WARNING: This will delete all sessions and frames for distance {distance_cm}cm")
    response = input("Are you sure? Type 'yes' to confirm: ")
    
    if response.lower() == 'yes':
        db = SmoothPursuitDB()
        db.clear_sessions_by_distance(distance_cm)
        db.close()
        print(f"Data for {distance_cm}cm cleared successfully")
    else:
        print("Operation cancelled")


def main():
    if len(sys.argv) < 2:
        print("Usage: python manage_db.py [command]")
        print()
        print("Commands:")
        print("  stats              - Show database statistics")
        print("  clear              - Clear ALL data (sessions, frames, models)")
        print("  clear-distance X   - Clear data for specific distance (e.g., clear-distance 30)")
        sys.exit(1)
    
    command = sys.argv[1].lower()
    
    if command == "stats":
        show_stats()
    elif command == "clear":
        clear_all()
    elif command == "clear-distance":
        if len(sys.argv) < 3:
            print("Error: Please specify distance. Usage: python manage_db.py clear-distance 30")
            sys.exit(1)
        try:
            distance = int(sys.argv[2])
            clear_distance(distance)
        except ValueError:
            print(f"Error: Invalid distance '{sys.argv[2]}'. Must be a number.")
            sys.exit(1)
    else:
        print(f"Unknown command: {command}")
        print("Use 'stats', 'clear', or 'clear-distance X'")
        sys.exit(1)


if __name__ == "__main__":
    main()



