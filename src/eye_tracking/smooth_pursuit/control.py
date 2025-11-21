#!/usr/bin/env python3
"""
Terminal control script for two-phone beam splitter recording.

Usage:
    python control.py start    # Start recording on both phones
    python control.py stop     # Stop recording
    python control.py status   # Check connection and recording status
"""

import sys
import requests
import json

# Backend URL - change this if using ngrok
BACKEND_URL = "http://localhost:8000"


def start_recording():
    """Start recording on both phones."""
    try:
        response = requests.post(f"{BACKEND_URL}/api/control/start")
        data = response.json()
        
        if response.status_code == 200:
            print(f"✓ {data['status']}")
            print(f"  {data.get('message', '')}")
        else:
            print(f"✗ Error: {data.get('error', 'Unknown error')}")
            
    except requests.exceptions.ConnectionError:
        print("✗ Error: Cannot connect to backend. Is it running?")
    except Exception as e:
        print(f"✗ Error: {e}")


def stop_recording():
    """Stop recording on both phones."""
    try:
        response = requests.post(f"{BACKEND_URL}/api/control/stop")
        data = response.json()
        
        if response.status_code == 200:
            print(f"✓ {data['status']}")
            print(f"  Frames captured: {data.get('frames_captured', 0)}")
        else:
            print(f"✗ Error: {data.get('error', 'Unknown error')}")
            
    except requests.exceptions.ConnectionError:
        print("✗ Error: Cannot connect to backend. Is it running?")
    except Exception as e:
        print(f"✗ Error: {e}")


def check_status():
    """Check connection and recording status."""
    try:
        response = requests.get(f"{BACKEND_URL}/api/control/status")
        data = response.json()
        
        print("=== System Status ===")
        print(f"Recording active:    {data['recording_active']}")
        print(f"Display connected:   {data['display_connected']}")
        print(f"Camera connected:    {data['camera_connected']}")
        print(f"Frames captured:     {data['frame_count']}")
        print(f"Dot buffer size:     {data['buffer_size']}")
        print("====================")
        
        # Connection warnings
        if not data['display_connected'] and not data['camera_connected']:
            print("\n⚠ Warning: No phones connected")
            print("  Open URLs on both phones:")
            print(f"  Display: {BACKEND_URL}/?mode=display&distance=30")
            print(f"  Camera:  {BACKEND_URL}/?mode=camera&distance=30")
        elif not data['display_connected']:
            print("\n⚠ Warning: Display phone not connected")
        elif not data['camera_connected']:
            print("\n⚠ Warning: Camera phone not connected")
        else:
            print("\n✓ Both phones connected and ready")
            
    except requests.exceptions.ConnectionError:
        print("✗ Error: Cannot connect to backend. Is it running?")
        print(f"\nTo start backend:")
        print("  cd /path/to/smooth_pursuit")
        print("  source venv/bin/activate")
        print("  python backend.py")
    except Exception as e:
        print(f"✗ Error: {e}")


def print_usage():
    """Print usage instructions."""
    print("Usage: python control.py [command]")
    print()
    print("Commands:")
    print("  start    - Start recording on both phones")
    print("  stop     - Stop recording")
    print("  status   - Check connection and recording status")
    print()
    print("Setup:")
    print(f"  1. Start backend: python backend.py")
    print(f"  2. Display phone: {BACKEND_URL}/?mode=display&distance=30")
    print(f"  3. Camera phone:  {BACKEND_URL}/?mode=camera&distance=30")
    print(f"  4. Control from terminal: python control.py start")


def main():
    if len(sys.argv) < 2:
        print_usage()
        sys.exit(1)
    
    command = sys.argv[1].lower()
    
    if command == "start":
        start_recording()
    elif command == "stop":
        stop_recording()
    elif command == "status":
        check_status()
    elif command in ["help", "-h", "--help"]:
        print_usage()
    else:
        print(f"Unknown command: {command}")
        print()
        print_usage()
        sys.exit(1)


if __name__ == "__main__":
    main()

