#!/usr/bin/env python3
"""
Terminal control script for two-phone beam splitter recording and inference.
Interactive CLI for seamless data collection and model testing.

Features:
- Pre-flight checks before any operation
- Required parameter validation
- Session naming and notes
- Real-time status monitoring
- Error prevention through guided workflows
"""

import sys
import os
import requests
import json
import time
import signal
import shutil
import sqlite3
from datetime import datetime
from typing import Optional, Dict, Any, List
from pathlib import Path

# Backend URL - change this if using ngrok
BACKEND_URL = os.environ.get("BACKEND_URL", "http://localhost:8000")

# =====================================================================
# UTILITY FUNCTIONS
# =====================================================================

def clear_screen():
    """Clear terminal screen."""
    os.system('cls' if os.name == 'nt' else 'clear')

def get_backend_status() -> Dict[str, Any]:
    """Get status from backend, handling connection errors."""
    try:
        response = requests.get(f"{BACKEND_URL}/api/control/status", timeout=2)
        if response.status_code == 200:
            return response.json()
        return {"error": f"Backend returned status {response.status_code}"}
    except requests.exceptions.ConnectionError:
        return {"error": "Cannot connect to backend. Is it running?"}
    except Exception as e:
        return {"error": str(e)}

def print_header(title: str):
    """Print a formatted header."""
    width = 50
    print("\n" + "=" * width)
    print(f" {title}".center(width))
    print("=" * width)

def print_status(data: Dict[str, Any], compact: bool = False):
    """Print formatted status."""
    if "error" in data:
        print(f"✗ Status Check Failed: {data['error']}")
        return False

    if compact:
        display = "✓" if data.get('display_connected') else "✗"
        camera = "✓" if data.get('camera_connected') else "✗"
        rec = "REC" if data.get('recording_active') else ""
        inf = "INF" if data.get('inference_active') else ""
        print(f"[Display:{display}] [Camera:{camera}] {rec}{inf}")
        return True

    print_header("SYSTEM STATUS")
    
    # Connection status
    print("\n📱 Phone Connections:")
    if data.get('display_connected'):
        print("   Display phone:  ✓ Connected")
    else:
        print("   Display phone:  ✗ NOT CONNECTED")
        print(f"                   → Open: {BACKEND_URL}/?mode=display")
    
    if data.get('camera_connected'):
        print("   Camera phone:   ✓ Connected")
    else:
        print("   Camera phone:   ✗ NOT CONNECTED")
        print(f"                   → Open: {BACKEND_URL}/?mode=camera")

    # Activity status
    print("\n📊 Current Activity:")
    if data.get('recording_active'):
        print(f"   Recording:      ✓ ACTIVE ({data.get('frame_count', 0)} frames)")
    elif data.get('inference_active'):
        print(f"   Inference:      ✓ ACTIVE ({data.get('inference_frame_count', 0)} frames)")
    else:
        print("   Activity:       None (idle)")
    
    # Configuration
    print("\n⚙️  Configuration:")
    dist = data.get('distance_cm')
    print(f"   Distance:       {dist}cm" if dist else "   Distance:       Not set")
    
    print()
    return data.get('display_connected') and data.get('camera_connected')

def get_input(prompt: str, default: Any = None, cast_type: type = str, required: bool = False) -> Any:
    """Get input with validation."""
    while True:
        if default is not None:
            user_input = input(f"{prompt} [{default}]: ").strip()
            if not user_input:
                return default
        else:
            user_input = input(f"{prompt}: ").strip()
        
        if not user_input:
            if required:
                print("✗ This field is required.")
                continue
            return None
            
        try:
            return cast_type(user_input)
        except ValueError:
            print(f"✗ Invalid input. Expected {cast_type.__name__}.")
            if not required:
                return None

def confirm(prompt: str, default: bool = True) -> bool:
    """Get yes/no confirmation."""
    default_str = "Y/n" if default else "y/N"
    response = input(f"{prompt} [{default_str}]: ").strip().lower()
    if not response:
        return default
    return response.startswith('y')

def wait_for_connections(timeout: int = 30) -> bool:
    """Wait for both phones to connect."""
    print("\n⏳ Waiting for phone connections...")
    print(f"   (timeout: {timeout}s)")
    
    start = time.time()
    while time.time() - start < timeout:
        status = get_backend_status()
        if status.get('display_connected') and status.get('camera_connected'):
            print("✓ Both phones connected!")
            return True
        
        display = "✓" if status.get('display_connected') else "⏳"
        camera = "✓" if status.get('camera_connected') else "⏳"
        print(f"\r   Display: {display}  Camera: {camera}  [{int(time.time()-start)}s]", end="", flush=True)
        time.sleep(1)
    
    print("\n✗ Timeout waiting for connections.")
    return False

# =====================================================================
# PRE-FLIGHT CHECKS
# =====================================================================

def preflight_check(require_model: bool = False) -> Optional[Dict[str, Any]]:
    """
    Comprehensive pre-flight check before recording/inference.
    Returns status dict if all checks pass, None otherwise.
    """
    print_header("PRE-FLIGHT CHECKS")
    all_pass = True
    
    # 1. Backend connectivity
    print("\n1️⃣  Backend Server...")
    status = get_backend_status()
    if "error" in status:
        print(f"   ✗ FAIL: {status['error']}")
        print("   → Make sure backend is running: python backend.py")
        return None
    print("   ✓ Connected")
    
    # 2. Phone connections
    print("\n2️⃣  Phone Connections...")
    if not status.get('display_connected'):
        print("   ✗ Display phone not connected")
        print(f"   → Open on display phone: {BACKEND_URL}/?mode=display")
        all_pass = False
    else:
        print("   ✓ Display phone connected")
    
    if not status.get('camera_connected'):
        print("   ✗ Camera phone not connected")
        print(f"   → Open on camera phone: {BACKEND_URL}/?mode=camera")
        all_pass = False
    else:
        print("   ✓ Camera phone connected")
    
    # 3. No conflicting activity
    print("\n3️⃣  Activity Status...")
    if status.get('recording_active'):
        print("   ⚠ Recording already in progress!")
        if not confirm("   Stop current recording?"):
            return None
        requests.post(f"{BACKEND_URL}/api/control/stop")
        print("   ✓ Previous recording stopped")
    elif status.get('inference_active'):
        print("   ⚠ Inference already in progress!")
        if not confirm("   Stop current inference?"):
            return None
        requests.post(f"{BACKEND_URL}/api/control/inference/stop")
        print("   ✓ Previous inference stopped")
    else:
        print("   ✓ System idle")
    
    # 4. Model availability (for inference)
    if require_model:
        print("\n4️⃣  Model Availability...")
        try:
            resp = requests.get(f"{BACKEND_URL}/api/models")
            models = resp.json().get('models', [])
            if not models:
                print("   ✗ No trained models found")
                print("   → Train a model first: python train_smooth_pursuit.py")
                all_pass = False
            else:
                print(f"   ✓ {len(models)} model(s) available")
        except:
            print("   ⚠ Could not check models")
    
    # Final result
    print()
    if not all_pass:
        print("❌ PRE-FLIGHT CHECKS FAILED")
        print("   Fix the issues above and try again.\n")
        return None
    
    print("✅ ALL PRE-FLIGHT CHECKS PASSED\n")
    return status

# =====================================================================
# DATA COLLECTION
# =====================================================================

def start_recording_interactive():
    """Guided workflow to start TRAINING DATA COLLECTION."""
    print_header("TRAINING DATA COLLECTION SETUP")
    
    # Pre-flight
    status = preflight_check(require_model=False)
    if status is None:
        if confirm("Wait for phone connections?"):
            if not wait_for_connections():
                return
            status = get_backend_status()
        else:
            return
    
    # Configuration
    print_header("BEAM SPLITTER DISTANCE CONFIGURATION")
    
    # 1. TOTAL DISTANCE (simplified - for 45° beam splitter, it's an isometric mapping)
    print("\n📏 Total Distance Measurement")
    print("   For a 45° beam splitter, the geometry is an isometric mapping.")
    print("   Measure the TOTAL distance from the eye to the virtual image position.")
    print("   This is: (Display → Splitter) + (Splitter → Eye)\n")
    print("   Example: If display is 10cm from splitter and eye is 30cm from splitter,")
    print("            total distance = 40cm\n")
    
    total_distance = None
    while total_distance is None or total_distance <= 0:
        total_distance = get_input("   Total distance: Eye to virtual image (cm)", 40, int, required=True)
        if total_distance <= 0:
            print("   ✗ Distance must be positive.")
    
    # 2. Session note (optional)
    print("\n📝 Session Notes (optional)")
    notes = get_input("   Add a note for this session", "")
    
    # 3. Confirmation
    print("\n" + "-" * 50)
    print("TRAINING DATA COLLECTION CONFIGURATION:")
    print(f"  • Total distance:  {total_distance} cm")
    print(f"  • Notes:            {notes or '(none)'}")
    print("-" * 50)
    
    if not confirm("\nStart TRAINING DATA COLLECTION with these settings?"):
        print("Cancelled.")
        return
    
    # Start
    print("\n⏳ Starting training data collection...")
    try:
        payload = {
            "total_distance_cm": total_distance,
        }
        if notes:
            payload["notes"] = notes
        
        response = requests.post(f"{BACKEND_URL}/api/control/start", json=payload, timeout=10)
        
        # Handle empty or invalid JSON responses
        try:
            data = response.json()
        except (ValueError, json.JSONDecodeError) as e:
            print(f"\n✗ Backend returned invalid response (Status {response.status_code})")
            print(f"  Response: {response.text[:500]}")
            print(f"  JSON Error: {e}")
            if response.status_code == 500:
                print("\n  → This is a server error. Check backend logs for details.")
            return
        
        # Check for errors in response
        if "error" in data:
            print(f"\n✗ Failed to start: {data['error']}")
            if "not initialized" in data['error'].lower():
                print("  → Make sure both phones are connected and camera phone has sent initialization data")
            elif "not connected" in data['error'].lower():
                print("  → Make sure both phones are connected to the backend")
            return
        
        if response.status_code == 200:
            print("\n" + "=" * 40)
            print("  ✓ TRAINING DATA COLLECTION STARTED")
            print("=" * 40)
            print(f"\n  Session: {data.get('session_id', 'Unknown')}")
            print(f"  Total distance: {total_distance}cm")
            print("\n  📹 Recording in progress...")
            print("  → Type 'stop' or press Ctrl+C to stop\n")
            
            # Monitor mode
            monitor_recording()
        else:
            print(f"\n✗ Failed to start (Status {response.status_code}): {data.get('error', 'Unknown error')}")
            
    except requests.exceptions.ConnectionError:
        print(f"\n✗ Cannot connect to backend at {BACKEND_URL}")
        print("  → Make sure backend is running: python backend.py")
    except requests.exceptions.Timeout:
        print(f"\n✗ Request timed out. Backend may be unresponsive.")
    except Exception as e:
        print(f"\n✗ Error: {e}")
        import traceback
        traceback.print_exc()

def monitor_recording():
    """Monitor recording progress with live updates."""
    last_count = 0
    try:
        while True:
            status = get_backend_status()
            if not status.get('recording_active'):
                print("\n✓ Recording stopped.")
                break
            
            count = status.get('frame_count', 0)
            if count != last_count:
                print(f"\r  Frames: {count}", end="", flush=True)
                last_count = count
            
            time.sleep(0.5)
    except KeyboardInterrupt:
        print("\n\n⏳ Stopping recording...")
        stop_recording()

def stop_recording():
    """Stop TRAINING DATA COLLECTION and show summary."""
    try:
        response = requests.post(f"{BACKEND_URL}/api/control/stop")
        data = response.json()
        
        if "error" in data:
            print(f"\n✗ Error: {data['error']}")
            return
        
        print("\n" + "=" * 50)
        print("  TRAINING DATA COLLECTION COMPLETE")
        print("=" * 50)
        print(f"\n  📹 Frames captured: {data.get('frames_captured', data.get('frame_count', 0))}")
        print(f"  💾 Session saved: {data.get('session_id', 'Unknown')}")
        print("\n  → Data is ready for training.")
        print("  → Run: python control.py train")
        print()
        
    except Exception as e:
        print(f"✗ Error stopping recording: {e}")

# =====================================================================
# INFERENCE / TESTING
# =====================================================================

def start_inference_interactive():
    """Guided workflow to start model testing."""
    print_header("MODEL TESTING (INFERENCE)")
    
    # Pre-flight
    status = preflight_check(require_model=True)
    if status is None:
        return
    
    # Get available models
    try:
        resp = requests.get(f"{BACKEND_URL}/api/models")
        models = resp.json().get('models', [])
    except:
        models = []
    
    if not models:
        print("✗ No models available. Train a model first.")
        return
    
    # Model selection
    print_header("SELECT MODEL")
    print("\nAvailable trained models:\n")
    
    # Normalize models to dict format (handle both string and dict responses)
    normalized_models = []
    for m in models:
        if isinstance(m, str):
            # Backward compatibility: if API returns strings, extract info from filename
            model_name = m
            distance = None
            if "cm" in model_name:
                try:
                    distance = int(model_name.split("_")[-1].replace("cm.pth", "").replace(".pth", ""))
                except:
                    pass
            normalized_models.append({
                "name": model_name,
                "distance_cm": distance,
                "best_val_loss": None,
            })
        else:
            # Already a dict
            normalized_models.append(m)
    
    for i, m in enumerate(normalized_models):
        dist = m.get('distance_cm') or '?'
        loss = m.get('best_val_loss')
        loss_str = f"{loss:.4f}" if loss is not None else "N/A"
        model_name = m.get('name') or m.get('model_name', 'Unknown')
        print(f"  {i+1}. {model_name}")
        print(f"     Distance: {dist}cm | Val Loss: {loss_str}")
        print()
    
    choice = get_input("Select model number", 1, int, required=True)
    if not (1 <= choice <= len(normalized_models)):
        print("✗ Invalid selection.")
        return
    
    selected_model = normalized_models[choice - 1]
    model_name = selected_model.get('name') or selected_model.get('model_name', 'Unknown')
    model_distance = selected_model.get('distance_cm') or 30
    
    # Distance configuration (for inference, use same total distance as training)
    print_header("BEAM SPLITTER DISTANCE CONFIGURATION")
    
    print(f"\n📏 Model '{model_name}' was trained at {model_distance}cm total distance.")
    print("   For accurate testing, use the SAME total distance as training.")
    print("   (For 45° beam splitter: total = display_to_splitter + splitter_to_eye)\n")
    
    total_distance = get_input("   Total distance: Eye to virtual image (cm)", model_distance, int, required=True)
    
    if total_distance != model_distance:
        print(f"\n   ⚠ Warning: Using {total_distance}cm with a model trained at {model_distance}cm")
        print("   Results may be less accurate.")
        if not confirm("   Continue anyway?"):
            return
    
    # Confirmation
    print("\n" + "-" * 50)
    print("INFERENCE CONFIGURATION:")
    print(f"  • Model:          {model_name}")
    print(f"  • Total distance:  {total_distance} cm")
    print("-" * 50)
    
    if not confirm("\nStart INFERENCE/TESTING?"):
        print("Cancelled.")
        return
    
    # Start
    print("\n⏳ Starting inference...")
    try:
        payload = {
            "model_name": model_name,
            "total_distance_cm": total_distance,
        }
        response = requests.post(f"{BACKEND_URL}/api/control/inference/start", json=payload)
        data = response.json()
        
        if response.status_code == 200 and "error" not in data:
            print("\n" + "=" * 40)
            print("  ✓ INFERENCE STARTED")
            print("=" * 40)
            print(f"\n  Model: {model_name}")
            print(f"  Total distance: {total_distance}cm")
            print("\n  🔍 Testing in progress...")
            print("  → Type 'stop' or press Ctrl+C to stop\n")
            
            # Monitor
            monitor_inference()
        else:
            print(f"\n✗ Failed to start: {data.get('error', 'Unknown error')}")
            
    except Exception as e:
        print(f"\n✗ Error: {e}")

def monitor_inference():
    """Monitor inference with live updates."""
    try:
        while True:
            status = get_backend_status()
            if not status.get('inference_active'):
                print("\n✓ Inference stopped.")
                break
            
            count = status.get('inference_frame_count', 0)
            print(f"\r  Frames processed: {count}", end="", flush=True)
            
            time.sleep(0.5)
    except KeyboardInterrupt:
        print("\n\n⏳ Stopping inference...")
        stop_inference()

def stop_inference():
    """Stop MODEL TESTING (INFERENCE) and show detailed results with diagnosis."""
    try:
        response = requests.post(f"{BACKEND_URL}/api/control/inference/stop")
        data = response.json()
        
        if "error" in data:
            print(f"\n✗ Error: {data['error']}")
            return
        
        print("\n" + "=" * 50)
        print("  MODEL TESTING (INFERENCE) COMPLETE - RESULTS")
        print("=" * 50)
        
        print(f"\n  Frames processed: {data.get('frames_processed', 0)}")
        
        metrics = data.get('metrics', {})
        if "error" not in metrics:
            print("\n  📊 ANGLE ERRORS:")
            print(f"     Mean Horizontal: {metrics.get('mean_error_horizontal', 0):.2f}°")
            print(f"     Mean Vertical:   {metrics.get('mean_error_vertical', 0):.2f}°")
            print(f"     RMS Error:       {metrics.get('rms_error', 0):.2f}°")
            
            if 'mean_velocity_error_horizontal' in metrics:
                print("\n  📈 VELOCITY ANALYSIS:")
                print(f"     Velocity Error (H): {metrics['mean_velocity_error_horizontal']:.2f}°/s")
                print(f"     Velocity Error (V): {metrics['mean_velocity_error_vertical']:.2f}°/s")
                
                if 'smooth_pursuit_gain_horizontal' in metrics:
                    gain = metrics['smooth_pursuit_gain_horizontal']
                    print(f"     Pursuit Gain:       {gain:.3f}")
            
            if 'lag_horizontal_ms' in metrics:
                print("\n  ⏱️  LAG ANALYSIS:")
                lag = metrics['lag_horizontal_ms']
                print(f"     Horizontal Lag:     {lag:.1f}ms")
                
                # Interpretation
                if abs(lag) < 50:
                    print("     → Good temporal alignment")
                elif lag > 0:
                    print(f"     → Predicted lags behind by {lag:.0f}ms")
                else:
                    print(f"     → Predicted leads by {abs(lag):.0f}ms")
            
            # Overall diagnosis
            print("\n  🩺 DIAGNOSIS:")
            rms = metrics.get('rms_error', 999)
            gain = metrics.get('smooth_pursuit_gain_horizontal', 1.0)
            
            if rms < 2.0 and 0.8 <= gain <= 1.2:
                print("     ✓ Normal smooth pursuit function")
            elif rms < 5.0:
                print("     ⚠ Mild smooth pursuit impairment")
            else:
                print("     ✗ Significant smooth pursuit deficit")
        
        print("\n" + "=" * 50)
        print()
    except Exception as e:
        print(f"✗ Error: {e}")

def stop_all():
    """Stop any active recording or inference, showing appropriate results."""
    # Check what's actually active first
    status = get_backend_status()
    
    if status.get("error"):
        print("✗ Could not connect to backend.")
        return
    
    # Check inference first (more specific)
    if status.get("inference_active"):
        print("⏳ Stopping inference...")
        stop_inference()  # This will show inference results
        return
    
    # Check recording
    if status.get("recording_active"):
        print("⏳ Stopping recording...")
        stop_recording()  # This will show recording summary
        return
    
    # Nothing active
    print("✓ No active recording or inference to stop.")

# =====================================================================
# LIST MODELS AND DATA
# =====================================================================

def list_models():
    """List all available trained models."""
    print_header("AVAILABLE MODELS")
    
    db_path = Path(__file__).parent / "Dataset" / "smooth_pursuit_data.db"
    if not db_path.exists():
        print("✗ No database found. No models available.")
        return
    
    try:
        from database import SmoothPursuitDB
        db = SmoothPursuitDB()
        models = db.get_all_models()
        db.close()
        
        if not models:
            print("\n✗ No models found in database.")
            print("  → Train a model first: python control.py train")
            return
        
        print(f"\nFound {len(models)} model(s):\n")
        
        for i, m in enumerate(models, 1):
            name = m["model_name"]
            dist = m["distance_cm"] if m["distance_cm"] is not None else "?"
            loss = m["best_val_loss"] if m["best_val_loss"] is not None else None
            loss_str = f"{loss:.4f}" if loss is not None else "N/A"
            created = m["created_at"] if m["created_at"] is not None else "?"
            
            print(f"  {i}. {name}")
            print(f"     Distance: {dist}cm | Val Loss: {loss_str} | Created: {created}")
            print()
        
        # Also check for model files
        models_dir = Path(__file__).parent.parent / "mobilenet" / "checkpoints"
        if models_dir.exists():
            model_files = list(models_dir.glob("model_*.pth"))
            db_model_names = {m["model_name"] for m in models}
            orphaned = [f.name for f in model_files if f.name not in db_model_names]
            
            if orphaned:
                print(f"\n⚠ Warning: {len(orphaned)} model file(s) not in database:")
                for name in orphaned:
                    print(f"     - {name}")
                print("  → These may be from old training runs.")
        
    except Exception as e:
        print(f"✗ Error loading models: {e}")

def list_data():
    """List all available training data (sessions)."""
    print_header("AVAILABLE TRAINING DATA")
    
    db_path = Path(__file__).parent / "Dataset" / "smooth_pursuit_data.db"
    if not db_path.exists():
        print("✗ No database found. No data available.")
        return
    
    try:
        from database import SmoothPursuitDB
        db = SmoothPursuitDB()
        
        # Get all sessions grouped by distance
        cursor = db.conn.cursor()
        cursor.execute("""
            SELECT 
                distance_cm,
                COUNT(*) as session_count,
                SUM(frame_count) as total_frames,
                MIN(created_at) as first_session,
                MAX(created_at) as last_session
            FROM sessions
            GROUP BY distance_cm
            ORDER BY distance_cm
        """)
        distance_groups = cursor.fetchall()
        
        if not distance_groups:
            print("\n✗ No training sessions found.")
            print("  → Collect data first: python control.py start")
            db.close()
            return
        
        print(f"\nTraining Data Summary:\n")
        
        total_sessions = 0
        total_frames = 0
        
        for group in distance_groups:
            dist = group["distance_cm"]
            sessions = group["session_count"]
            frames = group["total_frames"] or 0
            first = group["first_session"]
            last = group["last_session"]
            
            total_sessions += sessions
            total_frames += frames
            
            print(f"  📊 {dist}cm (total distance):")
            print(f"     Sessions: {sessions} | Frames: {frames}")
            print(f"     First: {first} | Last: {last}")
            print()
        
        print(f"  Total: {total_sessions} session(s), {total_frames} frame(s)")
        print()
        
        # Show individual sessions
        print("Individual Sessions:\n")
        cursor.execute("""
            SELECT id, session_id, distance_cm, frame_count, created_at, notes
            FROM sessions
            ORDER BY created_at DESC
            LIMIT 20
        """)
        sessions = cursor.fetchall()
        
        for i, s in enumerate(sessions, 1):
            session_id = s["session_id"]
            dist = s["distance_cm"]
            frames = s["frame_count"] or 0
            created = s["created_at"]
            notes = s["notes"] if "notes" in s else None
            
            print(f"  {i}. {session_id}")
            print(f"     Total Distance: {dist}cm | Frames: {frames} | Created: {created}")
            if notes:
                print(f"     Notes: {notes}")
            print()
        
        if len(sessions) == 20:
            cursor.execute("SELECT COUNT(*) FROM sessions")
            total = cursor.fetchone()[0]
            print(f"  ... and {total - 20} more session(s)")
        
        db.close()
        
    except Exception as e:
        print(f"✗ Error loading data: {e}")

# =====================================================================
# DELETE FUNCTIONALITY
# =====================================================================

def delete_interactive():
    """Interactive deletion of models or data."""
    print_header("DELETE MODELS OR DATA")
    
    db_path = Path(__file__).parent / "Dataset" / "smooth_pursuit_data.db"
    if not db_path.exists():
        print("✗ No database found.")
        return
    
    print("\nWhat would you like to delete?")
    print("  1. Models (trained model files and database records)")
    print("  2. Data (recording sessions and frames)")
    print("  3. Cancel")
    
    choice = get_input("\nSelect option (1-3)", 3, int, required=True)
    
    if choice == 1:
        delete_models_interactive()
    elif choice == 2:
        delete_data_interactive()
    else:
        print("Cancelled.")

def delete_models_interactive():
    """Interactive model deletion."""
    print_header("DELETE MODELS")
    
    try:
        from database import SmoothPursuitDB
        db = SmoothPursuitDB()
        models = db.get_all_models()
        
        if not models:
            print("\n✗ No models found.")
            db.close()
            return
        
        print(f"\nAvailable models ({len(models)}):\n")
        
        for i, m in enumerate(models, 1):
            name = m["model_name"]
            dist = m["distance_cm"] if m["distance_cm"] is not None else "?"
            loss = m["best_val_loss"] if m["best_val_loss"] is not None else None
            loss_str = f"{loss:.4f}" if loss is not None else "N/A"
            
            print(f"  {i}. {name}")
            print(f"     Distance: {dist}cm | Val Loss: {loss_str}")
            print()
        
        print("  0. Cancel")
        print()
        
        selections = get_input("Select model(s) to delete (comma-separated, e.g., 1,3,5)", "", str)
        
        if not selections or selections.strip() == "0":
            print("Cancelled.")
            db.close()
            return
        
        # Parse selections
        try:
            indices = [int(x.strip()) for x in selections.split(",")]
            indices = [i for i in indices if 1 <= i <= len(models)]
        except ValueError:
            print("✗ Invalid input. Use comma-separated numbers (e.g., 1,3,5)")
            db.close()
            return
        
        if not indices:
            print("✗ No valid selections.")
            db.close()
            return
        
        # Show what will be deleted
        print("\n" + "=" * 50)
        print("MODELS TO DELETE:")
        print("=" * 50)
        for idx in indices:
            m = models[idx - 1]
            print(f"  • {m['model_name']}")
        print("=" * 50)
        
        if not confirm("\n⚠ WARNING: This will delete model files and database records. Continue?"):
            print("Cancelled.")
            db.close()
            return
        
        # Delete models
        models_dir = Path(__file__).parent.parent / "mobilenet" / "checkpoints"
        deleted_count = 0
        
        for idx in indices:
            m = models[idx - 1]
            model_name = m["model_name"]
            model_id = m["id"]
            
            # Delete from database
            cursor = db.conn.cursor()
            cursor.execute("DELETE FROM training_runs WHERE model_id = ?", (model_id,))
            cursor.execute("DELETE FROM inference_sessions WHERE model_id = ?", (model_id,))
            cursor.execute("DELETE FROM models WHERE id = ?", (model_id,))
            db.conn.commit()
            
            # Delete file
            model_file = models_dir / model_name
            if model_file.exists():
                try:
                    model_file.unlink()
                    print(f"  ✓ Deleted: {model_name}")
                    deleted_count += 1
                except Exception as e:
                    print(f"  ✗ Failed to delete file {model_name}: {e}")
            else:
                print(f"  ⚠ File not found: {model_name} (database record deleted)")
                deleted_count += 1
        
        print(f"\n✓ Deleted {deleted_count} model(s).")
        db.close()
        
    except Exception as e:
        print(f"✗ Error: {e}")

def delete_data_interactive():
    """Interactive data deletion."""
    print_header("DELETE TRAINING DATA")
    
    try:
        from database import SmoothPursuitDB
        db = SmoothPursuitDB()
        
        # Get all sessions
        cursor = db.conn.cursor()
        cursor.execute("""
            SELECT id, session_id, distance_cm, frame_count, created_at, notes
            FROM sessions
            ORDER BY created_at DESC
        """)
        sessions = cursor.fetchall()
        
        if not sessions:
            print("\n✗ No sessions found.")
            db.close()
            return
        
        print(f"\nAvailable sessions ({len(sessions)}):\n")
        
        for i, s in enumerate(sessions, 1):
            session_id = s["session_id"]
            dist = s["distance_cm"]
            frames = s["frame_count"] or 0
            created = s["created_at"]
            notes = s["notes"] if "notes" in s else None
            
            print(f"  {i}. {session_id}")
            print(f"     Total Distance: {dist}cm | Frames: {frames} | Created: {created}")
            if notes:
                print(f"     Notes: {notes}")
            print()
        
        print("  0. Cancel")
        print("  all. Delete all sessions")
        print()
        
        selection = get_input("Select session(s) to delete (comma-separated, e.g., 1,3,5 or 'all')", "", str)
        
        if not selection or selection.strip() == "0":
            print("Cancelled.")
            db.close()
            return
        
        # Handle "all" case
        if selection.strip().lower() == "all":
            if not confirm("\n⚠ WARNING: This will delete ALL sessions and frames. Continue?"):
                print("Cancelled.")
                db.close()
                return
            
            # Delete all
            cursor.execute("DELETE FROM inference_results")
            cursor.execute("DELETE FROM inference_sessions")
            cursor.execute("DELETE FROM training_runs")
            cursor.execute("DELETE FROM frames")
            cursor.execute("DELETE FROM sessions")
            db.conn.commit()
            
            # Delete session directories
            data_dir = Path(__file__).parent / "Dataset" / "smooth_pursuit_data"
            if data_dir.exists():
                for session_dir in data_dir.glob("session_*"):
                    import shutil
                    try:
                        shutil.rmtree(session_dir)
                    except Exception as e:
                        print(f"  ⚠ Failed to delete directory {session_dir.name}: {e}")
            
            print("\n✓ Deleted all sessions and data.")
            db.close()
            return
        
        # Parse selections
        try:
            indices = [int(x.strip()) for x in selection.split(",")]
            indices = [i for i in indices if 1 <= i <= len(sessions)]
        except ValueError:
            print("✗ Invalid input. Use comma-separated numbers (e.g., 1,3,5) or 'all'")
            db.close()
            return
        
        if not indices:
            print("✗ No valid selections.")
            db.close()
            return
        
        # Show what will be deleted
        print("\n" + "=" * 50)
        print("SESSIONS TO DELETE:")
        print("=" * 50)
        total_frames = 0
        for idx in indices:
            s = sessions[idx - 1]
            print(f"  • {s['session_id']} ({s['frame_count'] or 0} frames)")
            total_frames += s['frame_count'] or 0
        print(f"\n  Total: {len(indices)} session(s), {total_frames} frame(s)")
        print("=" * 50)
        
        if not confirm("\n⚠ WARNING: This will delete sessions, frames, and image files. Continue?"):
            print("Cancelled.")
            db.close()
            return
        
        # Delete sessions
        session_ids = [sessions[idx - 1]["id"] for idx in indices]
        placeholders = ",".join("?" * len(session_ids))
        
        # Delete related data
        cursor.execute(f"DELETE FROM inference_results WHERE inference_session_id IN (SELECT id FROM inference_sessions WHERE distance_cm IN (SELECT distance_cm FROM sessions WHERE id IN ({placeholders})))", session_ids)
        cursor.execute(f"DELETE FROM training_runs WHERE session_id IN ({placeholders})", session_ids)
        cursor.execute(f"DELETE FROM frames WHERE session_id IN ({placeholders})", session_ids)
        
        # Get session directories before deleting
        session_dirs_to_delete = []
        for idx in indices:
            s = sessions[idx - 1]
            session_id_str = s["session_id"]
            dist = s["distance_cm"]
            session_dir = Path(__file__).parent / "Dataset" / "smooth_pursuit_data" / f"session_{session_id_str}_{dist}cm"
            if session_dir.exists():
                session_dirs_to_delete.append(session_dir)
        
        # Delete from database
        cursor.execute(f"DELETE FROM sessions WHERE id IN ({placeholders})", session_ids)
        db.conn.commit()
        
        # Delete session directories
        for session_dir in session_dirs_to_delete:
            import shutil
            try:
                shutil.rmtree(session_dir)
            except Exception as e:
                print(f"  ⚠ Failed to delete directory {session_dir.name}: {e}")
        
        print(f"\n✓ Deleted {len(indices)} session(s) and {total_frames} frame(s).")
        db.close()
        
    except Exception as e:
        print(f"✗ Error: {e}")

# =====================================================================
# TRAINING (calls train_smooth_pursuit.py)
# =====================================================================

def start_training_interactive():
    """Guided interactive workflow to start training."""
    print_header("MODEL TRAINING")
    
    # Check for training data
    db_path = Path(__file__).parent / "Dataset" / "smooth_pursuit_data.db"
    if not db_path.exists():
        print("✗ No training data found.")
        print("  → Collect data first using option 2.")
        return
    
    # Get available distances and their data
    try:
        import sqlite3
        conn = sqlite3.connect(str(db_path))
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()
        
        # Get distances with session and frame counts
        cursor.execute("""
            SELECT 
                s.distance_cm,
                COUNT(DISTINCT s.id) as session_count,
                COUNT(f.id) as frame_count
            FROM sessions s
            LEFT JOIN frames f ON f.session_id = s.id
            GROUP BY s.distance_cm
            ORDER BY s.distance_cm
        """)
        distances_data = cursor.fetchall()
        conn.close()
        
        if not distances_data:
            print("✗ No training sessions found in database.")
            return
        
        print("\n📊 Available Training Data (total distance):\n")
        distance_options = []
        for row in distances_data:
            dist = row["distance_cm"]
            sessions = row["session_count"]
            frames = row["frame_count"]
            print(f"  {len(distance_options) + 1}. {dist}cm: {sessions} session(s), {frames} frames")
            distance_options.append(dist)
        
    except Exception as e:
        print(f"⚠ Could not read database: {e}")
        return
    
    # Step 1: Select distance(s) to train on
    print("\n" + "-" * 50)
    print("STEP 1: SELECT DISTANCE TO TRAIN ON")
    print("-" * 50)
    
    if len(distance_options) == 1:
        distance = distance_options[0]
        print(f"\n✓ Only one distance available: {distance}cm")
    else:
        choice = get_input(f"\nSelect distance (1-{len(distance_options)})", 1, int, required=True)
        if not (1 <= choice <= len(distance_options)):
            print("✗ Invalid selection.")
            return
        distance = distance_options[choice - 1]
    
    print(f"\n✓ Selected: {distance}cm")
    
    # Step 2: Choose training mode (from scratch, resume, or fine-tune)
    print("\n" + "-" * 50)
    print("STEP 2: CHOOSE TRAINING MODE")
    print("-" * 50)
    print("\n  1. 🆕 Train from scratch (ImageNet pretrained)")
    print("  2. 🔄 Resume training (continue from checkpoint)")
    print("  3. 🔧 Fine-tune from existing model")
    
    mode_choice = get_input("\nSelect mode (1-3)", 1, int, required=True)
    if not (1 <= mode_choice <= 3):
        print("✗ Invalid selection.")
        return
    
    resume_model = None
    fine_tune_model = None
    
    if mode_choice == 2:  # Resume
        # Get available models for this distance
        try:
            conn = sqlite3.connect(str(db_path))
            conn.row_factory = sqlite3.Row
            cursor = conn.cursor()
            cursor.execute("""
                SELECT model_name, best_val_loss, created_at 
                FROM models 
                WHERE distance_cm = ?
                ORDER BY created_at DESC
            """, (distance,))
            models = cursor.fetchall()
            conn.close()
            
            if not models:
                print(f"\n✗ No existing models found for {distance}cm.")
                print("  → Use 'Train from scratch' instead.")
                return
            
            print(f"\n📦 Available models for {distance}cm:\n")
            for i, m in enumerate(models):
                loss = m["best_val_loss"] if "best_val_loss" in m else 0
                model_name = m["model_name"]
                print(f"  {i+1}. {model_name} (Val Loss: {loss:.4f})")
            
            model_choice = get_input("\nSelect model to resume from", 1, int, required=True)
            if not (1 <= model_choice <= len(models)):
                print("✗ Invalid selection.")
                return
            
            resume_model = models[model_choice - 1]["model_name"]
            print(f"✓ Will resume from: {resume_model}")
            
        except Exception as e:
            print(f"⚠ Error loading models: {e}")
            return
    
    elif mode_choice == 3:  # Fine-tune
        # Get all available models (any distance)
        try:
            conn = sqlite3.connect(str(db_path))
            conn.row_factory = sqlite3.Row
            cursor = conn.cursor()
            cursor.execute("""
                SELECT model_name, distance_cm, best_val_loss, created_at 
                FROM models 
                ORDER BY created_at DESC
            """)
            models = cursor.fetchall()
            conn.close()
            
            if not models:
                print("\n✗ No existing models found.")
                print("  → Use 'Train from scratch' instead.")
                return
            
            print(f"\n📦 Available models:\n")
            for i, m in enumerate(models):
                dist = m["distance_cm"]
                loss = m["best_val_loss"] if "best_val_loss" in m else 0
                model_name = m["model_name"]
                print(f"  {i+1}. {model_name} ({dist}cm, Val Loss: {loss:.4f})")
            
            model_choice = get_input("\nSelect model to fine-tune from", 1, int, required=True)
            if not (1 <= model_choice <= len(models)):
                print("✗ Invalid selection.")
                return
            
            fine_tune_model = models[model_choice - 1]["model_name"]
            model_dist = models[model_choice - 1]["distance_cm"]
            print(f"✓ Will fine-tune from: {fine_tune_model}")
            if model_dist != distance:
                print(f"  ⚠ Note: Model trained at {model_dist}cm, now training at {distance}cm")
            
        except Exception as e:
            print(f"⚠ Error loading models: {e}")
            return
    
    # Step 3: Training parameters
    print("\n" + "-" * 50)
    print("STEP 3: TRAINING PARAMETERS")
    print("-" * 50)
    
    epochs = get_input("\nNumber of epochs", 50, int)
    batch_size = get_input("Batch size", 32, int)
    learning_rate = get_input("Learning rate", 0.001, float)
    
    # Summary
    print("\n" + "=" * 50)
    print("TRAINING CONFIGURATION")
    print("=" * 50)
    print(f"  • Distance:        {distance}cm")
    print(f"  • Mode:            ", end="")
    if mode_choice == 1:
        print("Train from scratch")
    elif mode_choice == 2:
        print(f"Resume from {resume_model}")
    else:
        print(f"Fine-tune from {fine_tune_model}")
    print(f"  • Epochs:          {epochs}")
    print(f"  • Batch size:      {batch_size}")
    print(f"  • Learning rate:   {learning_rate}")
    print("=" * 50)
    
    if not confirm("\nStart training?"):
        print("Cancelled.")
        return
    
    # Build command
    print("\n⏳ Starting training...\n")
    import subprocess
    
    cmd = [
        sys.executable, 
        str(Path(__file__).parent / "train_smooth_pursuit.py"),
        "--data_dir", "Dataset/smooth_pursuit_data",
        "--distance", str(distance),
        "--epochs", str(epochs),
        "--batch_size", str(batch_size),
        "--lr", str(learning_rate),
    ]
    
    if resume_model:
        cmd.extend(["--resume", resume_model])
    elif fine_tune_model:
        cmd.extend(["--fine_tune", fine_tune_model])
    
    try:
        subprocess.run(cmd, cwd=str(Path(__file__).parent))
    except KeyboardInterrupt:
        print("\nTraining interrupted.")

# =====================================================================
# MAIN MENU
# =====================================================================

def main_menu():
    """Main interactive menu."""
    while True:
        print_header("SMOOTH PURSUIT EYE TRACKING")
        
        # Quick status
        status = get_backend_status()
        if "error" not in status:
            print_status(status, compact=True)
        else:
            print("⚠ Backend not connected")
        
        print("\n  1. 📊 Check System Status")
        print("  2. 📹 Collect Training Data")
        print("  3. 🧠 Train Model")
        print("  4. 🔍 Test Model (Inference)")
        print("  5. 🛑 Stop All")
        print("  6. 🚪 Exit")
        
        choice = get_input("\nSelect option", 1, int)
        
        if choice == 1:
            print_status(get_backend_status())
            input("\nPress Enter to continue...")
        elif choice == 2:
            start_recording_interactive()
        elif choice == 3:
            start_training_interactive()
        elif choice == 4:
            start_inference_interactive()
        elif choice == 5:
            stop_all()
        elif choice == 6:
            print("\nGoodbye! 👋\n")
            sys.exit(0)
        else:
            print("Invalid option.")

def main():
    """Entry point."""
    # Handle Ctrl+C gracefully
    signal.signal(signal.SIGINT, lambda s, f: (stop_all(), sys.exit(0)))
    
    if len(sys.argv) > 1:
        # Script mode for automation
        cmd = sys.argv[1].lower()
        
        if cmd == "start":
            # Use interactive mode for starting data collection (prevents mistakes)
            start_recording_interactive()
            
        elif cmd == "stop":
            stop_all()
            
        elif cmd == "test" or cmd == "inference":
            # Use interactive mode for inference (prevents mistakes)
            start_inference_interactive()
            
        elif cmd == "test-stop":
            stop_inference()
            
        elif cmd == "status":
            print_status(get_backend_status())
            
        elif cmd == "train":
            # Use interactive mode for training (easier and more flexible)
            start_training_interactive()
            
        elif cmd == "models":
            list_models()
            
        elif cmd == "data":
            list_data()
            
        elif cmd == "delete":
            delete_interactive()
            
        elif cmd in ["help", "-h", "--help"]:
            print("\nUsage: python control.py [command] [args]")
            print("\nCommands:")
            print("  (no args)     Interactive menu")
            print("  start         Interactive training data collection (prompts for distance)")
            print("  stop          Stop recording/inference")
            print("  test          Interactive model testing (select model, distance, etc.)")
            print("  test-stop     Stop inference and show results")
            print("  train         Interactive training (select distance, mode, etc.)")
            print("  models        List all available trained models")
            print("  data          List all available training data (sessions)")
            print("  delete        Interactive deletion (models or data)")
            print("  status        Show system status")
            print()
        else:
            print(f"Unknown command: {cmd}")
            print("Use 'python control.py help' for usage.")
    else:
        # Interactive mode
        try:
            main_menu()
        except KeyboardInterrupt:
            print("\n\nExiting...")
            stop_all()
            sys.exit(0)

if __name__ == "__main__":
    main()
