#!/usr/bin/env python3
"""
Quick script to load a model and show inference status.
"""

import sys
import requests
from pathlib import Path

BACKEND_URL = "http://localhost:8000"

def list_models():
    """List available trained models."""
    try:
        response = requests.get(f"{BACKEND_URL}/api/models")
        data = response.json()
        models = data.get("models", [])
        
        if models:
            print("Available models:")
            for model in models:
                print(f"  - {model}")
        else:
            print("No models found. Train one first:")
            print("  python train_smooth_pursuit.py --data_dir Dataset/smooth_pursuit_data --distance 30")
        return models
    except requests.exceptions.ConnectionError:
        print("Error: Cannot connect to backend. Is it running?")
        return []

def load_model(model_name):
    """Load a model for inference."""
    try:
        response = requests.post(f"{BACKEND_URL}/api/select-model/{model_name}")
        data = response.json()
        
        if "error" in data:
            print(f"Error: {data['error']}")
            return False
        else:
            print(f"✓ Model loaded: {model_name}")
            return True
    except requests.exceptions.ConnectionError:
        print("Error: Cannot connect to backend. Is it running?")
        return False

def main():
    if len(sys.argv) < 2:
        print("Usage: python quick_inference.py [command]")
        print()
        print("Commands:")
        print("  list              - List available models")
        print("  load <model>      - Load a model for inference")
        print("  status            - Check if model is loaded")
        print()
        print("Examples:")
        print("  python quick_inference.py list")
        print("  python quick_inference.py load model_30cm.pth")
        sys.exit(1)
    
    command = sys.argv[1].lower()
    
    if command == "list":
        list_models()
    elif command == "load":
        if len(sys.argv) < 3:
            print("Error: Please specify model name")
            print("Example: python quick_inference.py load model_30cm.pth")
            sys.exit(1)
        model_name = sys.argv[2]
        load_model(model_name)
        print("\nModel is ready for inference!")
        print("Open http://localhost:8000 and click 'START TEST'")
    elif command == "status":
        models = list_models()
        print("\nTo load a model:")
        print("  python quick_inference.py load model_30cm.pth")
    else:
        print(f"Unknown command: {command}")
        sys.exit(1)

if __name__ == "__main__":
    main()

