SMOOTH PURSUIT EYE TRACKING FOR CONCUSSION DETECTION

System for collecting eye tracking data and detecting smooth pursuit deficits.

QUICK START:
1. Install dependencies: ./install.sh
2. Deploy backend: ./run_with_ngrok.sh (or python3 backend.py for local)
3. Setup mobile app: see mobile_app/SETUP.txt
4. Update config.js with backend URL
5. Run app on phone

See DEPLOYMENT.txt for detailed deployment options.

WORKFLOW:

1. DATA COLLECTION
   - Run backend: cd smooth_pursuit && python backend.py
   - Update config.js with your backend IP address
   - Setup React Native app (see mobile_app/SETUP.txt)
   - Use Recording mode to collect eye images with calculated angles
   - Data saved in Dataset/smooth_pursuit_data/

2. TRAINING
   python train_smooth_pursuit.py --data_dir Dataset/smooth_pursuit_data --distance 30 --epochs 50
   - Trains model for specific distance (e.g., 30cm)
   - Saves to mobilenet/checkpoints/model_30cm.pth

3. INFERENCE
   - Load model via API: POST /api/select-model/model_30cm.pth
   - Use Inference mode in app to test smooth pursuit
   - Metrics calculated: error, gain, latency, saccade frequency

DEVICE CONFIGURATION:
Update mobile_app/app_code/config.js with your phone's specs:
- Screen physical dimensions (cm)
- Screen resolution (px)
- Front camera position (px)

BACKEND ENDPOINTS:
- GET / - Health check
- GET /api/models - List trained models
- POST /api/select-model/{name} - Load model for inference
- WebSocket /ws/recording - Data collection mode
- WebSocket /ws/inference - Testing mode

FILES:
- backend.py - FastAPI server with WebSocket endpoints
- train_smooth_pursuit.py - Training script for collected data
- requirements.txt - Python dependencies
- mobile_app/app_code/ - React Native app components

