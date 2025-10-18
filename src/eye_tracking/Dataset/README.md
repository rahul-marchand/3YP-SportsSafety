# Columbia Gaze Dataset Setup

## Download

Download the dataset (5,880 images, ~6k samples) from:
https://www.cs.columbia.edu/CAVE/databases/columbia_gaze/

## Installation

1. Download the ZIP file from the link above
2. Extract the contents to this directory (`src/eye_tracking/Dataset/`)
3. **Important**: The path extracted to must be exactly:
   ```
   src/eye_tracking/Dataset/
   ```

## Expected Directory Structure

After extraction, the structure should look like:
```
src/eye_tracking/Dataset/
├── README.md
├── dataloader.py
└── columbia_gaze_data_set/
    └── Columbia Gaze Data Set/
        ├── 0001/
        │   ├── 0001_2m_0P_0V_0H.jpg
        │   ├── 0001_2m_0P_0V_5H.jpg
        │   └── ...
        ├── 0002/
        └── ...
```

The dataloader expects this exact structure. If the path differs, the code will not find the images.
