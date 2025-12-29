"""Configuration constants for pupil segmentation module."""

# OpenEDS Dataset
OPENEDS_IMAGE_SIZE = (400, 640)  # (H, W)
NUM_CLASSES = 4
CLASS_NAMES = ["background", "sclera", "iris", "pupil"]
CLASS_IDS = {"background": 0, "sclera": 1, "iris": 2, "pupil": 3}

# Preprocessing (from RITnet paper)
GAMMA = 0.8
CLAHE_CLIP_LIMIT = 1.5
CLAHE_TILE_SIZE = (8, 8)
NORMALIZE_MEAN = 0.5
NORMALIZE_STD = 0.5

# RITnet architecture
RITNET_BASE_CHANNELS = 32
RITNET_IN_CHANNELS = 1  # grayscale
