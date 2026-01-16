"""
Configuration file for spatial audio navigation system
"""

import os
from pathlib import Path

# Paths
PROJECT_ROOT = Path(__file__).parent
VIDEO_INPUT_PATH = os.getenv("VIDEO_INPUT_PATH", "/kaggle/input/msrvtt/MSR-VTT/TrainValVideo")
OUTPUT_DIR = PROJECT_ROOT / "output"
AUDIO_OUTPUT_DIR = OUTPUT_DIR / "audio"
REPORTS_DIR = OUTPUT_DIR / "reports"
CACHE_DIR = OUTPUT_DIR / "cache"

# Create output directories
for dir_path in [OUTPUT_DIR, AUDIO_OUTPUT_DIR, REPORTS_DIR, CACHE_DIR]:
    dir_path.mkdir(parents=True, exist_ok=True)

# Video Processing
TARGET_FPS = 10
BATCH_SIZE = 5
MAX_FRAMES = None  # None for full video processing

# Object Detection (YOLO)
YOLO_MODEL = "yolov8n.pt"  # nano variant for speed
CONFIDENCE_THRESHOLD = 0.5
IOU_THRESHOLD = 0.45
MAX_DETECTIONS = 100

# Object Tracking (ByteTrack)
TRACK_THRESH = 0.5
TRACK_BUFFER = 30
MATCH_THRESH = 0.8
FRAME_RATE = TARGET_FPS

# Spatial Zones
# Horizontal zones (left, center, right)
LEFT_ZONE_END = 0.33
RIGHT_ZONE_START = 0.67

# Depth zones (near, middle, far) - based on bbox area relative to frame
NEAR_THRESHOLD = 0.15  # bbox area > 15% of frame = near
FAR_THRESHOLD = 0.05   # bbox area < 5% of frame = far

# Scene Understanding (Moondream)
MOONDREAM_MODEL = "vikhyatk/moondream2"
MOONDREAM_REVISION = "2024-08-26"
PROCESS_EVERY_N_FRAMES = 5  # Run Moondream every N frames to save compute
MOONDREAM_MAX_TOKENS = 50
SCENE_CACHE_THRESHOLD = 0.9  # Similarity threshold for caching

# Text-to-Speech
TTS_ENGINE = "pyttsx3"  # Options: pyttsx3, gtts
TTS_RATE = 150  # Words per minute
TTS_VOLUME = 0.9

# Spatial Audio
SAMPLE_RATE = 44100
AUDIO_CHANNELS = 2  # Stereo for spatial audio
PAN_STRENGTH = 0.8  # How much to pan (0-1)

# Distance audio cues
DISTANCE_VOLUME_NEAR = 1.0
DISTANCE_VOLUME_MID = 0.7
DISTANCE_VOLUME_FAR = 0.4
DISTANCE_PITCH_NEAR = 1.1
DISTANCE_PITCH_MID = 1.0
DISTANCE_PITCH_FAR = 0.9

# Temporal Consistency
MIN_POSITION_CHANGE = 0.1  # Minimum change in position to re-announce (10% of frame)
MIN_SIZE_CHANGE = 0.2  # Minimum change in size to re-announce distance
ANNOUNCEMENT_COOLDOWN = 3  # Seconds before re-announcing same object

# Priority Objects (safety-critical)
PRIORITY_OBJECTS = [
    "person", "car", "truck", "bus", "motorcycle", "bicycle",
    "stop sign", "traffic light", "fire hydrant", "bench",
    "chair", "door", "stairs", "obstacle"
]

# Performance
LOG_PERFORMANCE = True
PERFORMANCE_LOG_FILE = OUTPUT_DIR / "performance.log"

# Demo
DEMO_VIDEO_NAME = "video0.mp4"  # Sample video for demo
DEMO_MAX_FRAMES = 300  # Process first 30 seconds at 10 FPS
