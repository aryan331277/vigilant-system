# Quick Start Guide

Get up and running with the Spatial Audio Navigation System in 5 minutes!

## Prerequisites

- Python 3.8 or higher
- pip package manager
- (Optional) CUDA-capable GPU for faster processing

## Installation

### 1. Clone or Download

```bash
# If using git
git clone <repository-url>
cd spatial-audio-navigation

# Or simply navigate to the project directory
cd /path/to/project
```

### 2. Install Dependencies

```bash
pip install -r requirements.txt
```

This will install all required packages:
- OpenCV for video processing
- YOLOv8 for object detection
- ByteTrack for tracking
- Moondream for scene understanding
- pyttsx3 for text-to-speech
- scipy for audio processing

**Note**: First-time installation will download YOLO model weights (~6MB for nano variant).

### 3. Verify Installation

```bash
python -c "import cv2, numpy, torch, ultralytics; print('All dependencies OK!')"
```

## Running Your First Demo

### Option 1: With MSR-VTT Dataset

If you have access to the MSR-VTT dataset:

```bash
python demo.py --video-name video0.mp4 --max-frames 300
```

### Option 2: With Custom Video

If you have your own video file:

```bash
python demo.py --video /path/to/your/video.mp4 --max-frames 300
```

### Option 3: Create and Use Test Video

If you don't have any video:

```bash
# Create test videos
python create_test_video.py --create-defaults

# Run demo with test video
python demo.py --video test_videos/test_short.mp4
```

## Understanding the Output

After processing, you'll see:

### 1. Console Output

```
============================================================
PERFORMANCE SUMMARY
============================================================
Total Frames Processed: 300
Total Time: 24.50s
Average FPS: 12.24
Total Detections: 1250
Total Announcements: 45
Total Alerts: 3

Time Breakdown:
  Video Loading: 0.50s (2.0%)
  Detection: 12.30s (50.2%)
  Scene Reasoning: 8.20s (33.5%)
  Audio Generation: 3.50s (14.3%)
============================================================
```

### 2. Generated Files

```
output/
├── audio/                          # Spatial audio files
│   ├── spatial_0.50_person.wav
│   ├── spatial_1.20_chair.wav
│   ├── scene_2.50.wav
│   └── alert_3.00.wav
├── reports/                        # Processing reports
│   ├── video_report.json          # Summary
│   ├── video_frames.json          # Frame details
│   └── video_audio_sequence.json  # Audio sequence
└── cache/                          # Cached data
    └── moondream_cache.json
```

### 3. Audio Files

Listen to the generated spatial audio:
- Files are named by timestamp and object
- Stereo WAV format (44.1kHz)
- Left/right panning indicates position
- Volume indicates distance

### 4. Reports

Check `output/reports/video_report.json` for detailed metrics:

```json
{
  "performance_metrics": {
    "total_frames": 300,
    "avg_fps": 12.24,
    "total_announcements": 45
  },
  "audio_sequence": [
    {
      "timestamp": 0.50,
      "text": "person center close",
      "position": "center",
      "depth": "near"
    }
  ]
}
```

## Common Commands

### Process First 100 Frames

```bash
python demo.py --video my_video.mp4 --max-frames 100
```

### Process Full Video

```bash
python demo.py --video my_video.mp4
```

### Use GPU Acceleration

```bash
python demo.py --video my_video.mp4 --device cuda
```

### Quiet Mode (Less Output)

```bash
python demo.py --video my_video.mp4 --no-progress
```

## Testing the System

Run the test suite:

```bash
python test_pipeline.py
```

Or with pytest (if installed):

```bash
pytest test_pipeline.py -v
```

## Adjusting Settings

Edit `config.py` to customize behavior:

```python
# Process faster (lower quality)
TARGET_FPS = 5
PROCESS_EVERY_N_FRAMES = 10

# Process slower (higher quality)
TARGET_FPS = 15
PROCESS_EVERY_N_FRAMES = 3

# Adjust detection sensitivity
CONFIDENCE_THRESHOLD = 0.3  # More detections
CONFIDENCE_THRESHOLD = 0.7  # Fewer detections

# Adjust spatial zones
LEFT_ZONE_END = 0.25       # Smaller left zone
RIGHT_ZONE_START = 0.75    # Smaller right zone
```

## Troubleshooting

### Problem: "ultralytics not found"

```bash
pip install ultralytics
```

### Problem: "boxmot not found"

```bash
pip install boxmot
```

### Problem: "pyttsx3 initialization failed"

**Linux:**
```bash
sudo apt-get install espeak
pip install pyttsx3
```

**Mac:**
```bash
# Uses native TTS, should work out of the box
pip install pyttsx3
```

**Windows:**
```bash
# Uses native SAPI5, should work out of the box
pip install pyttsx3
```

### Problem: "CUDA out of memory"

```bash
# Use CPU instead
python demo.py --video my_video.mp4 --device cpu
```

### Problem: "Video file not found"

- Check the path is correct
- Use absolute path if relative path doesn't work
- Ensure video format is supported (MP4, AVI, MOV)

### Problem: "No audio generated"

- Check TTS engine is working: `python -c "import pyttsx3; e = pyttsx3.init(); e.say('test'); e.runAndWait()"`
- Check output directory permissions
- Look for errors in console output

## Next Steps

### 1. Explore the Code

- `spatial_audio_pipeline.py` - Main orchestration
- `detection_tracker.py` - Object detection and tracking
- `scene_reasoner.py` - Scene understanding
- `audio_generator.py` - Spatial audio generation

### 2. Customize for Your Use Case

- Add custom object classes in `config.py`
- Adjust spatial zones for your application
- Modify audio effects in `audio_generator.py`

### 3. Batch Process Videos

```python
from spatial_audio_pipeline import process_batch_videos

videos = ["video1.mp4", "video2.mp4", "video3.mp4"]
reports = process_batch_videos(videos, max_frames=500, device="cpu")
```

### 4. Integrate with Your Application

```python
from spatial_audio_pipeline import SpatialAudioPipeline

pipeline = SpatialAudioPipeline(device="cpu")
report = pipeline.process_video("input.mp4")

# Access results
print(f"Generated {len(report['audio_sequence'])} audio cues")
```

## Performance Tips

### For Maximum Speed (Real-time on CPU)

```python
# In config.py
TARGET_FPS = 5
PROCESS_EVERY_N_FRAMES = 10
YOLO_MODEL = "yolov8n.pt"
CONFIDENCE_THRESHOLD = 0.6
```

### For Maximum Accuracy

```python
# In config.py
TARGET_FPS = 15
PROCESS_EVERY_N_FRAMES = 3
YOLO_MODEL = "yolov8s.pt"
CONFIDENCE_THRESHOLD = 0.4
```

### For GPU Processing

```bash
# Use CUDA
python demo.py --video my_video.mp4 --device cuda

# In config.py, you can use larger models
YOLO_MODEL = "yolov8m.pt"  # Medium model
```

## Understanding Spatial Audio

The system generates stereo audio with three types of cues:

### 1. Horizontal Position (Panning)
- **Left objects**: More audio in left channel
- **Center objects**: Equal in both channels
- **Right objects**: More audio in right channel

### 2. Distance (Volume & Pitch)
- **Near**: Loud, slightly higher pitch
- **Middle**: Medium volume, normal pitch
- **Far**: Quiet, slightly lower pitch

### 3. Priority Alerts
- Safety-critical objects (person, car, etc.)
- Louder volume
- Announced with "Warning:" or "Caution:"

## Example Use Cases

### 1. Indoor Navigation
```bash
# Process indoor video with emphasis on obstacles
python demo.py --video indoor_walkthrough.mp4 --max-frames 600
```

### 2. Outdoor Scene Understanding
```bash
# Process outdoor video with vehicle detection
python demo.py --video street_view.mp4 --device cuda
```

### 3. Training Data Creation
```bash
# Process multiple videos for analysis
python -c "
from spatial_audio_pipeline import process_batch_videos
videos = ['train1.mp4', 'train2.mp4', 'train3.mp4']
process_batch_videos(videos, device='cuda')
"
```

## Getting Help

- Check `README.md` for comprehensive documentation
- See `ARCHITECTURE.md` for system design details
- Review `test_pipeline.py` for usage examples
- Open an issue on GitHub for bug reports

## What's Next?

Now that you have the system running:

1. ✅ Process your first video
2. ✅ Listen to generated spatial audio
3. ✅ Review performance reports
4. ⬜ Experiment with different settings
5. ⬜ Try batch processing
6. ⬜ Customize for your use case
7. ⬜ Integrate into your application

Happy coding! 🎉
