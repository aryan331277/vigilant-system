# Spatial Audio Navigation System for Visually Impaired

A complete end-to-end offline spatial audio navigation system that processes videos to generate real-time spatial audio cues describing detected objects and their positions, designed to assist visually impaired users.

## Overview

This system processes video input (from MSR-VTT dataset or any video source) and generates spatial audio announcements that provide contextual information about objects in the scene, including:
- Object detection and tracking
- Spatial positioning (left, center, right)
- Distance estimation (near, middle, far)
- Scene understanding with natural language descriptions
- Priority alerts for safety-critical objects

## Technical Stack

- **Object Detection**: YOLOv8 (nano variant for speed)
- **Object Tracking**: ByteTrack for temporal consistency
- **Scene Understanding**: Moondream vision-language model
- **Text-to-Speech**: pyttsx3 (offline TTS)
- **Spatial Audio**: Custom stereo panning and distance effects
- **Video Processing**: OpenCV

## Features

✅ End-to-end video processing pipeline  
✅ Real-time object detection with YOLOv8  
✅ Temporal tracking with ByteTrack (prevents duplicate announcements)  
✅ Spatial zone detection (left/center/right, near/mid/far)  
✅ Scene understanding with Moondream  
✅ Spatial audio generation with position and distance cues  
✅ Priority alerts for safety-critical objects  
✅ Performance monitoring and detailed reporting  
✅ Fully offline operation  
✅ Modular architecture for easy optimization  

## Installation

### Requirements

- Python 3.8+
- Recommended: GPU with CUDA support (optional, CPU works too)

### Install Dependencies

```bash
pip install -r requirements.txt
```

### Dependencies

```
opencv-python>=4.8.0
numpy>=1.24.0
ultralytics>=8.0.0
torch>=2.0.0
torchvision>=0.15.0
boxmot>=10.0.0
transformers>=4.35.0
Pillow>=10.0.0
pyttsx3>=2.90
scipy>=1.11.0
```

## Quick Start

### Run Demo

Process a sample video with default settings:

```bash
python demo.py
```

### Process Specific Video

```bash
python demo.py --video /path/to/video.mp4 --max-frames 300
```

### Process on GPU

```bash
python demo.py --device cuda
```

### Process Video from MSR-VTT Dataset

```bash
python demo.py --video-name video0.mp4 --max-frames 300
```

## Usage

### Basic Usage

```python
from spatial_audio_pipeline import process_single_video

# Process a video
report = process_single_video(
    video_path="/path/to/video.mp4",
    max_frames=300,  # Optional: limit frames
    device="cpu"     # or "cuda"
)

# Access results
print(f"Processed {report['performance_metrics']['total_frames']} frames")
print(f"Generated {report['performance_metrics']['total_announcements']} announcements")
```

### Advanced Usage

```python
from spatial_audio_pipeline import SpatialAudioPipeline

# Initialize pipeline
pipeline = SpatialAudioPipeline(device="cpu")

# Process multiple videos
for video_path in video_paths:
    report = pipeline.process_video(video_path, max_frames=None)
    print(f"Completed: {video_path}")
```

### Batch Processing

```python
from spatial_audio_pipeline import process_batch_videos

video_paths = [
    "/path/to/video1.mp4",
    "/path/to/video2.mp4",
    "/path/to/video3.mp4"
]

reports = process_batch_videos(
    video_paths,
    max_frames=500,
    device="cuda"
)
```

## Architecture

```
Video Frame → YOLO Detection → ByteTrack → Moondream Description → TTS → Spatial Audio
                                     ↓
                          (Temporal Consistency Check)
```

### Pipeline Components

1. **Video Loader** (`video_loader.py`)
   - Loads videos from MSR-VTT or any source
   - Extracts frames at configurable FPS
   - Normalizes input for processing

2. **Detection & Tracking** (`detection_tracker.py`)
   - YOLOv8 for object detection
   - ByteTrack for multi-object tracking
   - Spatial analysis (position and depth)
   - Temporal consistency checks

3. **Scene Reasoner** (`scene_reasoner.py`)
   - Moondream for scene understanding
   - Spatial relationship extraction
   - Priority alert identification
   - Description caching

4. **Audio Generator** (`audio_generator.py`)
   - Text-to-speech synthesis
   - Spatial audio effects (panning, volume, pitch)
   - Audio sequencing

5. **Main Pipeline** (`spatial_audio_pipeline.py`)
   - Orchestrates all components
   - Performance monitoring
   - Report generation

## Configuration

Edit `config.py` to customize:

```python
# Video processing
TARGET_FPS = 10

# Detection thresholds
CONFIDENCE_THRESHOLD = 0.5

# Spatial zones
LEFT_ZONE_END = 0.33
RIGHT_ZONE_START = 0.67

# Moondream processing
PROCESS_EVERY_N_FRAMES = 5

# Audio settings
TTS_RATE = 150
PAN_STRENGTH = 0.8

# Priority objects
PRIORITY_OBJECTS = ["person", "car", "door", "stairs", ...]
```

## Output

The system generates several outputs:

### Audio Files
- `output/audio/spatial_*.wav` - Spatial audio announcements
- `output/audio/scene_*.wav` - Scene descriptions
- `output/audio/alert_*.wav` - Priority alerts

### Reports
- `output/reports/*_report.json` - Processing summary and metrics
- `output/reports/*_frames.json` - Detailed frame-by-frame results
- `output/reports/*_audio_sequence.json` - Audio sequence metadata

### Performance Metrics
- Total frames processed
- Processing FPS
- Detection count
- Announcement count
- Time breakdown by component

## Examples

### Example Output

```
[0.00s] person center close
[0.50s] chair on your right at medium distance
[1.00s] door on your left far away
[1.20s] Warning: person center, very close
```

### Sample Report

```json
{
  "performance_metrics": {
    "total_frames": 300,
    "avg_fps": 12.5,
    "total_detections": 1250,
    "total_announcements": 45,
    "total_alerts": 3
  },
  "audio_sequence": [
    {
      "timestamp": 0.0,
      "text": "person center close",
      "detection": "person",
      "position": "center",
      "depth": "near"
    }
  ]
}
```

## Performance

Typical performance on CPU (Intel i7):
- **Processing FPS**: 10-15 FPS
- **Detection time**: ~50ms per frame
- **Tracking overhead**: ~5ms per frame
- **Moondream inference**: ~200ms (every 5 frames)
- **Audio generation**: ~100ms per announcement

GPU acceleration (CUDA) can achieve 30+ FPS.

## Dataset

### MSR-VTT Dataset

The system is designed to work with the MSR-VTT dataset:
- **Path**: `/kaggle/input/msrvtt/MSR-VTT/TrainValVideo`
- **Format**: MP4 videos
- **Resolution**: Variable (automatically handled)

### Custom Videos

Any video format supported by OpenCV can be used:

```python
python demo.py --video /path/to/custom_video.mp4
```

## Optimization Tips

### For Real-time Performance
1. Use YOLOv8 nano variant (`yolov8n.pt`)
2. Reduce `TARGET_FPS` (e.g., 5-10 FPS)
3. Increase `PROCESS_EVERY_N_FRAMES` (e.g., 10)
4. Use GPU acceleration
5. Lower `CONFIDENCE_THRESHOLD` to reduce detections

### For Accuracy
1. Use larger YOLO model (`yolov8s.pt` or `yolov8m.pt`)
2. Increase `TARGET_FPS`
3. Process more frames with Moondream
4. Lower detection thresholds

### For Mobile Deployment
1. Use quantized models
2. Reduce frame resolution
3. Batch process offline
4. Cache common descriptions

## Troubleshooting

### Common Issues

**Issue**: Models not downloading
```bash
# Manually download YOLO model
python -c "from ultralytics import YOLO; YOLO('yolov8n.pt')"
```

**Issue**: Audio not generating
```bash
# Test TTS engine
python -c "import pyttsx3; engine = pyttsx3.init(); engine.say('test'); engine.runAndWait()"
```

**Issue**: CUDA out of memory
```bash
# Use CPU or reduce batch size
python demo.py --device cpu
```

## Future Enhancements

- [ ] Real-time webcam processing
- [ ] Mobile app deployment
- [ ] Depth sensor integration
- [ ] Haptic feedback support
- [ ] Multi-language TTS
- [ ] Custom object training
- [ ] Cloud sync for model updates
- [ ] User preference learning

## Contributing

Contributions are welcome! Areas for improvement:
- Better spatial audio algorithms (HRTF)
- More efficient tracking
- Improved scene understanding
- Additional safety features
- Mobile optimization

## License

This project is provided as-is for educational and research purposes.

## Acknowledgments

- YOLOv8 by Ultralytics
- ByteTrack by ByteDance
- Moondream by vikhyatk
- MSR-VTT dataset by Microsoft Research

## Citation

If you use this system in your research, please cite:

```
@software{spatial_audio_navigation,
  title={Spatial Audio Navigation System for Visually Impaired},
  author={Your Name},
  year={2024},
  url={https://github.com/yourusername/spatial-audio-navigation}
}
```

## Contact

For questions or feedback, please open an issue on GitHub.
