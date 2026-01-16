# Architecture Documentation

## System Overview

The Spatial Audio Navigation System is designed as a modular pipeline that transforms video input into spatial audio cues for visually impaired users. The system operates entirely offline and processes video frames through multiple stages: detection, tracking, reasoning, and audio generation.

## Core Components

### 1. Configuration (`config.py`)

Central configuration module that defines all system parameters:

- **Paths**: Input/output directories
- **Video Processing**: FPS, batch size, frame limits
- **Detection**: YOLO model, confidence thresholds
- **Tracking**: ByteTrack parameters
- **Spatial Zones**: Left/center/right and near/mid/far boundaries
- **Moondream**: Model settings, processing frequency
- **Audio**: TTS engine, sample rate, spatial effects
- **Performance**: Logging and monitoring settings

### 2. Video Loader (`video_loader.py`)

**Purpose**: Load and preprocess videos from MSR-VTT or custom sources.

**Classes**:
- `VideoLoader`: Main video loading class
  - Opens video with OpenCV
  - Extracts metadata (FPS, resolution, duration)
  - Yields frames at target FPS
  - Handles frame skipping for performance
  
- `MSRVTTDataset`: Helper for MSR-VTT dataset
  - Lists available videos
  - Gets video paths by name
  - Provides sample videos

**Key Features**:
- Automatic frame rate adjustment
- RGB conversion for consistency
- Memory-efficient frame streaming
- Context manager support

### 3. Detection & Tracking (`detection_tracker.py`)

**Purpose**: Detect and track objects across frames with spatial analysis.

**Classes**:

#### `ObjectDetector`
- Wraps YOLOv8 model
- Performs object detection on frames
- Filters by confidence threshold
- Returns structured Detection objects

#### `ObjectTracker`
- Wraps ByteTrack algorithm
- Maintains object IDs across frames
- Tracks object history
- Handles track matching

#### `SpatialAnalyzer`
- Analyzes object positions
- Determines horizontal zones (left/center/right)
- Estimates depth (near/middle/far) based on bbox size
- Formats natural language descriptions

#### `DetectionTrackingPipeline`
- Combines detection, tracking, and spatial analysis
- Implements temporal consistency
- Decides when to announce objects
- Identifies priority objects

**Data Structures**:
- `Detection`: Bounding box, confidence, class, track ID
- `SpatialInfo`: Position zones, relative area, center coordinates

**Temporal Consistency**:
- Tracks last announcement time per object
- Only re-announces if position/size changes significantly
- Implements cooldown period
- Prevents duplicate announcements

### 4. Scene Reasoning (`scene_reasoner.py`)

**Purpose**: Understand scene context using Moondream vision-language model.

**Classes**:

#### `MoondreamReasoner`
- Loads Moondream model
- Generates scene descriptions
- Extracts spatial relationships
- Identifies priority alerts
- Caches descriptions

#### `DescriptionCache`
- LRU-style cache for descriptions
- Reduces redundant inference
- Evicts least-accessed items

**Data Structures**:
- `SceneDescription`: Description text, objects, relations, alerts, hash, timestamp

**Features**:
- Frame hashing for cache lookups
- Fallback rule-based descriptions
- Priority object identification
- Spatial relationship extraction

**Processing Strategy**:
- Processes every Nth frame (configurable)
- Uses Moondream when available
- Falls back to rule-based descriptions
- Caches results to disk

### 5. Audio Generation (`audio_generator.py`)

**Purpose**: Convert descriptions to spatial audio using TTS and audio effects.

**Classes**:

#### `TextToSpeech`
- Wraps pyttsx3 engine
- Synthesizes text to WAV files
- Configurable rate and volume
- Caches synthesized audio

#### `SpatialAudioProcessor`
- Applies stereo panning
- Adjusts volume by distance
- Modulates pitch for depth
- Processes spatial effects

#### `SpatialAudioGenerator`
- Orchestrates TTS and spatial processing
- Generates announcements for detections
- Creates scene description audio
- Produces priority alerts
- Tracks audio sequence

**Spatial Audio Effects**:

1. **Panning** (Horizontal Position)
   - Left: More audio in left channel
   - Center: Equal in both channels
   - Right: More audio in right channel
   - Uses constant-power panning

2. **Volume** (Distance)
   - Near: Full volume (1.0)
   - Middle: Reduced volume (0.7)
   - Far: Low volume (0.4)

3. **Pitch** (Distance)
   - Near: Slightly higher pitch (1.1x)
   - Middle: Normal pitch (1.0x)
   - Far: Slightly lower pitch (0.9x)

### 6. Main Pipeline (`spatial_audio_pipeline.py`)

**Purpose**: Orchestrate all components and manage the processing flow.

**Classes**:

#### `PerformanceMetrics`
- Tracks processing statistics
- Measures time per component
- Calculates average FPS
- Generates performance reports

#### `SpatialAudioPipeline`
- Initializes all components
- Processes videos end-to-end
- Manages frame-by-frame workflow
- Generates reports

**Processing Flow**:

```
1. Load video and extract metadata
2. Initialize detection, tracking, reasoning, audio components
3. For each frame:
   a. Detect objects with YOLO
   b. Update tracks with ByteTrack
   c. Analyze spatial information
   d. Check temporal consistency
   e. Generate announcements for new/changed objects
   f. Every N frames: Run Moondream for scene understanding
   g. Generate spatial audio
4. Compile results and generate reports
5. Cleanup resources
```

**Output**:
- Audio WAV files
- JSON reports (processing summary, frame details, audio sequence)
- Performance metrics

### 7. Demo Script (`demo.py`)

**Purpose**: User-friendly entry point for testing the system.

**Features**:
- Command-line interface
- Dependency checking
- Video selection (dataset or custom)
- Progress display
- Result summary

## Data Flow

```
┌─────────────┐
│ Video Input │
└──────┬──────┘
       │
       ▼
┌──────────────┐
│ Video Loader │ → Extract frames at target FPS
└──────┬───────┘
       │
       ▼
┌────────────────────┐
│ Object Detection   │ → YOLOv8 detects objects
└──────┬─────────────┘
       │
       ▼
┌────────────────────┐
│ Object Tracking    │ → ByteTrack maintains IDs
└──────┬─────────────┘
       │
       ▼
┌────────────────────┐
│ Spatial Analysis   │ → Determine position & depth
└──────┬─────────────┘
       │
       ├──────────────────────────┐
       │                          │
       ▼                          ▼
┌──────────────┐      ┌────────────────────┐
│ Temporal     │      │ Scene Reasoning    │
│ Consistency  │      │ (Every N frames)   │
│ Check        │      └──────┬─────────────┘
└──────┬───────┘             │
       │                     │
       │◄────────────────────┘
       │
       ▼
┌────────────────────┐
│ Should Announce?   │
└──────┬─────────────┘
       │ Yes
       ▼
┌────────────────────┐
│ Text-to-Speech     │
└──────┬─────────────┘
       │
       ▼
┌────────────────────┐
│ Spatial Audio      │
│ Processing         │
└──────┬─────────────┘
       │
       ▼
┌────────────────────┐
│ Audio Output       │
│ (WAV files)        │
└────────────────────┘
```

## Design Decisions

### Why YOLOv8?
- State-of-the-art detection accuracy
- Fast inference (real-time capable)
- Nano variant for resource-constrained devices
- Well-maintained and documented

### Why ByteTrack?
- Excellent tracking performance
- Handles occlusions well
- Low computational overhead
- Prevents object ID switches

### Why Moondream?
- Compact vision-language model
- Can run offline on CPU
- Good spatial understanding
- Reasonable inference time

### Why pyttsx3?
- Fully offline TTS
- Cross-platform support
- No API dependencies
- Fast synthesis

### Temporal Consistency Strategy

The system avoids announcement spam through:

1. **Track-based announcements**: Only announce when object first appears or changes
2. **Cooldown periods**: Wait N seconds before re-announcing same object
3. **Position thresholds**: Only re-announce if object moves significantly
4. **Size thresholds**: Only re-announce if object size changes significantly

This ensures natural, non-repetitive audio feedback.

### Spatial Zone Design

**Horizontal Zones**:
- Left: 0-33% of frame width
- Center: 33-67% of frame width
- Right: 67-100% of frame width

**Depth Zones** (based on bbox area relative to frame):
- Near: >15% of frame area
- Middle: 5-15% of frame area
- Far: <5% of frame area

These thresholds were chosen to provide useful spatial information while avoiding over-segmentation.

## Performance Optimizations

### Current Optimizations

1. **Frame skipping**: Process at target FPS instead of native video FPS
2. **Selective reasoning**: Run Moondream every N frames, not every frame
3. **Audio caching**: Reuse synthesized audio for identical text
4. **Description caching**: Cache Moondream outputs by frame hash
5. **Batch processing**: Process frames in batches when possible

### Future Optimization Opportunities

1. **Model quantization**: INT8 quantization for YOLO and Moondream
2. **TensorRT**: Use TensorRT for YOLO inference
3. **ONNX export**: Export models to ONNX for cross-platform optimization
4. **Multi-threading**: Parallel processing of detection and audio generation
5. **GPU streaming**: Use CUDA streams for overlapped execution
6. **Resolution scaling**: Reduce input resolution for faster inference

## Error Handling

The system implements graceful degradation:

1. **Missing dependencies**: Warns but continues with available features
2. **Model loading failures**: Falls back to alternative implementations
3. **Detection failures**: Skips frame and continues
4. **Audio generation failures**: Logs error but continues processing
5. **Invalid input**: Validates and reports clear error messages

## Testing Strategy

### Unit Tests (`test_pipeline.py`)

- Component isolation testing
- Data structure validation
- Spatial analysis accuracy
- Audio processing correctness
- Cache functionality

### Integration Tests

- End-to-end pipeline flow
- Component interaction
- Error propagation
- Performance metrics

### Test Video Generation

`create_test_video.py` generates synthetic videos for testing without requiring the full MSR-VTT dataset.

## Configuration Flexibility

All major parameters are configurable via `config.py`:

- Easy experimentation with different settings
- No code changes needed for tuning
- Environment variable support
- Sensible defaults provided

## Extensibility

The modular design allows easy extension:

1. **New detectors**: Swap `ObjectDetector` implementation
2. **New trackers**: Replace `ObjectTracker` implementation
3. **New reasoning**: Add alternative to `MoondreamReasoner`
4. **New audio effects**: Extend `SpatialAudioProcessor`
5. **New output formats**: Add custom report generators

## Dependencies

### Core
- `opencv-python`: Video I/O
- `numpy`: Array operations
- `ultralytics`: YOLOv8
- `boxmot`: ByteTrack
- `transformers`: Moondream
- `pyttsx3`: TTS
- `scipy`: Audio processing
- `torch`: Deep learning framework

### Optional
- `cuda`: GPU acceleration
- `pytest`: Testing
- `black`: Code formatting

## Deployment Considerations

### CPU Deployment
- Use YOLOv8 nano
- Process at 5-10 FPS
- Run Moondream every 10+ frames
- Expect 10-15 FPS throughput

### GPU Deployment
- Can use larger YOLO models
- Process at 20-30 FPS
- Run Moondream more frequently
- Expect 30+ FPS throughput

### Mobile Deployment
- Requires model quantization
- May need TFLite conversion
- Consider edge TPU support
- Batch offline processing recommended

### Edge Device Deployment
- Raspberry Pi: CPU only, YOLOv8 nano, 5 FPS
- Jetson Nano: GPU capable, 15-20 FPS
- Coral TPU: With TFLite models, 30 FPS

## Security & Privacy

- **Fully offline**: No data sent to external servers
- **No telemetry**: No usage tracking
- **Local processing**: All computation on-device
- **No cloud dependencies**: Works without internet

## Future Architecture Evolution

### Phase 2: Real-time Processing
- Webcam/camera input support
- Streaming architecture
- Ring buffer for frame management
- Reduced latency pipeline

### Phase 3: Multi-modal Input
- Depth sensor integration
- IMU data for head tracking
- GPS for outdoor navigation
- Integration with smart glasses

### Phase 4: Personalization
- User preference learning
- Custom object training
- Adaptive announcement strategies
- Context-aware filtering

## Maintenance

### Model Updates
- YOLO: Update via `ultralytics` package
- Moondream: Update via `transformers` model hub
- ByteTrack: Update via `boxmot` package

### Monitoring
- Performance logs in `output/performance.log`
- Processing reports track degradation
- Audio sequence for quality checks

### Debugging
- Verbose logging available
- Frame-by-frame reports
- Component timing breakdowns
- Test suite for validation
