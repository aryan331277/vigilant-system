# Installation Guide

Complete installation instructions for the Spatial Audio Navigation System.

## Prerequisites

### System Requirements

**Minimum:**
- Python 3.8 or higher
- 4GB RAM
- 2GB free disk space
- CPU with AVX support

**Recommended:**
- Python 3.10+
- 8GB RAM
- 5GB free disk space
- NVIDIA GPU with CUDA support (optional)
- Linux/macOS/Windows 10+

### Software Dependencies

- Python 3.8+
- pip (Python package manager)
- Git (for cloning repository)

## Installation Methods

### Method 1: Using pip (Recommended)

```bash
# Clone repository
git clone <repository-url>
cd spatial-audio-navigation

# Create virtual environment (recommended)
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Verify installation
python demo.py --help
```

### Method 2: Using setup.py

```bash
# Clone repository
git clone <repository-url>
cd spatial-audio-navigation

# Install package
pip install -e .

# Run demo
spatial-audio-demo --help
```

### Method 3: Development Setup

```bash
# Clone repository
git clone <repository-url>
cd spatial-audio-navigation

# Create virtual environment
python3 -m venv venv
source venv/bin/activate

# Install with development dependencies
pip install -r requirements.txt
pip install pytest pytest-cov black flake8 mypy

# Run tests
python test_pipeline.py
```

## Dependency Installation

### Core Dependencies

```bash
pip install opencv-python numpy ultralytics torch torchvision boxmot transformers pyttsx3 scipy
```

### Platform-Specific Dependencies

#### Linux (Ubuntu/Debian)

```bash
# Install system dependencies for pyttsx3
sudo apt-get update
sudo apt-get install espeak espeak-data libespeak1 libespeak-dev

# Install Python dependencies
pip install -r requirements.txt
```

#### macOS

```bash
# Install Homebrew (if not installed)
/bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"

# Install Python dependencies (macOS has built-in TTS)
pip install -r requirements.txt
```

#### Windows

```bash
# Windows has built-in SAPI5 TTS, just install Python dependencies
pip install -r requirements.txt
```

### GPU Support (CUDA)

If you have an NVIDIA GPU and want GPU acceleration:

```bash
# Install CUDA-enabled PyTorch (example for CUDA 11.8)
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118

# Verify CUDA availability
python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}')"
```

## Post-Installation Verification

### 1. Check Python Version

```bash
python --version
# Should output: Python 3.8.0 or higher
```

### 2. Verify Dependencies

```bash
python -c "import cv2, numpy, torch, ultralytics, transformers, pyttsx3, scipy; print('All imports successful!')"
```

### 3. Check YOLO Installation

```bash
python -c "from ultralytics import YOLO; model = YOLO('yolov8n.pt'); print('YOLO model loaded successfully!')"
```

This will download the YOLOv8 nano model (~6MB) on first run.

### 4. Test TTS Engine

```bash
python -c "import pyttsx3; engine = pyttsx3.init(); engine.say('Test'); engine.runAndWait(); print('TTS working!')"
```

### 5. Run Test Suite

```bash
python test_pipeline.py
```

## Configuration

### Environment Variables

You can set environment variables to customize paths:

```bash
export VIDEO_INPUT_PATH="/path/to/msrvtt/videos"
```

### Config File

Edit `config.py` to customize system behavior:

```python
# Video processing
TARGET_FPS = 10

# Detection
CONFIDENCE_THRESHOLD = 0.5
YOLO_MODEL = "yolov8n.pt"

# Audio
TTS_RATE = 150
PAN_STRENGTH = 0.8
```

## Testing Installation

### Quick Test

```bash
# Create a test video
python create_test_video.py --create-defaults

# Process test video
python demo.py --video test_videos/test_short.mp4
```

### Verify Output

Check that files are created in `output/` directory:

```bash
ls output/audio/        # Should contain WAV files
ls output/reports/      # Should contain JSON reports
```

## Troubleshooting

### Issue: "ModuleNotFoundError: No module named 'ultralytics'"

**Solution:**
```bash
pip install ultralytics
```

### Issue: "ModuleNotFoundError: No module named 'boxmot'"

**Solution:**
```bash
pip install boxmot
```

### Issue: "pyttsx3 initialization failed"

**Linux Solution:**
```bash
sudo apt-get install espeak
```

**macOS/Windows Solution:**
- These platforms have built-in TTS, so this shouldn't occur
- Try reinstalling: `pip uninstall pyttsx3 && pip install pyttsx3`

### Issue: "CUDA out of memory"

**Solution:**
```bash
# Use CPU instead
python demo.py --video my_video.mp4 --device cpu

# Or reduce batch size in config.py
BATCH_SIZE = 1
```

### Issue: "ImportError: cannot import name 'AutoModelForCausalLM'"

**Solution:**
```bash
pip install --upgrade transformers
```

### Issue: "opencv-python not found"

**Solution:**
```bash
pip install opencv-python
```

### Issue: Model download fails

**Solution:**
```bash
# Manually download YOLO model
mkdir -p ~/.cache/ultralytics
cd ~/.cache/ultralytics
wget https://github.com/ultralytics/assets/releases/download/v0.0.0/yolov8n.pt
```

### Issue: Permission denied when creating output directories

**Solution:**
```bash
# Create directories with proper permissions
mkdir -p output/audio output/reports output/cache
chmod -R 755 output
```

## Updating

### Update Dependencies

```bash
pip install --upgrade -r requirements.txt
```

### Update YOLO Model

```bash
python -c "from ultralytics import YOLO; YOLO('yolov8n.pt').download()"
```

### Update Moondream

```bash
python -c "from transformers import AutoModelForCausalLM; AutoModelForCausalLM.from_pretrained('vikhyatk/moondream2', revision='2024-08-26')"
```

## Uninstallation

### Remove Package

```bash
pip uninstall spatial-audio-navigation
```

### Remove Dependencies

```bash
pip uninstall -r requirements.txt -y
```

### Clean Cache

```bash
rm -rf ~/.cache/ultralytics
rm -rf ~/.cache/huggingface
rm -rf output/
```

## Advanced Installation

### Docker Installation (Future)

```dockerfile
FROM python:3.10-slim

WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt

COPY . .
CMD ["python", "demo.py"]
```

### Conda Installation

```bash
# Create conda environment
conda create -n spatial-audio python=3.10
conda activate spatial-audio

# Install dependencies
pip install -r requirements.txt
```

### Installing for Development

```bash
# Clone with development branch
git clone -b develop <repository-url>
cd spatial-audio-navigation

# Install in editable mode with dev dependencies
pip install -e ".[dev]"

# Install pre-commit hooks
pre-commit install
```

## Platform-Specific Notes

### Linux
- Requires `espeak` for TTS
- GPU acceleration works best on Linux
- Tested on Ubuntu 20.04, 22.04

### macOS
- Uses native macOS TTS (no extra dependencies)
- M1/M2 Macs: Use MPS backend for acceleration
- Tested on macOS 12+

### Windows
- Uses native SAPI5 TTS
- GPU acceleration requires CUDA installation
- Tested on Windows 10, 11

### Raspberry Pi
- Use YOLOv8 nano for best performance
- Reduce TARGET_FPS to 5
- Expect 3-5 FPS processing
- 4GB+ RAM recommended

## Verification Checklist

After installation, verify:

- [ ] Python 3.8+ installed
- [ ] All dependencies installed
- [ ] YOLO model downloaded
- [ ] TTS engine working
- [ ] Test suite passes
- [ ] Demo script runs
- [ ] Output files generated
- [ ] Audio files playable

## Getting Help

If you encounter issues:

1. Check this installation guide
2. Review troubleshooting section
3. Check [QUICKSTART.md](QUICKSTART.md)
4. Review [README.md](README.md)
5. Run test suite: `python test_pipeline.py`
6. Open an issue on GitHub

## Next Steps

After successful installation:

1. Read [QUICKSTART.md](QUICKSTART.md) to run your first demo
2. Review [README.md](README.md) for detailed usage
3. Check [ARCHITECTURE.md](ARCHITECTURE.md) for system design
4. Explore the code and customize for your needs

## Summary

Installation is complete when you can run:

```bash
python demo.py --video test_videos/test_short.mp4
```

And see:
- Console output with processing stats
- Audio files in `output/audio/`
- Reports in `output/reports/`

**Happy coding!** 🎉
