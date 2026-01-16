"""
Demo script for spatial audio navigation system
Processes a sample MSR-VTT video and generates spatial audio output
"""

import sys
import argparse
import logging
from pathlib import Path

import config
from video_loader import MSRVTTDataset, VideoLoader
from spatial_audio_pipeline import SpatialAudioPipeline, process_single_video

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def print_banner():
    """Print demo banner"""
    banner = """
    ╔═══════════════════════════════════════════════════════════════╗
    ║                                                               ║
    ║      Spatial Audio Navigation System for Visually Impaired   ║
    ║                                                               ║
    ║  Technology Stack:                                            ║
    ║    - YOLOv8: Object Detection                                 ║
    ║    - ByteTrack: Object Tracking                               ║
    ║    - Moondream: Scene Understanding                           ║
    ║    - TTS: Spatial Audio Generation                            ║
    ║                                                               ║
    ╚═══════════════════════════════════════════════════════════════╝
    """
    print(banner)


def setup_demo_environment():
    """Setup demo environment and check dependencies"""
    logger.info("Setting up demo environment...")
    
    # Check output directories
    for dir_path in [config.OUTPUT_DIR, config.AUDIO_OUTPUT_DIR, 
                     config.REPORTS_DIR, config.CACHE_DIR]:
        if not dir_path.exists():
            dir_path.mkdir(parents=True)
            logger.info(f"Created directory: {dir_path}")
    
    # Check dependencies
    dependencies = {
        'ultralytics': 'pip install ultralytics',
        'boxmot': 'pip install boxmot',
        'transformers': 'pip install transformers',
        'pyttsx3': 'pip install pyttsx3',
        'scipy': 'pip install scipy',
        'cv2': 'pip install opencv-python'
    }
    
    missing = []
    for package, install_cmd in dependencies.items():
        try:
            __import__(package)
            logger.info(f"✓ {package} available")
        except ImportError:
            logger.warning(f"✗ {package} not found. Install with: {install_cmd}")
            missing.append(package)
    
    if missing:
        logger.warning(f"\nMissing dependencies: {', '.join(missing)}")
        logger.warning("Some features may not work without these packages.")
        return False
    
    logger.info("All dependencies available!")
    return True


def select_demo_video(video_name: str = None) -> Path:
    """Select video for demo"""
    dataset = MSRVTTDataset()
    
    if video_name:
        # Use specified video
        video_path = dataset.get_video_path(video_name)
        if video_path:
            logger.info(f"Using specified video: {video_name}")
            return video_path
        else:
            logger.warning(f"Video not found: {video_name}")
    
    # Use sample video
    sample = dataset.get_sample_video()
    if sample:
        logger.info(f"Using sample video: {sample.name}")
        return sample
    
    # No videos found
    logger.error("No videos found in dataset path.")
    logger.error(f"Expected path: {config.VIDEO_INPUT_PATH}")
    logger.error("\nTo run demo with a custom video, use: python demo.py --video /path/to/video.mp4")
    return None


def run_demo(video_path: str, max_frames: int = None, device: str = "cpu",
             show_progress: bool = True):
    """
    Run the demo
    
    Args:
        video_path: Path to video file
        max_frames: Maximum frames to process
        device: Device to run on ('cpu' or 'cuda')
        show_progress: Show detailed progress
    """
    print_banner()
    
    logger.info("=" * 60)
    logger.info("DEMO CONFIGURATION")
    logger.info("=" * 60)
    logger.info(f"Video: {video_path}")
    logger.info(f"Max frames: {max_frames if max_frames else 'All'}")
    logger.info(f"Device: {device}")
    logger.info(f"Target FPS: {config.TARGET_FPS}")
    logger.info(f"Output directory: {config.OUTPUT_DIR}")
    logger.info("=" * 60)
    
    # Check dependencies
    deps_ok = setup_demo_environment()
    if not deps_ok:
        logger.warning("\nProceeding with available dependencies...")
    
    # Verify video exists
    video_path = Path(video_path)
    if not video_path.exists():
        logger.error(f"Video file not found: {video_path}")
        return
    
    # Get video info
    logger.info("\nLoading video information...")
    try:
        with VideoLoader(str(video_path), target_fps=config.TARGET_FPS) as loader:
            metadata = loader.get_metadata()
            
            logger.info("\nVideo Information:")
            logger.info(f"  Resolution: {metadata['width']}x{metadata['height']}")
            logger.info(f"  Duration: {metadata['duration']:.2f}s")
            logger.info(f"  Original FPS: {metadata['original_fps']:.2f}")
            logger.info(f"  Total Frames: {metadata['total_frames']}")
            logger.info(f"  Processing FPS: {config.TARGET_FPS}")
            
            if max_frames:
                estimated_duration = max_frames / config.TARGET_FPS
                logger.info(f"  Processing: First {max_frames} frames (~{estimated_duration:.1f}s)")
    
    except Exception as e:
        logger.error(f"Failed to load video: {e}")
        return
    
    # Process video
    logger.info("\n" + "=" * 60)
    logger.info("STARTING PROCESSING")
    logger.info("=" * 60)
    
    try:
        report = process_single_video(
            str(video_path),
            max_frames=max_frames,
            device=device
        )
        
        # Display results
        logger.info("\n" + "=" * 60)
        logger.info("PROCESSING COMPLETE")
        logger.info("=" * 60)
        
        logger.info("\nResults Summary:")
        logger.info(f"  Total Frames Processed: {report['performance_metrics']['total_frames']}")
        logger.info(f"  Total Detections: {report['performance_metrics']['total_detections']}")
        logger.info(f"  Audio Announcements: {report['performance_metrics']['total_announcements']}")
        logger.info(f"  Priority Alerts: {report['performance_metrics']['total_alerts']}")
        logger.info(f"  Average FPS: {report['performance_metrics']['avg_fps']:.2f}")
        
        logger.info("\nOutput Files:")
        logger.info(f"  Audio files: {config.AUDIO_OUTPUT_DIR}")
        logger.info(f"  Reports: {config.REPORTS_DIR}")
        
        # List generated audio files
        audio_files = list(config.AUDIO_OUTPUT_DIR.glob("*.wav"))
        logger.info(f"\nGenerated {len(audio_files)} audio files:")
        for audio_file in sorted(audio_files)[:10]:  # Show first 10
            logger.info(f"    {audio_file.name}")
        if len(audio_files) > 10:
            logger.info(f"    ... and {len(audio_files) - 10} more")
        
        # Show sample announcements
        if report['audio_sequence']:
            logger.info("\nSample Announcements:")
            for ann in report['audio_sequence'][:5]:
                logger.info(f"  [{ann['timestamp']:.2f}s] {ann['text']}")
            if len(report['audio_sequence']) > 5:
                logger.info(f"  ... and {len(report['audio_sequence']) - 5} more")
        
        logger.info("\n" + "=" * 60)
        logger.info("DEMO COMPLETED SUCCESSFULLY!")
        logger.info("=" * 60)
        
    except Exception as e:
        logger.error(f"\nDemo failed with error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(
        description='Spatial Audio Navigation System Demo',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Process sample video from MSR-VTT dataset
  python demo.py
  
  # Process specific video with limited frames
  python demo.py --video /path/to/video.mp4 --max-frames 100
  
  # Process on GPU
  python demo.py --device cuda
  
  # Process sample video from dataset by name
  python demo.py --video-name video0.mp4 --max-frames 300
        """
    )
    
    parser.add_argument(
        '--video',
        type=str,
        help='Path to video file'
    )
    
    parser.add_argument(
        '--video-name',
        type=str,
        default=config.DEMO_VIDEO_NAME,
        help='Video name from MSR-VTT dataset'
    )
    
    parser.add_argument(
        '--max-frames',
        type=int,
        default=config.DEMO_MAX_FRAMES,
        help=f'Maximum frames to process (default: {config.DEMO_MAX_FRAMES})'
    )
    
    parser.add_argument(
        '--device',
        type=str,
        default='cpu',
        choices=['cpu', 'cuda'],
        help='Device to run on (default: cpu)'
    )
    
    parser.add_argument(
        '--no-progress',
        action='store_true',
        help='Disable detailed progress output'
    )
    
    args = parser.parse_args()
    
    # Determine video path
    if args.video:
        video_path = args.video
    else:
        # Try to get from dataset
        video_path = select_demo_video(args.video_name)
        if video_path is None:
            logger.error("\nNo video available for demo.")
            logger.error("Please specify a video with --video /path/to/video.mp4")
            sys.exit(1)
    
    # Run demo
    run_demo(
        video_path=str(video_path),
        max_frames=args.max_frames,
        device=args.device,
        show_progress=not args.no_progress
    )


if __name__ == "__main__":
    main()
