"""
Utility script to create a test video for demo purposes
Useful when MSR-VTT dataset is not available
"""

import cv2
import numpy as np
from pathlib import Path
import argparse

import config


def create_test_video(output_path: str, duration: int = 10, fps: int = 30,
                     width: int = 640, height: int = 480):
    """
    Create a test video with moving colored rectangles
    
    Args:
        output_path: Path to save video
        duration: Duration in seconds
        fps: Frames per second
        width: Video width
        height: Video height
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Setup video writer
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(str(output_path), fourcc, fps, (width, height))
    
    total_frames = duration * fps
    
    print(f"Creating test video: {output_path}")
    print(f"Resolution: {width}x{height}, Duration: {duration}s, FPS: {fps}")
    print(f"Total frames: {total_frames}")
    
    for frame_idx in range(total_frames):
        # Create white background
        frame = np.ones((height, width, 3), dtype=np.uint8) * 255
        
        # Add moving rectangles (simulating objects)
        # Object 1: Red rectangle moving left to right
        obj1_x = int((frame_idx / total_frames) * width)
        obj1_y = height // 4
        cv2.rectangle(frame, (obj1_x, obj1_y), (obj1_x + 80, obj1_y + 80), (0, 0, 255), -1)
        cv2.putText(frame, "Person", (obj1_x + 5, obj1_y + 45), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        # Object 2: Blue rectangle moving right to left
        obj2_x = width - int((frame_idx / total_frames) * width)
        obj2_y = height // 2
        cv2.rectangle(frame, (obj2_x, obj2_y), (obj2_x + 60, obj2_y + 60), (255, 0, 0), -1)
        cv2.putText(frame, "Chair", (obj2_x + 5, obj2_y + 35),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        # Object 3: Green rectangle moving up and down
        obj3_x = width // 2
        obj3_y = int(abs(np.sin(frame_idx / fps * 2 * np.pi)) * (height - 100))
        cv2.rectangle(frame, (obj3_x, obj3_y), (obj3_x + 70, obj3_y + 70), (0, 255, 0), -1)
        cv2.putText(frame, "Door", (obj3_x + 10, obj3_y + 40),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        # Add frame counter
        cv2.putText(frame, f"Frame: {frame_idx}/{total_frames}", (10, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)
        
        # Convert RGB to BGR for OpenCV
        frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        
        # Write frame
        out.write(frame_bgr)
        
        if frame_idx % 30 == 0:
            print(f"Progress: {frame_idx}/{total_frames} frames")
    
    out.release()
    print(f"Test video created successfully: {output_path}")
    print(f"File size: {output_path.stat().st_size / (1024*1024):.2f} MB")


def create_default_test_videos():
    """Create default test videos"""
    output_dir = Path("test_videos")
    output_dir.mkdir(exist_ok=True)
    
    # Create short test video
    create_test_video(
        output_path=output_dir / "test_short.mp4",
        duration=5,
        fps=30,
        width=640,
        height=480
    )
    
    # Create medium test video
    create_test_video(
        output_path=output_dir / "test_medium.mp4",
        duration=15,
        fps=30,
        width=1280,
        height=720
    )
    
    print("\nTest videos created in 'test_videos/' directory")
    print("You can now run: python demo.py --video test_videos/test_short.mp4")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Create test videos for demo")
    
    parser.add_argument(
        '--output',
        type=str,
        default=None,
        help='Output video path'
    )
    
    parser.add_argument(
        '--duration',
        type=int,
        default=10,
        help='Duration in seconds (default: 10)'
    )
    
    parser.add_argument(
        '--fps',
        type=int,
        default=30,
        help='Frames per second (default: 30)'
    )
    
    parser.add_argument(
        '--width',
        type=int,
        default=640,
        help='Video width (default: 640)'
    )
    
    parser.add_argument(
        '--height',
        type=int,
        default=480,
        help='Video height (default: 480)'
    )
    
    parser.add_argument(
        '--create-defaults',
        action='store_true',
        help='Create default test videos'
    )
    
    args = parser.parse_args()
    
    if args.create_defaults:
        create_default_test_videos()
    elif args.output:
        create_test_video(
            output_path=args.output,
            duration=args.duration,
            fps=args.fps,
            width=args.width,
            height=args.height
        )
    else:
        print("Please specify --output or use --create-defaults")
        print("Example: python create_test_video.py --create-defaults")
        print("Example: python create_test_video.py --output my_test.mp4 --duration 5")
