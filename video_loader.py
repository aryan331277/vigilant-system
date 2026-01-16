"""
MSR-VTT dataset video loading and preprocessing
"""

import cv2
import numpy as np
from pathlib import Path
from typing import Generator, Tuple, Optional
import logging

import config

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class VideoLoader:
    """Handles loading and preprocessing of videos from MSR-VTT dataset"""
    
    def __init__(self, video_path: str, target_fps: int = config.TARGET_FPS):
        """
        Initialize video loader
        
        Args:
            video_path: Path to video file
            target_fps: Target frames per second for processing
        """
        self.video_path = Path(video_path)
        self.target_fps = target_fps
        self.cap = None
        self.original_fps = None
        self.frame_count = 0
        self.total_frames = 0
        self.width = 0
        self.height = 0
        self.frame_skip = 1
        
        self._initialize_video()
    
    def _initialize_video(self):
        """Initialize video capture and extract metadata"""
        if not self.video_path.exists():
            raise FileNotFoundError(f"Video not found: {self.video_path}")
        
        self.cap = cv2.VideoCapture(str(self.video_path))
        
        if not self.cap.isOpened():
            raise ValueError(f"Failed to open video: {self.video_path}")
        
        self.original_fps = self.cap.get(cv2.CAP_PROP_FPS)
        self.total_frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self.width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
        # Calculate frame skip to achieve target FPS
        if self.original_fps > 0:
            self.frame_skip = max(1, int(self.original_fps / self.target_fps))
        
        logger.info(f"Video loaded: {self.video_path.name}")
        logger.info(f"Resolution: {self.width}x{self.height}")
        logger.info(f"Original FPS: {self.original_fps:.2f}, Target FPS: {self.target_fps}")
        logger.info(f"Total frames: {self.total_frames}, Frame skip: {self.frame_skip}")
        logger.info(f"Duration: {self.get_duration():.2f}s")
    
    def get_duration(self) -> float:
        """Get video duration in seconds"""
        if self.original_fps > 0:
            return self.total_frames / self.original_fps
        return 0.0
    
    def get_metadata(self) -> dict:
        """Get video metadata"""
        return {
            "path": str(self.video_path),
            "name": self.video_path.name,
            "width": self.width,
            "height": self.height,
            "original_fps": self.original_fps,
            "target_fps": self.target_fps,
            "total_frames": self.total_frames,
            "duration": self.get_duration(),
            "frame_skip": self.frame_skip
        }
    
    def read_frames(self, max_frames: Optional[int] = None) -> Generator[Tuple[int, np.ndarray, float], None, None]:
        """
        Generator that yields frames from the video
        
        Args:
            max_frames: Maximum number of frames to process (None for all)
            
        Yields:
            Tuple of (frame_number, frame_array, timestamp)
        """
        frame_idx = 0
        processed_frames = 0
        
        while True:
            ret, frame = self.cap.read()
            
            if not ret:
                break
            
            # Skip frames to achieve target FPS
            if frame_idx % self.frame_skip == 0:
                timestamp = frame_idx / self.original_fps if self.original_fps > 0 else 0.0
                
                # Normalize frame (convert BGR to RGB)
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                
                yield frame_idx, frame_rgb, timestamp
                
                processed_frames += 1
                
                if max_frames and processed_frames >= max_frames:
                    logger.info(f"Reached max frames limit: {max_frames}")
                    break
            
            frame_idx += 1
        
        logger.info(f"Processed {processed_frames} frames from {frame_idx} total frames")
    
    def reset(self):
        """Reset video to beginning"""
        if self.cap:
            self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            self.frame_count = 0
    
    def release(self):
        """Release video capture"""
        if self.cap:
            self.cap.release()
            logger.info(f"Video released: {self.video_path.name}")
    
    def __enter__(self):
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.release()


class MSRVTTDataset:
    """Helper class to work with MSR-VTT dataset"""
    
    def __init__(self, dataset_path: str = config.VIDEO_INPUT_PATH):
        """
        Initialize MSR-VTT dataset handler
        
        Args:
            dataset_path: Path to MSR-VTT video directory
        """
        self.dataset_path = Path(dataset_path)
        self.video_files = []
        
        if self.dataset_path.exists():
            self.video_files = sorted(list(self.dataset_path.glob("*.mp4")))
            logger.info(f"Found {len(self.video_files)} videos in {dataset_path}")
        else:
            logger.warning(f"Dataset path not found: {dataset_path}")
    
    def get_video_path(self, video_name: str) -> Optional[Path]:
        """Get full path to a video by name"""
        video_path = self.dataset_path / video_name
        if video_path.exists():
            return video_path
        return None
    
    def list_videos(self, limit: Optional[int] = None) -> list:
        """List available videos"""
        videos = [v.name for v in self.video_files]
        if limit:
            videos = videos[:limit]
        return videos
    
    def get_sample_video(self) -> Optional[Path]:
        """Get a sample video for testing"""
        if self.video_files:
            return self.video_files[0]
        return None


def preprocess_frame_for_yolo(frame: np.ndarray) -> np.ndarray:
    """
    Preprocess frame for YOLO input
    
    Args:
        frame: RGB frame array
        
    Returns:
        Preprocessed frame
    """
    # YOLO expects RGB input, which we already have from VideoLoader
    return frame


if __name__ == "__main__":
    # Test video loading
    dataset = MSRVTTDataset()
    
    if dataset.video_files:
        sample_video = dataset.get_sample_video()
        print(f"Testing with sample video: {sample_video}")
        
        with VideoLoader(str(sample_video), target_fps=5) as loader:
            print("\nVideo Metadata:")
            for key, value in loader.get_metadata().items():
                print(f"  {key}: {value}")
            
            print("\nReading first 10 frames...")
            for frame_num, frame, timestamp in loader.read_frames(max_frames=10):
                print(f"  Frame {frame_num}: shape={frame.shape}, timestamp={timestamp:.2f}s")
    else:
        print("No videos found. Creating test dataset structure...")
        print(f"Expected path: {config.VIDEO_INPUT_PATH}")
