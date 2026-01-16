"""
Main orchestration pipeline for spatial audio navigation system
"""

import time
import logging
from pathlib import Path
from typing import Optional, Dict, List
import json
from dataclasses import dataclass, asdict

import config
from video_loader import VideoLoader
from detection_tracker import DetectionTrackingPipeline
from scene_reasoner import MoondreamReasoner
from audio_generator import SpatialAudioGenerator

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


@dataclass
class PerformanceMetrics:
    """Performance metrics for the pipeline"""
    total_frames: int = 0
    total_time: float = 0.0
    video_load_time: float = 0.0
    detection_time: float = 0.0
    tracking_time: float = 0.0
    reasoning_time: float = 0.0
    audio_generation_time: float = 0.0
    total_detections: int = 0
    total_announcements: int = 0
    total_alerts: int = 0
    avg_fps: float = 0.0
    
    def calculate_averages(self):
        """Calculate average metrics"""
        if self.total_frames > 0:
            self.avg_fps = self.total_frames / self.total_time if self.total_time > 0 else 0.0
    
    def to_dict(self) -> dict:
        """Convert to dictionary"""
        return asdict(self)
    
    def print_summary(self):
        """Print performance summary"""
        logger.info("=" * 60)
        logger.info("PERFORMANCE SUMMARY")
        logger.info("=" * 60)
        logger.info(f"Total Frames Processed: {self.total_frames}")
        logger.info(f"Total Time: {self.total_time:.2f}s")
        logger.info(f"Average FPS: {self.avg_fps:.2f}")
        logger.info(f"Total Detections: {self.total_detections}")
        logger.info(f"Total Announcements: {self.total_announcements}")
        logger.info(f"Total Alerts: {self.total_alerts}")
        logger.info("\nTime Breakdown:")
        logger.info(f"  Video Loading: {self.video_load_time:.2f}s ({self._percentage(self.video_load_time)}%)")
        logger.info(f"  Detection: {self.detection_time:.2f}s ({self._percentage(self.detection_time)}%)")
        logger.info(f"  Scene Reasoning: {self.reasoning_time:.2f}s ({self._percentage(self.reasoning_time)}%)")
        logger.info(f"  Audio Generation: {self.audio_generation_time:.2f}s ({self._percentage(self.audio_generation_time)}%)")
        logger.info("=" * 60)
    
    def _percentage(self, value: float) -> str:
        """Calculate percentage of total time"""
        if self.total_time > 0:
            return f"{(value / self.total_time * 100):.1f}"
        return "0.0"


class SpatialAudioPipeline:
    """Main pipeline orchestration"""
    
    def __init__(self, device: str = "cpu"):
        """
        Initialize pipeline
        
        Args:
            device: Device to run models on ('cpu' or 'cuda')
        """
        self.device = device
        self.detection_pipeline = None
        self.reasoner = None
        self.audio_generator = None
        self.metrics = PerformanceMetrics()
        
        logger.info(f"Initializing Spatial Audio Pipeline on {device}")
    
    def process_video(self, video_path: str, max_frames: Optional[int] = None,
                     output_name: Optional[str] = None) -> Dict:
        """
        Process a video end-to-end
        
        Args:
            video_path: Path to video file
            max_frames: Maximum frames to process (None for all)
            output_name: Name for output files
            
        Returns:
            Dictionary with processing results and metadata
        """
        start_time = time.time()
        
        # Initialize components
        logger.info(f"Processing video: {video_path}")
        
        # Load video
        load_start = time.time()
        video_loader = VideoLoader(video_path, target_fps=config.TARGET_FPS)
        video_metadata = video_loader.get_metadata()
        self.metrics.video_load_time = time.time() - load_start
        
        # Initialize detection pipeline with video dimensions
        self.detection_pipeline = DetectionTrackingPipeline(
            video_metadata['width'],
            video_metadata['height'],
            device=self.device
        )
        
        # Initialize reasoner
        self.reasoner = MoondreamReasoner(device=self.device)
        
        # Initialize audio generator
        self.audio_generator = SpatialAudioGenerator()
        
        # Process frames
        results = self._process_frames(video_loader, max_frames)
        
        # Finalize
        video_loader.release()
        self.reasoner.cleanup()
        self.audio_generator.cleanup()
        
        # Calculate metrics
        self.metrics.total_time = time.time() - start_time
        self.metrics.calculate_averages()
        
        # Generate report
        if output_name is None:
            output_name = Path(video_path).stem
        
        report = self._generate_report(video_metadata, results, output_name)
        
        # Print summary
        self.metrics.print_summary()
        
        return report
    
    def _process_frames(self, video_loader: VideoLoader, max_frames: Optional[int]) -> Dict:
        """Process video frames"""
        frame_results = []
        batch_detections = []
        last_scene_frame = -1
        
        logger.info("Starting frame processing...")
        
        for frame_num, frame, timestamp in video_loader.read_frames(max_frames=max_frames):
            self.metrics.total_frames += 1
            
            # Log progress
            if frame_num % 50 == 0:
                logger.info(f"Processing frame {frame_num} at {timestamp:.2f}s")
            
            # Detection and tracking
            det_start = time.time()
            detections_with_spatial = self.detection_pipeline.process_frame(frame, timestamp)
            self.metrics.detection_time += time.time() - det_start
            self.metrics.total_detections += len(detections_with_spatial)
            
            # Filter announcements based on temporal consistency
            announcements = []
            priority_alerts = []
            
            for detection, spatial_info in detections_with_spatial:
                should_announce = self.detection_pipeline.should_announce(
                    detection, spatial_info, timestamp
                )
                
                if should_announce:
                    # Check if priority object
                    is_priority = self.detection_pipeline.is_priority_object(detection)
                    
                    # Generate audio announcement
                    audio_start = time.time()
                    audio_path = self.audio_generator.generate_announcement(
                        detection, spatial_info, timestamp
                    )
                    self.metrics.audio_generation_time += time.time() - audio_start
                    
                    if audio_path:
                        self.metrics.total_announcements += 1
                        
                        announcement = {
                            'timestamp': timestamp,
                            'detection': detection.class_name,
                            'position': spatial_info.horizontal_zone,
                            'depth': spatial_info.depth_zone,
                            'confidence': detection.confidence,
                            'track_id': detection.track_id,
                            'is_priority': is_priority,
                            'audio_path': str(audio_path)
                        }
                        
                        if is_priority:
                            priority_alerts.append(announcement)
                            self.metrics.total_alerts += 1
                        else:
                            announcements.append(announcement)
            
            # Scene reasoning (every N frames)
            scene_description = None
            if frame_num - last_scene_frame >= config.PROCESS_EVERY_N_FRAMES:
                reason_start = time.time()
                scene_desc = self.reasoner.analyze_scene(frame, detections_with_spatial, timestamp)
                self.metrics.reasoning_time += time.time() - reason_start
                
                # Generate scene audio
                if scene_desc.description:
                    audio_start = time.time()
                    scene_audio_path = self.audio_generator.generate_scene_announcement(scene_desc)
                    self.metrics.audio_generation_time += time.time() - audio_start
                    
                    scene_description = {
                        'timestamp': timestamp,
                        'description': scene_desc.description,
                        'objects': scene_desc.objects_mentioned,
                        'spatial_relations': scene_desc.spatial_relations,
                        'audio_path': str(scene_audio_path) if scene_audio_path else None
                    }
                
                # Generate priority alert audio
                for alert in scene_desc.priority_alerts:
                    audio_start = time.time()
                    alert_audio = self.audio_generator.generate_priority_alert(alert, timestamp)
                    self.metrics.audio_generation_time += time.time() - audio_start
                
                last_scene_frame = frame_num
            
            # Store frame results
            frame_result = {
                'frame_num': frame_num,
                'timestamp': timestamp,
                'detections_count': len(detections_with_spatial),
                'announcements': announcements,
                'priority_alerts': priority_alerts,
                'scene_description': scene_description
            }
            
            frame_results.append(frame_result)
        
        logger.info(f"Frame processing complete. Processed {self.metrics.total_frames} frames.")
        
        return {
            'frames': frame_results,
            'audio_sequence': self.audio_generator.get_audio_sequence()
        }
    
    def _generate_report(self, video_metadata: Dict, results: Dict, output_name: str) -> Dict:
        """Generate processing report"""
        report_path = config.REPORTS_DIR / f"{output_name}_report.json"
        
        report = {
            'video_metadata': video_metadata,
            'performance_metrics': self.metrics.to_dict(),
            'processing_results': {
                'total_frames': len(results['frames']),
                'frames_with_detections': sum(1 for f in results['frames'] if f['detections_count'] > 0),
                'total_announcements': self.metrics.total_announcements,
                'total_alerts': self.metrics.total_alerts
            },
            'audio_sequence': results['audio_sequence']
        }
        
        # Save report
        with open(report_path, 'w') as f:
            json.dump(report, f, indent=2)
        
        logger.info(f"Report saved: {report_path}")
        
        # Save detailed frame results
        frames_path = config.REPORTS_DIR / f"{output_name}_frames.json"
        with open(frames_path, 'w') as f:
            json.dump(results['frames'], f, indent=2)
        
        logger.info(f"Frame details saved: {frames_path}")
        
        # Export audio sequence metadata
        audio_seq_path = config.REPORTS_DIR / f"{output_name}_audio_sequence.json"
        self.audio_generator.export_sequence_metadata(audio_seq_path)
        
        return report


def process_single_video(video_path: str, max_frames: Optional[int] = None,
                        device: str = "cpu") -> Dict:
    """
    Convenience function to process a single video
    
    Args:
        video_path: Path to video file
        max_frames: Maximum frames to process
        device: Device to run on
        
    Returns:
        Processing report
    """
    pipeline = SpatialAudioPipeline(device=device)
    report = pipeline.process_video(video_path, max_frames=max_frames)
    return report


def process_batch_videos(video_paths: List[str], max_frames: Optional[int] = None,
                        device: str = "cpu") -> List[Dict]:
    """
    Process multiple videos in batch
    
    Args:
        video_paths: List of video paths
        max_frames: Maximum frames per video
        device: Device to run on
        
    Returns:
        List of processing reports
    """
    pipeline = SpatialAudioPipeline(device=device)
    reports = []
    
    for i, video_path in enumerate(video_paths):
        logger.info(f"Processing video {i+1}/{len(video_paths)}: {video_path}")
        try:
            report = pipeline.process_video(video_path, max_frames=max_frames)
            reports.append(report)
        except Exception as e:
            logger.error(f"Failed to process {video_path}: {e}")
            continue
    
    return reports


if __name__ == "__main__":
    # Test pipeline
    import sys
    
    if len(sys.argv) > 1:
        video_path = sys.argv[1]
        max_frames = int(sys.argv[2]) if len(sys.argv) > 2 else None
    else:
        # Use demo video
        from video_loader import MSRVTTDataset
        
        dataset = MSRVTTDataset()
        sample_video = dataset.get_sample_video()
        
        if sample_video:
            video_path = str(sample_video)
            max_frames = 50  # Process first 50 frames for testing
        else:
            logger.error("No video found. Please provide video path as argument.")
            sys.exit(1)
    
    logger.info(f"Testing pipeline with: {video_path}")
    logger.info(f"Max frames: {max_frames}")
    
    try:
        report = process_single_video(video_path, max_frames=max_frames)
        logger.info("Pipeline test completed successfully!")
    except Exception as e:
        logger.error(f"Pipeline test failed: {e}")
        import traceback
        traceback.print_exc()
