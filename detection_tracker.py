"""
Object detection using YOLOv8 and tracking using ByteTrack
"""

import numpy as np
from typing import List, Dict, Tuple, Optional
import logging
from dataclasses import dataclass
from collections import defaultdict
import time

try:
    from ultralytics import YOLO
except ImportError:
    YOLO = None
    logging.warning("ultralytics not installed. Install with: pip install ultralytics")

try:
    from boxmot import ByteTrack
except ImportError:
    ByteTrack = None
    logging.warning("boxmot not installed. Install with: pip install boxmot")

import config

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class Detection:
    """Represents a detected object"""
    bbox: Tuple[float, float, float, float]  # x1, y1, x2, y2
    confidence: float
    class_id: int
    class_name: str
    track_id: Optional[int] = None


@dataclass
class SpatialInfo:
    """Spatial information about an object"""
    horizontal_zone: str  # left, center, right
    depth_zone: str  # near, middle, far
    center_x: float
    center_y: float
    bbox_area: float
    frame_area: float
    relative_area: float  # bbox_area / frame_area


class ObjectDetector:
    """YOLOv8-based object detector"""
    
    def __init__(self, model_name: str = config.YOLO_MODEL, device: str = "cpu"):
        """
        Initialize YOLO detector
        
        Args:
            model_name: YOLO model name (e.g., 'yolov8n.pt')
            device: Device to run on ('cpu' or 'cuda')
        """
        if YOLO is None:
            raise ImportError("ultralytics not installed. Install with: pip install ultralytics")
        
        self.model_name = model_name
        self.device = device
        self.model = None
        self.class_names = {}
        
        self._load_model()
    
    def _load_model(self):
        """Load YOLO model"""
        logger.info(f"Loading YOLO model: {self.model_name}")
        self.model = YOLO(self.model_name)
        self.class_names = self.model.names
        logger.info(f"YOLO model loaded with {len(self.class_names)} classes")
    
    def detect(self, frame: np.ndarray) -> List[Detection]:
        """
        Detect objects in frame
        
        Args:
            frame: RGB frame array
            
        Returns:
            List of Detection objects
        """
        results = self.model(
            frame,
            conf=config.CONFIDENCE_THRESHOLD,
            iou=config.IOU_THRESHOLD,
            max_det=config.MAX_DETECTIONS,
            verbose=False
        )
        
        detections = []
        
        for result in results:
            boxes = result.boxes
            
            for i in range(len(boxes)):
                bbox = boxes.xyxy[i].cpu().numpy()
                conf = float(boxes.conf[i].cpu().numpy())
                cls_id = int(boxes.cls[i].cpu().numpy())
                cls_name = self.class_names.get(cls_id, f"class_{cls_id}")
                
                detection = Detection(
                    bbox=tuple(bbox),
                    confidence=conf,
                    class_id=cls_id,
                    class_name=cls_name
                )
                detections.append(detection)
        
        return detections


class ObjectTracker:
    """ByteTrack-based object tracker"""
    
    def __init__(self):
        """Initialize ByteTrack tracker"""
        if ByteTrack is None:
            logger.warning("ByteTrack not available. Tracking will be disabled.")
            self.tracker = None
        else:
            self.tracker = ByteTrack(
                track_thresh=config.TRACK_THRESH,
                track_buffer=config.TRACK_BUFFER,
                match_thresh=config.MATCH_THRESH,
                frame_rate=config.FRAME_RATE
            )
            logger.info("ByteTrack tracker initialized")
        
        self.track_history = defaultdict(list)
    
    def update(self, detections: List[Detection], frame: np.ndarray) -> List[Detection]:
        """
        Update tracker with new detections
        
        Args:
            detections: List of Detection objects
            frame: Current frame
            
        Returns:
            List of Detection objects with track IDs
        """
        if self.tracker is None or not detections:
            return detections
        
        # Convert detections to format expected by ByteTrack
        # Format: [x1, y1, x2, y2, conf, class_id]
        det_array = np.array([
            [*det.bbox, det.confidence, det.class_id]
            for det in detections
        ])
        
        try:
            # Update tracker
            tracks = self.tracker.update(det_array, frame)
            
            # Match tracks to detections
            tracked_detections = []
            
            for track in tracks:
                # Track format: [x1, y1, x2, y2, track_id, conf, class_id, ...]
                x1, y1, x2, y2 = track[:4]
                track_id = int(track[4])
                
                # Find matching detection
                for det in detections:
                    det_bbox = det.bbox
                    # Check if bboxes are close enough
                    if (abs(det_bbox[0] - x1) < 10 and abs(det_bbox[1] - y1) < 10 and
                        abs(det_bbox[2] - x2) < 10 and abs(det_bbox[3] - y2) < 10):
                        det.track_id = track_id
                        tracked_detections.append(det)
                        
                        # Update track history
                        self.track_history[track_id].append({
                            'bbox': det.bbox,
                            'class_name': det.class_name,
                            'timestamp': time.time()
                        })
                        break
            
            return tracked_detections if tracked_detections else detections
            
        except Exception as e:
            logger.warning(f"Tracking failed: {e}. Returning detections without track IDs.")
            return detections
    
    def get_track_history(self, track_id: int) -> List[Dict]:
        """Get history for a specific track"""
        return self.track_history.get(track_id, [])


class SpatialAnalyzer:
    """Analyzes spatial position and depth of detected objects"""
    
    def __init__(self, frame_width: int, frame_height: int):
        """
        Initialize spatial analyzer
        
        Args:
            frame_width: Frame width in pixels
            frame_height: Frame height in pixels
        """
        self.frame_width = frame_width
        self.frame_height = frame_height
        self.frame_area = frame_width * frame_height
    
    def analyze(self, detection: Detection) -> SpatialInfo:
        """
        Analyze spatial information of detection
        
        Args:
            detection: Detection object
            
        Returns:
            SpatialInfo object
        """
        x1, y1, x2, y2 = detection.bbox
        
        # Calculate center
        center_x = (x1 + x2) / 2
        center_y = (y1 + y2) / 2
        
        # Normalize center to [0, 1]
        norm_center_x = center_x / self.frame_width
        
        # Determine horizontal zone
        if norm_center_x < config.LEFT_ZONE_END:
            horizontal_zone = "left"
        elif norm_center_x > config.RIGHT_ZONE_START:
            horizontal_zone = "right"
        else:
            horizontal_zone = "center"
        
        # Calculate bbox area
        bbox_area = (x2 - x1) * (y2 - y1)
        relative_area = bbox_area / self.frame_area
        
        # Determine depth zone based on relative size
        if relative_area > config.NEAR_THRESHOLD:
            depth_zone = "near"
        elif relative_area < config.FAR_THRESHOLD:
            depth_zone = "far"
        else:
            depth_zone = "middle"
        
        return SpatialInfo(
            horizontal_zone=horizontal_zone,
            depth_zone=depth_zone,
            center_x=norm_center_x,
            center_y=center_y / self.frame_height,
            bbox_area=bbox_area,
            frame_area=self.frame_area,
            relative_area=relative_area
        )
    
    @staticmethod
    def format_spatial_description(detection: Detection, spatial_info: SpatialInfo) -> str:
        """
        Format spatial information into natural language
        
        Args:
            detection: Detection object
            spatial_info: SpatialInfo object
            
        Returns:
            Natural language description
        """
        # Build description
        parts = [detection.class_name]
        
        # Add position
        if spatial_info.horizontal_zone != "center":
            parts.append(f"on your {spatial_info.horizontal_zone}")
        else:
            parts.append("ahead")
        
        # Add distance
        distance_map = {
            "near": "close",
            "middle": "at medium distance",
            "far": "far away"
        }
        parts.append(distance_map[spatial_info.depth_zone])
        
        return " ".join(parts)


class DetectionTrackingPipeline:
    """Combined detection and tracking pipeline"""
    
    def __init__(self, frame_width: int, frame_height: int, device: str = "cpu"):
        """
        Initialize detection and tracking pipeline
        
        Args:
            frame_width: Frame width in pixels
            frame_height: Frame height in pixels
            device: Device to run on ('cpu' or 'cuda')
        """
        self.detector = ObjectDetector(device=device)
        self.tracker = ObjectTracker()
        self.spatial_analyzer = SpatialAnalyzer(frame_width, frame_height)
        
        self.last_announcements = {}  # track_id -> (timestamp, spatial_info)
    
    def process_frame(self, frame: np.ndarray, frame_timestamp: float) -> List[Tuple[Detection, SpatialInfo]]:
        """
        Process frame: detect, track, and analyze spatial info
        
        Args:
            frame: RGB frame array
            frame_timestamp: Timestamp of frame in seconds
            
        Returns:
            List of (Detection, SpatialInfo) tuples
        """
        # Detect objects
        detections = self.detector.detect(frame)
        
        # Track objects
        tracked_detections = self.tracker.update(detections, frame)
        
        # Analyze spatial information
        results = []
        for detection in tracked_detections:
            spatial_info = self.spatial_analyzer.analyze(detection)
            results.append((detection, spatial_info))
        
        return results
    
    def should_announce(self, detection: Detection, spatial_info: SpatialInfo, 
                       frame_timestamp: float) -> bool:
        """
        Determine if object should be announced based on temporal consistency
        
        Args:
            detection: Detection object
            spatial_info: SpatialInfo object
            frame_timestamp: Current frame timestamp
            
        Returns:
            True if should announce, False otherwise
        """
        if detection.track_id is None:
            return True
        
        track_id = detection.track_id
        
        # Check if we've announced this object recently
        if track_id in self.last_announcements:
            last_time, last_spatial = self.last_announcements[track_id]
            
            # Check cooldown period
            if frame_timestamp - last_time < config.ANNOUNCEMENT_COOLDOWN:
                # Check if position or size changed significantly
                pos_change = abs(spatial_info.center_x - last_spatial.center_x)
                size_change = abs(spatial_info.relative_area - last_spatial.relative_area)
                
                if (pos_change < config.MIN_POSITION_CHANGE and 
                    size_change < config.MIN_SIZE_CHANGE):
                    return False
        
        # Update last announcement
        self.last_announcements[track_id] = (frame_timestamp, spatial_info)
        return True
    
    def is_priority_object(self, detection: Detection) -> bool:
        """Check if object is safety-critical"""
        return detection.class_name.lower() in [obj.lower() for obj in config.PRIORITY_OBJECTS]


if __name__ == "__main__":
    # Test detection and tracking
    print("Testing detection and tracking pipeline...")
    
    # Create dummy frame
    dummy_frame = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
    
    try:
        pipeline = DetectionTrackingPipeline(640, 480)
        results = pipeline.process_frame(dummy_frame, 0.0)
        
        print(f"Detected {len(results)} objects")
        
        for detection, spatial_info in results:
            desc = SpatialAnalyzer.format_spatial_description(detection, spatial_info)
            print(f"  - {desc} (confidence: {detection.confidence:.2f})")
            
    except Exception as e:
        print(f"Error: {e}")
        print("Make sure to install: pip install ultralytics boxmot")
