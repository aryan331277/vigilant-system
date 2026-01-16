"""
Scene understanding using Moondream vision-language model
"""

import numpy as np
from typing import Dict, List, Optional, Tuple
import logging
from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
import torch
from PIL import Image

try:
    from transformers import AutoModelForCausalLM, AutoTokenizer
except ImportError:
    AutoModelForCausalLM = None
    AutoTokenizer = None
    logging.warning("transformers not installed. Install with: pip install transformers")

import config
from detection_tracker import Detection, SpatialInfo

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class SceneDescription:
    """Represents a scene description from Moondream"""
    description: str
    objects_mentioned: List[str]
    spatial_relations: List[str]
    priority_alerts: List[str]
    frame_hash: str
    timestamp: float


class MoondreamReasoner:
    """Scene understanding using Moondream model"""
    
    def __init__(self, model_name: str = config.MOONDREAM_MODEL, 
                 revision: str = config.MOONDREAM_REVISION,
                 device: str = "cpu"):
        """
        Initialize Moondream reasoner
        
        Args:
            model_name: Moondream model name
            revision: Model revision
            device: Device to run on ('cpu' or 'cuda')
        """
        self.model_name = model_name
        self.revision = revision
        self.device = device
        self.model = None
        self.tokenizer = None
        self.cache = {}
        self.cache_file = config.CACHE_DIR / "moondream_cache.json"
        
        self._load_model()
        self._load_cache()
    
    def _load_model(self):
        """Load Moondream model"""
        if AutoModelForCausalLM is None or AutoTokenizer is None:
            logger.warning("Moondream will not be available. Install transformers.")
            return
        
        try:
            logger.info(f"Loading Moondream model: {self.model_name}")
            
            self.tokenizer = AutoTokenizer.from_pretrained(
                self.model_name,
                revision=self.revision,
                trust_remote_code=True
            )
            
            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_name,
                revision=self.revision,
                trust_remote_code=True,
                torch_dtype=torch.float32 if self.device == "cpu" else torch.float16
            ).to(self.device)
            
            self.model.eval()
            
            logger.info("Moondream model loaded successfully")
            
        except Exception as e:
            logger.error(f"Failed to load Moondream: {e}")
            self.model = None
            self.tokenizer = None
    
    def _load_cache(self):
        """Load cached descriptions"""
        if self.cache_file.exists():
            try:
                with open(self.cache_file, 'r') as f:
                    self.cache = json.load(f)
                logger.info(f"Loaded {len(self.cache)} cached descriptions")
            except Exception as e:
                logger.warning(f"Failed to load cache: {e}")
                self.cache = {}
    
    def _save_cache(self):
        """Save cache to disk"""
        try:
            with open(self.cache_file, 'w') as f:
                json.dump(self.cache, f, indent=2)
        except Exception as e:
            logger.warning(f"Failed to save cache: {e}")
    
    def _compute_frame_hash(self, frame: np.ndarray) -> str:
        """Compute hash of frame for caching"""
        # Use a subset of pixels for speed
        subset = frame[::10, ::10, :]
        return hashlib.md5(subset.tobytes()).hexdigest()
    
    def analyze_scene(self, frame: np.ndarray, detections: List[Tuple[Detection, SpatialInfo]], 
                     timestamp: float) -> SceneDescription:
        """
        Analyze scene using Moondream
        
        Args:
            frame: RGB frame array
            detections: List of (Detection, SpatialInfo) tuples
            timestamp: Frame timestamp
            
        Returns:
            SceneDescription object
        """
        # Check cache
        frame_hash = self._compute_frame_hash(frame)
        
        if frame_hash in self.cache:
            cached = self.cache[frame_hash]
            logger.debug(f"Using cached description for frame {frame_hash[:8]}")
            return SceneDescription(
                description=cached['description'],
                objects_mentioned=cached['objects_mentioned'],
                spatial_relations=cached['spatial_relations'],
                priority_alerts=cached['priority_alerts'],
                frame_hash=frame_hash,
                timestamp=timestamp
            )
        
        # Generate description using detections
        description = self._generate_description(frame, detections)
        
        # Extract information
        objects_mentioned = list(set([det.class_name for det, _ in detections]))
        spatial_relations = self._extract_spatial_relations(detections)
        priority_alerts = self._identify_priority_alerts(detections)
        
        scene_desc = SceneDescription(
            description=description,
            objects_mentioned=objects_mentioned,
            spatial_relations=spatial_relations,
            priority_alerts=priority_alerts,
            frame_hash=frame_hash,
            timestamp=timestamp
        )
        
        # Cache result
        self.cache[frame_hash] = {
            'description': description,
            'objects_mentioned': objects_mentioned,
            'spatial_relations': spatial_relations,
            'priority_alerts': priority_alerts
        }
        
        # Periodically save cache
        if len(self.cache) % 10 == 0:
            self._save_cache()
        
        return scene_desc
    
    def _generate_description(self, frame: np.ndarray, 
                            detections: List[Tuple[Detection, SpatialInfo]]) -> str:
        """Generate natural language description"""
        if not detections:
            return "No objects detected in the scene"
        
        # If Moondream is available, use it
        if self.model is not None and self.tokenizer is not None:
            try:
                return self._generate_with_moondream(frame, detections)
            except Exception as e:
                logger.warning(f"Moondream inference failed: {e}. Using fallback.")
        
        # Fallback: use rule-based description
        return self._generate_fallback_description(detections)
    
    def _generate_with_moondream(self, frame: np.ndarray, 
                                detections: List[Tuple[Detection, SpatialInfo]]) -> str:
        """Generate description using Moondream model"""
        # Convert frame to PIL Image
        image = Image.fromarray(frame)
        
        # Create prompt based on detections
        objects = [det.class_name for det, _ in detections[:5]]  # Top 5 objects
        prompt = (f"Describe the spatial layout of this scene focusing on navigation for "
                 f"visually impaired users. Detected objects: {', '.join(objects)}. "
                 f"Provide brief, clear spatial descriptions.")
        
        # Encode image and prompt
        with torch.no_grad():
            # Use Moondream's answer method
            if hasattr(self.model, 'answer_question'):
                description = self.model.answer_question(
                    self.tokenizer.encode_image(image),
                    prompt,
                    self.tokenizer,
                    max_new_tokens=config.MOONDREAM_MAX_TOKENS
                )
            else:
                # Fallback if API changed
                description = self._generate_fallback_description(detections)
        
        return description
    
    def _generate_fallback_description(self, detections: List[Tuple[Detection, SpatialInfo]]) -> str:
        """Generate rule-based description as fallback"""
        if not detections:
            return "Clear path ahead"
        
        # Group by zones
        zones = {'left': [], 'center': [], 'right': []}
        depths = {'near': [], 'middle': [], 'far': []}
        
        for det, spatial in detections:
            zones[spatial.horizontal_zone].append(det.class_name)
            depths[spatial.depth_zone].append(det.class_name)
        
        parts = []
        
        # Priority: near objects first
        if depths['near']:
            near_objects = ', '.join(list(set(depths['near']))[:3])
            parts.append(f"Close: {near_objects}")
        
        # Then by position
        if zones['center']:
            center_objects = ', '.join(list(set(zones['center']))[:2])
            parts.append(f"ahead: {center_objects}")
        
        if zones['left']:
            left_objects = ', '.join(list(set(zones['left']))[:2])
            parts.append(f"left: {left_objects}")
        
        if zones['right']:
            right_objects = ', '.join(list(set(zones['right']))[:2])
            parts.append(f"right: {right_objects}")
        
        return ". ".join(parts) if parts else "Objects detected in scene"
    
    def _extract_spatial_relations(self, detections: List[Tuple[Detection, SpatialInfo]]) -> List[str]:
        """Extract spatial relationships between objects"""
        relations = []
        
        # Sort by horizontal position
        sorted_dets = sorted(detections, key=lambda x: x[1].center_x)
        
        for i in range(len(sorted_dets) - 1):
            det1, spatial1 = sorted_dets[i]
            det2, spatial2 = sorted_dets[i + 1]
            
            # Check if objects are side by side
            if abs(spatial1.center_y - spatial2.center_y) < 0.2:  # Similar vertical position
                if spatial1.horizontal_zone == 'left' and spatial2.horizontal_zone == 'right':
                    relations.append(f"{det1.class_name} to left of {det2.class_name}")
        
        return relations
    
    def _identify_priority_alerts(self, detections: List[Tuple[Detection, SpatialInfo]]) -> List[str]:
        """Identify safety-critical alerts"""
        alerts = []
        
        priority_classes = [obj.lower() for obj in config.PRIORITY_OBJECTS]
        
        for det, spatial in detections:
            if det.class_name.lower() in priority_classes:
                # Near objects are high priority
                if spatial.depth_zone == 'near':
                    alert = f"Warning: {det.class_name} {spatial.horizontal_zone}, very close"
                    alerts.append(alert)
                # Center objects at medium distance
                elif spatial.horizontal_zone == 'center' and spatial.depth_zone == 'middle':
                    alert = f"Caution: {det.class_name} ahead"
                    alerts.append(alert)
        
        return alerts
    
    def cleanup(self):
        """Save cache and cleanup"""
        self._save_cache()
        logger.info("Moondream reasoner cleanup complete")


class DescriptionCache:
    """Cache for scene descriptions to avoid redundant processing"""
    
    def __init__(self, max_size: int = 1000):
        self.cache = {}
        self.max_size = max_size
        self.access_count = {}
    
    def get(self, key: str) -> Optional[SceneDescription]:
        """Get cached description"""
        if key in self.cache:
            self.access_count[key] = self.access_count.get(key, 0) + 1
            return self.cache[key]
        return None
    
    def put(self, key: str, value: SceneDescription):
        """Store description in cache"""
        # Evict least accessed if cache is full
        if len(self.cache) >= self.max_size:
            least_accessed = min(self.access_count.items(), key=lambda x: x[1])[0]
            del self.cache[least_accessed]
            del self.access_count[least_accessed]
        
        self.cache[key] = value
        self.access_count[key] = 0
    
    def clear(self):
        """Clear cache"""
        self.cache.clear()
        self.access_count.clear()


if __name__ == "__main__":
    # Test scene reasoning
    print("Testing Moondream scene reasoner...")
    
    # Create dummy frame
    dummy_frame = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
    
    # Create dummy detections
    from detection_tracker import Detection, SpatialInfo
    
    dummy_detections = [
        (
            Detection(bbox=(100, 100, 200, 200), confidence=0.9, class_id=0, class_name="person"),
            SpatialInfo(horizontal_zone="center", depth_zone="near", center_x=0.5, center_y=0.5,
                       bbox_area=10000, frame_area=307200, relative_area=0.03)
        ),
        (
            Detection(bbox=(400, 150, 500, 250), confidence=0.85, class_id=1, class_name="chair"),
            SpatialInfo(horizontal_zone="right", depth_zone="middle", center_x=0.7, center_y=0.4,
                       bbox_area=10000, frame_area=307200, relative_area=0.03)
        )
    ]
    
    try:
        reasoner = MoondreamReasoner()
        scene_desc = reasoner.analyze_scene(dummy_frame, dummy_detections, 0.0)
        
        print("\nScene Description:")
        print(f"  Description: {scene_desc.description}")
        print(f"  Objects: {', '.join(scene_desc.objects_mentioned)}")
        print(f"  Spatial Relations: {', '.join(scene_desc.spatial_relations)}")
        print(f"  Priority Alerts: {', '.join(scene_desc.priority_alerts)}")
        
        reasoner.cleanup()
        
    except Exception as e:
        print(f"Error: {e}")
        print("Note: Moondream requires transformers. Install with: pip install transformers")
