"""
Unit tests for spatial audio navigation system components
"""

import unittest
import numpy as np
from pathlib import Path
import tempfile
import shutil

import config


class TestVideoLoader(unittest.TestCase):
    """Test video loading functionality"""
    
    def setUp(self):
        """Setup test environment"""
        self.temp_dir = Path(tempfile.mkdtemp())
    
    def tearDown(self):
        """Cleanup test environment"""
        if self.temp_dir.exists():
            shutil.rmtree(self.temp_dir)
    
    def test_video_loader_init(self):
        """Test video loader initialization"""
        from video_loader import MSRVTTDataset
        
        dataset = MSRVTTDataset()
        self.assertIsInstance(dataset.video_files, list)
    
    def test_frame_preprocessing(self):
        """Test frame preprocessing"""
        from video_loader import preprocess_frame_for_yolo
        
        # Create dummy frame
        frame = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
        processed = preprocess_frame_for_yolo(frame)
        
        self.assertEqual(processed.shape, frame.shape)
        self.assertEqual(processed.dtype, frame.dtype)


class TestDetectionTracker(unittest.TestCase):
    """Test detection and tracking components"""
    
    def test_detection_dataclass(self):
        """Test Detection dataclass"""
        from detection_tracker import Detection
        
        det = Detection(
            bbox=(100, 100, 200, 200),
            confidence=0.9,
            class_id=0,
            class_name="person"
        )
        
        self.assertEqual(det.class_name, "person")
        self.assertEqual(det.confidence, 0.9)
        self.assertIsNone(det.track_id)
    
    def test_spatial_info(self):
        """Test SpatialInfo dataclass"""
        from detection_tracker import SpatialInfo
        
        spatial = SpatialInfo(
            horizontal_zone="left",
            depth_zone="near",
            center_x=0.3,
            center_y=0.5,
            bbox_area=10000,
            frame_area=307200,
            relative_area=0.03
        )
        
        self.assertEqual(spatial.horizontal_zone, "left")
        self.assertEqual(spatial.depth_zone, "near")
    
    def test_spatial_analyzer(self):
        """Test spatial analysis"""
        from detection_tracker import SpatialAnalyzer, Detection
        
        analyzer = SpatialAnalyzer(640, 480)
        
        # Test left zone
        det_left = Detection(
            bbox=(50, 100, 150, 200),
            confidence=0.9,
            class_id=0,
            class_name="person"
        )
        spatial_left = analyzer.analyze(det_left)
        self.assertEqual(spatial_left.horizontal_zone, "left")
        
        # Test center zone
        det_center = Detection(
            bbox=(270, 100, 370, 200),
            confidence=0.9,
            class_id=0,
            class_name="person"
        )
        spatial_center = analyzer.analyze(det_center)
        self.assertEqual(spatial_center.horizontal_zone, "center")
        
        # Test right zone
        det_right = Detection(
            bbox=(500, 100, 600, 200),
            confidence=0.9,
            class_id=0,
            class_name="person"
        )
        spatial_right = analyzer.analyze(det_right)
        self.assertEqual(spatial_right.horizontal_zone, "right")
    
    def test_spatial_description(self):
        """Test spatial description formatting"""
        from detection_tracker import SpatialAnalyzer, Detection, SpatialInfo
        
        det = Detection(
            bbox=(100, 100, 200, 200),
            confidence=0.9,
            class_id=0,
            class_name="person"
        )
        
        spatial = SpatialInfo(
            horizontal_zone="left",
            depth_zone="near",
            center_x=0.3,
            center_y=0.5,
            bbox_area=10000,
            frame_area=307200,
            relative_area=0.03
        )
        
        desc = SpatialAnalyzer.format_spatial_description(det, spatial)
        
        self.assertIn("person", desc)
        self.assertIn("left", desc)
        self.assertIn("close", desc)


class TestSceneReasoner(unittest.TestCase):
    """Test scene understanding components"""
    
    def test_scene_description(self):
        """Test SceneDescription dataclass"""
        from scene_reasoner import SceneDescription
        
        desc = SceneDescription(
            description="Test scene",
            objects_mentioned=["person", "chair"],
            spatial_relations=["person left of chair"],
            priority_alerts=["Warning: person ahead"],
            frame_hash="abc123",
            timestamp=0.0
        )
        
        self.assertEqual(desc.description, "Test scene")
        self.assertEqual(len(desc.objects_mentioned), 2)
    
    def test_description_cache(self):
        """Test description caching"""
        from scene_reasoner import DescriptionCache, SceneDescription
        
        cache = DescriptionCache(max_size=2)
        
        desc1 = SceneDescription(
            description="Scene 1",
            objects_mentioned=["person"],
            spatial_relations=[],
            priority_alerts=[],
            frame_hash="hash1",
            timestamp=0.0
        )
        
        # Test put and get
        cache.put("key1", desc1)
        retrieved = cache.get("key1")
        
        self.assertIsNotNone(retrieved)
        self.assertEqual(retrieved.description, "Scene 1")
        
        # Test cache eviction
        desc2 = SceneDescription(
            description="Scene 2",
            objects_mentioned=["chair"],
            spatial_relations=[],
            priority_alerts=[],
            frame_hash="hash2",
            timestamp=1.0
        )
        
        desc3 = SceneDescription(
            description="Scene 3",
            objects_mentioned=["table"],
            spatial_relations=[],
            priority_alerts=[],
            frame_hash="hash3",
            timestamp=2.0
        )
        
        cache.put("key2", desc2)
        cache.put("key3", desc3)
        
        # key1 should be evicted (least accessed)
        self.assertIsNone(cache.get("key1"))
        self.assertIsNotNone(cache.get("key2"))


class TestAudioGenerator(unittest.TestCase):
    """Test audio generation components"""
    
    def test_spatial_audio_processor(self):
        """Test spatial audio processing"""
        from audio_generator import SpatialAudioProcessor
        
        processor = SpatialAudioProcessor()
        
        # Create dummy audio
        audio = np.random.randn(44100).astype(np.float32)  # 1 second
        
        # Apply spatial effects
        spatial_audio = processor.apply_spatial_effects(audio, "left", "near")
        
        # Should be stereo
        self.assertEqual(len(spatial_audio.shape), 2)
        self.assertEqual(spatial_audio.shape[1], 2)
        
        # Left channel should be louder for left position
        left_power = np.sum(spatial_audio[:, 0] ** 2)
        right_power = np.sum(spatial_audio[:, 1] ** 2)
        self.assertGreater(left_power, right_power)
    
    def test_panning(self):
        """Test stereo panning"""
        from audio_generator import SpatialAudioProcessor
        
        processor = SpatialAudioProcessor()
        audio = np.ones(1000, dtype=np.float32)
        
        # Test left panning
        left_l, left_r = processor._apply_panning(audio, "left")
        self.assertGreater(np.mean(left_l), np.mean(left_r))
        
        # Test right panning
        right_l, right_r = processor._apply_panning(audio, "right")
        self.assertGreater(np.mean(right_r), np.mean(right_l))
        
        # Test center (equal)
        center_l, center_r = processor._apply_panning(audio, "center")
        np.testing.assert_almost_equal(np.mean(center_l), np.mean(center_r), decimal=5)


class TestPipelineIntegration(unittest.TestCase):
    """Test pipeline integration"""
    
    def test_performance_metrics(self):
        """Test performance metrics"""
        from spatial_audio_pipeline import PerformanceMetrics
        
        metrics = PerformanceMetrics()
        metrics.total_frames = 100
        metrics.total_time = 10.0
        metrics.total_detections = 500
        metrics.total_announcements = 50
        
        metrics.calculate_averages()
        
        self.assertEqual(metrics.avg_fps, 10.0)
        self.assertEqual(metrics.total_frames, 100)
    
    def test_config_loading(self):
        """Test configuration loading"""
        self.assertIsNotNone(config.TARGET_FPS)
        self.assertIsNotNone(config.CONFIDENCE_THRESHOLD)
        self.assertIsNotNone(config.YOLO_MODEL)
        self.assertEqual(config.AUDIO_CHANNELS, 2)


class TestEndToEnd(unittest.TestCase):
    """End-to-end integration tests"""
    
    def test_dummy_pipeline(self):
        """Test pipeline with dummy data"""
        # This test would require actual video file
        # For now, just test imports
        from spatial_audio_pipeline import SpatialAudioPipeline
        
        # Just test initialization
        pipeline = SpatialAudioPipeline(device="cpu")
        self.assertIsNotNone(pipeline)


def run_tests():
    """Run all tests"""
    # Create test suite
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()
    
    # Add test classes
    suite.addTests(loader.loadTestsFromTestCase(TestVideoLoader))
    suite.addTests(loader.loadTestsFromTestCase(TestDetectionTracker))
    suite.addTests(loader.loadTestsFromTestCase(TestSceneReasoner))
    suite.addTests(loader.loadTestsFromTestCase(TestAudioGenerator))
    suite.addTests(loader.loadTestsFromTestCase(TestPipelineIntegration))
    suite.addTests(loader.loadTestsFromTestCase(TestEndToEnd))
    
    # Run tests
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    
    return result.wasSuccessful()


if __name__ == "__main__":
    import sys
    success = run_tests()
    sys.exit(0 if success else 1)
