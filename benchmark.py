"""
Benchmark script for performance testing
"""

import time
import argparse
from pathlib import Path
import json
import numpy as np

from spatial_audio_pipeline import SpatialAudioPipeline
from video_loader import VideoLoader
import config


def benchmark_video_loading(video_path: str, num_runs: int = 3) -> dict:
    """Benchmark video loading performance"""
    print(f"\nBenchmarking Video Loading...")
    
    times = []
    
    for i in range(num_runs):
        start = time.time()
        
        with VideoLoader(video_path, target_fps=config.TARGET_FPS) as loader:
            frame_count = 0
            for frame_num, frame, timestamp in loader.read_frames(max_frames=100):
                frame_count += 1
        
        elapsed = time.time() - start
        times.append(elapsed)
        print(f"  Run {i+1}: {elapsed:.2f}s, {frame_count/elapsed:.2f} FPS")
    
    return {
        'avg_time': np.mean(times),
        'std_time': np.std(times),
        'avg_fps': 100 / np.mean(times)
    }


def benchmark_detection(video_path: str, num_frames: int = 50) -> dict:
    """Benchmark detection performance"""
    print(f"\nBenchmarking Object Detection...")
    
    from detection_tracker import ObjectDetector
    
    with VideoLoader(video_path, target_fps=config.TARGET_FPS) as loader:
        detector = ObjectDetector(device="cpu")
        
        times = []
        detection_counts = []
        
        for frame_num, frame, timestamp in loader.read_frames(max_frames=num_frames):
            start = time.time()
            detections = detector.detect(frame)
            elapsed = time.time() - start
            
            times.append(elapsed)
            detection_counts.append(len(detections))
            
            if frame_num % 10 == 0:
                print(f"  Frame {frame_num}: {elapsed*1000:.1f}ms, {len(detections)} detections")
    
    return {
        'avg_time_ms': np.mean(times) * 1000,
        'std_time_ms': np.std(times) * 1000,
        'max_time_ms': np.max(times) * 1000,
        'avg_detections': np.mean(detection_counts),
        'total_frames': len(times)
    }


def benchmark_tracking(video_path: str, num_frames: int = 50) -> dict:
    """Benchmark tracking performance"""
    print(f"\nBenchmarking Object Tracking...")
    
    from detection_tracker import DetectionTrackingPipeline
    
    with VideoLoader(video_path, target_fps=config.TARGET_FPS) as loader:
        metadata = loader.get_metadata()
        pipeline = DetectionTrackingPipeline(
            metadata['width'],
            metadata['height'],
            device="cpu"
        )
        
        times = []
        
        for frame_num, frame, timestamp in loader.read_frames(max_frames=num_frames):
            start = time.time()
            results = pipeline.process_frame(frame, timestamp)
            elapsed = time.time() - start
            
            times.append(elapsed)
            
            if frame_num % 10 == 0:
                print(f"  Frame {frame_num}: {elapsed*1000:.1f}ms, {len(results)} tracked objects")
    
    return {
        'avg_time_ms': np.mean(times) * 1000,
        'std_time_ms': np.std(times) * 1000,
        'max_time_ms': np.max(times) * 1000,
        'total_frames': len(times)
    }


def benchmark_audio_generation(num_samples: int = 10) -> dict:
    """Benchmark audio generation"""
    print(f"\nBenchmarking Audio Generation...")
    
    from audio_generator import SpatialAudioGenerator
    from detection_tracker import Detection, SpatialInfo
    
    generator = SpatialAudioGenerator()
    
    # Create test detections
    test_objects = [
        ("person", "center", "near"),
        ("chair", "left", "middle"),
        ("door", "right", "far"),
        ("car", "center", "middle"),
        ("table", "left", "near")
    ]
    
    times = []
    
    for i in range(num_samples):
        obj_name, position, depth = test_objects[i % len(test_objects)]
        
        detection = Detection(
            bbox=(100, 100, 200, 200),
            confidence=0.9,
            class_id=0,
            class_name=obj_name
        )
        
        spatial_info = SpatialInfo(
            horizontal_zone=position,
            depth_zone=depth,
            center_x=0.5,
            center_y=0.5,
            bbox_area=10000,
            frame_area=307200,
            relative_area=0.03
        )
        
        start = time.time()
        audio_path = generator.generate_announcement(detection, spatial_info, float(i))
        elapsed = time.time() - start
        
        times.append(elapsed)
        
        print(f"  Sample {i+1}: {elapsed*1000:.1f}ms")
    
    generator.cleanup()
    
    return {
        'avg_time_ms': np.mean(times) * 1000,
        'std_time_ms': np.std(times) * 1000,
        'max_time_ms': np.max(times) * 1000,
        'total_samples': len(times)
    }


def benchmark_full_pipeline(video_path: str, num_frames: int = 100, device: str = "cpu") -> dict:
    """Benchmark full pipeline"""
    print(f"\nBenchmarking Full Pipeline...")
    
    pipeline = SpatialAudioPipeline(device=device)
    
    start = time.time()
    report = pipeline.process_video(video_path, max_frames=num_frames)
    elapsed = time.time() - start
    
    metrics = report['performance_metrics']
    
    return {
        'total_time': elapsed,
        'avg_fps': metrics['avg_fps'],
        'total_frames': metrics['total_frames'],
        'total_detections': metrics['total_detections'],
        'total_announcements': metrics['total_announcements'],
        'detection_time': metrics['detection_time'],
        'reasoning_time': metrics['reasoning_time'],
        'audio_time': metrics['audio_generation_time']
    }


def run_benchmark_suite(video_path: str, device: str = "cpu"):
    """Run complete benchmark suite"""
    print("=" * 60)
    print("SPATIAL AUDIO NAVIGATION SYSTEM - BENCHMARK SUITE")
    print("=" * 60)
    print(f"Video: {video_path}")
    print(f"Device: {device}")
    print("=" * 60)
    
    results = {}
    
    # Video loading
    try:
        results['video_loading'] = benchmark_video_loading(video_path)
    except Exception as e:
        print(f"Video loading benchmark failed: {e}")
        results['video_loading'] = None
    
    # Detection
    try:
        results['detection'] = benchmark_detection(video_path, num_frames=50)
    except Exception as e:
        print(f"Detection benchmark failed: {e}")
        results['detection'] = None
    
    # Tracking
    try:
        results['tracking'] = benchmark_tracking(video_path, num_frames=50)
    except Exception as e:
        print(f"Tracking benchmark failed: {e}")
        results['tracking'] = None
    
    # Audio generation
    try:
        results['audio_generation'] = benchmark_audio_generation(num_samples=10)
    except Exception as e:
        print(f"Audio generation benchmark failed: {e}")
        results['audio_generation'] = None
    
    # Full pipeline
    try:
        results['full_pipeline'] = benchmark_full_pipeline(video_path, num_frames=100, device=device)
    except Exception as e:
        print(f"Full pipeline benchmark failed: {e}")
        results['full_pipeline'] = None
    
    # Print summary
    print("\n" + "=" * 60)
    print("BENCHMARK SUMMARY")
    print("=" * 60)
    
    if results.get('video_loading'):
        print(f"\nVideo Loading:")
        print(f"  Average FPS: {results['video_loading']['avg_fps']:.2f}")
    
    if results.get('detection'):
        print(f"\nObject Detection:")
        print(f"  Average time: {results['detection']['avg_time_ms']:.1f}ms")
        print(f"  Max time: {results['detection']['max_time_ms']:.1f}ms")
        print(f"  Average detections: {results['detection']['avg_detections']:.1f}")
    
    if results.get('tracking'):
        print(f"\nObject Tracking (Detection + Tracking):")
        print(f"  Average time: {results['tracking']['avg_time_ms']:.1f}ms")
        print(f"  Max time: {results['tracking']['max_time_ms']:.1f}ms")
    
    if results.get('audio_generation'):
        print(f"\nAudio Generation:")
        print(f"  Average time: {results['audio_generation']['avg_time_ms']:.1f}ms")
        print(f"  Max time: {results['audio_generation']['max_time_ms']:.1f}ms")
    
    if results.get('full_pipeline'):
        print(f"\nFull Pipeline:")
        print(f"  Total time: {results['full_pipeline']['total_time']:.2f}s")
        print(f"  Average FPS: {results['full_pipeline']['avg_fps']:.2f}")
        print(f"  Frames processed: {results['full_pipeline']['total_frames']}")
        print(f"  Total detections: {results['full_pipeline']['total_detections']}")
        print(f"  Total announcements: {results['full_pipeline']['total_announcements']}")
        print(f"\n  Time breakdown:")
        print(f"    Detection: {results['full_pipeline']['detection_time']:.2f}s")
        print(f"    Reasoning: {results['full_pipeline']['reasoning_time']:.2f}s")
        print(f"    Audio: {results['full_pipeline']['audio_time']:.2f}s")
    
    print("=" * 60)
    
    # Save results
    output_path = config.OUTPUT_DIR / "benchmark_results.json"
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\nResults saved to: {output_path}")
    
    return results


def main():
    parser = argparse.ArgumentParser(description="Benchmark spatial audio navigation system")
    
    parser.add_argument(
        '--video',
        type=str,
        required=True,
        help='Path to video file for benchmarking'
    )
    
    parser.add_argument(
        '--device',
        type=str,
        default='cpu',
        choices=['cpu', 'cuda'],
        help='Device to run on (default: cpu)'
    )
    
    parser.add_argument(
        '--component',
        type=str,
        choices=['all', 'loading', 'detection', 'tracking', 'audio', 'pipeline'],
        default='all',
        help='Component to benchmark (default: all)'
    )
    
    args = parser.parse_args()
    
    video_path = Path(args.video)
    
    if not video_path.exists():
        print(f"Error: Video not found: {video_path}")
        return
    
    if args.component == 'all':
        run_benchmark_suite(str(video_path), device=args.device)
    elif args.component == 'loading':
        benchmark_video_loading(str(video_path))
    elif args.component == 'detection':
        benchmark_detection(str(video_path))
    elif args.component == 'tracking':
        benchmark_tracking(str(video_path))
    elif args.component == 'audio':
        benchmark_audio_generation()
    elif args.component == 'pipeline':
        benchmark_full_pipeline(str(video_path), device=args.device)


if __name__ == "__main__":
    main()
