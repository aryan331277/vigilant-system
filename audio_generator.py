"""
Text-to-speech and spatial audio generation
"""

import numpy as np
from typing import Optional, Tuple
import logging
from pathlib import Path
import wave
import struct
import io

try:
    import pyttsx3
except ImportError:
    pyttsx3 = None
    logging.warning("pyttsx3 not installed. Install with: pip install pyttsx3")

try:
    from scipy.io import wavfile
    from scipy import signal
except ImportError:
    wavfile = None
    signal = None
    logging.warning("scipy not installed. Install with: pip install scipy")

import config
from scene_reasoner import SceneDescription
from detection_tracker import Detection, SpatialInfo

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class TextToSpeech:
    """Text-to-speech engine wrapper"""
    
    def __init__(self, engine_name: str = config.TTS_ENGINE):
        """
        Initialize TTS engine
        
        Args:
            engine_name: TTS engine to use ('pyttsx3' or 'gtts')
        """
        self.engine_name = engine_name
        self.engine = None
        self.temp_dir = config.CACHE_DIR / "tts_temp"
        self.temp_dir.mkdir(exist_ok=True)
        
        self._initialize_engine()
    
    def _initialize_engine(self):
        """Initialize TTS engine"""
        if self.engine_name == "pyttsx3":
            if pyttsx3 is None:
                logger.error("pyttsx3 not available")
                return
            
            try:
                self.engine = pyttsx3.init()
                self.engine.setProperty('rate', config.TTS_RATE)
                self.engine.setProperty('volume', config.TTS_VOLUME)
                logger.info("pyttsx3 TTS engine initialized")
            except Exception as e:
                logger.error(f"Failed to initialize pyttsx3: {e}")
                self.engine = None
        else:
            logger.warning(f"Unsupported TTS engine: {self.engine_name}")
    
    def synthesize(self, text: str, output_path: Optional[Path] = None) -> Optional[Path]:
        """
        Synthesize text to speech
        
        Args:
            text: Text to synthesize
            output_path: Optional output path (if None, generates temp file)
            
        Returns:
            Path to generated audio file, or None if failed
        """
        if not text or not self.engine:
            return None
        
        if output_path is None:
            # Generate temp filename
            import hashlib
            text_hash = hashlib.md5(text.encode()).hexdigest()[:8]
            output_path = self.temp_dir / f"tts_{text_hash}.wav"
        
        # Skip if already exists
        if output_path.exists():
            logger.debug(f"Using cached TTS: {output_path.name}")
            return output_path
        
        try:
            if self.engine_name == "pyttsx3":
                self.engine.save_to_file(text, str(output_path))
                self.engine.runAndWait()
            
            if output_path.exists():
                logger.debug(f"Generated TTS: {text[:50]}...")
                return output_path
            else:
                logger.warning(f"TTS file not created: {output_path}")
                return None
                
        except Exception as e:
            logger.error(f"TTS synthesis failed: {e}")
            return None
    
    def cleanup(self):
        """Cleanup resources"""
        if self.engine and hasattr(self.engine, 'stop'):
            try:
                self.engine.stop()
            except:
                pass


class SpatialAudioProcessor:
    """Applies spatial audio effects to audio signals"""
    
    def __init__(self, sample_rate: int = config.SAMPLE_RATE):
        """
        Initialize spatial audio processor
        
        Args:
            sample_rate: Audio sample rate
        """
        self.sample_rate = sample_rate
    
    def apply_spatial_effects(self, audio_data: np.ndarray, horizontal_zone: str, 
                             depth_zone: str) -> np.ndarray:
        """
        Apply spatial audio effects based on position
        
        Args:
            audio_data: Mono audio data
            horizontal_zone: Horizontal position (left, center, right)
            depth_zone: Depth position (near, middle, far)
            
        Returns:
            Stereo audio data with spatial effects
        """
        # Ensure mono input
        if len(audio_data.shape) > 1:
            audio_data = audio_data.mean(axis=1)
        
        # Apply volume based on distance
        volume = self._get_volume_for_depth(depth_zone)
        audio_data = audio_data * volume
        
        # Apply pitch shift for depth (subtle effect)
        pitch_factor = self._get_pitch_for_depth(depth_zone)
        audio_data = self._pitch_shift(audio_data, pitch_factor)
        
        # Apply panning for horizontal position
        left_channel, right_channel = self._apply_panning(audio_data, horizontal_zone)
        
        # Combine channels
        stereo_audio = np.stack([left_channel, right_channel], axis=1)
        
        return stereo_audio
    
    def _get_volume_for_depth(self, depth_zone: str) -> float:
        """Get volume multiplier based on depth"""
        volume_map = {
            'near': config.DISTANCE_VOLUME_NEAR,
            'middle': config.DISTANCE_VOLUME_MID,
            'far': config.DISTANCE_VOLUME_FAR
        }
        return volume_map.get(depth_zone, 1.0)
    
    def _get_pitch_for_depth(self, depth_zone: str) -> float:
        """Get pitch factor based on depth"""
        pitch_map = {
            'near': config.DISTANCE_PITCH_NEAR,
            'middle': config.DISTANCE_PITCH_MID,
            'far': config.DISTANCE_PITCH_FAR
        }
        return pitch_map.get(depth_zone, 1.0)
    
    def _pitch_shift(self, audio_data: np.ndarray, factor: float) -> np.ndarray:
        """Apply pitch shifting (simple resampling approach)"""
        if factor == 1.0 or signal is None:
            return audio_data
        
        try:
            # Simple pitch shift using resampling
            num_samples = int(len(audio_data) / factor)
            shifted = signal.resample(audio_data, num_samples)
            
            # Pad or trim to original length
            if len(shifted) < len(audio_data):
                shifted = np.pad(shifted, (0, len(audio_data) - len(shifted)))
            else:
                shifted = shifted[:len(audio_data)]
            
            return shifted
        except Exception as e:
            logger.warning(f"Pitch shift failed: {e}")
            return audio_data
    
    def _apply_panning(self, audio_data: np.ndarray, horizontal_zone: str) -> Tuple[np.ndarray, np.ndarray]:
        """Apply stereo panning"""
        # Calculate pan position (-1 = full left, 0 = center, 1 = full right)
        pan_map = {
            'left': -config.PAN_STRENGTH,
            'center': 0.0,
            'right': config.PAN_STRENGTH
        }
        pan = pan_map.get(horizontal_zone, 0.0)
        
        # Apply constant power panning
        left_gain = np.sqrt(0.5 * (1 - pan))
        right_gain = np.sqrt(0.5 * (1 + pan))
        
        left_channel = audio_data * left_gain
        right_channel = audio_data * right_gain
        
        return left_channel, right_channel
    
    def load_audio(self, audio_path: Path) -> Tuple[np.ndarray, int]:
        """Load audio file"""
        try:
            if wavfile is not None:
                sample_rate, audio_data = wavfile.read(audio_path)
                # Convert to float
                if audio_data.dtype == np.int16:
                    audio_data = audio_data.astype(np.float32) / 32768.0
                elif audio_data.dtype == np.int32:
                    audio_data = audio_data.astype(np.float32) / 2147483648.0
                return audio_data, sample_rate
            else:
                logger.error("scipy not available for audio loading")
                return None, None
        except Exception as e:
            logger.error(f"Failed to load audio: {e}")
            return None, None
    
    def save_audio(self, audio_data: np.ndarray, output_path: Path, sample_rate: Optional[int] = None):
        """Save audio file"""
        if sample_rate is None:
            sample_rate = self.sample_rate
        
        try:
            # Convert to int16
            if audio_data.dtype == np.float32 or audio_data.dtype == np.float64:
                audio_data = (audio_data * 32767).astype(np.int16)
            
            if wavfile is not None:
                wavfile.write(output_path, sample_rate, audio_data)
                logger.debug(f"Saved audio: {output_path.name}")
            else:
                # Fallback: use wave module
                self._save_audio_wave(audio_data, output_path, sample_rate)
        except Exception as e:
            logger.error(f"Failed to save audio: {e}")
    
    def _save_audio_wave(self, audio_data: np.ndarray, output_path: Path, sample_rate: int):
        """Save audio using wave module"""
        try:
            with wave.open(str(output_path), 'w') as wav_file:
                n_channels = audio_data.shape[1] if len(audio_data.shape) > 1 else 1
                wav_file.setnchannels(n_channels)
                wav_file.setsampwidth(2)  # 16-bit
                wav_file.setframerate(sample_rate)
                
                # Convert to bytes
                audio_bytes = audio_data.tobytes()
                wav_file.writeframes(audio_bytes)
            
            logger.debug(f"Saved audio (wave): {output_path.name}")
        except Exception as e:
            logger.error(f"Failed to save audio with wave module: {e}")


class SpatialAudioGenerator:
    """Main spatial audio generation pipeline"""
    
    def __init__(self):
        """Initialize spatial audio generator"""
        self.tts = TextToSpeech()
        self.spatial_processor = SpatialAudioProcessor()
        self.audio_sequence = []
    
    def generate_announcement(self, detection: Detection, spatial_info: SpatialInfo, 
                            frame_timestamp: float) -> Optional[Path]:
        """
        Generate spatial audio announcement for a detection
        
        Args:
            detection: Detection object
            spatial_info: SpatialInfo object
            frame_timestamp: Frame timestamp
            
        Returns:
            Path to generated audio file
        """
        # Format text
        from detection_tracker import SpatialAnalyzer
        text = SpatialAnalyzer.format_spatial_description(detection, spatial_info)
        
        # Generate TTS
        tts_path = self.tts.synthesize(text)
        
        if tts_path is None:
            return None
        
        # Load audio
        audio_data, sample_rate = self.spatial_processor.load_audio(tts_path)
        
        if audio_data is None:
            return None
        
        # Apply spatial effects
        spatial_audio = self.spatial_processor.apply_spatial_effects(
            audio_data, 
            spatial_info.horizontal_zone,
            spatial_info.depth_zone
        )
        
        # Save spatial audio
        output_path = config.AUDIO_OUTPUT_DIR / f"spatial_{frame_timestamp:.2f}_{detection.class_name}.wav"
        self.spatial_processor.save_audio(spatial_audio, output_path, sample_rate)
        
        # Track in sequence
        self.audio_sequence.append({
            'timestamp': frame_timestamp,
            'text': text,
            'audio_path': str(output_path),
            'detection': detection.class_name,
            'position': spatial_info.horizontal_zone,
            'depth': spatial_info.depth_zone
        })
        
        return output_path
    
    def generate_scene_announcement(self, scene_desc: SceneDescription) -> Optional[Path]:
        """
        Generate audio for scene description
        
        Args:
            scene_desc: SceneDescription object
            
        Returns:
            Path to generated audio file
        """
        text = scene_desc.description
        
        # Generate TTS
        tts_path = self.tts.synthesize(text)
        
        if tts_path is None:
            return None
        
        # For scene descriptions, use center position with no extreme effects
        audio_data, sample_rate = self.spatial_processor.load_audio(tts_path)
        
        if audio_data is None:
            return None
        
        # Apply minimal spatial effects (center, middle)
        spatial_audio = self.spatial_processor.apply_spatial_effects(
            audio_data, 'center', 'middle'
        )
        
        # Save
        output_path = config.AUDIO_OUTPUT_DIR / f"scene_{scene_desc.timestamp:.2f}.wav"
        self.spatial_processor.save_audio(spatial_audio, output_path, sample_rate)
        
        self.audio_sequence.append({
            'timestamp': scene_desc.timestamp,
            'text': text,
            'audio_path': str(output_path),
            'type': 'scene_description'
        })
        
        return output_path
    
    def generate_priority_alert(self, alert_text: str, timestamp: float) -> Optional[Path]:
        """
        Generate priority alert audio
        
        Args:
            alert_text: Alert text
            timestamp: Timestamp
            
        Returns:
            Path to generated audio file
        """
        # Generate TTS with emphasis
        tts_path = self.tts.synthesize(alert_text)
        
        if tts_path is None:
            return None
        
        audio_data, sample_rate = self.spatial_processor.load_audio(tts_path)
        
        if audio_data is None:
            return None
        
        # For alerts, use near depth with higher volume
        spatial_audio = self.spatial_processor.apply_spatial_effects(
            audio_data, 'center', 'near'
        )
        
        # Boost volume slightly for alerts
        spatial_audio = spatial_audio * 1.2
        spatial_audio = np.clip(spatial_audio, -32767, 32767)
        
        output_path = config.AUDIO_OUTPUT_DIR / f"alert_{timestamp:.2f}.wav"
        self.spatial_processor.save_audio(spatial_audio, output_path, sample_rate)
        
        self.audio_sequence.append({
            'timestamp': timestamp,
            'text': alert_text,
            'audio_path': str(output_path),
            'type': 'priority_alert'
        })
        
        return output_path
    
    def get_audio_sequence(self) -> list:
        """Get sequence of all generated audio"""
        return self.audio_sequence
    
    def export_sequence_metadata(self, output_path: Path):
        """Export audio sequence metadata to JSON"""
        import json
        
        with open(output_path, 'w') as f:
            json.dump(self.audio_sequence, f, indent=2)
        
        logger.info(f"Exported audio sequence metadata: {output_path}")
    
    def cleanup(self):
        """Cleanup resources"""
        self.tts.cleanup()


if __name__ == "__main__":
    # Test audio generation
    print("Testing spatial audio generation...")
    
    from detection_tracker import Detection, SpatialInfo
    
    # Create dummy detection
    detection = Detection(
        bbox=(100, 100, 200, 200),
        confidence=0.9,
        class_id=0,
        class_name="person"
    )
    
    spatial_info = SpatialInfo(
        horizontal_zone="left",
        depth_zone="near",
        center_x=0.3,
        center_y=0.5,
        bbox_area=10000,
        frame_area=307200,
        relative_area=0.03
    )
    
    try:
        generator = SpatialAudioGenerator()
        
        audio_path = generator.generate_announcement(detection, spatial_info, 0.0)
        
        if audio_path:
            print(f"Generated spatial audio: {audio_path}")
            print(f"Audio sequence length: {len(generator.get_audio_sequence())}")
        else:
            print("Audio generation failed. Check dependencies.")
        
        generator.cleanup()
        
    except Exception as e:
        print(f"Error: {e}")
        print("Note: Requires pyttsx3 and scipy. Install with: pip install pyttsx3 scipy")
