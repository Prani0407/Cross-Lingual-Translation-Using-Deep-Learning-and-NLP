import cv2
import numpy as np
import os
from typing import Optional, Tuple
import logging
import time
import sounddevice as sd
import scipy.io.wavfile as wavfile
from speaker_detection import SpeakerDetector
from translation_models import TranslationModel
from speech_processing import SpeechProcessor
import torch

try:
    from pydub import AudioSegment
except ImportError as e:
    logging.warning(f"Failed to import pydub: {e}")
    AudioSegment = None

class VideoProcessor:
    def __init__(self, translation_system: TranslationModel, output_folder: str = None):
        self.translation_system = translation_system
        self.speaker_detector = SpeakerDetector()
        self.logger = logging.getLogger(__name__)
        self.output_dir = output_folder if output_folder else os.path.join(os.path.dirname(__file__), 'outputs')
        os.makedirs(self.output_dir, exist_ok=True)
        self.speech_processor = SpeechProcessor()

    def process_realtime_webcam(
        self,
        audio_path: str = None,
        model_choice: str = "mbart50",
        speaker_detect: bool = False,
        frame: np.ndarray = None
    ) -> Tuple[Optional[str], Optional[str], Optional[str]]:
        audio_data = None
        if audio_path and os.path.exists(audio_path):
            try:
                audio = AudioSegment.from_file(audio_path)
                self.logger.info(f"Audio loaded: {audio_path}, duration: {audio.duration_seconds} seconds")
                audio = audio.set_channels(1).set_frame_rate(16000)
                audio = audio.normalize()
                audio = audio.high_pass_filter(300.0)
                audio = audio + 10
                audio.export(audio_path, format="wav")  # Overwrite with processed audio
                self.logger.info("Audio extracted and processed successfully from uploaded file")
            except Exception as e:
                self.logger.error(f"Failed to extract or process audio from file: {e}")
                return None, None, None
        else:
            self.logger.error("No valid audio path provided for real-time processing")
            return None, None, None

        recognized_text = None
        translated_text = None
        speaker_info = None
        if os.path.getsize(audio_path) > 0:
            try:
                recognized_text = self.speech_processor.recognize_speech(audio_path)
                self.logger.info(f"Recognized text: {recognized_text}")
                if recognized_text and recognized_text != "No speech detected":
                    translated_text = self.translation_system.translate(recognized_text, source_lang='en', target_lang='hi', model_choice=model_choice)
                    self.logger.info(f"Translated text: {translated_text}")
                else:
                    self.logger.info("No speech detected in audio")
            except Exception as e:
                self.logger.error(f"Speech processing failed: {e}")
        else:
            self.logger.warning("Audio file is empty")

        if speaker_detect and recognized_text:
            if frame is not None and frame.size > 0:
                speaker_info = self.speaker_detector.detect_speaker(frame, bool(recognized_text))
                self.logger.info(f"Speaker info: {speaker_info}")
            else:
                speaker_info = "Not Speaking (No face detected or invalid frame)"
                self.logger.warning("No valid video frame provided for speaker detection")

        self.logger.info("Real-time webcam processing complete")
        # Clean up temporary file
        if os.path.exists(audio_path):
            try:
                os.remove(audio_path)
                self.logger.debug(f"Removed temp file: {audio_path}")
            except Exception as e:
                self.logger.warning(f"Error deleting {audio_path}: {e}")

        # Clear PyTorch cache to free up GPU/CPU memory
        if 'cuda' in self.translation_system.device:
            torch.cuda.empty_cache()
        self.logger.debug("Cleared PyTorch cache after real-time processing")

        return recognized_text or "No speech detected", translated_text or "No translation available", speaker_info

    def process_video_uploaded(
        self,
        input_path: str,
        model_choice: str = "mbart50",
        speaker_detect: bool = False
    ) -> Tuple[Optional[str], Optional[str], Optional[str]]:
        if AudioSegment is None:
            self.logger.error("PyDub not available, cannot process audio")
            return None, None, None

        audio_data = None
        try:
            self.logger.info(f"Extracting audio from video: {input_path}")
            video_audio = AudioSegment.from_file(input_path, format="mp4")
            self.logger.info(f"Audio extracted: duration {video_audio.duration_seconds} seconds, channels {video_audio.channels}, frame rate {video_audio.frame_rate}")
            video_audio = video_audio.set_channels(1).set_frame_rate(16000)
            video_audio = video_audio.normalize()
            video_audio = video_audio.high_pass_filter(300.0)
            video_audio = video_audio + 10
            audio_path = os.path.join(self.output_dir, f"temp_audio_{int(time.time())}.wav")
            video_audio.export(audio_path, format="wav")
            self.logger.info(f"Audio data prepared, length: {len(np.array(video_audio.get_array_of_samples()))} samples")
        except Exception as e:
            self.logger.error(f"Failed to extract audio with PyDub: {e}")
            return None, None, None

        recognized_text = None
        translated_text = None
        speaker_info = None
        if os.path.getsize(audio_path) > 0:
            try:
                recognized_text = self.speech_processor.recognize_speech(audio_path)
                if recognized_text and recognized_text != "No speech detected":
                    translated_text = self.translation_system.translate(recognized_text, source_lang='en', target_lang='hi', model_choice=model_choice)
                else:
                    self.logger.info("No speech detected in uploaded video")
            except Exception as e:
                self.logger.error(f"Speech processing failed for uploaded video: {e}")
        else:
            self.logger.info("No audio data extracted from video")

        if speaker_detect and recognized_text:
            cap = cv2.VideoCapture(input_path)
            ret, frame = cap.read()
            cap.release()
            if ret:
                speaker_info = self.speaker_detector.detect_speaker(frame, bool(recognized_text))
            else:
                self.logger.warning("Could not read frame from video")

        if os.path.exists(audio_path):
            os.remove(audio_path)
            self.logger.debug(f"Cleaned up temp file: {audio_path}")

        # Clear PyTorch cache to free up GPU/CPU memory
        if 'cuda' in self.translation_system.device:
            torch.cuda.empty_cache()
        self.logger.debug("Cleared PyTorch cache after video upload processing")

        self.logger.info("Uploaded video processing complete")
        return recognized_text or "No speech detected", translated_text or "No translation available", speaker_info