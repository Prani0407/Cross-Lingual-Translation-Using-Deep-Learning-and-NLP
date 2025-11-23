import os
import logging
import numpy as np
import soundfile as sf
from pydub import AudioSegment
import speech_recognition as sr

logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

class SpeechProcessor:
    def __init__(self):
        self.recognizer = sr.Recognizer()
        self.recognizer.energy_threshold = 300
        self.recognizer.dynamic_energy_threshold = True

    def adjust_for_ambient_noise(self, audio_path):
        try:
            with sr.AudioFile(audio_path) as source:
                self.recognizer.adjust_for_ambient_noise(source, duration=0.5)  # Shorten duration
                logger.debug("Adjusted for ambient noise")
        except Exception as e:
            logger.error(f"Failed to adjust for ambient noise: {str(e)}")

    def recognize_speech(self, audio_path):
        logger.debug(f"Starting speech recognition from: {audio_path}")
        try:
            with sf.SoundFile(audio_path) as audio_file:
                duration_sec = len(audio_file) / audio_file.samplerate
                logger.debug(f"Audio file duration: {duration_sec:.2f} seconds")

            audio = AudioSegment.from_wav(audio_path)
            audio = audio.set_channels(1).set_frame_rate(16000)
            processed_audio_path = audio_path.replace(".wav", "_processed.wav")
            audio.export(processed_audio_path, format="wav")
            logger.debug(f"Processed audio saved to: {processed_audio_path}")

            self.adjust_for_ambient_noise(processed_audio_path)

            with sr.AudioFile(processed_audio_path) as source:
                audio_data = self.recognizer.record(source)  # Process entire file
                logger.debug("Audio recorded from file.")

            text = self.recognizer.recognize_google(audio_data, language='en-IN')
            logger.debug(f"Recognized text: {text}")
            return text

        except sr.UnknownValueError:
            logger.warning("No speech detected")
            return "No speech detected"
        except sr.RequestError as e:
            logger.error(f"Google API error: {str(e)}")
            return f"Error: Could not request results; {str(e)}"
        except Exception as e:
            logger.error(f"General recognition error: {str(e)}")
            return f"Error: {str(e)}"
        finally:
            if os.path.exists(processed_audio_path):
                try:
                    os.remove(processed_audio_path)
                    logger.debug(f"Cleaned up temp file: {processed_audio_path}")
                except Exception as e:
                    logger.warning(f"Could not delete {processed_audio_path}: {e}")