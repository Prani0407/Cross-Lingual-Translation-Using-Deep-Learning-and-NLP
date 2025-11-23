import os
import uuid
import logging
import numpy as np
from flask import Flask, request, jsonify, send_from_directory, render_template
from werkzeug.utils import secure_filename
import speech_recognition as sr
from transformers import MBartForConditionalGeneration, MBart50TokenizerFast
import torch
from video_processing import VideoProcessor
from translation_models import TranslationModel
import base64
from io import BytesIO
from PIL import Image

# --- Logging Configuration ---
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

# --- Flask App Setup ---
app = Flask(__name__, template_folder='templates')
app.config.from_object('config.Config')

# --- Ensure Folders Exist ---
for folder in [app.config['UPLOAD_FOLDER'], app.config['OUTPUT_FOLDER']]:
    os.makedirs(folder, exist_ok=True)
    logger.info(f"Directory ensured: {folder}")

# --- Initialize Modules ---
device = "cuda" if torch.cuda.is_available() else "cpu"
translation_system = TranslationModel()  # Initialize once, but model loading happens per request
video_processor = VideoProcessor(translation_system, app.config['OUTPUT_FOLDER'])
recognizer = sr.Recognizer()

# === ROUTES ===

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/static/<path:filename>')
def serve_static(filename):
    return send_from_directory(app.static_folder, filename)

@app.route('/static/weights/<path:filename>')
def serve_weights(filename):
    return send_from_directory(os.path.join(app.static_folder, 'weights'), filename)

@app.route('/uploads/<path:filename>')
def serve_uploaded(filename):
    file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    if os.path.exists(file_path):
        return send_from_directory(app.config['UPLOAD_FOLDER'], filename)
    return jsonify({'error': 'File not found'}), 404

@app.route('/process_realtime_webcam', methods=['POST'])
def process_realtime_webcam():
    try:
        audio_file = request.files.get('audio')
        video_frame_data = request.form.get('video_frame')
        model = request.form.get('model', 'mbart50')
        speaker_detect = request.form.get('speaker_detect', 'false').lower() == 'true'

        if not audio_file:
            logger.warning("No audio file provided.")
            return jsonify({'error': 'No audio file provided'}), 400

        audio_filename = f"audio_{uuid.uuid4()}.wav"
        audio_path = os.path.join(app.config['UPLOAD_FOLDER'], secure_filename(audio_filename))
        audio_file.save(audio_path)
        if os.path.getsize(audio_path) == 0:
            logger.error(f"Empty audio file received: {audio_path}")
            os.remove(audio_path)
            return jsonify({'error': 'Empty audio file received'}), 400
        logger.debug(f"Saved audio to {audio_path} with size {os.path.getsize(audio_path)} bytes")

        # Decode video frame if provided
        frame = None
        if video_frame_data:
            frame_data = base64.b64decode(video_frame_data.split(',')[1])
            frame = np.array(Image.open(BytesIO(frame_data)))

        recognized_text, translated_text, speaker_info = video_processor.process_realtime_webcam(
            audio_path, model, speaker_detect, frame
        )

        return jsonify({
            'recognized_text': recognized_text,
            'translated_text': translated_text,
            'speaker_info': speaker_info
        })

    except Exception as e:
        logger.error(f"Realtime processing failed: {str(e)}")
        return jsonify({'error': str(e)}), 500
    finally:
        if os.path.exists(audio_path):
            try:
                os.remove(audio_path)
                logger.debug(f"Removed temp file: {audio_path}")
            except Exception as e:
                logger.warning(f"Error deleting {audio_path}: {e}")

        # Clear any residual session data (e.g., in-memory buffers)
        if 'cuda' in device:
            torch.cuda.empty_cache()
        logger.debug("Cleared PyTorch cache after real-time processing")

@app.route('/translate', methods=['POST'])
def translate_text_endpoint():
    try:
        data = request.get_json()
        text = data.get('text', '')
        model = data.get('model', 'mbart50')

        if not text.strip():
            return jsonify({'error': 'No text provided'}), 400

        # Use the TranslationModel's translate method with a fresh instance
        translated_text = translation_system.translate(text, source_lang='en', target_lang='hi', model_choice=model)

        return jsonify({
            'translated_text': translated_text
        })

    except Exception as e:
        logger.error(f"Text translation failed: {str(e)}")
        return jsonify({'error': str(e)}), 500
    finally:
        # Clear any residual session data (e.g., in-memory buffers)
        if 'cuda' in device:
            torch.cuda.empty_cache()
        logger.debug("Cleared PyTorch cache after text translation")

# --- Run the App ---
if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)