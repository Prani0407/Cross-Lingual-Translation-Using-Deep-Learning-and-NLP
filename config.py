import os
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent

class Config:
    SECRET_KEY = os.environ.get('SECRET_KEY', 'your-secret-key-here')
    BASE_DIR = BASE_DIR
    UPLOAD_FOLDER = str(BASE_DIR / 'uploads')
    OUTPUT_FOLDER = str(BASE_DIR / 'outputs')
    STATIC_FOLDER = str(BASE_DIR / 'static')
    MAX_CONTENT_LENGTH = 100 * 1024 * 1024
    TEMPLATE_FOLDER = str(BASE_DIR / 'templates')

    @staticmethod
    def init_app(app):
        for folder in [Config.UPLOAD_FOLDER, Config.OUTPUT_FOLDER]:
            os.makedirs(folder, exist_ok=True)
            print(f"Ensured directory exists: {folder}")