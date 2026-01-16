"""
Configuration settings for Face Recognition System
"""
import os
from pathlib import Path

# Base directory
BASE_DIR = Path(__file__).parent.absolute()

# Database settings
DATABASE_PATH = BASE_DIR / "data" / "faces.db"

# Model settings
FACE_DETECTION_CONFIDENCE = 0.9
FACE_RECOGNITION_THRESHOLD = 0.5  # Cosine similarity threshold

# Anti-spoofing settings
BLINK_THRESHOLD = 0.25  # Eye Aspect Ratio threshold for blink detection
TEXTURE_THRESHOLD = 0.5  # LBP texture analysis threshold

# Image processing
MAX_IMAGE_SIZE = (1920, 1080)
FACE_SIZE = (112, 112)  # Standard size for InsightFace

# API settings
API_HOST = os.getenv("API_HOST", "0.0.0.0")
API_PORT = int(os.getenv("API_PORT", "8000"))

# Create data directory if not exists
DATA_DIR = BASE_DIR / "data"
DATA_DIR.mkdir(exist_ok=True)
