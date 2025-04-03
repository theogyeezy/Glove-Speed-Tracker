"""
Configuration settings for the Glove Speed Tracker application.
"""

import os

# Project paths
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, 'data')
MODELS_DIR = os.path.join(BASE_DIR, 'models')
OUTPUT_DIR = os.path.join(BASE_DIR, 'output')

# Create directories if they don't exist
for directory in [DATA_DIR, MODELS_DIR, OUTPUT_DIR]:
    os.makedirs(directory, exist_ok=True)

# Video processing settings
SUPPORTED_FORMATS = ['.mp4', '.avi', '.mov', '.mkv']
DEFAULT_FRAME_RATE = 30  # Default frame rate if not available in video metadata
FRAME_RESIZE = (640, 480)  # Width, Height for resizing frames

# Detection and tracking settings
CONFIDENCE_THRESHOLD = 0.5  # Minimum confidence for object detection
IOU_THRESHOLD = 0.4  # Intersection over Union threshold for non-max suppression
MAX_TRACKING_FAILURES = 5  # Maximum number of frames to continue tracking after detection loss

# Speed calculation settings
PIXELS_PER_METER = 100  # Default pixels per meter (will be calibrated)
SMOOTHING_WINDOW = 5  # Number of frames for moving average smoothing

# UI settings
MAX_UPLOAD_SIZE_MB = 500  # Maximum upload size in MB
ALLOWED_EXTENSIONS = set(['mp4', 'avi', 'mov', 'mkv'])

# Deployment settings
PORT = int(os.environ.get('PORT', 5000))
DEBUG = os.environ.get('DEBUG', 'False').lower() == 'true'
