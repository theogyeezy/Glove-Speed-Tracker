"""
App Engine specific configuration for the Glove Speed Tracker application.
This module adapts the Flask application for deployment on Google Cloud App Engine.
"""

import os
import logging
from src.ui import app, create_app_directories, create_template_files, create_static_files

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger('app_engine')

# Create necessary directories and files
create_app_directories()
create_template_files()
create_static_files()

# Log startup information
logger.info(f"Starting Glove Speed Tracker application on App Engine")
logger.info(f"Application directory: {os.path.dirname(os.path.abspath(__file__))}")

# App Engine uses the 'app' variable directly
# No need to call app.run() as App Engine manages this
