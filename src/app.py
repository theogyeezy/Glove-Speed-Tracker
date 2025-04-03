"""
Main application entry point for the Glove Speed Tracker application.
Initializes and runs the Flask web application.
"""

import os
import logging
from src.ui import app, create_app_directories, create_template_files, create_static_files

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger('app')

if __name__ == "__main__":
    # Create necessary directories and files
    create_app_directories()
    create_template_files()
    create_static_files()
    
    # Get port from environment variable or use default
    port = int(os.environ.get("PORT", 5000))
    
    # Log startup information
    logger.info(f"Starting Glove Speed Tracker application on port {port}")
    logger.info(f"Application directory: {os.path.dirname(os.path.abspath(__file__))}")
    
    # Run the application
    app.run(host='0.0.0.0', port=port, debug=False)
