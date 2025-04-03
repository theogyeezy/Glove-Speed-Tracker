"""
Static file serving configuration for App Engine deployment.
Creates necessary static files and directories for the web application.
"""

import os
import shutil
from pathlib import Path

# Define static directory
STATIC_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'static')

def create_static_directories():
    """Create static directories for the web application."""
    os.makedirs(os.path.join(STATIC_DIR, 'css'), exist_ok=True)
    os.makedirs(os.path.join(STATIC_DIR, 'js'), exist_ok=True)
    os.makedirs(os.path.join(STATIC_DIR, 'img'), exist_ok=True)

def create_css_file():
    """Create CSS file for the web application."""
    css_content = """
    body {
        font-family: 'Roboto', Arial, sans-serif;
        line-height: 1.6;
        margin: 0;
        padding: 0;
        color: #333;
        background-color: #f8f9fa;
    }
    
    .container {
        width: 90%;
        max-width: 1200px;
        margin: 0 auto;
        padding: 20px;
    }
    
    header {
        background-color: #2c3e50;
        color: white;
        padding: 1rem 0;
        box-shadow: 0 2px 5px rgba(0,0,0,0.1);
    }
    
    header .container {
        display: flex;
        justify-content: space-between;
        align-items: center;
    }
    
    .logo {
        font-size: 1.5rem;
        font-weight: bold;
    }
    
    nav ul {
        display: flex;
        list-style: none;
        margin: 0;
        padding: 0;
    }
    
    nav ul li {
        margin-left: 20px;
    }
    
    nav ul li a {
        color: white;
        text-decoration: none;
        transition: color 0.3s;
    }
    
    nav ul li a:hover {
        color: #3498db;
    }
    
    .hero {
        background-color: #3498db;
        color: white;
        padding: 3rem 0;
        text-align: center;
    }
    
    .hero h1 {
        font-size: 2.5rem;
        margin-bottom: 1rem;
    }
    
    .hero p {
        font-size: 1.2rem;
        max-width: 800px;
        margin: 0 auto;
    }
    
    .btn {
        display: inline-block;
        background-color: #2c3e50;
        color: white;
        padding: 10px 20px;
        border-radius: 5px;
        text-decoration: none;
        font-weight: bold;
        transition: background-color 0.3s;
        margin-top: 20px;
    }
    
    .btn:hover {
        background-color: #1a252f;
    }
    
    .features {
        padding: 3rem 0;
    }
    
    .features-grid {
        display: grid;
        grid-template-columns: repeat(auto-fit, minmax(300px, 1fr));
        gap: 30px;
        margin-top: 2rem;
    }
    
    .feature-card {
        background-color: white;
        border-radius: 5px;
        padding: 20px;
        box-shadow: 0 2px 5px rgba(0,0,0,0.1);
        transition: transform 0.3s, box-shadow 0.3s;
    }
    
    .feature-card:hover {
        transform: translateY(-5px);
        box-shadow: 0 5px 15px rgba(0,0,0,0.1);
    }
    
    .feature-card h3 {
        color: #2c3e50;
        margin-top: 0;
    }
    
    .upload-section {
        background-color: white;
        border-radius: 5px;
        padding: 30px;
        box-shadow: 0 2px 5px rgba(0,0,0,0.1);
        margin-bottom: 30px;
    }
    
    .upload-form {
        max-width: 600px;
        margin: 0 auto;
    }
    
    .form-group {
        margin-bottom: 20px;
    }
    
    .form-group label {
        display: block;
        margin-bottom: 5px;
        font-weight: bold;
    }
    
    .form-control {
        width: 100%;
        padding: 10px;
        border: 1px solid #ddd;
        border-radius: 4px;
        font-size: 16px;
    }
    
    .results-section {
        background-color: white;
        border-radius: 5px;
        padding: 30px;
        box-shadow: 0 2px 5px rgba(0,0,0,0.1);
    }
    
    .results-container {
        display: grid;
        grid-template-columns: 1fr 1fr;
        gap: 20px;
    }
    
    .video-container {
        position: relative;
        padding-bottom: 56.25%; /* 16:9 aspect ratio */
        height: 0;
        overflow: hidden;
    }
    
    .video-container video {
        position: absolute;
        top: 0;
        left: 0;
        width: 100%;
        height: 100%;
    }
    
    .stats-container {
        background-color: #f8f9fa;
        padding: 20px;
        border-radius: 5px;
    }
    
    .stat-item {
        margin-bottom: 15px;
    }
    
    .stat-label {
        font-weight: bold;
        color: #2c3e50;
    }
    
    .chart-container {
        margin-top: 30px;
        height: 300px;
    }
    
    footer {
        background-color: #2c3e50;
        color: white;
        padding: 2rem 0;
        margin-top: 3rem;
    }
    
    .footer-content {
        display: flex;
        justify-content: space-between;
        flex-wrap: wrap;
    }
    
    .footer-section {
        flex: 1;
        min-width: 200px;
        margin-bottom: 20px;
    }
    
    .footer-section h3 {
        margin-top: 0;
        border-bottom: 2px solid #3498db;
        padding-bottom: 10px;
        display: inline-block;
    }
    
    .footer-section ul {
        list-style: none;
        padding: 0;
    }
    
    .footer-section ul li {
        margin-bottom: 10px;
    }
    
    .footer-section ul li a {
        color: #ecf0f1;
        text-decoration: none;
        transition: color 0.3s;
    }
    
    .footer-section ul li a:hover {
        color: #3498db;
    }
    
    .copyright {
        text-align: center;
        margin-top: 20px;
        padding-top: 20px;
        border-top: 1px solid rgba(255,255,255,0.1);
    }
    
    /* Responsive adjustments */
    @media (max-width: 768px) {
        header .container {
            flex-direction: column;
            text-align: center;
        }
        
        nav ul {
            margin-top: 15px;
            justify-content: center;
        }
        
        .results-container {
            grid-template-columns: 1fr;
        }
        
        .footer-content {
            flex-direction: column;
        }
    }
    
    /* Loading spinner */
    .spinner {
        border: 4px solid rgba(0, 0, 0, 0.1);
        width: 36px;
        height: 36px;
        border-radius: 50%;
        border-left-color: #3498db;
        animation: spin 1s linear infinite;
        margin: 20px auto;
    }
    
    @keyframes spin {
        0% { transform: rotate(0deg); }
        100% { transform: rotate(360deg); }
    }
    
    .hidden {
        display: none;
    }
    """
    
    with open(os.path.join(STATIC_DIR, 'css', 'style.css'), 'w') as f:
        f.write(css_content)

def create_js_file():
    """Create JavaScript file for the web application."""
    js_content = """
    document.addEventListener('DOMContentLoaded', function() {
        // Form submission handling
        const uploadForm = document.getElementById('upload-form');
        const loadingSpinner = document.getElementById('loading-spinner');
        const resultsSection = document.getElementById('results-section');
        
        if (uploadForm) {
            uploadForm.addEventListener('submit', function(e) {
                // Show loading spinner
                if (loadingSpinner) {
                    loadingSpinner.classList.remove('hidden');
                }
                
                // Hide results section during processing
                if (resultsSection) {
                    resultsSection.classList.add('hidden');
                }
                
                // Form will submit normally
            });
        }
        
        // Initialize charts if they exist and data is available
        const speedChartElement = document.getElementById('speed-chart');
        if (speedChartElement && window.speedData) {
            initializeSpeedChart(speedChartElement, window.speedData);
        }
        
        const trajectoryChartElement = document.getElementById('trajectory-chart');
        if (trajectoryChartElement && window.trajectoryData) {
            initializeTrajectoryChart(trajectoryChartElement, window.trajectoryData);
        }
    });
    
    function initializeSpeedChart(canvas, data) {
        const ctx = canvas.getContext('2d');
        new Chart(ctx, {
            type: 'line',
            data: {
                labels: data.frames,
                datasets: [{
                    label: 'Speed (mph)',
                    data: data.speeds,
                    borderColor: '#3498db',
                    backgroundColor: 'rgba(52, 152, 219, 0.1)',
                    borderWidth: 2,
                    tension: 0.3,
                    fill: true
                }]
            },
            options: {
                responsive: true,
                maintainAspectRatio: false,
                scales: {
                    y: {
                        beginAtZero: true,
                        title: {
                            display: true,
                            text: 'Speed (mph)'
                        }
                    },
                    x: {
                        title: {
                            display: true,
                            text: 'Frame'
                        }
                    }
                }
            }
        });
    }
    
    function initializeTrajectoryChart(canvas, data) {
        const ctx = canvas.getContext('2d');
        new Chart(ctx, {
            type: 'scatter',
            data: {
                datasets: [{
                    label: 'Glove Trajectory',
                    data: data.map((point, index) => ({
                        x: point.x,
                        y: point.y
                    })),
                    backgroundColor: '#2c3e50',
                    borderColor: '#2c3e50',
                    pointRadius: 3,
                    pointHoverRadius: 5,
                    showLine: true,
                    tension: 0.3
                }]
            },
            options: {
                responsive: true,
                maintainAspectRatio: false,
                scales: {
                    y: {
                        reverse: true, // Reverse Y axis to match image coordinates
                        title: {
                            display: true,
                            text: 'Y Position'
                        }
                    },
                    x: {
                        title: {
                            display: true,
                            text: 'X Position'
                        }
                    }
                }
            }
        });
    }
    """
    
    with open(os.path.join(STATIC_DIR, 'js', 'main.js'), 'w') as f:
        f.write(js_content)

def create_placeholder_image():
    """Create a placeholder image for the web application."""
    # This is a simple function to create a placeholder image
    # In a real application, you would use actual images
    try:
        # Try to create a simple colored rectangle as a placeholder
        import numpy as np
        from PIL import Image, ImageDraw, ImageFont
        
        # Create a blank image with a blue background
        img = Image.new('RGB', (800, 400), color=(52, 152, 219))
        draw = ImageDraw.Draw(img)
        
        # Add text
        try:
            font = ImageFont.truetype("arial.ttf", 40)
        except IOError:
            font = ImageFont.load_default()
            
        draw.text((400, 200), "Glove Speed Tracker", fill=(255, 255, 255), font=font, anchor="mm")
        
        # Save the image
        img.save(os.path.join(STATIC_DIR, 'img', 'hero-banner.jpg'))
        
        # Create a logo image
        logo_img = Image.new('RGB', (200, 200), color=(44, 62, 80))
        logo_draw = ImageDraw.Draw(logo_img)
        
        # Draw a simple glove shape
        points = [(100, 50), (150, 100), (130, 150), (100, 170), (70, 150), (50, 100)]
        logo_draw.polygon(points, fill=(52, 152, 219))
        
        # Save the logo
        logo_img.save(os.path.join(STATIC_DIR, 'img', 'logo.png'))
        
    except ImportError:
        # If PIL is not available, create empty files
        with open(os.path.join(STATIC_DIR, 'img', 'hero-banner.jpg'), 'w') as f:
            f.write("Placeholder for hero banner image")
            
        with open(os.path.join(STATIC_DIR, 'img', 'logo.png'), 'w') as f:
            f.write("Placeholder for logo image")

def setup_static_files():
    """Set up all static files for the web application."""
    # Create directories
    create_static_directories()
    
    # Create CSS file
    create_css_file()
    
    # Create JavaScript file
    create_js_file()
    
    # Create placeholder images
    create_placeholder_image()
    
    print("Static files have been set up successfully.")

if __name__ == "__main__":
    setup_static_files()
