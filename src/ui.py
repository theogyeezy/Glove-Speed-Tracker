"""
User interface module for the Glove Speed Tracker application.
Provides a web-based interface for uploading videos, analyzing glove speed, and viewing results.
"""

import os
import sys
import logging
import json
import uuid
import threading
import time
from datetime import datetime
from pathlib import Path
import cv2
import numpy as np
from flask import Flask, request, render_template, jsonify, send_from_directory, Response

# Import project modules
from config import OUTPUT_DIR, DATA_DIR
from video_processor import VideoProcessor
from detector import GloveDetector
from tracker import GloveTracker
from speed_calculator import SpeedCalculator
from data_analyzer import GloveDataAnalyzer
from visualizer import GloveVisualizer

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger('ui')

# Create Flask app
app = Flask(__name__, 
            static_folder=os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'static'),
            template_folder=os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'templates'))

# Global variables for tracking processing jobs
processing_jobs = {}

class ProcessingJob:
    """Class to track the status of a video processing job."""
    
    def __init__(self, job_id, video_path, output_dir):
        self.job_id = job_id
        self.video_path = video_path
        self.output_dir = output_dir
        self.status = "initializing"
        self.progress = 0
        self.results = {}
        self.error = None
        self.start_time = datetime.now()
        
    def update_status(self, status, progress=None):
        self.status = status
        if progress is not None:
            self.progress = progress
        logger.info(f"Job {self.job_id}: {status} ({self.progress}%)")
        
    def set_error(self, error_message):
        self.status = "error"
        self.error = error_message
        logger.error(f"Job {self.job_id}: {error_message}")
        
    def set_results(self, results):
        self.status = "completed"
        self.progress = 100
        self.results = results
        logger.info(f"Job {self.job_id}: Processing completed")
        
    def get_info(self):
        return {
            "job_id": self.job_id,
            "status": self.status,
            "progress": self.progress,
            "video_path": self.video_path,
            "start_time": self.start_time.isoformat(),
            "elapsed_time": (datetime.now() - self.start_time).total_seconds(),
            "error": self.error,
            "results": self.results
        }

def process_video(job):
    """
    Process a video file to track glove movement and calculate speed.
    
    Args:
        job (ProcessingJob): The processing job object
    """
    try:
        # Create output directory if it doesn't exist
        os.makedirs(job.output_dir, exist_ok=True)
        
        # Initialize components
        job.update_status("initializing components", 5)
        video_processor = VideoProcessor()
        detector = GloveDetector()
        tracker = GloveTracker()
        
        # Load video
        job.update_status("loading video", 10)
        video_processor.load_video(job.video_path)
        fps = video_processor.get_fps()
        frame_count = video_processor.get_frame_count()
        frame_width = video_processor.get_frame_width()
        frame_height = video_processor.get_frame_height()
        
        # Initialize tracking data storage
        tracking_data = []
        
        # Process video frames
        job.update_status("processing video frames", 15)
        frame_idx = 0
        bbox = None
        
        while True:
            # Read frame
            success, frame = video_processor.read_frame()
            if not success:
                break
                
            # Update progress
            progress = 15 + (frame_idx / frame_count) * 50
            if frame_idx % 10 == 0:
                job.update_status("processing video frames", progress)
            
            # Detect glove in first frame or if tracking is lost
            if frame_idx == 0 or bbox is None:
                detections = detector.detect(frame)
                if detections:
                    bbox = detections[0]
                    tracker.init(frame, bbox)
            else:
                # Track glove
                success, bbox = tracker.update(frame)
                if not success:
                    # If tracking fails, try detection again
                    detections = detector.detect(frame)
                    if detections:
                        bbox = detections[0]
                        tracker.init(frame, bbox)
            
            # Store tracking data if bbox is valid
            if bbox is not None:
                x, y, w, h = bbox
                center_x = x + w/2
                center_y = y + h/2
                tracking_data.append({
                    'frame': frame_idx,
                    'x': x,
                    'y': y,
                    'width': w,
                    'height': h,
                    'center_x': center_x,
                    'center_y': center_y
                })
            
            frame_idx += 1
        
        # Convert tracking data to DataFrame
        job.update_status("calculating speed", 65)
        import pandas as pd
        tracking_df = pd.DataFrame(tracking_data)
        
        # Save tracking data
        tracking_csv_path = os.path.join(job.output_dir, f"tracking_data_{job.job_id}.csv")
        tracking_df.to_csv(tracking_csv_path, index=False)
        
        # Calculate speed
        calculator = SpeedCalculator(fps=fps)
        calculator.set_tracking_data(tracking_df)
        speed_data = calculator.calculate_speed(smoothing=True)
        calculator.calculate_acceleration(smoothing=True)
        
        # Save speed data
        speed_csv_path = os.path.join(job.output_dir, f"speed_data_{job.job_id}.csv")
        calculator.save_speed_data(speed_csv_path)
        
        # Generate speed report
        job.update_status("generating reports", 75)
        report_path = calculator.generate_speed_report(
            os.path.join(job.output_dir, f"speed_report_{job.job_id}.html")
        )
        
        # Create visualization
        job.update_status("creating visualizations", 85)
        visualizer = GloveVisualizer(fps=fps)
        visualizer.set_frame_size(frame_width, frame_height)
        visualizer.set_tracking_data(tracking_df)
        visualizer.set_speed_data(speed_data)
        
        # Generate visualization video
        vis_video_path = os.path.join(job.output_dir, f"visualization_{job.job_id}.mp4")
        visualizer.create_visualization_video(vis_video_path)
        
        # Generate interactive report
        interactive_report_path = os.path.join(job.output_dir, f"interactive_report_{job.job_id}.html")
        visualizer.create_interactive_report(interactive_report_path)
        
        # Perform data analysis
        job.update_status("analyzing data", 95)
        analyzer = GloveDataAnalyzer()
        analyzer.set_speed_data(speed_data)
        analyzer.extract_features()
        analyzer.cluster_movements()
        analyzer.perform_pca()
        analyzer.train_speed_prediction_model()
        
        # Generate analysis report
        analysis_report_path = os.path.join(job.output_dir, f"analysis_report_{job.job_id}.html")
        analyzer.generate_analysis_report(analysis_report_path)
        
        # Get statistics
        max_speed = calculator.get_max_speed()
        avg_speed = calculator.get_average_speed()
        
        # Set results
        job.update_status("finalizing", 99)
        job.set_results({
            "tracking_data": tracking_csv_path,
            "speed_data": speed_csv_path,
            "speed_report": report_path,
            "visualization_video": vis_video_path,
            "interactive_report": interactive_report_path,
            "analysis_report": analysis_report_path,
            "statistics": {
                "max_speed_mps": max_speed[0],
                "max_speed_mph": max_speed[1],
                "max_speed_frame": int(max_speed[2]),
                "avg_speed_mps": avg_speed[0],
                "avg_speed_mph": avg_speed[1],
                "total_frames": frame_count,
                "tracked_frames": len(tracking_data),
                "duration_seconds": frame_count / fps
            }
        })
        
    except Exception as e:
        logger.exception("Error processing video")
        job.set_error(str(e))

# Flask routes
@app.route('/')
def index():
    """Render the main page."""
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload_video():
    """Handle video upload and start processing."""
    if 'video' not in request.files:
        return jsonify({"error": "No video file provided"}), 400
        
    video_file = request.files['video']
    if video_file.filename == '':
        return jsonify({"error": "No video file selected"}), 400
        
    # Generate unique job ID
    job_id = str(uuid.uuid4())
    
    # Create directories if they don't exist
    os.makedirs(DATA_DIR, exist_ok=True)
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    # Save uploaded video
    video_path = os.path.join(DATA_DIR, f"upload_{job_id}_{video_file.filename}")
    video_file.save(video_path)
    
    # Create job
    job = ProcessingJob(job_id, video_path, OUTPUT_DIR)
    processing_jobs[job_id] = job
    
    # Start processing in a separate thread
    threading.Thread(target=process_video, args=(job,)).start()
    
    return jsonify({
        "job_id": job_id,
        "message": "Video upload successful, processing started",
        "status_url": f"/status/{job_id}"
    })

@app.route('/status/<job_id>')
def get_job_status(job_id):
    """Get the status of a processing job."""
    if job_id not in processing_jobs:
        return jsonify({"error": "Job not found"}), 404
        
    job = processing_jobs[job_id]
    return jsonify(job.get_info())

@app.route('/results/<job_id>')
def get_job_results(job_id):
    """Get the results of a completed job."""
    if job_id not in processing_jobs:
        return jsonify({"error": "Job not found"}), 404
        
    job = processing_jobs[job_id]
    if job.status != "completed":
        return jsonify({"error": "Job not completed yet", "status": job.status, "progress": job.progress}), 400
        
    return jsonify(job.results)

@app.route('/view/<job_id>')
def view_results(job_id):
    """Render the results page for a specific job."""
    if job_id not in processing_jobs:
        return render_template('error.html', message="Job not found")
        
    job = processing_jobs[job_id]
    if job.status != "completed":
        return render_template('processing.html', job=job.get_info())
        
    return render_template('results.html', job=job.get_info())

@app.route('/output/<path:filename>')
def serve_output(filename):
    """Serve files from the output directory."""
    return send_from_directory(OUTPUT_DIR, filename)

@app.route('/data/<path:filename>')
def serve_data(filename):
    """Serve files from the data directory."""
    return send_from_directory(DATA_DIR, filename)

@app.route('/video_feed/<job_id>')
def video_feed(job_id):
    """Stream the visualization video."""
    if job_id not in processing_jobs:
        return jsonify({"error": "Job not found"}), 404
        
    job = processing_jobs[job_id]
    if job.status != "completed" or "visualization_video" not in job.results:
        return jsonify({"error": "Video not available"}), 400
        
    def generate():
        video_path = job.results["visualization_video"]
        cap = cv2.VideoCapture(video_path)
        
        while True:
            success, frame = cap.read()
            if not success:
                break
                
            # Convert frame to JPEG
            ret, buffer = cv2.imencode('.jpg', frame)
            frame_bytes = buffer.tobytes()
            
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')
                   
            # Control frame rate
            time.sleep(1/30)
            
        cap.release()
        
    return Response(generate(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/jobs')
def list_jobs():
    """List all processing jobs."""
    jobs_info = {job_id: job.get_info() for job_id, job in processing_jobs.items()}
    return jsonify(jobs_info)

@app.route('/demo')
def demo():
    """Start a demo processing job with sample video."""
    # Find a sample video in the data directory
    sample_videos = [f for f in os.listdir(DATA_DIR) if f.endswith(('.mp4', '.avi', '.mov'))]
    
    if not sample_videos:
        # Create a synthetic test video if no samples exist
        from video_processor import create_test_video
        sample_video_path = os.path.join(DATA_DIR, "sample_video.mp4")
        create_test_video(sample_video_path)
    else:
        sample_video_path = os.path.join(DATA_DIR, sample_videos[0])
    
    # Generate unique job ID
    job_id = str(uuid.uuid4())
    
    # Create job
    job = ProcessingJob(job_id, sample_video_path, OUTPUT_DIR)
    processing_jobs[job_id] = job
    
    # Start processing in a separate thread
    threading.Thread(target=process_video, args=(job,)).start()
    
    return jsonify({
        "job_id": job_id,
        "message": "Demo processing started",
        "status_url": f"/status/{job_id}"
    })

def create_app_directories():
    """Create necessary directories for the application."""
    os.makedirs(DATA_DIR, exist_ok=True)
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    os.makedirs(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'static'), exist_ok=True)
    os.makedirs(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'templates'), exist_ok=True)

def create_template_files():
    """Create HTML template files for the web interface."""
    templates_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'templates')
    
    # Create index.html
    with open(os.path.join(templates_dir, 'index.html'), 'w') as f:
        f.write("""
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Glove Speed Tracker</title>
    <link href="https://cdn.jsdelivr.net/npm/bootstrap@5.3.0-alpha1/dist/css/bootstrap.min.css" rel="stylesheet">
    <style>
        body {
            padding-top: 2rem;
            padding-bottom: 2rem;
            background-color: #f8f9fa;
        }
        .header {
            text-align: center;
            margin-bottom: 2rem;
        }
        .upload-container {
            background-color: white;
            border-radius: 10px;
            padding: 2rem;
            box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
            margin-bottom: 2rem;
        }
        .progress {
            height: 25px;
            margin-top: 1rem;
        }
        .features {
            margin-top: 3rem;
        }
        .feature-card {
            height: 100%;
            transition: transform 0.3s;
        }
        .feature-card:hover {
            transform: translateY(-5px);
        }
        .feature-icon {
            font-size: 2.5rem;
            margin-bottom: 1rem;
            color: #0d6efd;
        }
        #dropArea {
            border: 2px dashed #ccc;
            border-radius: 8px;
            padding: 3rem;
            text-align: center;
            cursor: pointer;
            margin-bottom: 1rem;
            transition: background-color 0.3s;
        }
        #dropArea.highlight {
            background-color: #f0f8ff;
            border-color: #0d6efd;
        }
        #uploadForm {
            display: none;
        }
        #jobStatus {
            display: none;
            margin-top: 2rem;
        }
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1 class="display-4">Glove Speed Tracker</h1>
            <p class="lead">Upload baseball catcher videos to analyze glove speed and movement patterns</p>
        </div>
        
        <div class="row justify-content-center">
            <div class="col-md-8">
                <div class="upload-container">
                    <h2 class="mb-4">Upload Video</h2>
                    
                    <div id="dropArea">
                        <svg xmlns="http://www.w3.org/2000/svg" width="64" height="64" fill="currentColor" class="bi bi-cloud-upload mb-3" viewBox="0 0 16 16">
                            <path fill-rule="evenodd" d="M4.406 1.342A5.53 5.53 0 0 1 8 0c2.69 0 4.923 2 5.166 4.579C14.758 4.804 16 6.137 16 7.773 16 9.569 14.502 11 12.687 11H10a.5.5 0 0 1 0-1h2.688C13.979 10 15 8.988 15 7.773c0-1.216-1.02-2.228-2.313-2.228h-.5v-.5C12.188 2.825 10.328 1 8 1a4.53 4.53 0 0 0-2.941 1.1c-.757.652-1.153 1.438-1.153 2.055v.448l-.445.049C2.064 4.805 1 5.952 1 7.318 1 8.785 2.23 10 3.781 10H6a.5.5 0 0 1 0 1H3.781C1.708 11 0 9.366 0 7.318c0-1.763 1.266-3.223 2.942-3.593.143-.863.698-1.723 1.464-2.383z"/>
                            <path fill-rule="evenodd" d="M7.646 4.146a.5.5 0 0 1 .708 0l3 3a.5.5 0 0 1-.708.708L8.5 5.707V14.5a.5.5 0 0 1-1 0V5.707L5.354 7.854a.5.5 0 1 1-.708-.708l3-3z"/>
                        </svg>
                        <h4>Drag & Drop Video File Here</h4>
                        <p>or click to select a file</p>
                        <p class="text-muted small">Supported formats: MP4, AVI, MOV</p>
                    </div>
                    
                    <form id="uploadForm" enctype="multipart/form-data">
                        <input type="file" id="videoFile" name="video" accept="video/*" class="form-control">
                        <div class="d-grid gap-2 mt-3">
                            <button type="submit" id="uploadButton" class="btn btn-primary">Upload & Process</button>
                        </div>
                    </form>
                    
                    <div class="text-center mt-4">
                        <p>Don't have a video? <a href="#" id="demoLink">Try a demo</a></p>
                    </div>
                    
                    <div id="jobStatus">
                        <h4>Processing Status</h4>
                        <div class="progress">
                            <div id="progressBar" class="progress-bar progress-bar-striped progress-bar-animated" role="progressbar" style="width: 0%"></div>
                        </div>
                        <p id="statusMessage" class="mt-2">Initializing...</p>
                        <div id="processingTime" class="text-muted small"></div>
                    </div>
                </div>
            </div>
        </div>
        
        <div class="features">
            <h2 class="text-center mb-4">Features</h2>
            <div class="row g-4">
                <div class="col-md-4">
                    <div class="card feature-card">
                        <div class="card-body text-center">
                            <div class="feature-icon">
                                <svg xmlns="http://www.w3.org/2000/svg" width="1em" height="1em" fill="currentColor" class="bi bi-camera-video" viewBox="0 0 16 16">
                                    <path fill-rule="evenodd" d="M0 5a2 2 0 0 1 2-2h7.5a2 2 0 0 1 1.983 1.738l3.11-1.382A1 1 0 0 1 16 4.269v7.462a1 1 0 0 1-1.406.913l-3.111-1.382A2 2 0 0 1 9.5 13H2a2 2 0 0 1-2-2V5zm11.5 5.175 3.5 1.556V4.269l-3.5 1.556v4.35zM2 4a1 1 0 0 0-1 1v6a1 1 0 0 0 1 1h7.5a1 1 0 0 0 1-1V5a1 1 0 0 0-1-1H2z"/>
                                </svg>
                            </div>
                            <h5 class="card-title">Video Processing</h5>
                            <p class="card-text">Process videos in various formats to extract individual frames for analysis.</p>
                        </div>
                    </div>
                </div>
                <div class="col-md-4">
                    <div class="card feature-card">
                        <div class="card-body text-center">
                            <div class="feature-icon">
                                <svg xmlns="http://www.w3.org/2000/svg" width="1em" height="1em" fill="currentColor" class="bi bi-speedometer2" viewBox="0 0 16 16">
                                    <path d="M8 4a.5.5 0 0 1 .5.5V6a.5.5 0 0 1-1 0V4.5A.5.5 0 0 1 8 4zM3.732 5.732a.5.5 0 0 1 .707 0l.915.914a.5.5 0 1 1-.708.708l-.914-.915a.5.5 0 0 1 0-.707zM2 10a.5.5 0 0 1 .5-.5h1.586a.5.5 0 0 1 0 1H2.5A.5.5 0 0 1 2 10zm9.5 0a.5.5 0 0 1 .5-.5h1.5a.5.5 0 0 1 0 1H12a.5.5 0 0 1-.5-.5zm.754-4.246a.389.389 0 0 0-.527-.02L7.547 9.31a.91.91 0 1 0 1.302 1.258l3.434-4.297a.389.389 0 0 0-.029-.518z"/>
                                    <path fill-rule="evenodd" d="M0 10a8 8 0 1 1 15.547 2.661c-.442 1.253-1.845 1.602-2.932 1.25C11.309 13.488 9.475 13 8 13c-1.474 0-3.31.488-4.615.911-1.087.352-2.49.003-2.932-1.25A7.988 7.988 0 0 1 0 10zm8-7a7 7 0 0 0-6.603 9.329c.203.575.923.876 1.68.63C4.397 12.533 6.358 12 8 12s3.604.532 4.923.96c.757.245 1.477-.056 1.68-.631A7 7 0 0 0 8 3z"/>
                                </svg>
                            </div>
                            <h5 class="card-title">Speed Calculation</h5>
                            <p class="card-text">Calculate glove speed in real-time with accurate measurements in mph and m/s.</p>
                        </div>
                    </div>
                </div>
                <div class="col-md-4">
                    <div class="card feature-card">
                        <div class="card-body text-center">
                            <div class="feature-icon">
                                <svg xmlns="http://www.w3.org/2000/svg" width="1em" height="1em" fill="currentColor" class="bi bi-graph-up" viewBox="0 0 16 16">
                                    <path fill-rule="evenodd" d="M0 0h1v15h15v1H0V0Zm14.817 3.113a.5.5 0 0 1 .07.704l-4.5 5.5a.5.5 0 0 1-.74.037L7.06 6.767l-3.656 5.027a.5.5 0 0 1-.808-.588l4-5.5a.5.5 0 0 1 .758-.06l2.609 2.61 4.15-5.073a.5.5 0 0 1 .704-.07Z"/>
                                </svg>
                            </div>
                            <h5 class="card-title">Data Analysis</h5>
                            <p class="card-text">Advanced analytics with movement pattern recognition and performance insights.</p>
                        </div>
                    </div>
                </div>
            </div>
        </div>
    </div>
    
    <script src="https://cdn.jsdelivr.net/npm/bootstrap@5.3.0-alpha1/dist/js/bootstrap.bundle.min.js"></script>
    <script>
        document.addEventListener('DOMContentLoaded', function() {
            const dropArea = document.getElementById('dropArea');
            const fileInput = document.getElementById('videoFile');
            const uploadForm = document.getElementById('uploadForm');
            const uploadButton = document.getElementById('uploadButton');
            const jobStatus = document.getElementById('jobStatus');
            const progressBar = document.getElementById('progressBar');
            const statusMessage = document.getElementById('statusMessage');
            const processingTime = document.getElementById('processingTime');
            const demoLink = document.getElementById('demoLink');
            
            let currentJobId = null;
            let statusCheckInterval = null;
            
            // Handle drag and drop events
            ['dragenter', 'dragover', 'dragleave', 'drop'].forEach(eventName => {
                dropArea.addEventListener(eventName, preventDefaults, false);
            });
            
            function preventDefaults(e) {
                e.preventDefault();
                e.stopPropagation();
            }
            
            ['dragenter', 'dragover'].forEach(eventName => {
                dropArea.addEventListener(eventName, highlight, false);
            });
            
            ['dragleave', 'drop'].forEach(eventName => {
                dropArea.addEventListener(eventName, unhighlight, false);
            });
            
            function highlight() {
                dropArea.classList.add('highlight');
            }
            
            function unhighlight() {
                dropArea.classList.remove('highlight');
            }
            
            // Handle file drop
            dropArea.addEventListener('drop', handleDrop, false);
            
            function handleDrop(e) {
                const dt = e.dataTransfer;
                const files = dt.files;
                
                if (files.length > 0) {
                    fileInput.files = files;
                    uploadForm.style.display = 'block';
                }
            }
            
            // Handle click on drop area
            dropArea.addEventListener('click', function() {
                fileInput.click();
            });
            
            // Show form when file is selected
            fileInput.addEventListener('change', function() {
                if (fileInput.files.length > 0) {
                    uploadForm.style.display = 'block';
                }
            });
            
            // Handle form submission
            uploadForm.addEventListener('submit', function(e) {
                e.preventDefault();
                
                if (fileInput.files.length === 0) {
                    alert('Please select a video file');
                    return;
                }
                
                const formData = new FormData();
                formData.append('video', fileInput.files[0]);
                
                // Show status section
                jobStatus.style.display = 'block';
                uploadButton.disabled = true;
                progressBar.style.width = '0%';
                statusMessage.textContent = 'Uploading video...';
                
                // Upload video
                fetch('/upload', {
                    method: 'POST',
                    body: formData
                })
                .then(response => response.json())
                .then(data => {
                    if (data.error) {
                        throw new Error(data.error);
                    }
                    
                    currentJobId = data.job_id;
                    statusMessage.textContent = data.message;
                    
                    // Start checking status
                    startStatusCheck();
                })
                .catch(error => {
                    statusMessage.textContent = 'Error: ' + error.message;
                    uploadButton.disabled = false;
                });
            });
            
            // Handle demo link
            demoLink.addEventListener('click', function(e) {
                e.preventDefault();
                
                // Show status section
                jobStatus.style.display = 'block';
                uploadButton.disabled = true;
                progressBar.style.width = '0%';
                statusMessage.textContent = 'Starting demo...';
                
                // Start demo
                fetch('/demo')
                .then(response => response.json())
                .then(data => {
                    if (data.error) {
                        throw new Error(data.error);
                    }
                    
                    currentJobId = data.job_id;
                    statusMessage.textContent = data.message;
                    
                    // Start checking status
                    startStatusCheck();
                })
                .catch(error => {
                    statusMessage.textContent = 'Error: ' + error.message;
                    uploadButton.disabled = false;
                });
            });
            
            function startStatusCheck() {
                if (statusCheckInterval) {
                    clearInterval(statusCheckInterval);
                }
                
                statusCheckInterval = setInterval(checkJobStatus, 1000);
            }
            
            function checkJobStatus() {
                if (!currentJobId) return;
                
                fetch(`/status/${currentJobId}`)
                .then(response => response.json())
                .then(data => {
                    if (data.error) {
                        throw new Error(data.error);
                    }
                    
                    // Update progress
                    progressBar.style.width = `${data.progress}%`;
                    statusMessage.textContent = data.status;
                    
                    // Update processing time
                    const elapsedSeconds = Math.round(data.elapsed_time);
                    processingTime.textContent = `Processing time: ${elapsedSeconds} seconds`;
                    
                    // Check if completed
                    if (data.status === 'completed') {
                        clearInterval(statusCheckInterval);
                        statusMessage.textContent = 'Processing completed!';
                        
                        // Redirect to results page
                        setTimeout(() => {
                            window.location.href = `/view/${currentJobId}`;
                        }, 1000);
                    }
                    
                    // Check if error
                    if (data.status === 'error') {
                        clearInterval(statusCheckInterval);
                        statusMessage.textContent = `Error: ${data.error}`;
                        uploadButton.disabled = false;
                    }
                })
                .catch(error => {
                    statusMessage.textContent = 'Error checking status: ' + error.message;
                    clearInterval(statusCheckInterval);
                    uploadButton.disabled = false;
                });
            }
        });
    </script>
</body>
</html>
        """)
    
    # Create results.html
    with open(os.path.join(templates_dir, 'results.html'), 'w') as f:
        f.write("""
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Analysis Results - Glove Speed Tracker</title>
    <link href="https://cdn.jsdelivr.net/npm/bootstrap@5.3.0-alpha1/dist/css/bootstrap.min.css" rel="stylesheet">
    <style>
        body {
            padding-top: 2rem;
            padding-bottom: 2rem;
            background-color: #f8f9fa;
        }
        .header {
            text-align: center;
            margin-bottom: 2rem;
        }
        .results-container {
            background-color: white;
            border-radius: 10px;
            padding: 2rem;
            box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
            margin-bottom: 2rem;
        }
        .stats-card {
            text-align: center;
            padding: 1.5rem;
            border-radius: 8px;
            margin-bottom: 1rem;
            background-color: #f8f9fa;
            transition: transform 0.3s;
        }
        .stats-card:hover {
            transform: translateY(-5px);
        }
        .stats-value {
            font-size: 2rem;
            font-weight: bold;
            margin: 0.5rem 0;
        }
        .tab-content {
            padding: 1.5rem;
            background-color: white;
            border: 1px solid #dee2e6;
            border-top: none;
            border-radius: 0 0 0.25rem 0.25rem;
        }
        .video-container {
            position: relative;
            padding-bottom: 56.25%; /* 16:9 aspect ratio */
            height: 0;
            overflow: hidden;
            margin-bottom: 1rem;
        }
        .video-container img, .video-container video {
            position: absolute;
            top: 0;
            left: 0;
            width: 100%;
            height: 100%;
            object-fit: contain;
        }
        .report-iframe {
            width: 100%;
            height: 600px;
            border: none;
        }
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1 class="display-4">Analysis Results</h1>
            <p class="lead">Glove speed tracking and movement analysis</p>
            <a href="/" class="btn btn-outline-primary">Back to Home</a>
        </div>
        
        <div class="results-container">
            <h2 class="mb-4">Key Statistics</h2>
            
            <div class="row">
                <div class="col-md-4">
                    <div class="stats-card">
                        <h5>Maximum Speed</h5>
                        <div class="stats-value" id="maxSpeed">--</div>
                        <p class="text-muted" id="maxSpeedFrame">Frame: --</p>
                    </div>
                </div>
                <div class="col-md-4">
                    <div class="stats-card">
                        <h5>Average Speed</h5>
                        <div class="stats-value" id="avgSpeed">--</div>
                        <p class="text-muted">Overall average</p>
                    </div>
                </div>
                <div class="col-md-4">
                    <div class="stats-card">
                        <h5>Tracking Accuracy</h5>
                        <div class="stats-value" id="trackingAccuracy">--</div>
                        <p class="text-muted" id="framesTracked">Frames: -- / --</p>
                    </div>
                </div>
            </div>
        </div>
        
        <div class="results-container">
            <ul class="nav nav-tabs" id="resultsTabs" role="tablist">
                <li class="nav-item" role="presentation">
                    <button class="nav-link active" id="visualization-tab" data-bs-toggle="tab" data-bs-target="#visualization" type="button" role="tab" aria-controls="visualization" aria-selected="true">Visualization</button>
                </li>
                <li class="nav-item" role="presentation">
                    <button class="nav-link" id="interactive-tab" data-bs-toggle="tab" data-bs-target="#interactive" type="button" role="tab" aria-controls="interactive" aria-selected="false">Interactive Report</button>
                </li>
                <li class="nav-item" role="presentation">
                    <button class="nav-link" id="analysis-tab" data-bs-toggle="tab" data-bs-target="#analysis" type="button" role="tab" aria-controls="analysis" aria-selected="false">Data Analysis</button>
                </li>
                <li class="nav-item" role="presentation">
                    <button class="nav-link" id="download-tab" data-bs-toggle="tab" data-bs-target="#download" type="button" role="tab" aria-controls="download" aria-selected="false">Download Data</button>
                </li>
            </ul>
            
            <div class="tab-content" id="resultsTabContent">
                <div class="tab-pane fade show active" id="visualization" role="tabpanel" aria-labelledby="visualization-tab">
                    <h3 class="mb-3">Glove Tracking Visualization</h3>
                    <div class="video-container">
                        <img id="loadingVideo" src="https://via.placeholder.com/640x360.png?text=Loading+Video..." alt="Loading...">
                        <video id="visualizationVideo" controls style="display: none;">
                            <source id="videoSource" src="" type="video/mp4">
                            Your browser does not support the video tag.
                        </video>
                    </div>
                    <p class="text-muted">The visualization shows the tracked glove position, trajectory, and real-time speed measurements.</p>
                </div>
                
                <div class="tab-pane fade" id="interactive" role="tabpanel" aria-labelledby="interactive-tab">
                    <h3 class="mb-3">Interactive Speed Report</h3>
                    <iframe id="interactiveReport" class="report-iframe" src="about:blank"></iframe>
                </div>
                
                <div class="tab-pane fade" id="analysis" role="tabpanel" aria-labelledby="analysis-tab">
                    <h3 class="mb-3">Movement Pattern Analysis</h3>
                    <iframe id="analysisReport" class="report-iframe" src="about:blank"></iframe>
                </div>
                
                <div class="tab-pane fade" id="download" role="tabpanel" aria-labelledby="download-tab">
                    <h3 class="mb-3">Download Results</h3>
                    <div class="list-group">
                        <a href="#" id="downloadVisualization" class="list-group-item list-group-item-action" target="_blank">
                            <div class="d-flex w-100 justify-content-between">
                                <h5 class="mb-1">Visualization Video</h5>
                                <small>MP4</small>
                            </div>
                            <p class="mb-1">Download the visualization video with tracking overlay</p>
                        </a>
                        <a href="#" id="downloadInteractiveReport" class="list-group-item list-group-item-action" target="_blank">
                            <div class="d-flex w-100 justify-content-between">
                                <h5 class="mb-1">Interactive Report</h5>
                                <small>HTML</small>
                            </div>
                            <p class="mb-1">Download the interactive HTML report with charts and visualizations</p>
                        </a>
                        <a href="#" id="downloadAnalysisReport" class="list-group-item list-group-item-action" target="_blank">
                            <div class="d-flex w-100 justify-content-between">
                                <h5 class="mb-1">Analysis Report</h5>
                                <small>HTML</small>
                            </div>
                            <p class="mb-1">Download the data analysis report with movement patterns</p>
                        </a>
                        <a href="#" id="downloadTrackingData" class="list-group-item list-group-item-action" target="_blank">
                            <div class="d-flex w-100 justify-content-between">
                                <h5 class="mb-1">Tracking Data</h5>
                                <small>CSV</small>
                            </div>
                            <p class="mb-1">Download the raw tracking data for further analysis</p>
                        </a>
                        <a href="#" id="downloadSpeedData" class="list-group-item list-group-item-action" target="_blank">
                            <div class="d-flex w-100 justify-content-between">
                                <h5 class="mb-1">Speed Data</h5>
                                <small>CSV</small>
                            </div>
                            <p class="mb-1">Download the calculated speed data for further analysis</p>
                        </a>
                    </div>
                </div>
            </div>
        </div>
    </div>
    
    <script src="https://cdn.jsdelivr.net/npm/bootstrap@5.3.0-alpha1/dist/js/bootstrap.bundle.min.js"></script>
    <script>
        document.addEventListener('DOMContentLoaded', function() {
            const jobId = '{{ job.job_id }}';
            
            // Load job results
            fetch(`/results/${jobId}`)
            .then(response => response.json())
            .then(data => {
                // Update statistics
                if (data.statistics) {
                    document.getElementById('maxSpeed').textContent = `${data.statistics.max_speed_mph.toFixed(2)} mph`;
                    document.getElementById('maxSpeedFrame').textContent = `Frame: ${data.statistics.max_speed_frame}`;
                    document.getElementById('avgSpeed').textContent = `${data.statistics.avg_speed_mph.toFixed(2)} mph`;
                    
                    const accuracy = (data.statistics.tracked_frames / data.statistics.total_frames * 100).toFixed(1);
                    document.getElementById('trackingAccuracy').textContent = `${accuracy}%`;
                    document.getElementById('framesTracked').textContent = `Frames: ${data.statistics.tracked_frames} / ${data.statistics.total_frames}`;
                }
                
                // Set video source
                if (data.visualization_video) {
                    const videoPath = data.visualization_video.replace(/^.*[\\\/]/, '');
                    document.getElementById('videoSource').src = `/output/${videoPath}`;
                    const video = document.getElementById('visualizationVideo');
                    video.load();
                    video.style.display = 'block';
                    document.getElementById('loadingVideo').style.display = 'none';
                }
                
                // Set report iframes
                if (data.interactive_report) {
                    const reportPath = data.interactive_report.replace(/^.*[\\\/]/, '');
                    document.getElementById('interactiveReport').src = `/output/${reportPath}`;
                }
                
                if (data.analysis_report) {
                    const analysisPath = data.analysis_report.replace(/^.*[\\\/]/, '');
                    document.getElementById('analysisReport').src = `/output/${analysisPath}`;
                }
                
                // Set download links
                if (data.visualization_video) {
                    const videoPath = data.visualization_video.replace(/^.*[\\\/]/, '');
                    document.getElementById('downloadVisualization').href = `/output/${videoPath}`;
                }
                
                if (data.interactive_report) {
                    const reportPath = data.interactive_report.replace(/^.*[\\\/]/, '');
                    document.getElementById('downloadInteractiveReport').href = `/output/${reportPath}`;
                }
                
                if (data.analysis_report) {
                    const analysisPath = data.analysis_report.replace(/^.*[\\\/]/, '');
                    document.getElementById('downloadAnalysisReport').href = `/output/${analysisPath}`;
                }
                
                if (data.tracking_data) {
                    const trackingPath = data.tracking_data.replace(/^.*[\\\/]/, '');
                    document.getElementById('downloadTrackingData').href = `/output/${trackingPath}`;
                }
                
                if (data.speed_data) {
                    const speedPath = data.speed_data.replace(/^.*[\\\/]/, '');
                    document.getElementById('downloadSpeedData').href = `/output/${speedPath}`;
                }
            })
            .catch(error => {
                console.error('Error loading results:', error);
            });
        });
    </script>
</body>
</html>
        """)
    
    # Create processing.html
    with open(os.path.join(templates_dir, 'processing.html'), 'w') as f:
        f.write("""
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Processing - Glove Speed Tracker</title>
    <link href="https://cdn.jsdelivr.net/npm/bootstrap@5.3.0-alpha1/dist/css/bootstrap.min.css" rel="stylesheet">
    <style>
        body {
            padding-top: 2rem;
            padding-bottom: 2rem;
            background-color: #f8f9fa;
        }
        .header {
            text-align: center;
            margin-bottom: 2rem;
        }
        .processing-container {
            background-color: white;
            border-radius: 10px;
            padding: 2rem;
            box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
            margin-bottom: 2rem;
            text-align: center;
        }
        .progress {
            height: 25px;
            margin: 2rem 0;
        }
        .spinner-border {
            width: 5rem;
            height: 5rem;
            margin-bottom: 2rem;
        }
    </style>
    <meta http-equiv="refresh" content="5">
</head>
<body>
    <div class="container">
        <div class="header">
            <h1 class="display-4">Processing Video</h1>
            <p class="lead">Please wait while we analyze your video</p>
        </div>
        
        <div class="processing-container">
            <div class="spinner-border text-primary" role="status">
                <span class="visually-hidden">Loading...</span>
            </div>
            
            <h2 id="statusMessage">{{ job.status }}</h2>
            
            <div class="progress">
                <div class="progress-bar progress-bar-striped progress-bar-animated" role="progressbar" style="width: {{ job.progress }}%" aria-valuenow="{{ job.progress }}" aria-valuemin="0" aria-valuemax="100"></div>
            </div>
            
            <p id="processingTime" class="text-muted">Processing time: {{ job.elapsed_time }} seconds</p>
            
            <p>This page will automatically refresh every 5 seconds. You will be redirected to the results page when processing is complete.</p>
            
            <a href="/" class="btn btn-outline-secondary mt-3">Back to Home</a>
        </div>
    </div>
    
    <script src="https://cdn.jsdelivr.net/npm/bootstrap@5.3.0-alpha1/dist/js/bootstrap.bundle.min.js"></script>
    <script>
        document.addEventListener('DOMContentLoaded', function() {
            const jobId = '{{ job.job_id }}';
            
            // Check if job is completed
            fetch(`/status/${jobId}`)
            .then(response => response.json())
            .then(data => {
                if (data.status === 'completed') {
                    window.location.href = `/view/${jobId}`;
                }
            })
            .catch(error => {
                console.error('Error checking status:', error);
            });
        });
    </script>
</body>
</html>
        """)
    
    # Create error.html
    with open(os.path.join(templates_dir, 'error.html'), 'w') as f:
        f.write("""
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Error - Glove Speed Tracker</title>
    <link href="https://cdn.jsdelivr.net/npm/bootstrap@5.3.0-alpha1/dist/css/bootstrap.min.css" rel="stylesheet">
    <style>
        body {
            padding-top: 2rem;
            padding-bottom: 2rem;
            background-color: #f8f9fa;
        }
        .header {
            text-align: center;
            margin-bottom: 2rem;
        }
        .error-container {
            background-color: white;
            border-radius: 10px;
            padding: 2rem;
            box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
            margin-bottom: 2rem;
            text-align: center;
        }
        .error-icon {
            font-size: 5rem;
            color: #dc3545;
            margin-bottom: 1rem;
        }
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1 class="display-4">Error</h1>
        </div>
        
        <div class="error-container">
            <div class="error-icon">
                <svg xmlns="http://www.w3.org/2000/svg" width="1em" height="1em" fill="currentColor" class="bi bi-exclamation-triangle-fill" viewBox="0 0 16 16">
                    <path d="M8.982 1.566a1.13 1.13 0 0 0-1.96 0L.165 13.233c-.457.778.091 1.767.98 1.767h13.713c.889 0 1.438-.99.98-1.767L8.982 1.566zM8 5c.535 0 .954.462.9.995l-.35 3.507a.552.552 0 0 1-1.1 0L7.1 5.995A.905.905 0 0 1 8 5zm.002 6a1 1 0 1 1 0 2 1 1 0 0 1 0-2z"/>
                </svg>
            </div>
            
            <h2 class="mb-4">{{ message }}</h2>
            
            <p>Something went wrong. Please try again or contact support if the problem persists.</p>
            
            <a href="/" class="btn btn-primary mt-3">Back to Home</a>
        </div>
    </div>
    
    <script src="https://cdn.jsdelivr.net/npm/bootstrap@5.3.0-alpha1/dist/js/bootstrap.bundle.min.js"></script>
</body>
</html>
        """)

def create_static_files():
    """Create static files for the web interface."""
    static_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'static')
    
    # Create CSS file
    with open(os.path.join(static_dir, 'style.css'), 'w') as f:
        f.write("""
/* Main styles for Glove Speed Tracker */
body {
    font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
    line-height: 1.6;
    color: #333;
    background-color: #f8f9fa;
    margin: 0;
    padding: 0;
}

.container {
    max-width: 1200px;
    margin: 0 auto;
    padding: 20px;
}

.header {
    text-align: center;
    margin-bottom: 30px;
}

.header h1 {
    font-size: 2.5rem;
    margin-bottom: 10px;
}

.header p {
    font-size: 1.2rem;
    color: #6c757d;
}

.card {
    background-color: #fff;
    border-radius: 8px;
    box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
    margin-bottom: 20px;
    overflow: hidden;
    transition: transform 0.3s ease;
}

.card:hover {
    transform: translateY(-5px);
}

.card-header {
    background-color: #f8f9fa;
    padding: 15px 20px;
    border-bottom: 1px solid #e9ecef;
}

.card-body {
    padding: 20px;
}

.btn {
    display: inline-block;
    font-weight: 400;
    text-align: center;
    white-space: nowrap;
    vertical-align: middle;
    user-select: none;
    border: 1px solid transparent;
    padding: 0.375rem 0.75rem;
    font-size: 1rem;
    line-height: 1.5;
    border-radius: 0.25rem;
    transition: color 0.15s ease-in-out, background-color 0.15s ease-in-out, border-color 0.15s ease-in-out, box-shadow 0.15s ease-in-out;
    cursor: pointer;
}

.btn-primary {
    color: #fff;
    background-color: #007bff;
    border-color: #007bff;
}

.btn-primary:hover {
    color: #fff;
    background-color: #0069d9;
    border-color: #0062cc;
}

.btn-secondary {
    color: #fff;
    background-color: #6c757d;
    border-color: #6c757d;
}

.btn-secondary:hover {
    color: #fff;
    background-color: #5a6268;
    border-color: #545b62;
}

.form-group {
    margin-bottom: 1rem;
}

.form-control {
    display: block;
    width: 100%;
    padding: 0.375rem 0.75rem;
    font-size: 1rem;
    line-height: 1.5;
    color: #495057;
    background-color: #fff;
    background-clip: padding-box;
    border: 1px solid #ced4da;
    border-radius: 0.25rem;
    transition: border-color 0.15s ease-in-out, box-shadow 0.15s ease-in-out;
}

.progress {
    display: flex;
    height: 1rem;
    overflow: hidden;
    font-size: 0.75rem;
    background-color: #e9ecef;
    border-radius: 0.25rem;
}

.progress-bar {
    display: flex;
    flex-direction: column;
    justify-content: center;
    color: #fff;
    text-align: center;
    white-space: nowrap;
    background-color: #007bff;
    transition: width 0.6s ease;
}

.alert {
    position: relative;
    padding: 0.75rem 1.25rem;
    margin-bottom: 1rem;
    border: 1px solid transparent;
    border-radius: 0.25rem;
}

.alert-success {
    color: #155724;
    background-color: #d4edda;
    border-color: #c3e6cb;
}

.alert-danger {
    color: #721c24;
    background-color: #f8d7da;
    border-color: #f5c6cb;
}

.alert-info {
    color: #0c5460;
    background-color: #d1ecf1;
    border-color: #bee5eb;
}

/* Responsive adjustments */
@media (max-width: 768px) {
    .container {
        padding: 10px;
    }
    
    .header h1 {
        font-size: 2rem;
    }
    
    .header p {
        font-size: 1rem;
    }
}
        """)
    
    # Create JavaScript file
    with open(os.path.join(static_dir, 'app.js'), 'w') as f:
        f.write("""
// Main JavaScript for Glove Speed Tracker

// Wait for DOM to be fully loaded
document.addEventListener('DOMContentLoaded', function() {
    console.log('Glove Speed Tracker application initialized');
    
    // Initialize file upload functionality if present
    initFileUpload();
    
    // Initialize tabs if present
    initTabs();
    
    // Initialize charts if present
    initCharts();
});

// Initialize file upload functionality
function initFileUpload() {
    const fileInput = document.getElementById('videoFile');
    const uploadForm = document.getElementById('uploadForm');
    
    if (!fileInput || !uploadForm) return;
    
    uploadForm.addEventListener('submit', function(e) {
        e.preventDefault();
        
        if (fileInput.files.length === 0) {
            showAlert('Please select a video file', 'danger');
            return;
        }
        
        const formData = new FormData();
        formData.append('video', fileInput.files[0]);
        
        // Show loading state
        const submitButton = uploadForm.querySelector('button[type="submit"]');
        const originalText = submitButton.textContent;
        submitButton.disabled = true;
        submitButton.innerHTML = '<span class="spinner-border spinner-border-sm" role="status" aria-hidden="true"></span> Uploading...';
        
        // Upload file
        fetch('/upload', {
            method: 'POST',
            body: formData
        })
        .then(response => {
            if (!response.ok) {
                throw new Error('Network response was not ok');
            }
            return response.json();
        })
        .then(data => {
            if (data.error) {
                throw new Error(data.error);
            }
            
            // Redirect to status page
            window.location.href = `/view/${data.job_id}`;
        })
        .catch(error => {
            showAlert('Error: ' + error.message, 'danger');
            submitButton.disabled = false;
            submitButton.textContent = originalText;
        });
    });
    
    // File input change event
    fileInput.addEventListener('change', function() {
        const fileLabel = document.querySelector('.custom-file-label');
        if (fileLabel) {
            fileLabel.textContent = fileInput.files.length > 0 ? 
                fileInput.files[0].name : 'Choose file';
        }
    });
}

// Initialize tabs functionality
function initTabs() {
    const tabLinks = document.querySelectorAll('[data-bs-toggle="tab"]');
    
    if (tabLinks.length === 0) return;
    
    tabLinks.forEach(tabLink => {
        tabLink.addEventListener('click', function(e) {
            e.preventDefault();
            
            // Remove active class from all tabs
            tabLinks.forEach(link => {
                link.classList.remove('active');
                const tabContent = document.querySelector(link.getAttribute('data-bs-target'));
                if (tabContent) {
                    tabContent.classList.remove('active', 'show');
                }
            });
            
            // Add active class to clicked tab
            this.classList.add('active');
            const targetTab = document.querySelector(this.getAttribute('data-bs-target'));
            if (targetTab) {
                targetTab.classList.add('active', 'show');
            }
        });
    });
}

// Initialize charts if Chart.js is available
function initCharts() {
    if (typeof Chart === 'undefined') return;
    
    // Example: Initialize speed chart if canvas exists
    const speedChartCanvas = document.getElementById('speedChart');
    if (speedChartCanvas) {
        const ctx = speedChartCanvas.getContext('2d');
        new Chart(ctx, {
            type: 'line',
            data: {
                labels: [], // Will be populated with time data
                datasets: [{
                    label: 'Speed (mph)',
                    data: [], // Will be populated with speed data
                    borderColor: 'rgb(75, 192, 192)',
                    tension: 0.1
                }]
            },
            options: {
                responsive: true,
                scales: {
                    y: {
                        beginAtZero: true
                    }
                }
            }
        });
    }
}

// Helper function to show alerts
function showAlert(message, type = 'info') {
    const alertsContainer = document.getElementById('alerts');
    if (!alertsContainer) return;
    
    const alert = document.createElement('div');
    alert.className = `alert alert-${type} alert-dismissible fade show`;
    alert.role = 'alert';
    
    alert.innerHTML = `
        ${message}
        <button type="button" class="btn-close" data-bs-dismiss="alert" aria-label="Close"></button>
    `;
    
    alertsContainer.appendChild(alert);
    
    // Auto-dismiss after 5 seconds
    setTimeout(() => {
        alert.classList.remove('show');
        setTimeout(() => {
            alertsContainer.removeChild(alert);
        }, 150);
    }, 5000);
}
        """)

def run_app(host='0.0.0.0', port=5000, debug=True):
    """Run the Flask application."""
    # Create necessary directories and files
    create_app_directories()
    create_template_files()
    create_static_files()
    
    # Run the app
    app.run(host=host, port=port, debug=debug)

if __name__ == "__main__":
    # Create necessary directories and files
    create_app_directories()
    create_template_files()
    create_static_files()
    
    # Test the UI module
    print("\nGlove Speed Tracker UI module initialized")
    print("To start the web application, run:")
    print("  python -m src.ui")
    print("\nThe application will be available at http://localhost:5000")
