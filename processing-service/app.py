import os
from flask import Flask, request, jsonify
import tempfile
import json
import requests
from detector import GloveDetector
from tracker import GloveTracker
from speed_calculator import SpeedCalculator
from utils.supabase_client import SupabaseClient
from utils.video_utils import download_video, extract_frames

app = Flask(__name__)

@app.route('/health', methods=['GET'])
def health_check():
    return jsonify({
        'status': 'healthy',
        'version': '1.0.0'
    })

@app.route('/process', methods=['POST'])
def process_video():
    try:
        # Get request data
        data = request.json
        video_id = data.get('videoId')
        file_path = data.get('filePath')
        supabase_url = data.get('supabaseUrl')
        supabase_key = data.get('supabaseKey')
        
        if not all([video_id, file_path, supabase_url, supabase_key]):
            return jsonify({
                'success': False,
                'error': 'Missing required parameters'
            }), 400
        
        # Initialize Supabase client
        supabase = SupabaseClient(supabase_url, supabase_key)
        
        # Update video status to processing
        supabase.update_video_status(video_id, 'processing')
        
        # Create temporary directory for processing
        with tempfile.TemporaryDirectory() as temp_dir:
            # Download video from Supabase storage
            video_path = os.path.join(temp_dir, os.path.basename(file_path))
            download_video(supabase, file_path, video_path)
            
            # Extract frames from video
            frames_dir = os.path.join(temp_dir, 'frames')
            os.makedirs(frames_dir, exist_ok=True)
            frame_paths, fps, frame_width, frame_height = extract_frames(video_path, frames_dir)
            
            # Initialize detector and tracker
            detector = GloveDetector()
            tracker = GloveTracker()
            
            # Process frames
            detections = []
            for frame_path in frame_paths:
                # Detect glove in frame
                detection = detector.detect_glove(frame_path)
                if detection:
                    detections.append(detection)
                    # Track glove
                    tracker.update(detection)
            
            # Calculate speed
            calculator = SpeedCalculator(fps=fps, frame_width=frame_width, frame_height=frame_height)
            speeds = calculator.calculate_speeds(tracker.get_tracks())
            
            # Analyze movement patterns
            movement_patterns = calculator.analyze_movement_patterns()
            
            # Prepare results
            results = {
                'max_speed': max(speeds) if speeds else 0,
                'avg_speed': sum(speeds) / len(speeds) if speeds else 0,
                'top_acceleration': calculator.get_max_acceleration(),
                'movement_patterns': movement_patterns
            }
            
            # Store results in Supabase
            supabase.store_analysis_results(video_id, results)
            
            # Update video status to completed
            supabase.update_video_status(video_id, 'completed')
            
            return jsonify({
                'success': True,
                'videoId': video_id,
                'results': results
            })
            
    except Exception as e:
        # Log the error
        print(f"Error processing video: {str(e)}")
        
        # Update video status to error if possible
        try:
            if 'video_id' in locals() and 'supabase' in locals():
                supabase.update_video_status(video_id, 'error', str(e))
        except Exception as update_error:
            print(f"Error updating video status: {str(update_error)}")
        
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 8080))
    debug = os.environ.get('DEBUG', 'False').lower() == 'true'
    app.run(host='0.0.0.0', port=port, debug=debug)
