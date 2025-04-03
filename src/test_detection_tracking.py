"""
Test script for the glove detection and tracking module.
This script demonstrates the integration of video processing, detection, and tracking.
"""

import os
import sys
import argparse
import logging
import cv2
import numpy as np
from datetime import datetime

# Add the src directory to the path so we can import our modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.video_processor import VideoProcessor
from src.detector import GloveDetector
from src.tracker import MultiTracker
from src.config import DATA_DIR, OUTPUT_DIR

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger('test_detection_tracking')

def test_detection_tracking(video_path=None):
    """
    Test the integration of video processing, detection, and tracking.
    
    Args:
        video_path (str, optional): Path to a test video file
    """
    # Create a test video if not provided
    if not video_path or not os.path.exists(video_path):
        from src.test_video_processor import create_test_video
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        video_path = os.path.join(DATA_DIR, f"test_video_{timestamp}.mp4")
        create_test_video(video_path, duration=5, fps=30, size=(640, 480))
    
    logger.info(f"Testing detection and tracking with video: {video_path}")
    
    # Initialize video processor
    processor = VideoProcessor(video_path)
    
    # Extract frames
    frames = processor.extract_frames(start_time=0, end_time=None, step=1)
    logger.info(f"Extracted {len(frames)} frames")
    
    # Preprocess frames
    processed_frames = processor.preprocess_frames(frames)
    logger.info(f"Preprocessed {len(processed_frames)} frames")
    
    # Initialize detector
    detector = GloveDetector()
    
    # Initialize multi-tracker
    multi_tracker = MultiTracker(detector)
    
    # Process the first frame to initialize tracking
    first_frame = processed_frames[0]
    detections = detector.detect_glove(first_frame)
    
    if not detections:
        logger.error("No glove detected in the first frame")
        return False
    
    # Use the detection with highest confidence
    best_detection = max(detections, key=lambda x: x[4])
    x1, y1, x2, y2 = best_detection[:4]
    bbox = (x1, y1, x2 - x1, y2 - y1)
    
    # Initialize tracker
    success = multi_tracker.init(first_frame, bbox)
    if not success:
        logger.error("Failed to initialize tracker")
        return False
    
    # Process remaining frames
    tracked_frames = []
    tracking_data = []
    
    # Add first frame with detection visualization
    vis_frame = detector.visualize_detections(first_frame, [best_detection])
    vis_frame = multi_tracker.visualize_tracking(vis_frame, bbox)
    tracked_frames.append(vis_frame)
    
    # Track in remaining frames
    for i, frame in enumerate(processed_frames[1:], 1):
        # Update tracker
        success, bbox = multi_tracker.update(frame)
        
        if success:
            # Store tracking data
            x, y, w, h = bbox
            center_x = x + w/2
            center_y = y + h/2
            tracking_data.append((i, center_x, center_y, w, h))
            
            # Visualize tracking
            vis_frame = multi_tracker.visualize_tracking(frame, bbox)
            tracked_frames.append(vis_frame)
        else:
            logger.warning(f"Tracking failed on frame {i}")
            tracked_frames.append(frame)
    
    # Save tracked frames as a video
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_path = os.path.join(OUTPUT_DIR, f"tracked_video_{timestamp}.mp4")
    
    height, width = tracked_frames[0].shape[:2]
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, processor.fps, (width, height))
    
    for frame in tracked_frames:
        # Convert RGB to BGR for OpenCV
        out.write(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
    
    out.release()
    logger.info(f"Tracked video saved to {output_path}")
    
    # Save tracking data
    tracking_data_path = os.path.join(OUTPUT_DIR, f"tracking_data_{timestamp}.csv")
    with open(tracking_data_path, 'w') as f:
        f.write("frame,center_x,center_y,width,height\n")
        for data in tracking_data:
            f.write(f"{data[0]},{data[1]},{data[2]},{data[3]},{data[4]}\n")
    
    logger.info(f"Tracking data saved to {tracking_data_path}")
    
    # Release resources
    processor.release()
    
    logger.info("Detection and tracking test completed successfully")
    return {
        "original_video": video_path,
        "tracked_video": output_path,
        "tracking_data": tracking_data_path,
        "frame_count": len(frames),
        "tracked_frames": len(tracking_data)
    }

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Test the detection and tracking module")
    parser.add_argument("--video", type=str, help="Path to a test video file")
    args = parser.parse_args()
    
    test_results = test_detection_tracking(args.video)
    
    if test_results:
        print("\nTest Results:")
        print(f"Original video: {test_results['original_video']}")
        print(f"Tracked video: {test_results['tracked_video']}")
        print(f"Tracking data: {test_results['tracking_data']}")
        print(f"Total frames: {test_results['frame_count']}")
        print(f"Successfully tracked frames: {test_results['tracked_frames']}")
