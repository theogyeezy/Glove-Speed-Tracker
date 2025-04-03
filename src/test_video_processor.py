"""
Test script for the video processing module.
This script demonstrates the functionality of the VideoProcessor class.
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
from src.config import DATA_DIR, OUTPUT_DIR

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger('test_video_processor')

def create_test_video(output_path, duration=5, fps=30, size=(640, 480)):
    """
    Create a test video with a moving circle to simulate a glove.
    
    Args:
        output_path (str): Path to save the test video
        duration (int): Duration of the video in seconds
        fps (int): Frames per second
        size (tuple): Video dimensions (width, height)
        
    Returns:
        str: Path to the created test video
    """
    width, height = size
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, size)
    
    # Total frames to generate
    total_frames = duration * fps
    
    # Circle parameters
    circle_radius = 30
    circle_color = (0, 0, 255)  # Red
    
    # Starting position
    x = circle_radius + 50
    y = height // 2
    
    # Movement parameters
    dx = 5  # x velocity
    dy = 3  # y velocity
    
    logger.info(f"Creating test video with {total_frames} frames at {fps} FPS")
    
    for i in range(total_frames):
        # Create a blank frame
        frame = np.zeros((height, width, 3), dtype=np.uint8)
        
        # Update position with some randomness to simulate natural movement
        x += dx + np.random.randint(-2, 3)
        y += dy + np.random.randint(-2, 3)
        
        # Bounce off edges
        if x <= circle_radius or x >= width - circle_radius:
            dx = -dx
            x += dx * 2  # Prevent sticking to the edge
        
        if y <= circle_radius or y >= height - circle_radius:
            dy = -dy
            y += dy * 2  # Prevent sticking to the edge
        
        # Draw the circle
        cv2.circle(frame, (int(x), int(y)), circle_radius, circle_color, -1)
        
        # Add frame number text
        cv2.putText(frame, f"Frame: {i}", (20, 30), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        # Write the frame
        out.write(frame)
    
    out.release()
    logger.info(f"Test video created: {output_path}")
    return output_path

def test_video_processor(video_path=None):
    """
    Test the VideoProcessor class functionality.
    
    Args:
        video_path (str, optional): Path to a test video file
    """
    # Create a test video if not provided
    if not video_path or not os.path.exists(video_path):
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        video_path = os.path.join(DATA_DIR, f"test_video_{timestamp}.mp4")
        create_test_video(video_path)
    
    logger.info(f"Testing VideoProcessor with video: {video_path}")
    
    # Initialize video processor
    processor = VideoProcessor(video_path)
    
    # Extract frames (every 5th frame from the first 3 seconds)
    frames = processor.extract_frames(start_time=0, end_time=3, step=5)
    logger.info(f"Extracted {len(frames)} frames")
    
    # Preprocess frames
    processed_frames = processor.preprocess_frames(frames)
    logger.info(f"Preprocessed {len(processed_frames)} frames")
    
    # Save sample frames
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = os.path.join(OUTPUT_DIR, f"test_frames_{timestamp}")
    frame_paths = processor.save_frames(processed_frames[:10], output_dir=output_dir)
    logger.info(f"Saved {len(frame_paths)} frames to {output_dir}")
    
    # Create a sample video
    sample_video_path = processor.create_sample_video(
        processed_frames[:50], 
        output_path=os.path.join(OUTPUT_DIR, f"sample_{timestamp}.mp4")
    )
    logger.info(f"Created sample video: {sample_video_path}")
    
    # Release resources
    processor.release()
    
    logger.info("Video processor test completed successfully")
    return {
        "original_video": video_path,
        "sample_video": sample_video_path,
        "frame_dir": output_dir,
        "frame_count": len(frames),
        "sample_frames": frame_paths
    }

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Test the video processing module")
    parser.add_argument("--video", type=str, help="Path to a test video file")
    args = parser.parse_args()
    
    test_results = test_video_processor(args.video)
    
    print("\nTest Results:")
    print(f"Original video: {test_results['original_video']}")
    print(f"Sample video: {test_results['sample_video']}")
    print(f"Frame directory: {test_results['frame_dir']}")
    print(f"Extracted frames: {test_results['frame_count']}")
    print(f"Sample frames: {len(test_results['sample_frames'])}")
