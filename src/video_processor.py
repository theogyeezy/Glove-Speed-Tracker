"""
Video processing module for the Glove Speed Tracker application.
Handles video input, frame extraction, and preprocessing.
"""

import os
import cv2
import numpy as np
from tqdm import tqdm
import logging
from datetime import datetime

# Import configuration
from config import SUPPORTED_FORMATS, FRAME_RESIZE, DATA_DIR, OUTPUT_DIR

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger('video_processor')

class VideoProcessor:
    """
    Class for handling video processing operations including loading,
    frame extraction, and preprocessing.
    """
    
    def __init__(self, video_path=None):
        """
        Initialize the video processor.
        
        Args:
            video_path (str, optional): Path to the video file
        """
        self.video_path = video_path
        self.cap = None
        self.frame_count = 0
        self.fps = 0
        self.width = 0
        self.height = 0
        self.duration = 0
        self.frames = []
        
        if video_path and os.path.exists(video_path):
            self.load_video(video_path)
    
    def load_video(self, video_path):
        """
        Load a video file and extract its metadata.
        
        Args:
            video_path (str): Path to the video file
            
        Returns:
            bool: True if video loaded successfully, False otherwise
        """
        if not os.path.exists(video_path):
            logger.error(f"Video file not found: {video_path}")
            return False
            
        # Check if the file format is supported
        _, ext = os.path.splitext(video_path)
        if ext.lower() not in SUPPORTED_FORMATS:
            logger.error(f"Unsupported video format: {ext}")
            return False
            
        # Open the video file
        self.cap = cv2.VideoCapture(video_path)
        if not self.cap.isOpened():
            logger.error(f"Failed to open video: {video_path}")
            return False
            
        # Extract video metadata
        self.video_path = video_path
        self.frame_count = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self.fps = self.cap.get(cv2.CAP_PROP_FPS)
        self.width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.duration = self.frame_count / self.fps if self.fps > 0 else 0
        
        logger.info(f"Video loaded: {os.path.basename(video_path)}")
        logger.info(f"Dimensions: {self.width}x{self.height}, FPS: {self.fps:.2f}")
        logger.info(f"Frame count: {self.frame_count}, Duration: {self.duration:.2f}s")
        
        return True
    
    def extract_frames(self, start_time=0, end_time=None, step=1):
        """
        Extract frames from the video.
        
        Args:
            start_time (float): Start time in seconds
            end_time (float, optional): End time in seconds
            step (int): Extract every nth frame
            
        Returns:
            list: List of extracted frames
        """
        if not self.cap or not self.cap.isOpened():
            logger.error("No video loaded")
            return []
            
        # Calculate frame indices
        start_frame = int(start_time * self.fps)
        end_frame = int(end_time * self.fps) if end_time else self.frame_count
        
        # Validate frame range
        start_frame = max(0, min(start_frame, self.frame_count - 1))
        end_frame = max(start_frame + 1, min(end_frame, self.frame_count))
        
        # Set the starting position
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
        
        # Extract frames
        frames = []
        frame_indices = range(start_frame, end_frame, step)
        
        logger.info(f"Extracting frames {start_frame} to {end_frame} with step {step}")
        
        for i in tqdm(frame_indices, desc="Extracting frames"):
            ret, frame = self.cap.read()
            if not ret:
                break
                
            # Skip frames according to step
            if step > 1 and i % step != 0:
                continue
                
            frames.append(frame)
            
            # Skip to next frame position if step > 1
            if step > 1:
                next_pos = self.cap.get(cv2.CAP_PROP_POS_FRAMES) + step - 1
                if next_pos < end_frame:
                    self.cap.set(cv2.CAP_PROP_POS_FRAMES, next_pos)
        
        self.frames = frames
        logger.info(f"Extracted {len(frames)} frames")
        return frames
    
    def preprocess_frames(self, frames=None, resize=True):
        """
        Preprocess the extracted frames.
        
        Args:
            frames (list, optional): List of frames to preprocess
            resize (bool): Whether to resize the frames
            
        Returns:
            list: List of preprocessed frames
        """
        if frames is None:
            frames = self.frames
            
        if not frames:
            logger.error("No frames to preprocess")
            return []
            
        preprocessed_frames = []
        
        logger.info(f"Preprocessing {len(frames)} frames")
        
        for frame in tqdm(frames, desc="Preprocessing frames"):
            # Resize frame if needed
            if resize and FRAME_RESIZE:
                frame = cv2.resize(frame, FRAME_RESIZE)
                
            # Convert to RGB (OpenCV uses BGR by default)
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            
            # Apply additional preprocessing as needed
            # (e.g., normalization, contrast enhancement, etc.)
            
            preprocessed_frames.append(frame_rgb)
            
        logger.info(f"Preprocessed {len(preprocessed_frames)} frames")
        return preprocessed_frames
    
    def save_frames(self, frames=None, output_dir=None, prefix="frame"):
        """
        Save frames to disk.
        
        Args:
            frames (list, optional): List of frames to save
            output_dir (str, optional): Directory to save frames
            prefix (str): Prefix for frame filenames
            
        Returns:
            list: List of saved frame paths
        """
        if frames is None:
            frames = self.frames
            
        if not frames:
            logger.error("No frames to save")
            return []
            
        if output_dir is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_dir = os.path.join(OUTPUT_DIR, f"frames_{timestamp}")
            
        os.makedirs(output_dir, exist_ok=True)
        
        frame_paths = []
        
        logger.info(f"Saving {len(frames)} frames to {output_dir}")
        
        for i, frame in enumerate(tqdm(frames, desc="Saving frames")):
            # Convert back to BGR for saving with OpenCV
            if len(frame.shape) == 3 and frame.shape[2] == 3:
                if frame.dtype == np.float32 or frame.dtype == np.float64:
                    # Normalize if the frame is in float format
                    frame = (frame * 255).astype(np.uint8)
                
                # Check if the frame is in RGB format and convert to BGR
                frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
            else:
                frame_bgr = frame
                
            # Save the frame
            frame_path = os.path.join(output_dir, f"{prefix}_{i:06d}.jpg")
            cv2.imwrite(frame_path, frame_bgr)
            frame_paths.append(frame_path)
            
        logger.info(f"Saved {len(frame_paths)} frames")
        return frame_paths
    
    def create_sample_video(self, frames=None, output_path=None, fps=None):
        """
        Create a sample video from frames.
        
        Args:
            frames (list, optional): List of frames to use
            output_path (str, optional): Path to save the video
            fps (float, optional): Frames per second
            
        Returns:
            str: Path to the created video
        """
        if frames is None:
            frames = self.frames
            
        if not frames:
            logger.error("No frames to create video")
            return None
            
        if output_path is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_path = os.path.join(OUTPUT_DIR, f"sample_{timestamp}.mp4")
            
        if fps is None:
            fps = self.fps if self.fps > 0 else 30
            
        # Get frame dimensions
        height, width = frames[0].shape[:2]
        
        # Create video writer
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
        
        logger.info(f"Creating sample video with {len(frames)} frames at {fps} FPS")
        
        for frame in tqdm(frames, desc="Creating video"):
            # Convert to BGR for OpenCV
            if len(frame.shape) == 3 and frame.shape[2] == 3:
                if frame.dtype == np.float32 or frame.dtype == np.float64:
                    # Normalize if the frame is in float format
                    frame = (frame * 255).astype(np.uint8)
                
                # Check if the frame is in RGB format and convert to BGR
                frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
            else:
                frame_bgr = frame
                
            out.write(frame_bgr)
            
        out.release()
        logger.info(f"Sample video created: {output_path}")
        return output_path
    
    def release(self):
        """
        Release video resources.
        """
        if self.cap and self.cap.isOpened():
            self.cap.release()
            logger.info("Video resources released")

def test_video_processor():
    """
    Test function for the VideoProcessor class.
    """
    # Create a test video if needed
    test_video_path = os.path.join(DATA_DIR, "test_video.mp4")
    
    if not os.path.exists(test_video_path):
        logger.info("Test video not found. Please provide a test video.")
        return
    
    # Initialize video processor
    processor = VideoProcessor(test_video_path)
    
    # Extract frames
    frames = processor.extract_frames(start_time=0, end_time=5, step=5)
    
    # Preprocess frames
    processed_frames = processor.preprocess_frames(frames)
    
    # Save sample frames
    processor.save_frames(processed_frames[:10])
    
    # Create a sample video
    processor.create_sample_video(processed_frames[:100])
    
    # Release resources
    processor.release()
    
    logger.info("Video processor test completed")

if __name__ == "__main__":
    test_video_processor()
