import os
import cv2
import numpy as np

def download_video(supabase_client, file_path, destination):
    """
    Download a video from Supabase storage
    
    Args:
        supabase_client: Initialized SupabaseClient
        file_path (str): Path to the file in Supabase storage
        destination (str): Local path to save the file
    
    Returns:
        bool: True if successful, False otherwise
    """
    # Extract bucket and path
    parts = file_path.split('/')
    bucket = 'videos'  # Assuming 'videos' is the bucket name
    path = file_path
    
    return supabase_client.download_file(bucket, path, destination)

def extract_frames(video_path, output_dir, max_frames=300):
    """
    Extract frames from a video file
    
    Args:
        video_path (str): Path to the video file
        output_dir (str): Directory to save extracted frames
        max_frames (int): Maximum number of frames to extract
    
    Returns:
        tuple: (frame_paths, fps, frame_width, frame_height)
    """
    # Open the video file
    cap = cv2.VideoCapture(video_path)
    
    # Check if video opened successfully
    if not cap.isOpened():
        raise Exception(f"Error opening video file: {video_path}")
    
    # Get video properties
    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    # Calculate frame extraction interval
    if total_frames > max_frames:
        interval = total_frames // max_frames
    else:
        interval = 1
    
    frame_paths = []
    frame_count = 0
    
    while True:
        # Read a frame
        ret, frame = cap.read()
        
        # Break the loop if we've reached the end of the video
        if not ret:
            break
        
        # Extract frames at the specified interval
        if frame_count % interval == 0:
            # Save the frame
            frame_path = os.path.join(output_dir, f"frame_{frame_count:06d}.jpg")
            cv2.imwrite(frame_path, frame)
            frame_paths.append(frame_path)
        
        frame_count += 1
    
    # Release the video capture object
    cap.release()
    
    return frame_paths, fps, frame_width, frame_height

def preprocess_frame(frame_path, target_size=(640, 480)):
    """
    Preprocess a frame for glove detection
    
    Args:
        frame_path (str): Path to the frame image
        target_size (tuple): Target size for resizing
    
    Returns:
        numpy.ndarray: Preprocessed frame
    """
    # Read the frame
    frame = cv2.imread(frame_path)
    
    # Resize to target size
    frame = cv2.resize(frame, target_size)
    
    # Convert to RGB (from BGR)
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    
    return frame
