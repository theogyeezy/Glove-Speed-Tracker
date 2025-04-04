import numpy as np
import cv2

class GloveDetector:
    """
    Class for detecting baseball catcher's glove in video frames
    """
    
    def __init__(self, confidence_threshold=0.5):
        """
        Initialize the glove detector
        
        Args:
            confidence_threshold (float): Confidence threshold for detections
        """
        self.confidence_threshold = confidence_threshold
        # In a real implementation, this would load a trained model
        # For this example, we'll use a simple color-based detection
    
    def detect_glove(self, frame_path):
        """
        Detect glove in a frame
        
        Args:
            frame_path (str): Path to the frame image
        
        Returns:
            dict: Detection result with bounding box and confidence
        """
        # Load the frame
        frame = cv2.imread(frame_path)
        if frame is None:
            return None
        
        # Convert to HSV color space
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        
        # Define range for brown/tan color (typical glove color)
        lower_brown = np.array([10, 50, 50])
        upper_brown = np.array([30, 255, 255])
        
        # Create a mask for brown/tan colors
        mask = cv2.inRange(hsv, lower_brown, upper_brown)
        
        # Find contours in the mask
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        # If no contours found, return None
        if not contours:
            return None
        
        # Find the largest contour (likely to be the glove)
        largest_contour = max(contours, key=cv2.contourArea)
        
        # Get bounding box for the contour
        x, y, w, h = cv2.boundingRect(largest_contour)
        
        # Calculate confidence based on contour area
        area = cv2.contourArea(largest_contour)
        max_area = frame.shape[0] * frame.shape[1]
        confidence = min(area / (max_area * 0.1), 1.0)  # Normalize to [0, 1]
        
        # If confidence is below threshold, return None
        if confidence < self.confidence_threshold:
            return None
        
        # Return detection result
        return {
            'bbox': [x, y, w, h],
            'confidence': float(confidence),
            'center': [x + w/2, y + h/2]
        }
