"""
Detector module for the Glove Speed Tracker application.
Handles glove detection using deep learning models.
"""

import os
import cv2
import numpy as np
import tensorflow as tf
import logging
from tqdm import tqdm

# Import configuration
from config import CONFIDENCE_THRESHOLD, IOU_THRESHOLD, MODELS_DIR

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger('detector')

class GloveDetector:
    """
    Class for detecting baseball catcher's glove in video frames.
    Uses TensorFlow for object detection.
    """
    
    def __init__(self, model_path=None):
        """
        Initialize the glove detector.
        
        Args:
            model_path (str, optional): Path to a pre-trained model
        """
        self.model = None
        self.model_path = model_path
        self.input_size = (416, 416)  # Default input size for the model
        self.classes = ['glove']  # Only detecting gloves
        
        # Load model if provided
        if model_path and os.path.exists(model_path):
            self.load_model(model_path)
        else:
            logger.warning("No model provided or model not found. Using placeholder detection.")
    
    def load_model(self, model_path):
        """
        Load a pre-trained TensorFlow model.
        
        Args:
            model_path (str): Path to the model
            
        Returns:
            bool: True if model loaded successfully, False otherwise
        """
        try:
            # Load saved model
            self.model = tf.saved_model.load(model_path)
            self.model_path = model_path
            logger.info(f"Model loaded from {model_path}")
            return True
        except Exception as e:
            logger.error(f"Failed to load model: {str(e)}")
            return False
    
    def preprocess_image(self, image):
        """
        Preprocess an image for the detection model.
        
        Args:
            image (numpy.ndarray): Input image
            
        Returns:
            numpy.ndarray: Preprocessed image
        """
        # Resize image to the input size expected by the model
        resized = cv2.resize(image, self.input_size)
        
        # Convert to RGB if needed
        if len(resized.shape) == 3 and resized.shape[2] == 3:
            if image.dtype == np.float32 or image.dtype == np.float64:
                # Ensure pixel values are in the correct range
                resized = resized.astype(np.float32) / 255.0
            else:
                # Convert to float and normalize
                resized = resized.astype(np.float32) / 255.0
        
        # Expand dimensions to create batch of size 1
        preprocessed = np.expand_dims(resized, axis=0)
        
        return preprocessed
    
    def detect_glove(self, image, confidence_threshold=None):
        """
        Detect glove in an image.
        
        Args:
            image (numpy.ndarray): Input image
            confidence_threshold (float, optional): Detection confidence threshold
            
        Returns:
            list: List of detection results [x1, y1, x2, y2, confidence, class_id]
        """
        if confidence_threshold is None:
            confidence_threshold = CONFIDENCE_THRESHOLD
        
        # If no model is loaded, use placeholder detection
        if self.model is None:
            return self._placeholder_detection(image)
        
        # Preprocess the image
        preprocessed = self.preprocess_image(image)
        
        # Perform detection
        try:
            # Get input and output tensor names
            input_tensor = self.model.signatures['serving_default'].inputs[0].name
            output_tensors = list(self.model.signatures['serving_default'].outputs.keys())
            
            # Run inference
            results = self.model.signatures['serving_default'](tf.convert_to_tensor(preprocessed))
            
            # Process results
            detections = []
            
            # Extract detection boxes, scores, and classes
            boxes = results['detection_boxes'].numpy()[0]
            scores = results['detection_scores'].numpy()[0]
            classes = results['detection_classes'].numpy()[0].astype(np.int32)
            
            # Filter by confidence and class
            image_height, image_width = image.shape[:2]
            
            for i in range(len(scores)):
                if scores[i] >= confidence_threshold and classes[i] == 1:  # Assuming class 1 is glove
                    # Convert normalized coordinates to pixel coordinates
                    y1, x1, y2, x2 = boxes[i]
                    x1 = int(x1 * image_width)
                    y1 = int(y1 * image_height)
                    x2 = int(x2 * image_width)
                    y2 = int(y2 * image_height)
                    
                    detections.append([x1, y1, x2, y2, scores[i], classes[i]])
            
            return detections
            
        except Exception as e:
            logger.error(f"Detection error: {str(e)}")
            return self._placeholder_detection(image)
    
    def _placeholder_detection(self, image):
        """
        Placeholder detection method when no model is available.
        Uses simple color-based detection for red objects (assuming red glove).
        
        Args:
            image (numpy.ndarray): Input image
            
        Returns:
            list: List of detection results [x1, y1, x2, y2, confidence, class_id]
        """
        # Convert to HSV color space
        hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
        
        # Define range for red color detection
        lower_red1 = np.array([0, 100, 100])
        upper_red1 = np.array([10, 255, 255])
        lower_red2 = np.array([160, 100, 100])
        upper_red2 = np.array([180, 255, 255])
        
        # Create masks for red color
        mask1 = cv2.inRange(hsv, lower_red1, upper_red1)
        mask2 = cv2.inRange(hsv, lower_red2, upper_red2)
        mask = cv2.bitwise_or(mask1, mask2)
        
        # Find contours in the mask
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        detections = []
        
        if contours:
            # Find the largest contour (assuming it's the glove)
            largest_contour = max(contours, key=cv2.contourArea)
            area = cv2.contourArea(largest_contour)
            
            # Only consider if area is significant
            if area > 500:
                x, y, w, h = cv2.boundingRect(largest_contour)
                confidence = min(area / 10000, 0.9)  # Normalize confidence
                detections.append([x, y, x + w, y + h, confidence, 1])
        
        return detections
    
    def detect_in_frames(self, frames, confidence_threshold=None):
        """
        Detect glove in multiple frames.
        
        Args:
            frames (list): List of input frames
            confidence_threshold (float, optional): Detection confidence threshold
            
        Returns:
            list: List of detection results for each frame
        """
        if confidence_threshold is None:
            confidence_threshold = CONFIDENCE_THRESHOLD
        
        all_detections = []
        
        logger.info(f"Detecting gloves in {len(frames)} frames")
        
        for frame in tqdm(frames, desc="Detecting gloves"):
            detections = self.detect_glove(frame, confidence_threshold)
            all_detections.append(detections)
        
        logger.info(f"Completed detection on {len(frames)} frames")
        return all_detections
    
    def visualize_detections(self, image, detections, color=(0, 255, 0), thickness=2):
        """
        Visualize detections on an image.
        
        Args:
            image (numpy.ndarray): Input image
            detections (list): List of detection results [x1, y1, x2, y2, confidence, class_id]
            color (tuple): BGR color for bounding box
            thickness (int): Line thickness
            
        Returns:
            numpy.ndarray: Image with visualized detections
        """
        # Create a copy of the image
        vis_image = image.copy()
        
        for detection in detections:
            x1, y1, x2, y2, confidence, class_id = detection
            
            # Draw bounding box
            cv2.rectangle(vis_image, (int(x1), int(y1)), (int(x2), int(y2)), color, thickness)
            
            # Draw label
            label = f"{self.classes[int(class_id)-1]}: {confidence:.2f}"
            cv2.putText(vis_image, label, (int(x1), int(y1) - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, thickness)
        
        return vis_image

def test_detector():
    """
    Test function for the GloveDetector class.
    """
    # Create a test image with a red circle to simulate a glove
    image = np.zeros((480, 640, 3), dtype=np.uint8)
    cv2.circle(image, (320, 240), 50, (0, 0, 255), -1)  # Red circle
    
    # Initialize detector
    detector = GloveDetector()
    
    # Detect glove
    detections = detector.detect_glove(image)
    
    # Visualize detections
    vis_image = detector.visualize_detections(image, detections)
    
    # Save the result
    output_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 
                              "output", "test_detection.jpg")
    cv2.imwrite(output_path, cv2.cvtColor(vis_image, cv2.COLOR_RGB2BGR))
    
    logger.info(f"Test detection result saved to {output_path}")
    logger.info(f"Detections: {detections}")
    
    return detections

if __name__ == "__main__":
    test_detector()
