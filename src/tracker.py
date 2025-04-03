"""
Tracker module for the Glove Speed Tracker application.
Handles tracking of detected gloves across video frames.
Uses a custom implementation for compatibility with various OpenCV versions.
"""

import os
import cv2
import numpy as np
import logging
from tqdm import tqdm

# Import configuration
from config import MAX_TRACKING_FAILURES

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger('tracker')

class GloveTracker:
    """
    Class for tracking baseball catcher's glove across video frames.
    Implements a custom tracking approach for compatibility with various OpenCV versions.
    """
    
    def __init__(self, tracker_type='Custom'):
        """
        Initialize the glove tracker.
        
        Args:
            tracker_type (str): Type of tracker to use (mostly for logging purposes)
        """
        self.tracker_type = tracker_type
        self.bbox = None
        self.tracking_failures = 0
        self.tracking_history = []
        self.is_initialized = False
        self.prev_frame = None
        self.prev_bbox = None
        self.feature_params = dict(maxCorners=100, qualityLevel=0.3, minDistance=7, blockSize=7)
        self.lk_params = dict(winSize=(15, 15), maxLevel=2, criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))
        self.prev_points = None
        
        logger.info(f"Initialized {tracker_type} tracker")
    
    def init(self, frame, bbox):
        """
        Initialize the tracker with a frame and bounding box.
        
        Args:
            frame (numpy.ndarray): Initial frame
            bbox (tuple): Initial bounding box (x, y, width, height)
            
        Returns:
            bool: True if initialization successful, False otherwise
        """
        # Convert bbox from [x1, y1, x2, y2] to [x, y, width, height] if needed
        if len(bbox) == 4 and bbox[2] > bbox[0] and bbox[3] > bbox[1]:
            x, y, x2, y2 = bbox
            bbox = (int(x), int(y), int(x2 - x), int(y2 - y))
        
        try:
            # Store the frame and bbox
            self.prev_frame = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
            self.bbox = bbox
            self.prev_bbox = bbox
            
            # Extract region of interest
            x, y, w, h = bbox
            roi = self.prev_frame[y:y+h, x:x+w]
            
            # Find good features to track within the ROI
            points = cv2.goodFeaturesToTrack(roi, mask=None, **self.feature_params)
            
            if points is not None and len(points) > 0:
                # Adjust points to global coordinates
                self.prev_points = points + np.array([[x, y]], dtype=np.float32)
                self.is_initialized = True
                self.tracking_failures = 0
                self.tracking_history = [bbox]
                logger.info(f"Tracker initialized with bbox: {bbox}")
                return True
            else:
                logger.error("Failed to find features to track")
                return False
                
        except Exception as e:
            logger.error(f"Error initializing tracker: {str(e)}")
            return False
    
    def update(self, frame):
        """
        Update the tracker with a new frame.
        
        Args:
            frame (numpy.ndarray): New frame
            
        Returns:
            tuple: (success, bbox) where bbox is (x, y, width, height)
        """
        if not self.is_initialized or self.prev_frame is None or self.prev_points is None:
            logger.error("Tracker not initialized")
            return False, None
        
        try:
            # Convert frame to grayscale
            gray_frame = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
            
            # Calculate optical flow
            new_points, status, _ = cv2.calcOpticalFlowPyrLK(
                self.prev_frame, gray_frame, self.prev_points, None, **self.lk_params
            )
            
            # Filter out points that couldn't be tracked
            if new_points is not None and status is not None:
                good_new = new_points[status == 1]
                good_old = self.prev_points[status == 1]
            else:
                self.tracking_failures += 1
                logger.warning(f"Tracking failure #{self.tracking_failures}: No points tracked")
                return False, None
            
            if len(good_new) < 5:  # Need at least a few points for reliable tracking
                self.tracking_failures += 1
                logger.warning(f"Tracking failure #{self.tracking_failures}: Too few points tracked")
                return False, None
            
            # Calculate movement
            movement = good_new - good_old
            median_movement = np.median(movement, axis=0)
            
            # Update bounding box
            x, y, w, h = self.prev_bbox
            new_x = int(x + median_movement[0])
            new_y = int(y + median_movement[1])
            
            # Ensure the box stays within the frame
            height, width = frame.shape[:2]
            new_x = max(0, min(new_x, width - w))
            new_y = max(0, min(new_y, height - h))
            
            new_bbox = (new_x, new_y, w, h)
            
            # Update state
            self.prev_frame = gray_frame
            self.prev_points = good_new.reshape(-1, 1, 2)
            self.prev_bbox = new_bbox
            self.bbox = new_bbox
            self.tracking_failures = 0
            
            # Convert bbox from (x, y, width, height) to (x1, y1, x2, y2) for history
            bbox_xyxy = (new_x, new_y, new_x + w, new_y + h)
            self.tracking_history.append(bbox_xyxy)
            
            # Find new points to track within the updated bounding box
            roi = gray_frame[new_y:new_y+h, new_x:new_x+w]
            new_roi_points = cv2.goodFeaturesToTrack(roi, mask=None, **self.feature_params)
            
            if new_roi_points is not None and len(new_roi_points) > 0:
                # Adjust points to global coordinates
                self.prev_points = new_roi_points + np.array([[new_x, new_y]], dtype=np.float32)
            
            return True, new_bbox
            
        except Exception as e:
            logger.error(f"Error updating tracker: {str(e)}")
            self.tracking_failures += 1
            return False, None
    
    def predict_next_position(self):
        """
        Predict the next position based on tracking history.
        
        Returns:
            tuple: Predicted bounding box (x1, y1, x2, y2) or None if not enough history
        """
        if len(self.tracking_history) < 2:
            return None
        
        # Get the last two positions
        prev_pos = self.tracking_history[-2]
        curr_pos = self.tracking_history[-1]
        
        # Calculate velocity
        dx = curr_pos[0] - prev_pos[0]
        dy = curr_pos[1] - prev_pos[1]
        dw = (curr_pos[2] - curr_pos[0]) - (prev_pos[2] - prev_pos[0])
        dh = (curr_pos[3] - curr_pos[1]) - (prev_pos[3] - prev_pos[1])
        
        # Predict next position
        next_x1 = curr_pos[0] + dx
        next_y1 = curr_pos[1] + dy
        next_x2 = curr_pos[2] + dx + dw
        next_y2 = curr_pos[3] + dy + dh
        
        return (next_x1, next_y1, next_x2, next_y2)
    
    def reset(self):
        """
        Reset the tracker.
        """
        self.bbox = None
        self.prev_bbox = None
        self.prev_frame = None
        self.prev_points = None
        self.tracking_failures = 0
        self.is_initialized = False
        logger.info("Tracker reset")

class MultiTracker:
    """
    Class for managing multiple tracking algorithms and detection integration.
    Combines detection and tracking for robust glove tracking.
    """
    
    def __init__(self, detector=None):
        """
        Initialize the multi-tracker.
        
        Args:
            detector: Detector instance for reinitialization
        """
        self.primary_tracker = GloveTracker('Primary')
        self.backup_tracker = GloveTracker('Backup')
        self.detector = detector
        self.current_tracker = 'primary'
        self.track_history = []
        self.detection_frequency = 10  # Run detector every N frames
        self.frame_count = 0
        
        logger.info("MultiTracker initialized")
    
    def init(self, frame, bbox=None):
        """
        Initialize the multi-tracker with a frame and optional bounding box.
        If no bbox is provided and a detector is available, it will be used.
        
        Args:
            frame (numpy.ndarray): Initial frame
            bbox (tuple, optional): Initial bounding box
            
        Returns:
            bool: True if initialization successful, False otherwise
        """
        # If no bbox provided, try to detect
        if bbox is None and self.detector is not None:
            detections = self.detector.detect_glove(frame)
            if detections and len(detections) > 0:
                # Use the detection with highest confidence
                best_detection = max(detections, key=lambda x: x[4])
                x1, y1, x2, y2 = best_detection[:4]
                bbox = (x1, y1, x2 - x1, y2 - y1)
                logger.info(f"Initialized with detection: {bbox}")
            else:
                logger.error("No detection found for initialization")
                return False
        
        # Initialize both trackers
        primary_success = self.primary_tracker.init(frame, bbox)
        backup_success = self.backup_tracker.init(frame, bbox)
        
        if primary_success and backup_success:
            self.current_tracker = 'primary'
            self.track_history = [(self.frame_count, bbox)]
            self.frame_count = 0
            logger.info("MultiTracker initialized successfully")
            return True
        elif primary_success:
            self.current_tracker = 'primary'
            self.track_history = [(self.frame_count, bbox)]
            self.frame_count = 0
            logger.warning("Only primary tracker initialized successfully")
            return True
        elif backup_success:
            self.current_tracker = 'backup'
            self.track_history = [(self.frame_count, bbox)]
            self.frame_count = 0
            logger.warning("Only backup tracker initialized successfully")
            return True
        else:
            logger.error("Failed to initialize both trackers")
            return False
    
    def update(self, frame):
        """
        Update the multi-tracker with a new frame.
        
        Args:
            frame (numpy.ndarray): New frame
            
        Returns:
            tuple: (success, bbox) where bbox is (x, y, width, height)
        """
        self.frame_count += 1
        
        # Periodically run detector for verification if available
        if self.detector is not None and self.frame_count % self.detection_frequency == 0:
            detections = self.detector.detect_glove(frame)
            if detections and len(detections) > 0:
                # Use the detection with highest confidence
                best_detection = max(detections, key=lambda x: x[4])
                x1, y1, x2, y2 = best_detection[:4]
                detection_bbox = (x1, y1, x2 - x1, y2 - y1)
                
                # Compare with current tracking
                if self.current_tracker == 'primary':
                    current_bbox = self.primary_tracker.bbox
                else:
                    current_bbox = self.backup_tracker.bbox
                
                if current_bbox is not None:
                    # Calculate IoU between detection and tracking
                    iou = self._calculate_iou(detection_bbox, current_bbox)
                    
                    # If IoU is low, reinitialize with detection
                    if iou < 0.3:
                        logger.info(f"Low IoU ({iou:.2f}), reinitializing trackers with detection")
                        self.init(frame, detection_bbox)
        
        # Update primary tracker
        primary_success, primary_bbox = self.primary_tracker.update(frame)
        
        # Update backup tracker
        backup_success, backup_bbox = self.backup_tracker.update(frame)
        
        # Decide which tracker to use
        if primary_success and backup_success:
            # Both successful, use primary
            self.current_tracker = 'primary'
            bbox = primary_bbox
        elif primary_success:
            # Only primary successful
            self.current_tracker = 'primary'
            bbox = primary_bbox
        elif backup_success:
            # Only backup successful
            self.current_tracker = 'backup'
            bbox = backup_bbox
        else:
            # Both failed, try to reinitialize with detector
            if self.detector is not None:
                detections = self.detector.detect_glove(frame)
                if detections and len(detections) > 0:
                    # Use the detection with highest confidence
                    best_detection = max(detections, key=lambda x: x[4])
                    x1, y1, x2, y2 = best_detection[:4]
                    detection_bbox = (x1, y1, x2 - x1, y2 - y1)
                    
                    # Reinitialize trackers
                    self.init(frame, detection_bbox)
                    logger.info("Reinitialized trackers with new detection")
                    return True, detection_bbox
            
            # If we get here, tracking has failed
            logger.warning("Both trackers failed, no detection available")
            return False, None
        
        # Add to tracking history
        self.track_history.append((self.frame_count, bbox))
        
        return True, bbox
    
    def _calculate_iou(self, bbox1, bbox2):
        """
        Calculate Intersection over Union between two bounding boxes.
        
        Args:
            bbox1 (tuple): First bounding box (x, y, width, height)
            bbox2 (tuple): Second bounding box (x, y, width, height)
            
        Returns:
            float: IoU value
        """
        # Convert to x1, y1, x2, y2 format
        x1_1, y1_1, w1, h1 = bbox1
        x1_2, y1_2, w2, h2 = bbox2
        
        x2_1, y2_1 = x1_1 + w1, y1_1 + h1
        x2_2, y2_2 = x1_2 + w2, y1_2 + h2
        
        # Calculate intersection area
        x_left = max(x1_1, x1_2)
        y_top = max(y1_1, y1_2)
        x_right = min(x2_1, x2_2)
        y_bottom = min(y2_1, y2_2)
        
        if x_right < x_left or y_bottom < y_top:
            return 0.0
        
        intersection_area = (x_right - x_left) * (y_bottom - y_top)
        
        # Calculate union area
        bbox1_area = w1 * h1
        bbox2_area = w2 * h2
        union_area = bbox1_area + bbox2_area - intersection_area
        
        # Calculate IoU
        iou = intersection_area / union_area if union_area > 0 else 0
        
        return iou
    
    def get_trajectory(self):
        """
        Get the complete trajectory of the tracked object.
        
        Returns:
            list: List of (frame_number, bbox) tuples
        """
        return self.track_history
    
    def visualize_tracking(self, frame, bbox, color=(0, 255, 0), thickness=2):
        """
        Visualize tracking on a frame.
        
        Args:
            frame (numpy.ndarray): Input frame
            bbox (tuple): Bounding box (x, y, width, height)
            color (tuple): BGR color for bounding box
            thickness (int): Line thickness
            
        Returns:
            numpy.ndarray: Frame with visualized tracking
        """
        # Create a copy of the frame
        vis_frame = frame.copy()
        
        # Draw bounding box
        x, y, w, h = [int(v) for v in bbox]
        cv2.rectangle(vis_frame, (x, y), (x + w, y + h), color, thickness)
        
        # Draw tracker type
        tracker_text = f"Tracker: {self.current_tracker}"
        cv2.putText(vis_frame, tracker_text, (x, y - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, thickness)
        
        # Draw trajectory if we have enough history
        if len(self.track_history) > 1:
            # Get the last 20 positions or all if less than 20
            history = self.track_history[-20:]
            
            # Draw trajectory line
            for i in range(1, len(history)):
                prev_frame_num, prev_bbox = history[i-1]
                curr_frame_num, curr_bbox = history[i]
                
                # Get center points
                prev_x = prev_bbox[0] + prev_bbox[2] // 2
                prev_y = prev_bbox[1] + prev_bbox[3] // 2
                curr_x = curr_bbox[0] + curr_bbox[2] // 2
                curr_y = curr_bbox[1] + curr_bbox[3] // 2
                
                # Draw line between centers
                cv2.line(vis_frame, (int(prev_x), int(prev_y)), 
                         (int(curr_x), int(curr_y)), (0, 0, 255), 2)
        
        return vis_frame
    
    def reset(self):
        """
        Reset the multi-tracker.
        """
        self.primary_tracker.reset()
        self.backup_tracker.reset()
        self.current_tracker = 'primary'
        self.track_history = []
        self.frame_count = 0
        logger.info("MultiTracker reset")

def test_tracker():
    """
    Test function for the GloveTracker and MultiTracker classes.
    """
    from detector import GloveDetector
    
    # Create a sequence of test images with a moving circle
    frames = []
    width, height = 640, 480
    circle_radius = 30
    circle_color = (0, 0, 255)  # Red
    
    # Starting position
    x, y = circle_radius + 50, height // 2
    
    # Movement parameters
    dx, dy = 5, 3
    
    # Generate 50 frames
    for i in range(50):
        # Create a blank frame
        frame = np.zeros((height, width, 3), dtype=np.uint8)
        
        # Update position with some randomness
        x += dx + np.random.randint(-2, 3)
        y += dy + np.random.randint(-2, 3)
        
        # Bounce off edges
        if x <= circle_radius or x >= width - circle_radius:
            dx = -dx
            x += dx * 2
        
        if y <= circle_radius or y >= height - circle_radius:
            dy = -dy
            y += dy * 2
        
        # Draw the circle
        cv2.circle(frame, (int(x), int(y)), circle_radius, circle_color, -1)
        
        # Add frame number
        cv2.putText(frame, f"Frame: {i}", (20, 30), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        frames.append(frame)
    
    # Initialize detector
    detector = GloveDetector()
    
    # Initialize multi-tracker
    multi_tracker = MultiTracker(detector)
    
    # Initialize with first frame
    first_frame = frames[0]
    detections = detector.detect_glove(first_frame)
    
    if detections:
        # Use the first detection
        x1, y1, x2, y2, _, _ = detections[0]
        bbox = (x1, y1, x2 - x1, y2 - y1)
        
        # Initialize tracker
        multi_tracker.init(first_frame, bbox)
        
        # Process remaining frames
        tracked_frames = []
        
        for i, frame in enumerate(frames):
            # Update tracker
            success, bbox = multi_tracker.update(frame)
            
            if success:
                # Visualize tracking
                vis_frame = multi_tracker.visualize_tracking(frame, bbox)
                tracked_frames.append(vis_frame)
            else:
                logger.warning(f"Tracking failed on frame {i}")
                tracked_frames.append(frame)
        
        # Save tracked frames as a video
        output_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        output_path = os.path.join(output_dir, "output", "test_tracking.mp4")
        
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_path, fourcc, 10, (width, height))
        
        for frame in tracked_frames:
            out.write(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
        
        out.release()
        
        logger.info(f"Test tracking video saved to {output_path}")
        return True
    else:
        logger.error("No detection found in first frame")
        return False

if __name__ == "__main__":
    test_tracker()
