"""
Visualization and reporting module for the Glove Speed Tracker application.
Provides real-time visualization of glove tracking and comprehensive reporting.
"""

import os
import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import logging
from datetime import datetime
import json
import base64
from pathlib import Path

# Import project modules
from config import OUTPUT_DIR
from speed_calculator import SpeedCalculator

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger('visualizer')

class GloveVisualizer:
    """
    Class for visualizing glove tracking and speed data in real-time and generating reports.
    """
    
    def __init__(self, fps=30, pixels_per_meter=100):
        """
        Initialize the visualizer.
        
        Args:
            fps (float): Frames per second of the video
            pixels_per_meter (float): Conversion factor from pixels to meters
        """
        self.fps = fps
        self.pixels_per_meter = pixels_per_meter
        self.tracking_data = None
        self.speed_data = None
        self.frame_width = 640
        self.frame_height = 480
        self.font = cv2.FONT_HERSHEY_SIMPLEX
        self.font_scale = 0.6
        self.font_thickness = 2
        self.colors = {
            'bbox': (0, 255, 0),       # Green
            'text': (255, 255, 255),   # White
            'background': (0, 0, 0),   # Black
            'trajectory': (0, 165, 255), # Orange
            'speed_low': (0, 255, 0),  # Green
            'speed_medium': (0, 255, 255), # Yellow
            'speed_high': (0, 0, 255)  # Red
        }
        
        logger.info(f"GloveVisualizer initialized with FPS: {fps}, Pixels/meter: {pixels_per_meter}")
    
    def set_frame_size(self, width, height):
        """
        Set the frame size for visualization.
        
        Args:
            width (int): Frame width in pixels
            height (int): Frame height in pixels
        """
        self.frame_width = width
        self.frame_height = height
        logger.info(f"Frame size set to {width}x{height}")
    
    def load_tracking_data(self, tracking_data_path):
        """
        Load tracking data from a CSV file.
        
        Args:
            tracking_data_path (str): Path to the tracking data CSV file
            
        Returns:
            bool: True if data loaded successfully, False otherwise
        """
        try:
            self.tracking_data = pd.read_csv(tracking_data_path)
            logger.info(f"Loaded tracking data from {tracking_data_path} with {len(self.tracking_data)} records")
            return True
        except Exception as e:
            logger.error(f"Error loading tracking data: {str(e)}")
            return False
    
    def load_speed_data(self, speed_data_path):
        """
        Load speed data from a CSV file.
        
        Args:
            speed_data_path (str): Path to the speed data CSV file
            
        Returns:
            bool: True if data loaded successfully, False otherwise
        """
        try:
            self.speed_data = pd.read_csv(speed_data_path)
            logger.info(f"Loaded speed data from {speed_data_path} with {len(self.speed_data)} records")
            return True
        except Exception as e:
            logger.error(f"Error loading speed data: {str(e)}")
            return False
    
    def set_tracking_data(self, tracking_data):
        """
        Set tracking data directly.
        
        Args:
            tracking_data (DataFrame): Tracking data
            
        Returns:
            bool: True if data set successfully, False otherwise
        """
        try:
            self.tracking_data = tracking_data
            logger.info(f"Set tracking data with {len(tracking_data)} records")
            return True
        except Exception as e:
            logger.error(f"Error setting tracking data: {str(e)}")
            return False
    
    def set_speed_data(self, speed_data):
        """
        Set speed data directly.
        
        Args:
            speed_data (DataFrame): Speed data
            
        Returns:
            bool: True if data set successfully, False otherwise
        """
        try:
            self.speed_data = speed_data
            logger.info(f"Set speed data with {len(speed_data)} records")
            return True
        except Exception as e:
            logger.error(f"Error setting speed data: {str(e)}")
            return False
    
    def calculate_speed_data(self):
        """
        Calculate speed data from tracking data if not already loaded.
        
        Returns:
            bool: True if calculation successful, False otherwise
        """
        if self.tracking_data is None:
            logger.error("No tracking data available for speed calculation")
            return False
            
        if self.speed_data is not None:
            logger.info("Speed data already available, skipping calculation")
            return True
            
        try:
            # Initialize speed calculator
            calculator = SpeedCalculator(fps=self.fps, pixels_per_meter=self.pixels_per_meter)
            
            # Set tracking data
            calculator.set_tracking_data(self.tracking_data)
            
            # Calculate speed
            self.speed_data = calculator.calculate_speed(smoothing=True)
            
            logger.info(f"Calculated speed data with {len(self.speed_data)} records")
            return True
        except Exception as e:
            logger.error(f"Error calculating speed data: {str(e)}")
            return False
    
    def draw_bounding_box(self, frame, bbox, speed=None, frame_number=None):
        """
        Draw a bounding box on a frame with speed information.
        
        Args:
            frame (numpy.ndarray): Frame to draw on
            bbox (tuple): Bounding box coordinates (x, y, w, h)
            speed (float, optional): Speed to display
            frame_number (int, optional): Frame number
            
        Returns:
            numpy.ndarray: Frame with bounding box
        """
        try:
            x, y, w, h = bbox
            
            # Draw bounding box
            cv2.rectangle(frame, (int(x), int(y)), (int(x + w), int(y + h)), self.colors['bbox'], 2)
            
            # Add speed information if available
            if speed is not None:
                # Determine color based on speed
                if speed < 2.0:  # m/s
                    color = self.colors['speed_low']
                elif speed < 5.0:
                    color = self.colors['speed_medium']
                else:
                    color = self.colors['speed_high']
                
                # Create speed text
                speed_text = f"{speed:.1f} m/s"
                
                # Get text size
                (text_width, text_height), _ = cv2.getTextSize(
                    speed_text, self.font, self.font_scale, self.font_thickness
                )
                
                # Draw background rectangle for text
                cv2.rectangle(
                    frame, 
                    (int(x), int(y - text_height - 5)), 
                    (int(x + text_width + 5), int(y)), 
                    color, 
                    -1
                )
                
                # Draw speed text
                cv2.putText(
                    frame, 
                    speed_text, 
                    (int(x), int(y - 5)), 
                    self.font, 
                    self.font_scale, 
                    self.colors['text'], 
                    self.font_thickness
                )
            
            # Add frame number if available
            if frame_number is not None:
                frame_text = f"Frame: {frame_number}"
                
                # Draw frame number at top-left corner
                cv2.putText(
                    frame, 
                    frame_text, 
                    (10, 30), 
                    self.font, 
                    self.font_scale, 
                    self.colors['text'], 
                    self.font_thickness
                )
            
            return frame
        except Exception as e:
            logger.error(f"Error drawing bounding box: {str(e)}")
            return frame
    
    def draw_trajectory(self, frame, trajectory_points, max_points=30):
        """
        Draw the glove trajectory on a frame.
        
        Args:
            frame (numpy.ndarray): Frame to draw on
            trajectory_points (list): List of (x, y) trajectory points
            max_points (int, optional): Maximum number of points to draw
            
        Returns:
            numpy.ndarray: Frame with trajectory
        """
        try:
            # Limit the number of points to avoid cluttering
            if len(trajectory_points) > max_points:
                trajectory_points = trajectory_points[-max_points:]
            
            # Draw lines connecting trajectory points
            for i in range(1, len(trajectory_points)):
                pt1 = (int(trajectory_points[i-1][0]), int(trajectory_points[i-1][1]))
                pt2 = (int(trajectory_points[i][0]), int(trajectory_points[i][1]))
                cv2.line(frame, pt1, pt2, self.colors['trajectory'], 2)
            
            # Draw points
            for point in trajectory_points:
                cv2.circle(frame, (int(point[0]), int(point[1])), 3, self.colors['trajectory'], -1)
            
            return frame
        except Exception as e:
            logger.error(f"Error drawing trajectory: {str(e)}")
            return frame
    
    def draw_speed_gauge(self, frame, speed, max_speed=10.0, position=(50, 50), size=100):
        """
        Draw a speed gauge on a frame.
        
        Args:
            frame (numpy.ndarray): Frame to draw on
            speed (float): Current speed in m/s
            max_speed (float, optional): Maximum speed for gauge scaling
            position (tuple, optional): Position of the gauge (x, y)
            size (int, optional): Size of the gauge
            
        Returns:
            numpy.ndarray: Frame with speed gauge
        """
        try:
            x, y = position
            radius = size // 2
            
            # Draw gauge background
            cv2.circle(frame, (x + radius, y + radius), radius, self.colors['background'], -1)
            cv2.circle(frame, (x + radius, y + radius), radius, self.colors['text'], 2)
            
            # Draw gauge ticks
            for i in range(0, 11, 2):
                angle = np.pi * (0.75 + 1.5 * i / 10)
                end_x = int(x + radius + 0.8 * radius * np.cos(angle))
                end_y = int(y + radius + 0.8 * radius * np.sin(angle))
                start_x = int(x + radius + 0.9 * radius * np.cos(angle))
                start_y = int(y + radius + 0.9 * radius * np.sin(angle))
                cv2.line(frame, (start_x, start_y), (end_x, end_y), self.colors['text'], 2)
                
                # Add tick labels
                label_x = int(x + radius + 0.7 * radius * np.cos(angle))
                label_y = int(y + radius + 0.7 * radius * np.sin(angle))
                cv2.putText(
                    frame, 
                    f"{i * max_speed / 10:.0f}", 
                    (label_x, label_y), 
                    self.font, 
                    0.4, 
                    self.colors['text'], 
                    1
                )
            
            # Draw gauge needle
            speed_ratio = min(speed / max_speed, 1.0)
            angle = np.pi * (0.75 + 1.5 * speed_ratio)
            end_x = int(x + radius + 0.8 * radius * np.cos(angle))
            end_y = int(y + radius + 0.8 * radius * np.sin(angle))
            
            # Determine needle color based on speed
            if speed < max_speed * 0.3:
                needle_color = self.colors['speed_low']
            elif speed < max_speed * 0.7:
                needle_color = self.colors['speed_medium']
            else:
                needle_color = self.colors['speed_high']
                
            cv2.line(frame, (x + radius, y + radius), (end_x, end_y), needle_color, 3)
            
            # Draw center point
            cv2.circle(frame, (x + radius, y + radius), 5, needle_color, -1)
            
            # Draw speed text
            cv2.putText(
                frame, 
                f"{speed:.1f} m/s", 
                (x + radius - 30, y + radius + 30), 
                self.font, 
                0.6, 
                self.colors['text'], 
                2
            )
            
            return frame
        except Exception as e:
            logger.error(f"Error drawing speed gauge: {str(e)}")
            return frame
    
    def create_visualization_frame(self, frame_number, base_frame=None):
        """
        Create a visualization frame with tracking and speed information.
        
        Args:
            frame_number (int): Frame number to visualize
            base_frame (numpy.ndarray, optional): Base frame to draw on
            
        Returns:
            numpy.ndarray: Visualization frame
        """
        if self.tracking_data is None:
            logger.error("No tracking data available for visualization")
            return None
            
        try:
            # Get tracking data for the frame
            frame_data = self.tracking_data[self.tracking_data['frame'] == frame_number]
            
            if len(frame_data) == 0:
                logger.warning(f"No tracking data for frame {frame_number}")
                return None
            
            # Create base frame if not provided
            if base_frame is None:
                frame = np.zeros((self.frame_height, self.frame_width, 3), dtype=np.uint8)
            else:
                frame = base_frame.copy()
            
            # Get bounding box
            if all(col in frame_data.columns for col in ['x', 'y', 'width', 'height']):
                bbox = (
                    frame_data['x'].values[0],
                    frame_data['y'].values[0],
                    frame_data['width'].values[0],
                    frame_data['height'].values[0]
                )
            else:
                # Try alternative column names
                bbox = None
                if all(col in frame_data.columns for col in ['center_x', 'center_y']):
                    # Estimate bbox from center coordinates
                    center_x = frame_data['center_x'].values[0]
                    center_y = frame_data['center_y'].values[0]
                    # Use default size if width/height not available
                    width = frame_data['width'].values[0] if 'width' in frame_data.columns else 60
                    height = frame_data['height'].values[0] if 'height' in frame_data.columns else 60
                    bbox = (center_x - width/2, center_y - height/2, width, height)
            
            # Get speed if available
            speed = None
            if self.speed_data is not None:
                speed_frame_data = self.speed_data[self.speed_data['frame'] == frame_number]
                if len(speed_frame_data) > 0:
                    if 'smooth_speed_mps' in speed_frame_data.columns:
                        speed = speed_frame_data['smooth_speed_mps'].values[0]
                    elif 'speed_mps' in speed_frame_data.columns:
                        speed = speed_frame_data['speed_mps'].values[0]
            
            # Get trajectory points
            trajectory_points = []
            if all(col in self.tracking_data.columns for col in ['center_x', 'center_y']):
                # Get data up to current frame
                trajectory_data = self.tracking_data[self.tracking_data['frame'] <= frame_number]
                trajectory_points = list(zip(trajectory_data['center_x'], trajectory_data['center_y']))
            
            # Draw bounding box
            if bbox is not None:
                frame = self.draw_bounding_box(frame, bbox, speed, frame_number)
            
            # Draw trajectory
            if trajectory_points:
                frame = self.draw_trajectory(frame, trajectory_points)
            
            # Draw speed gauge
            if speed is not None:
                frame = self.draw_speed_gauge(frame, speed, position=(self.frame_width - 150, 20))
            
            return frame
        except Exception as e:
            logger.error(f"Error creating visualization frame: {str(e)}")
            return None
    
    def create_visualization_video(self, output_path=None, base_video_path=None):
        """
        Create a visualization video with tracking and speed information.
        
        Args:
            output_path (str, optional): Path to save the video
            base_video_path (str, optional): Path to base video
            
        Returns:
            str: Path to the saved video or None if creation failed
        """
        if self.tracking_data is None:
            logger.error("No tracking data available for visualization")
            return None
            
        try:
            # Calculate speed data if not available
            if self.speed_data is None:
                self.calculate_speed_data()
            
            # Create output path if not provided
            if output_path is None:
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                output_path = os.path.join(OUTPUT_DIR, f"visualization_{timestamp}.mp4")
            
            # Get frame range
            min_frame = int(self.tracking_data['frame'].min())
            max_frame = int(self.tracking_data['frame'].max())
            
            # Set up video writer
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            video_writer = cv2.VideoWriter(
                output_path, 
                fourcc, 
                self.fps, 
                (self.frame_width, self.frame_height)
            )
            
            # Open base video if provided
            base_video = None
            if base_video_path and os.path.exists(base_video_path):
                base_video = cv2.VideoCapture(base_video_path)
                
                # Update frame size based on base video
                if base_video.isOpened():
                    self.frame_width = int(base_video.get(cv2.CAP_PROP_FRAME_WIDTH))
                    self.frame_height = int(base_video.get(cv2.CAP_PROP_FRAME_HEIGHT))
                    
                    # Recreate video writer with updated frame size
                    video_writer = cv2.VideoWriter(
                        output_path, 
                        fourcc, 
                        self.fps, 
                        (self.frame_width, self.frame_height)
                    )
            
            # Create visualization frames
            for frame_number in range(min_frame, max_frame + 1):
                # Get base frame if available
                base_frame = None
                if base_video and base_video.isOpened():
                    ret, base_frame = base_video.read()
                    if not ret:
                        logger.warning(f"Could not read frame {frame_number} from base video")
                        break
                
                # Create visualization frame
                vis_frame = self.create_visualization_frame(frame_number, base_frame)
                
                if vis_frame is not None:
                    video_writer.write(vis_frame)
                
                # Log progress
                if frame_number % 50 == 0 or frame_number == max_frame:
                    progress = (frame_number - min_frame + 1) / (max_frame - min_frame + 1) * 100
                    logger.info(f"Visualization progress: {progress:.1f}% (frame {frame_number}/{max_frame})")
            
            # Release resources
            video_writer.release()
            if base_video and base_video.isOpened():
                base_video.release()
            
            logger.info(f"Visualization video saved to {output_path}")
            return output_path
        except Exception as e:
            logger.error(f"Error creating visualization video: {str(e)}")
            return None
    
    def create_interactive_report(self, output_path=None):
        """
        Create an interactive HTML report with visualizations and analysis.
        
        Args:
            output_path (str, optional): Path to save the report
            
        Returns:
            str: Path to the saved report or None if creation failed
        """
        if self.tracking_data is None:
            logger.error("No tracking data available for report")
            return None
            
        try:
            # Calculate speed data if not available
            if self.speed_data is None:
                self.calculate_speed_data()
            
            # Create output path if not provided
            if output_path is None:
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                output_path = os.path.join(OUTPUT_DIR, f"interactive_report_{timestamp}.html")
            
            # Create speed calculator for statistics and plots
            calculator = SpeedCalculator(fps=self.fps, pixels_per_meter=self.pixels_per_meter)
            calculator.set_tracking_data(self.tracking_data)
            
            if self.speed_data is not None:
                calculator.speed_data = self.speed_data
            else:
                calculator.calculate_speed(smoothing=True)
            
            # Generate plots
            trajectory_plot = calculator.plot_trajectory()
            speed_plot = calculator.plot_speed()
            acceleration_plot = calculator.plot_acceleration()
            
            # Get statistics
            max_speed = calculator.get_max_speed()
            avg_speed = calculator.get_average_speed()
            
            # Create sample visualization frames
            sample_frames = []
            frame_count = len(self.tracking_data['frame'].unique())
            
            # Select frames at regular intervals
            sample_indices = [
                int(self.tracking_data['frame'].min()),  # First frame
                int(self.tracking_data['frame'].min() + frame_count * 0.25),  # 25%
                int(self.tracking_data['frame'].min() + frame_count * 0.5),   # 50%
                int(self.tracking_data['frame'].min() + frame_count * 0.75),  # 75%
                int(self.tracking_data['frame'].max())   # Last frame
            ]
            
            for frame_number in sample_indices:
                vis_frame = self.create_visualization_frame(frame_number)
                if vis_frame is not None:
                    # Convert frame to base64 for embedding in HTML
                    _, buffer = cv2.imencode('.png', vis_frame)
                    img_str = base64.b64encode(buffer).decode('utf-8')
                    sample_frames.append({
                        'frame': frame_number,
                        'image': img_str
                    })
            
            # Create interactive HTML report
            with open(output_path, 'w') as f:
                f.write(f"""
                <!DOCTYPE html>
                <html>
                <head>
                    <title>Glove Speed Tracking - Interactive Report</title>
                    <style>
                        body {{ font-family: Arial, sans-serif; margin: 20px; }}
                        h1, h2, h3 {{ color: #2c3e50; }}
                        .container {{ max-width: 1200px; margin: 0 auto; }}
                        .section {{ background-color: #f8f9fa; padding: 15px; border-radius: 5px; margin-bottom: 20px; }}
                        .plot {{ margin-bottom: 30px; }}
                        .plot img {{ max-width: 100%; border: 1px solid #ddd; }}
                        .stats {{ display: flex; flex-wrap: wrap; }}
                        .stat-box {{ flex: 1; min-width: 200px; margin: 10px; padding: 15px; background-color: #e9ecef; border-radius: 5px; text-align: center; }}
                        .stat-value {{ font-size: 24px; font-weight: bold; margin: 10px 0; }}
                        .frames-container {{ display: flex; flex-wrap: wrap; justify-content: space-between; }}
                        .frame-box {{ width: 18%; margin-bottom: 20px; }}
                        .frame-box img {{ width: 100%; border: 1px solid #ddd; }}
                        table {{ border-collapse: collapse; width: 100%; }}
                        th, td {{ border: 1px solid #ddd; padding: 8px; text-align: left; }}
                        th {{ background-color: #f2f2f2; }}
                        tr:nth-child(even) {{ background-color: #f9f9f9; }}
                        .chart-container {{ height: 400px; margin-bottom: 30px; }}
                        
                        /* Interactive elements */
                        .tab {{ overflow: hidden; border: 1px solid #ccc; background-color: #f1f1f1; }}
                        .tab button {{ background-color: inherit; float: left; border: none; outline: none; cursor: pointer; padding: 14px 16px; transition: 0.3s; }}
                        .tab button:hover {{ background-color: #ddd; }}
                        .tab button.active {{ background-color: #ccc; }}
                        .tabcontent {{ display: none; padding: 6px 12px; border: 1px solid #ccc; border-top: none; }}
                        .tabcontent.active {{ display: block; }}
                    </style>
                    <script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
                </head>
                <body>
                    <div class="container">
                        <h1>Glove Speed Tracking - Interactive Report</h1>
                        <p>Generated on {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}</p>
                        
                        <div class="section">
                            <h2>Key Statistics</h2>
                            <div class="stats">
                                <div class="stat-box">
                                    <h3>Maximum Speed</h3>
                                    <div class="stat-value">{max_speed[1]:.2f} mph</div>
                                    <p>({max_speed[0]:.2f} m/s) at frame {max_speed[2]}</p>
                                </div>
                                <div class="stat-box">
                                    <h3>Average Speed</h3>
                                    <div class="stat-value">{avg_speed[1]:.2f} mph</div>
                                    <p>({avg_speed[0]:.2f} m/s)</p>
                                </div>
                                <div class="stat-box">
                                    <h3>Total Frames</h3>
                                    <div class="stat-value">{frame_count}</div>
                                    <p>({frame_count/self.fps:.1f} seconds)</p>
                                </div>
                            </div>
                        </div>
                        
                        <div class="section">
                            <h2>Visualization</h2>
                            <div class="tab">
                                <button class="tablinks active" onclick="openTab(event, 'Frames')">Key Frames</button>
                                <button class="tablinks" onclick="openTab(event, 'Trajectory')">Trajectory</button>
                                <button class="tablinks" onclick="openTab(event, 'SpeedChart')">Speed Chart</button>
                                <button class="tablinks" onclick="openTab(event, 'AccelerationChart')">Acceleration Chart</button>
                            </div>
                            
                            <div id="Frames" class="tabcontent active">
                                <h3>Key Frames</h3>
                                <div class="frames-container">
                """)
                
                # Add sample frames
                for frame_data in sample_frames:
                    f.write(f"""
                                    <div class="frame-box">
                                        <img src="data:image/png;base64,{frame_data['image']}" alt="Frame {frame_data['frame']}">
                                        <p>Frame {frame_data['frame']}</p>
                                    </div>
                    """)
                
                f.write("""
                                </div>
                            </div>
                            
                            <div id="Trajectory" class="tabcontent">
                                <h3>Glove Trajectory</h3>
                """)
                
                if trajectory_plot:
                    # Convert plot to base64
                    with open(trajectory_plot, 'rb') as img_file:
                        trajectory_b64 = base64.b64encode(img_file.read()).decode('utf-8')
                    
                    f.write(f"""
                                <div class="plot">
                                    <img src="data:image/png;base64,{trajectory_b64}" alt="Glove Trajectory">
                                </div>
                    """)
                
                f.write("""
                            </div>
                            
                            <div id="SpeedChart" class="tabcontent">
                                <h3>Speed Over Time</h3>
                                <div class="chart-container">
                                    <canvas id="speedChart"></canvas>
                                </div>
                """)
                
                if speed_plot:
                    # Convert plot to base64
                    with open(speed_plot, 'rb') as img_file:
                        speed_b64 = base64.b64encode(img_file.read()).decode('utf-8')
                    
                    f.write(f"""
                                <div class="plot">
                                    <img src="data:image/png;base64,{speed_b64}" alt="Speed Chart">
                                </div>
                    """)
                
                f.write("""
                            </div>
                            
                            <div id="AccelerationChart" class="tabcontent">
                                <h3>Acceleration Over Time</h3>
                                <div class="chart-container">
                                    <canvas id="accelerationChart"></canvas>
                                </div>
                """)
                
                if acceleration_plot:
                    # Convert plot to base64
                    with open(acceleration_plot, 'rb') as img_file:
                        acceleration_b64 = base64.b64encode(img_file.read()).decode('utf-8')
                    
                    f.write(f"""
                                <div class="plot">
                                    <img src="data:image/png;base64,{acceleration_b64}" alt="Acceleration Chart">
                                </div>
                    """)
                
                f.write("""
                            </div>
                        </div>
                        
                        <div class="section">
                            <h2>Data Table</h2>
                            <div class="tab">
                                <button class="tablinks active" onclick="openTab(event, 'SpeedTable')">Speed Data</button>
                                <button class="tablinks" onclick="openTab(event, 'TrackingTable')">Tracking Data</button>
                            </div>
                            
                            <div id="SpeedTable" class="tabcontent active">
                                <h3>Speed Data</h3>
                                <table>
                                    <tr>
                                        <th>Frame</th>
                                        <th>Time (s)</th>
                                        <th>Speed (m/s)</th>
                                        <th>Speed (mph)</th>
                                    </tr>
                """)
                
                # Add speed data rows (limit to every 5th frame to keep the report manageable)
                if self.speed_data is not None:
                    for i, row in self.speed_data.iloc[::5].iterrows():
                        if i in self.speed_data.index:
                            speed_mps = row['smooth_speed_mps'] if 'smooth_speed_mps' in self.speed_data.columns else row['speed_mps']
                            speed_mph = row['smooth_speed_mph'] if 'smooth_speed_mph' in self.speed_data.columns else speed_mps * 2.23694
                            
                            f.write(f"""
                                    <tr>
                                        <td>{int(row['frame'])}</td>
                                        <td>{row['time']:.2f}</td>
                                        <td>{speed_mps:.2f}</td>
                                        <td>{speed_mph:.2f}</td>
                                    </tr>
                            """)
                
                f.write("""
                                </table>
                            </div>
                            
                            <div id="TrackingTable" class="tabcontent">
                                <h3>Tracking Data</h3>
                                <table>
                                    <tr>
                                        <th>Frame</th>
                """)
                
                # Add column headers based on available columns
                tracking_columns = []
                if 'center_x' in self.tracking_data.columns and 'center_y' in self.tracking_data.columns:
                    tracking_columns.extend(['center_x', 'center_y'])
                    f.write("<th>Center X</th><th>Center Y</th>")
                
                if 'width' in self.tracking_data.columns and 'height' in self.tracking_data.columns:
                    tracking_columns.extend(['width', 'height'])
                    f.write("<th>Width</th><th>Height</th>")
                
                f.write("""
                                    </tr>
                """)
                
                # Add tracking data rows (limit to every 5th frame)
                for i, row in self.tracking_data.iloc[::5].iterrows():
                    if i in self.tracking_data.index:
                        f.write(f"<tr><td>{int(row['frame'])}</td>")
                        
                        for col in tracking_columns:
                            f.write(f"<td>{row[col]:.1f}</td>")
                        
                        f.write("</tr>")
                
                f.write("""
                                </table>
                            </div>
                        </div>
                    </div>
                    
                    <script>
                        // Tab functionality
                        function openTab(evt, tabName) {
                            var i, tabcontent, tablinks;
                            tabcontent = document.getElementsByClassName("tabcontent");
                            for (i = 0; i < tabcontent.length; i++) {
                                tabcontent[i].className = tabcontent[i].className.replace(" active", "");
                            }
                            tablinks = document.getElementsByClassName("tablinks");
                            for (i = 0; i < tablinks.length; i++) {
                                tablinks[i].className = tablinks[i].className.replace(" active", "");
                            }
                            document.getElementById(tabName).className += " active";
                            evt.currentTarget.className += " active";
                        }
                        
                        // Initialize charts
                        document.addEventListener('DOMContentLoaded', function() {
                """)
                
                # Add speed chart data
                if self.speed_data is not None:
                    # Sample data for the chart (every 3rd point to avoid overcrowding)
                    times = self.speed_data['time'].iloc[::3].tolist()
                    speeds = []
                    
                    if 'smooth_speed_mps' in self.speed_data.columns:
                        speeds = self.speed_data['smooth_speed_mps'].iloc[::3].tolist()
                    elif 'speed_mps' in self.speed_data.columns:
                        speeds = self.speed_data['speed_mps'].iloc[::3].tolist()
                    
                    f.write(f"""
                            // Speed chart
                            var speedCtx = document.getElementById('speedChart').getContext('2d');
                            var speedChart = new Chart(speedCtx, {{
                                type: 'line',
                                data: {{
                                    labels: {json.dumps(times)},
                                    datasets: [{{
                                        label: 'Speed (m/s)',
                                        data: {json.dumps(speeds)},
                                        borderColor: 'rgb(75, 192, 192)',
                                        tension: 0.1,
                                        fill: false
                                    }}]
                                }},
                                options: {{
                                    responsive: true,
                                    maintainAspectRatio: false,
                                    scales: {{
                                        y: {{
                                            beginAtZero: true,
                                            title: {{
                                                display: true,
                                                text: 'Speed (m/s)'
                                            }}
                                        }},
                                        x: {{
                                            title: {{
                                                display: true,
                                                text: 'Time (s)'
                                            }}
                                        }}
                                    }}
                                }}
                            }});
                    """)
                
                # Add acceleration chart data
                if self.speed_data is not None and 'smooth_acceleration_mps2' in self.speed_data.columns:
                    # Sample data for the chart (every 3rd point)
                    times = self.speed_data['time'].iloc[::3].tolist()
                    accels = self.speed_data['smooth_acceleration_mps2'].iloc[::3].tolist()
                    
                    f.write(f"""
                            // Acceleration chart
                            var accelCtx = document.getElementById('accelerationChart').getContext('2d');
                            var accelChart = new Chart(accelCtx, {{
                                type: 'line',
                                data: {{
                                    labels: {json.dumps(times)},
                                    datasets: [{{
                                        label: 'Acceleration (m/s²)',
                                        data: {json.dumps(accels)},
                                        borderColor: 'rgb(255, 99, 132)',
                                        tension: 0.1,
                                        fill: false
                                    }}]
                                }},
                                options: {{
                                    responsive: true,
                                    maintainAspectRatio: false,
                                    scales: {{
                                        y: {{
                                            title: {{
                                                display: true,
                                                text: 'Acceleration (m/s²)'
                                            }}
                                        }},
                                        x: {{
                                            title: {{
                                                display: true,
                                                text: 'Time (s)'
                                            }}
                                        }}
                                    }}
                                }}
                            }});
                    """)
                
                f.write("""
                        });
                    </script>
                </body>
                </html>
                """)
            
            logger.info(f"Interactive report saved to {output_path}")
            return output_path
        except Exception as e:
            logger.error(f"Error creating interactive report: {str(e)}")
            return None
    
    def create_json_report(self, output_path=None):
        """
        Create a JSON report with tracking and speed data.
        
        Args:
            output_path (str, optional): Path to save the report
            
        Returns:
            str: Path to the saved report or None if creation failed
        """
        if self.tracking_data is None:
            logger.error("No tracking data available for JSON report")
            return None
            
        try:
            # Calculate speed data if not available
            if self.speed_data is None:
                self.calculate_speed_data()
            
            # Create output path if not provided
            if output_path is None:
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                output_path = os.path.join(OUTPUT_DIR, f"report_{timestamp}.json")
            
            # Create speed calculator for statistics
            calculator = SpeedCalculator(fps=self.fps, pixels_per_meter=self.pixels_per_meter)
            calculator.set_tracking_data(self.tracking_data)
            
            if self.speed_data is not None:
                calculator.speed_data = self.speed_data
            else:
                calculator.calculate_speed(smoothing=True)
            
            # Get statistics
            max_speed = calculator.get_max_speed()
            avg_speed = calculator.get_average_speed()
            
            # Prepare data for JSON
            report_data = {
                "metadata": {
                    "generated_at": datetime.now().isoformat(),
                    "fps": self.fps,
                    "pixels_per_meter": self.pixels_per_meter,
                    "frame_count": len(self.tracking_data['frame'].unique()),
                    "duration_seconds": self.tracking_data['frame'].max() / self.fps
                },
                "statistics": {
                    "max_speed_mps": max_speed[0],
                    "max_speed_mph": max_speed[1],
                    "max_speed_frame": int(max_speed[2]),
                    "avg_speed_mps": avg_speed[0],
                    "avg_speed_mph": avg_speed[1]
                },
                "frames": []
            }
            
            # Add frame data
            for frame_number in sorted(self.tracking_data['frame'].unique()):
                frame_data = self.tracking_data[self.tracking_data['frame'] == frame_number]
                
                # Get speed data if available
                speed_data = None
                if self.speed_data is not None:
                    speed_frame_data = self.speed_data[self.speed_data['frame'] == frame_number]
                    if len(speed_frame_data) > 0:
                        speed_data = {}
                        
                        if 'smooth_speed_mps' in speed_frame_data.columns:
                            speed_data['speed_mps'] = float(speed_frame_data['smooth_speed_mps'].values[0])
                        elif 'speed_mps' in speed_frame_data.columns:
                            speed_data['speed_mps'] = float(speed_frame_data['speed_mps'].values[0])
                            
                        if 'smooth_speed_mph' in speed_frame_data.columns:
                            speed_data['speed_mph'] = float(speed_frame_data['smooth_speed_mph'].values[0])
                        elif 'speed_mph' in speed_frame_data.columns:
                            speed_data['speed_mph'] = float(speed_frame_data['speed_mph'].values[0])
                            
                        if 'smooth_acceleration_mps2' in speed_frame_data.columns:
                            speed_data['acceleration_mps2'] = float(speed_frame_data['smooth_acceleration_mps2'].values[0])
                        elif 'acceleration_mps2' in speed_frame_data.columns:
                            speed_data['acceleration_mps2'] = float(speed_frame_data['acceleration_mps2'].values[0])
                
                # Create frame entry
                frame_entry = {
                    "frame": int(frame_number),
                    "time": float(frame_number / self.fps)
                }
                
                # Add position data
                if 'center_x' in frame_data.columns and 'center_y' in frame_data.columns:
                    frame_entry["position"] = {
                        "center_x": float(frame_data['center_x'].values[0]),
                        "center_y": float(frame_data['center_y'].values[0])
                    }
                
                # Add bounding box data
                if 'width' in frame_data.columns and 'height' in frame_data.columns:
                    if 'position' in frame_entry:
                        frame_entry["position"]["width"] = float(frame_data['width'].values[0])
                        frame_entry["position"]["height"] = float(frame_data['height'].values[0])
                
                # Add speed data
                if speed_data:
                    frame_entry["speed"] = speed_data
                
                report_data["frames"].append(frame_entry)
            
            # Write JSON report
            with open(output_path, 'w') as f:
                json.dump(report_data, f, indent=2)
            
            logger.info(f"JSON report saved to {output_path}")
            return output_path
        except Exception as e:
            logger.error(f"Error creating JSON report: {str(e)}")
            return None

def test_visualizer():
    """
    Test function for the GloveVisualizer class.
    """
    import glob
    
    # Find the most recent tracking and speed data files
    tracking_files = glob.glob(os.path.join(OUTPUT_DIR, "tracking_data_*.csv"))
    speed_files = glob.glob(os.path.join(OUTPUT_DIR, "speed_data_*.csv"))
    
    if not tracking_files:
        logger.error("No tracking data files found")
        return False
    
    # Sort by modification time (newest first)
    tracking_file = sorted(tracking_files, key=os.path.getmtime, reverse=True)[0]
    speed_file = None
    if speed_files:
        speed_file = sorted(speed_files, key=os.path.getmtime, reverse=True)[0]
    
    logger.info(f"Testing GloveVisualizer with tracking data: {tracking_file}")
    if speed_file:
        logger.info(f"Using speed data: {speed_file}")
    
    # Initialize visualizer
    visualizer = GloveVisualizer(fps=30, pixels_per_meter=100)
    
    # Load tracking data
    success = visualizer.load_tracking_data(tracking_file)
    if not success:
        logger.error("Failed to load tracking data")
        return False
    
    # Load speed data if available
    if speed_file:
        success = visualizer.load_speed_data(speed_file)
        if not success:
            logger.warning("Failed to load speed data, will calculate from tracking data")
    
    # Get frame size from tracking data
    if 'center_x' in visualizer.tracking_data.columns and 'center_y' in visualizer.tracking_data.columns:
        max_x = visualizer.tracking_data['center_x'].max() + 100  # Add margin
        max_y = visualizer.tracking_data['center_y'].max() + 100
        visualizer.set_frame_size(int(max_x), int(max_y))
    
    # Create sample visualization frame
    middle_frame = int((visualizer.tracking_data['frame'].min() + visualizer.tracking_data['frame'].max()) / 2)
    sample_frame = visualizer.create_visualization_frame(middle_frame)
    
    if sample_frame is not None:
        # Save sample frame
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        sample_path = os.path.join(OUTPUT_DIR, f"sample_visualization_{timestamp}.png")
        cv2.imwrite(sample_path, sample_frame)
        logger.info(f"Sample visualization frame saved to {sample_path}")
    
    # Create visualization video
    video_path = visualizer.create_visualization_video()
    
    # Create interactive report
    report_path = visualizer.create_interactive_report()
    
    # Create JSON report
    json_path = visualizer.create_json_report()
    
    logger.info("GloveVisualizer test completed successfully")
    return {
        "tracking_data": tracking_file,
        "speed_data": speed_file,
        "sample_frame": sample_path if sample_frame is not None else None,
        "video": video_path,
        "report": report_path,
        "json": json_path
    }

if __name__ == "__main__":
    test_results = test_visualizer()
    
    if test_results:
        print("\nTest Results:")
        print(f"Tracking data: {test_results['tracking_data']}")
        print(f"Speed data: {test_results['speed_data']}")
        print(f"Sample frame: {test_results['sample_frame']}")
        print(f"Visualization video: {test_results['video']}")
        print(f"Interactive report: {test_results['report']}")
        print(f"JSON report: {test_results['json']}")
