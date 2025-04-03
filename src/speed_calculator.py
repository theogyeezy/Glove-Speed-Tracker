"""
Speed calculation module for the Glove Speed Tracker application.
Calculates glove speed based on tracking data and video metadata.
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import logging
from scipy.signal import savgol_filter
from datetime import datetime

# Import configuration
from config import PIXELS_PER_METER, SMOOTHING_WINDOW, OUTPUT_DIR

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger('speed_calculator')

class SpeedCalculator:
    """
    Class for calculating glove speed from tracking data.
    """
    
    def __init__(self, fps=30, pixels_per_meter=None):
        """
        Initialize the speed calculator.
        
        Args:
            fps (float): Frames per second of the video
            pixels_per_meter (float, optional): Conversion factor from pixels to meters
        """
        self.fps = fps
        self.pixels_per_meter = pixels_per_meter if pixels_per_meter else PIXELS_PER_METER
        self.tracking_data = None
        self.speed_data = None
        self.acceleration_data = None
        
        logger.info(f"SpeedCalculator initialized with FPS: {fps}, Pixels/meter: {self.pixels_per_meter}")
    
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
    
    def set_tracking_data(self, tracking_data):
        """
        Set tracking data directly from a list or DataFrame.
        
        Args:
            tracking_data (list/DataFrame): Tracking data with frame, center_x, center_y columns
            
        Returns:
            bool: True if data set successfully, False otherwise
        """
        try:
            if isinstance(tracking_data, list):
                # Convert list to DataFrame
                if tracking_data and len(tracking_data) > 0:
                    if isinstance(tracking_data[0], tuple) and len(tracking_data[0]) >= 3:
                        # Assuming format: [(frame_num, (x, y, w, h)), ...]
                        data = []
                        for frame_num, bbox in tracking_data:
                            x, y, w, h = bbox
                            center_x = x + w/2
                            center_y = y + h/2
                            data.append({
                                'frame': frame_num,
                                'center_x': center_x,
                                'center_y': center_y,
                                'width': w,
                                'height': h
                            })
                        self.tracking_data = pd.DataFrame(data)
                    else:
                        logger.error("Invalid tracking data format")
                        return False
                else:
                    logger.error("Empty tracking data")
                    return False
            else:
                # Assume it's already a DataFrame
                self.tracking_data = tracking_data
                
            logger.info(f"Set tracking data with {len(self.tracking_data)} records")
            return True
        except Exception as e:
            logger.error(f"Error setting tracking data: {str(e)}")
            return False
    
    def calculate_speed(self, smoothing=True, window_size=None):
        """
        Calculate speed from tracking data.
        
        Args:
            smoothing (bool): Whether to apply smoothing to the position data
            window_size (int, optional): Window size for smoothing filter
            
        Returns:
            pandas.DataFrame: DataFrame with speed data
        """
        if self.tracking_data is None or len(self.tracking_data) < 2:
            logger.error("Insufficient tracking data for speed calculation")
            return None
        
        # Make a copy of the tracking data
        data = self.tracking_data.copy()
        
        # Ensure data is sorted by frame number
        data = data.sort_values('frame')
        
        # Calculate time in seconds for each frame
        data['time'] = data['frame'] / self.fps
        
        # Apply smoothing to position data if requested
        if smoothing:
            if window_size is None:
                window_size = min(SMOOTHING_WINDOW, len(data) // 2)
                # Ensure window size is odd
                if window_size % 2 == 0:
                    window_size += 1
            
            # Ensure window size is valid
            if window_size >= 3 and window_size <= len(data):
                try:
                    # Apply Savitzky-Golay filter for smoothing
                    data['smooth_x'] = savgol_filter(data['center_x'], window_size, 2)
                    data['smooth_y'] = savgol_filter(data['center_y'], window_size, 2)
                    
                    # Use smoothed positions for calculations
                    pos_x = data['smooth_x']
                    pos_y = data['smooth_y']
                    
                    logger.info(f"Applied position smoothing with window size {window_size}")
                except Exception as e:
                    logger.warning(f"Smoothing failed: {str(e)}. Using raw positions.")
                    pos_x = data['center_x']
                    pos_y = data['center_y']
            else:
                logger.warning(f"Invalid window size {window_size}. Using raw positions.")
                pos_x = data['center_x']
                pos_y = data['center_y']
        else:
            # Use raw positions
            pos_x = data['center_x']
            pos_y = data['center_y']
        
        # Calculate displacement between consecutive frames
        data['dx'] = pos_x.diff()
        data['dy'] = pos_y.diff()
        
        # Calculate Euclidean distance (displacement magnitude)
        data['displacement_pixels'] = np.sqrt(data['dx']**2 + data['dy']**2)
        
        # Convert displacement from pixels to meters
        data['displacement_meters'] = data['displacement_pixels'] / self.pixels_per_meter
        
        # Calculate time difference between frames
        data['dt'] = data['time'].diff()
        
        # Calculate speed in meters per second
        data['speed_mps'] = data['displacement_meters'] / data['dt']
        
        # Convert to miles per hour
        data['speed_mph'] = data['speed_mps'] * 2.23694
        
        # Apply smoothing to speed data if requested
        if smoothing and window_size >= 3 and window_size <= len(data):
            try:
                data['smooth_speed_mps'] = savgol_filter(
                    data['speed_mps'].fillna(0), window_size, 2
                )
                data['smooth_speed_mph'] = savgol_filter(
                    data['speed_mph'].fillna(0), window_size, 2
                )
                logger.info(f"Applied speed smoothing with window size {window_size}")
            except Exception as e:
                logger.warning(f"Speed smoothing failed: {str(e)}. Using raw speeds.")
                data['smooth_speed_mps'] = data['speed_mps']
                data['smooth_speed_mph'] = data['speed_mph']
        else:
            data['smooth_speed_mps'] = data['speed_mps']
            data['smooth_speed_mph'] = data['speed_mph']
        
        # Store the speed data
        self.speed_data = data
        
        logger.info(f"Calculated speed for {len(data)} frames")
        
        return data
    
    def calculate_acceleration(self, smoothing=True, window_size=None):
        """
        Calculate acceleration from speed data.
        
        Args:
            smoothing (bool): Whether to apply smoothing to the acceleration data
            window_size (int, optional): Window size for smoothing filter
            
        Returns:
            pandas.DataFrame: DataFrame with acceleration data
        """
        if self.speed_data is None:
            # Calculate speed first if not already done
            self.calculate_speed(smoothing, window_size)
            
        if self.speed_data is None or len(self.speed_data) < 3:
            logger.error("Insufficient speed data for acceleration calculation")
            return None
        
        # Make a copy of the speed data
        data = self.speed_data.copy()
        
        # Calculate acceleration in meters per second squared
        data['acceleration_mps2'] = data['smooth_speed_mps'].diff() / data['dt']
        
        # Apply smoothing to acceleration data if requested
        if smoothing:
            if window_size is None:
                window_size = min(SMOOTHING_WINDOW, len(data) // 2)
                # Ensure window size is odd
                if window_size % 2 == 0:
                    window_size += 1
            
            # Ensure window size is valid
            if window_size >= 3 and window_size <= len(data):
                try:
                    # Apply Savitzky-Golay filter for smoothing
                    data['smooth_acceleration_mps2'] = savgol_filter(
                        data['acceleration_mps2'].fillna(0), window_size, 2
                    )
                    logger.info(f"Applied acceleration smoothing with window size {window_size}")
                except Exception as e:
                    logger.warning(f"Acceleration smoothing failed: {str(e)}. Using raw acceleration.")
                    data['smooth_acceleration_mps2'] = data['acceleration_mps2']
            else:
                logger.warning(f"Invalid window size {window_size}. Using raw acceleration.")
                data['smooth_acceleration_mps2'] = data['acceleration_mps2']
        else:
            data['smooth_acceleration_mps2'] = data['acceleration_mps2']
        
        # Store the acceleration data
        self.acceleration_data = data
        
        logger.info(f"Calculated acceleration for {len(data)} frames")
        
        return data
    
    def get_max_speed(self):
        """
        Get the maximum speed.
        
        Returns:
            tuple: (max_speed_mps, max_speed_mph, frame_number)
        """
        if self.speed_data is None:
            logger.error("No speed data available")
            return None
        
        # Find the row with maximum smoothed speed
        max_speed_row = self.speed_data.loc[self.speed_data['smooth_speed_mps'].idxmax()]
        
        max_speed_mps = max_speed_row['smooth_speed_mps']
        max_speed_mph = max_speed_row['smooth_speed_mph']
        frame_number = max_speed_row['frame']
        
        logger.info(f"Maximum speed: {max_speed_mps:.2f} m/s ({max_speed_mph:.2f} mph) at frame {frame_number}")
        
        return (max_speed_mps, max_speed_mph, frame_number)
    
    def get_average_speed(self):
        """
        Get the average speed.
        
        Returns:
            tuple: (avg_speed_mps, avg_speed_mph)
        """
        if self.speed_data is None:
            logger.error("No speed data available")
            return None
        
        # Calculate average of smoothed speed, ignoring NaN values
        avg_speed_mps = self.speed_data['smooth_speed_mps'].mean()
        avg_speed_mph = self.speed_data['smooth_speed_mph'].mean()
        
        logger.info(f"Average speed: {avg_speed_mps:.2f} m/s ({avg_speed_mph:.2f} mph)")
        
        return (avg_speed_mps, avg_speed_mph)
    
    def get_speed_at_frame(self, frame_number):
        """
        Get the speed at a specific frame.
        
        Args:
            frame_number (int): Frame number
            
        Returns:
            tuple: (speed_mps, speed_mph)
        """
        if self.speed_data is None:
            logger.error("No speed data available")
            return None
        
        # Find the row with the specified frame number
        frame_data = self.speed_data[self.speed_data['frame'] == frame_number]
        
        if len(frame_data) == 0:
            logger.warning(f"No data for frame {frame_number}")
            return None
        
        speed_mps = frame_data['smooth_speed_mps'].values[0]
        speed_mph = frame_data['smooth_speed_mph'].values[0]
        
        logger.info(f"Speed at frame {frame_number}: {speed_mps:.2f} m/s ({speed_mph:.2f} mph)")
        
        return (speed_mps, speed_mph)
    
    def plot_trajectory(self, output_path=None):
        """
        Plot the glove trajectory.
        
        Args:
            output_path (str, optional): Path to save the plot
            
        Returns:
            str: Path to the saved plot or None if plotting failed
        """
        if self.tracking_data is None:
            logger.error("No tracking data available")
            return None
        
        try:
            plt.figure(figsize=(10, 8))
            
            # Plot the trajectory
            plt.plot(self.tracking_data['center_x'], self.tracking_data['center_y'], 'b-', linewidth=2)
            plt.plot(self.tracking_data['center_x'], self.tracking_data['center_y'], 'ro', markersize=4)
            
            # Mark start and end points
            plt.plot(self.tracking_data['center_x'].iloc[0], self.tracking_data['center_y'].iloc[0], 'go', markersize=8, label='Start')
            plt.plot(self.tracking_data['center_x'].iloc[-1], self.tracking_data['center_y'].iloc[-1], 'mo', markersize=8, label='End')
            
            # Add labels and title
            plt.xlabel('X Position (pixels)')
            plt.ylabel('Y Position (pixels)')
            plt.title('Glove Trajectory')
            plt.legend()
            plt.grid(True)
            
            # Invert y-axis to match image coordinates (origin at top-left)
            plt.gca().invert_yaxis()
            
            # Save the plot if output path is provided
            if output_path is None:
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                output_path = os.path.join(OUTPUT_DIR, f"trajectory_{timestamp}.png")
            
            plt.savefig(output_path)
            plt.close()
            
            logger.info(f"Trajectory plot saved to {output_path}")
            return output_path
        except Exception as e:
            logger.error(f"Error plotting trajectory: {str(e)}")
            return None
    
    def plot_speed(self, output_path=None):
        """
        Plot the glove speed over time.
        
        Args:
            output_path (str, optional): Path to save the plot
            
        Returns:
            str: Path to the saved plot or None if plotting failed
        """
        if self.speed_data is None:
            logger.error("No speed data available")
            return None
        
        try:
            plt.figure(figsize=(12, 6))
            
            # Plot raw and smoothed speed
            plt.plot(self.speed_data['time'], self.speed_data['speed_mph'], 'b-', alpha=0.4, label='Raw Speed')
            plt.plot(self.speed_data['time'], self.speed_data['smooth_speed_mph'], 'r-', linewidth=2, label='Smoothed Speed')
            
            # Mark maximum speed
            max_speed = self.get_max_speed()
            if max_speed:
                max_speed_mph = max_speed[1]
                max_frame = max_speed[2]
                max_time = self.speed_data[self.speed_data['frame'] == max_frame]['time'].values[0]
                plt.plot(max_time, max_speed_mph, 'go', markersize=8)
                plt.annotate(f"{max_speed_mph:.1f} mph", 
                             (max_time, max_speed_mph),
                             xytext=(10, 10),
                             textcoords='offset points',
                             arrowprops=dict(arrowstyle='->'))
            
            # Add labels and title
            plt.xlabel('Time (seconds)')
            plt.ylabel('Speed (mph)')
            plt.title('Glove Speed Over Time')
            plt.legend()
            plt.grid(True)
            
            # Save the plot if output path is provided
            if output_path is None:
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                output_path = os.path.join(OUTPUT_DIR, f"speed_{timestamp}.png")
            
            plt.savefig(output_path)
            plt.close()
            
            logger.info(f"Speed plot saved to {output_path}")
            return output_path
        except Exception as e:
            logger.error(f"Error plotting speed: {str(e)}")
            return None
    
    def plot_acceleration(self, output_path=None):
        """
        Plot the glove acceleration over time.
        
        Args:
            output_path (str, optional): Path to save the plot
            
        Returns:
            str: Path to the saved plot or None if plotting failed
        """
        if self.acceleration_data is None:
            # Calculate acceleration first if not already done
            self.calculate_acceleration()
            
        if self.acceleration_data is None:
            logger.error("No acceleration data available")
            return None
        
        try:
            plt.figure(figsize=(12, 6))
            
            # Plot raw and smoothed acceleration
            plt.plot(self.acceleration_data['time'], self.acceleration_data['acceleration_mps2'], 'b-', alpha=0.4, label='Raw Acceleration')
            plt.plot(self.acceleration_data['time'], self.acceleration_data['smooth_acceleration_mps2'], 'r-', linewidth=2, label='Smoothed Acceleration')
            
            # Add zero line
            plt.axhline(y=0, color='k', linestyle='-', alpha=0.3)
            
            # Add labels and title
            plt.xlabel('Time (seconds)')
            plt.ylabel('Acceleration (m/s²)')
            plt.title('Glove Acceleration Over Time')
            plt.legend()
            plt.grid(True)
            
            # Save the plot if output path is provided
            if output_path is None:
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                output_path = os.path.join(OUTPUT_DIR, f"acceleration_{timestamp}.png")
            
            plt.savefig(output_path)
            plt.close()
            
            logger.info(f"Acceleration plot saved to {output_path}")
            return output_path
        except Exception as e:
            logger.error(f"Error plotting acceleration: {str(e)}")
            return None
    
    def generate_speed_report(self, output_path=None):
        """
        Generate a comprehensive speed analysis report.
        
        Args:
            output_path (str, optional): Path to save the report
            
        Returns:
            str: Path to the saved report or None if generation failed
        """
        if self.speed_data is None:
            logger.error("No speed data available")
            return None
        
        try:
            # Calculate statistics if not already done
            if self.acceleration_data is None:
                self.calculate_acceleration()
            
            max_speed = self.get_max_speed()
            avg_speed = self.get_average_speed()
            
            # Generate plots
            trajectory_plot = self.plot_trajectory()
            speed_plot = self.plot_speed()
            acceleration_plot = self.plot_acceleration()
            
            # Create report path if not provided
            if output_path is None:
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                output_path = os.path.join(OUTPUT_DIR, f"speed_report_{timestamp}.html")
            
            # Generate HTML report
            with open(output_path, 'w') as f:
                f.write(f"""
                <!DOCTYPE html>
                <html>
                <head>
                    <title>Glove Speed Analysis Report</title>
                    <style>
                        body {{ font-family: Arial, sans-serif; margin: 20px; }}
                        h1, h2 {{ color: #2c3e50; }}
                        .container {{ max-width: 1200px; margin: 0 auto; }}
                        .stats {{ background-color: #f8f9fa; padding: 15px; border-radius: 5px; margin-bottom: 20px; }}
                        .plot {{ margin-bottom: 30px; }}
                        .plot img {{ max-width: 100%; border: 1px solid #ddd; }}
                        table {{ border-collapse: collapse; width: 100%; }}
                        th, td {{ border: 1px solid #ddd; padding: 8px; text-align: left; }}
                        th {{ background-color: #f2f2f2; }}
                        tr:nth-child(even) {{ background-color: #f9f9f9; }}
                    </style>
                </head>
                <body>
                    <div class="container">
                        <h1>Glove Speed Analysis Report</h1>
                        <p>Generated on {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}</p>
                        
                        <div class="stats">
                            <h2>Speed Statistics</h2>
                            <p><strong>Maximum Speed:</strong> {max_speed[1]:.2f} mph ({max_speed[0]:.2f} m/s) at frame {max_speed[2]}</p>
                            <p><strong>Average Speed:</strong> {avg_speed[1]:.2f} mph ({avg_speed[0]:.2f} m/s)</p>
                            <p><strong>Total Frames Analyzed:</strong> {len(self.speed_data)}</p>
                            <p><strong>Video Duration:</strong> {self.speed_data['time'].max():.2f} seconds</p>
                            <p><strong>Frame Rate:</strong> {self.fps} fps</p>
                        </div>
                        
                        <h2>Trajectory Analysis</h2>
                        <div class="plot">
                            <img src="{os.path.basename(trajectory_plot)}" alt="Glove Trajectory">
                            <p>The plot above shows the path of the glove throughout the video.</p>
                        </div>
                        
                        <h2>Speed Analysis</h2>
                        <div class="plot">
                            <img src="{os.path.basename(speed_plot)}" alt="Glove Speed">
                            <p>The plot above shows the speed of the glove over time. The red line represents the smoothed speed.</p>
                        </div>
                        
                        <h2>Acceleration Analysis</h2>
                        <div class="plot">
                            <img src="{os.path.basename(acceleration_plot)}" alt="Glove Acceleration">
                            <p>The plot above shows the acceleration of the glove over time. Positive values indicate acceleration, while negative values indicate deceleration.</p>
                        </div>
                        
                        <h2>Detailed Data</h2>
                        <table>
                            <tr>
                                <th>Frame</th>
                                <th>Time (s)</th>
                                <th>Speed (mph)</th>
                                <th>Acceleration (m/s²)</th>
                            </tr>
                """)
                
                # Add rows for each frame (limit to every 5th frame to keep the report manageable)
                for i, row in self.acceleration_data.iloc[::5].iterrows():
                    if i in self.acceleration_data.index:
                        f.write(f"""
                            <tr>
                                <td>{int(row['frame'])}</td>
                                <td>{row['time']:.2f}</td>
                                <td>{row['smooth_speed_mph']:.2f}</td>
                                <td>{row['smooth_acceleration_mps2']:.2f}</td>
                            </tr>
                        """)
                
                f.write("""
                        </table>
                    </div>
                </body>
                </html>
                """)
            
            logger.info(f"Speed report generated and saved to {output_path}")
            return output_path
        except Exception as e:
            logger.error(f"Error generating speed report: {str(e)}")
            return None
    
    def save_speed_data(self, output_path=None):
        """
        Save speed data to a CSV file.
        
        Args:
            output_path (str, optional): Path to save the data
            
        Returns:
            str: Path to the saved data or None if saving failed
        """
        if self.speed_data is None:
            logger.error("No speed data available")
            return None
        
        try:
            # Create output path if not provided
            if output_path is None:
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                output_path = os.path.join(OUTPUT_DIR, f"speed_data_{timestamp}.csv")
            
            # Save to CSV
            self.speed_data.to_csv(output_path, index=False)
            
            logger.info(f"Speed data saved to {output_path}")
            return output_path
        except Exception as e:
            logger.error(f"Error saving speed data: {str(e)}")
            return None

def test_speed_calculator():
    """
    Test function for the SpeedCalculator class.
    """
    import glob
    
    # Find the most recent tracking data file
    tracking_files = glob.glob(os.path.join(OUTPUT_DIR, "tracking_data_*.csv"))
    if not tracking_files:
        logger.error("No tracking data files found")
        return False
    
    # Sort by modification time (newest first)
    tracking_file = sorted(tracking_files, key=os.path.getmtime, reverse=True)[0]
    
    logger.info(f"Testing SpeedCalculator with tracking data: {tracking_file}")
    
    # Initialize calculator with test FPS
    calculator = SpeedCalculator(fps=30, pixels_per_meter=100)
    
    # Load tracking data
    success = calculator.load_tracking_data(tracking_file)
    if not success:
        logger.error("Failed to load tracking data")
        return False
    
    # Calculate speed
    speed_data = calculator.calculate_speed(smoothing=True)
    if speed_data is None:
        logger.error("Failed to calculate speed")
        return False
    
    # Calculate acceleration
    acceleration_data = calculator.calculate_acceleration(smoothing=True)
    if acceleration_data is None:
        logger.error("Failed to calculate acceleration")
        return False
    
    # Get statistics
    max_speed = calculator.get_max_speed()
    avg_speed = calculator.get_average_speed()
    
    logger.info(f"Maximum speed: {max_speed[1]:.2f} mph at frame {max_speed[2]}")
    logger.info(f"Average speed: {avg_speed[1]:.2f} mph")
    
    # Generate plots
    trajectory_plot = calculator.plot_trajectory()
    speed_plot = calculator.plot_speed()
    acceleration_plot = calculator.plot_acceleration()
    
    # Generate report
    report_path = calculator.generate_speed_report()
    
    # Save data
    data_path = calculator.save_speed_data()
    
    logger.info("SpeedCalculator test completed successfully")
    return {
        "tracking_data": tracking_file,
        "max_speed_mph": max_speed[1],
        "avg_speed_mph": avg_speed[1],
        "trajectory_plot": trajectory_plot,
        "speed_plot": speed_plot,
        "acceleration_plot": acceleration_plot,
        "report": report_path,
        "data": data_path
    }

if __name__ == "__main__":
    test_results = test_speed_calculator()
    
    if test_results:
        print("\nTest Results:")
        print(f"Tracking data: {test_results['tracking_data']}")
        print(f"Maximum speed: {test_results['max_speed_mph']:.2f} mph")
        print(f"Average speed: {test_results['avg_speed_mph']:.2f} mph")
        print(f"Trajectory plot: {test_results['trajectory_plot']}")
        print(f"Speed plot: {test_results['speed_plot']}")
        print(f"Acceleration plot: {test_results['acceleration_plot']}")
        print(f"Report: {test_results['report']}")
        print(f"Data: {test_results['data']}")
