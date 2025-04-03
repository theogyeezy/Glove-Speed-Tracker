"""
Testing and optimization module for the Glove Speed Tracker application.
Provides functionality to test and optimize the application components.
"""

import os
import sys
import logging
import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
import cv2
import json
from datetime import datetime

# Import project modules
from config import DATA_DIR, OUTPUT_DIR
from video_processor import VideoProcessor, create_test_video
from detector import GloveDetector
from tracker import GloveTracker
from speed_calculator import SpeedCalculator
from data_analyzer import GloveDataAnalyzer
from visualizer import GloveVisualizer

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger('tester')

class PerformanceTester:
    """
    Class for testing and optimizing the performance of the Glove Speed Tracker application.
    """
    
    def __init__(self, data_dir=None, output_dir=None):
        """
        Initialize the performance tester.
        
        Args:
            data_dir (str, optional): Path to the data directory
            output_dir (str, optional): Path to the output directory
        """
        self.data_dir = data_dir if data_dir else DATA_DIR
        self.output_dir = output_dir if output_dir else OUTPUT_DIR
        
        # Create directories if they don't exist
        os.makedirs(self.data_dir, exist_ok=True)
        os.makedirs(self.output_dir, exist_ok=True)
        
        # Initialize test results
        self.test_results = {
            "timestamp": datetime.now().isoformat(),
            "video_processing": {},
            "detection": {},
            "tracking": {},
            "speed_calculation": {},
            "data_analysis": {},
            "visualization": {},
            "end_to_end": {}
        }
        
        logger.info(f"PerformanceTester initialized with data_dir: {self.data_dir}, output_dir: {self.output_dir}")
    
    def create_test_videos(self, num_videos=3, durations=[3, 5, 10], resolutions=[(640, 480), (1280, 720), (1920, 1080)]):
        """
        Create test videos with different durations and resolutions.
        
        Args:
            num_videos (int, optional): Number of test videos to create
            durations (list, optional): List of video durations in seconds
            resolutions (list, optional): List of video resolutions (width, height)
            
        Returns:
            list: Paths to the created test videos
        """
        logger.info(f"Creating {num_videos} test videos")
        
        video_paths = []
        
        for i in range(num_videos):
            # Select duration and resolution
            duration = durations[i % len(durations)]
            resolution = resolutions[i % len(resolutions)]
            
            # Create video path
            video_path = os.path.join(self.data_dir, f"test_video_{i+1}_{resolution[0]}x{resolution[1]}_{duration}s.mp4")
            
            # Create test video
            create_test_video(
                output_path=video_path,
                width=resolution[0],
                height=resolution[1],
                duration=duration,
                fps=30,
                moving_object=True,
                object_size=(60, 60)
            )
            
            video_paths.append(video_path)
            logger.info(f"Created test video: {video_path}")
        
        return video_paths
    
    def test_video_processing(self, video_paths=None):
        """
        Test the video processing module.
        
        Args:
            video_paths (list, optional): Paths to test videos
            
        Returns:
            dict: Test results
        """
        logger.info("Testing video processing module")
        
        # Create test videos if not provided
        if not video_paths:
            video_paths = self.create_test_videos()
        
        results = {
            "videos": [],
            "total_processing_time": 0,
            "average_fps": 0,
            "memory_usage": 0
        }
        
        total_frames = 0
        total_time = 0
        
        for video_path in video_paths:
            video_result = {
                "path": video_path,
                "processing_time": 0,
                "fps": 0,
                "frame_count": 0,
                "resolution": ""
            }
            
            # Initialize video processor
            processor = VideoProcessor()
            
            # Load video
            start_time = time.time()
            processor.load_video(video_path)
            load_time = time.time() - start_time
            
            # Get video properties
            frame_count = processor.get_frame_count()
            fps = processor.get_fps()
            width = processor.get_frame_width()
            height = processor.get_frame_height()
            
            video_result["frame_count"] = frame_count
            video_result["resolution"] = f"{width}x{height}"
            
            # Process all frames
            start_time = time.time()
            frames = []
            
            while True:
                success, frame = processor.read_frame()
                if not success:
                    break
                frames.append(frame)
            
            processing_time = time.time() - start_time
            
            video_result["processing_time"] = processing_time + load_time
            video_result["fps"] = frame_count / processing_time if processing_time > 0 else 0
            
            results["videos"].append(video_result)
            
            total_frames += frame_count
            total_time += processing_time
            
            logger.info(f"Processed video {video_path}: {frame_count} frames in {processing_time:.2f}s ({video_result['fps']:.2f} fps)")
        
        # Calculate overall results
        results["total_processing_time"] = total_time
        results["average_fps"] = total_frames / total_time if total_time > 0 else 0
        
        # Estimate memory usage (rough approximation)
        # Assuming 3 channels (RGB), 8 bits per channel
        avg_width = sum(int(v["resolution"].split("x")[0]) for v in results["videos"]) / len(results["videos"])
        avg_height = sum(int(v["resolution"].split("x")[1]) for v in results["videos"]) / len(results["videos"])
        avg_frame_size = avg_width * avg_height * 3  # bytes
        results["memory_usage"] = avg_frame_size * 10  # Assuming 10 frames in memory at once
        
        logger.info(f"Video processing test completed: {total_frames} total frames, {results['average_fps']:.2f} average fps")
        
        # Store results
        self.test_results["video_processing"] = results
        
        return results
    
    def test_detection(self, video_paths=None, num_frames=100):
        """
        Test the glove detection module.
        
        Args:
            video_paths (list, optional): Paths to test videos
            num_frames (int, optional): Number of frames to test per video
            
        Returns:
            dict: Test results
        """
        logger.info("Testing glove detection module")
        
        # Create test videos if not provided
        if not video_paths:
            video_paths = self.create_test_videos(num_videos=1)
        
        results = {
            "videos": [],
            "total_detection_time": 0,
            "average_detection_time": 0,
            "detection_rate": 0
        }
        
        total_frames = 0
        total_detections = 0
        total_time = 0
        
        for video_path in video_paths:
            video_result = {
                "path": video_path,
                "frames_tested": 0,
                "detections_found": 0,
                "detection_rate": 0,
                "average_detection_time": 0
            }
            
            # Initialize video processor and detector
            processor = VideoProcessor()
            detector = GloveDetector()
            
            # Load video
            processor.load_video(video_path)
            
            # Process frames
            frames_tested = 0
            detections_found = 0
            detection_times = []
            
            while frames_tested < num_frames:
                success, frame = processor.read_frame()
                if not success:
                    break
                
                # Run detection
                start_time = time.time()
                detections = detector.detect(frame)
                detection_time = time.time() - start_time
                
                frames_tested += 1
                if detections:
                    detections_found += 1
                
                detection_times.append(detection_time)
            
            video_result["frames_tested"] = frames_tested
            video_result["detections_found"] = detections_found
            video_result["detection_rate"] = detections_found / frames_tested if frames_tested > 0 else 0
            video_result["average_detection_time"] = sum(detection_times) / len(detection_times) if detection_times else 0
            
            results["videos"].append(video_result)
            
            total_frames += frames_tested
            total_detections += detections_found
            total_time += sum(detection_times)
            
            logger.info(f"Tested detection on {video_path}: {detections_found}/{frames_tested} detections ({video_result['detection_rate']*100:.2f}%), {video_result['average_detection_time']*1000:.2f} ms/frame")
        
        # Calculate overall results
        results["total_detection_time"] = total_time
        results["average_detection_time"] = total_time / total_frames if total_frames > 0 else 0
        results["detection_rate"] = total_detections / total_frames if total_frames > 0 else 0
        
        logger.info(f"Detection test completed: {total_detections}/{total_frames} detections ({results['detection_rate']*100:.2f}%), {results['average_detection_time']*1000:.2f} ms/frame")
        
        # Store results
        self.test_results["detection"] = results
        
        return results
    
    def test_tracking(self, video_paths=None):
        """
        Test the glove tracking module.
        
        Args:
            video_paths (list, optional): Paths to test videos
            
        Returns:
            dict: Test results
        """
        logger.info("Testing glove tracking module")
        
        # Create test videos if not provided
        if not video_paths:
            video_paths = self.create_test_videos(num_videos=1)
        
        results = {
            "videos": [],
            "total_tracking_time": 0,
            "average_tracking_time": 0,
            "tracking_success_rate": 0
        }
        
        total_frames = 0
        total_successful_tracks = 0
        total_time = 0
        
        for video_path in video_paths:
            video_result = {
                "path": video_path,
                "frames_processed": 0,
                "successful_tracks": 0,
                "tracking_success_rate": 0,
                "average_tracking_time": 0,
                "tracking_data": []
            }
            
            # Initialize video processor, detector, and tracker
            processor = VideoProcessor()
            detector = GloveDetector()
            tracker = GloveTracker()
            
            # Load video
            processor.load_video(video_path)
            
            # Process frames
            frame_idx = 0
            bbox = None
            tracking_times = []
            tracking_data = []
            
            while True:
                success, frame = processor.read_frame()
                if not success:
                    break
                
                # Detect or track
                if frame_idx == 0 or bbox is None:
                    # Initial detection
                    detections = detector.detect(frame)
                    if detections:
                        bbox = detections[0]
                        tracker.init(frame, bbox)
                        
                        # Record tracking data
                        x, y, w, h = bbox
                        tracking_data.append({
                            'frame': frame_idx,
                            'x': x,
                            'y': y,
                            'width': w,
                            'height': h,
                            'center_x': x + w/2,
                            'center_y': y + h/2,
                            'detection': True,
                            'tracking': False
                        })
                else:
                    # Track
                    start_time = time.time()
                    success, bbox = tracker.update(frame)
                    tracking_time = time.time() - start_time
                    
                    tracking_times.append(tracking_time)
                    
                    if success:
                        total_successful_tracks += 1
                        video_result["successful_tracks"] += 1
                        
                        # Record tracking data
                        x, y, w, h = bbox
                        tracking_data.append({
                            'frame': frame_idx,
                            'x': x,
                            'y': y,
                            'width': w,
                            'height': h,
                            'center_x': x + w/2,
                            'center_y': y + h/2,
                            'detection': False,
                            'tracking': True
                        })
                    else:
                        # Tracking failed, try detection
                        detections = detector.detect(frame)
                        if detections:
                            bbox = detections[0]
                            tracker.init(frame, bbox)
                            
                            # Record tracking data
                            x, y, w, h = bbox
                            tracking_data.append({
                                'frame': frame_idx,
                                'x': x,
                                'y': y,
                                'width': w,
                                'height': h,
                                'center_x': x + w/2,
                                'center_y': y + h/2,
                                'detection': True,
                                'tracking': False
                            })
                        else:
                            bbox = None
                
                frame_idx += 1
            
            video_result["frames_processed"] = frame_idx
            video_result["tracking_success_rate"] = video_result["successful_tracks"] / (frame_idx - 1) if frame_idx > 1 else 0
            video_result["average_tracking_time"] = sum(tracking_times) / len(tracking_times) if tracking_times else 0
            video_result["tracking_data"] = tracking_data
            
            # Save tracking data
            tracking_df = pd.DataFrame(tracking_data)
            tracking_csv_path = os.path.join(self.output_dir, f"tracking_test_{Path(video_path).stem}.csv")
            tracking_df.to_csv(tracking_csv_path, index=False)
            
            results["videos"].append(video_result)
            
            total_frames += frame_idx
            total_time += sum(tracking_times)
            
            logger.info(f"Tested tracking on {video_path}: {video_result['successful_tracks']}/{frame_idx-1} successful tracks ({video_result['tracking_success_rate']*100:.2f}%), {video_result['average_tracking_time']*1000:.2f} ms/frame")
        
        # Calculate overall results
        results["total_tracking_time"] = total_time
        results["average_tracking_time"] = total_time / (total_frames - len(video_paths)) if total_frames > len(video_paths) else 0
        results["tracking_success_rate"] = total_successful_tracks / (total_frames - len(video_paths)) if total_frames > len(video_paths) else 0
        
        logger.info(f"Tracking test completed: {total_successful_tracks}/{total_frames-len(video_paths)} successful tracks ({results['tracking_success_rate']*100:.2f}%), {results['average_tracking_time']*1000:.2f} ms/frame")
        
        # Store results
        self.test_results["tracking"] = results
        
        return results
    
    def test_speed_calculation(self, tracking_data=None):
        """
        Test the speed calculation module.
        
        Args:
            tracking_data (DataFrame, optional): Tracking data
            
        Returns:
            dict: Test results
        """
        logger.info("Testing speed calculation module")
        
        # Use tracking data from tracking test if not provided
        if tracking_data is None:
            # Check if tracking test has been run
            if not self.test_results["tracking"] or not self.test_results["tracking"]["videos"]:
                logger.warning("No tracking data available. Running tracking test first.")
                self.test_tracking()
            
            # Get tracking data from the first video
            tracking_data = pd.DataFrame(self.test_results["tracking"]["videos"][0]["tracking_data"])
        
        results = {
            "calculation_time": 0,
            "frames_processed": 0,
            "average_calculation_time": 0,
            "max_speed": 0,
            "avg_speed": 0,
            "speed_data": None
        }
        
        # Initialize speed calculator
        calculator = SpeedCalculator(fps=30)
        
        # Calculate speed
        start_time = time.time()
        calculator.set_tracking_data(tracking_data)
        speed_data = calculator.calculate_speed(smoothing=True)
        calculator.calculate_acceleration(smoothing=True)
        calculation_time = time.time() - start_time
        
        # Get statistics
        max_speed = calculator.get_max_speed()
        avg_speed = calculator.get_average_speed()
        
        # Save speed data
        speed_csv_path = os.path.join(self.output_dir, f"speed_test_data.csv")
        calculator.save_speed_data(speed_csv_path)
        
        # Store results
        results["calculation_time"] = calculation_time
        results["frames_processed"] = len(tracking_data)
        results["average_calculation_time"] = calculation_time / len(tracking_data) if len(tracking_data) > 0 else 0
        results["max_speed"] = {
            "mps": max_speed[0],
            "mph": max_speed[1],
            "frame": int(max_speed[2])
        }
        results["avg_speed"] = {
            "mps": avg_speed[0],
            "mph": avg_speed[1]
        }
        results["speed_data"] = speed_csv_path
        
        logger.info(f"Speed calculation test completed: {len(tracking_data)} frames in {calculation_time:.4f}s ({results['average_calculation_time']*1000:.2f} ms/frame)")
        logger.info(f"Max speed: {max_speed[0]:.2f} m/s ({max_speed[1]:.2f} mph) at frame {int(max_speed[2])}")
        logger.info(f"Avg speed: {avg_speed[0]:.2f} m/s ({avg_speed[1]:.2f} mph)")
        
        # Store results
        self.test_results["speed_calculation"] = results
        
        return results
    
    def test_data_analysis(self, speed_data=None):
        """
        Test the data analysis module.
        
        Args:
            speed_data (DataFrame, optional): Speed data
            
        Returns:
            dict: Test results
        """
        logger.info("Testing data analysis module")
        
        # Use speed data from speed calculation test if not provided
        if speed_data is None:
            # Check if speed calculation test has been run
            if not self.test_results["speed_calculation"] or not self.test_results["speed_calculation"]["speed_data"]:
                logger.warning("No speed data available. Running speed calculation test first.")
                self.test_speed_calculation()
            
            # Load speed data
            speed_data = pd.read_csv(self.test_results["speed_calculation"]["speed_data"])
        
        results = {
            "analysis_time": 0,
            "feature_extraction_time": 0,
            "clustering_time": 0,
            "pca_time": 0,
            "model_training_time": 0,
            "model_score": 0,
            "num_features": 0,
            "num_clusters": 0,
            "pca_variance_explained": 0
        }
        
        # Initialize data analyzer
        analyzer = GloveDataAnalyzer()
        
        # Set speed data
        analyzer.set_speed_data(speed_data)
        
        # Extract features
        start_time = time.time()
        analyzer.extract_features()
        feature_extraction_time = time.time() - start_time
        
        # Cluster movements
        start_time = time.time()
        analyzer.cluster_movements()
        clustering_time = time.time() - start_time
        
        # Perform PCA
        start_time = time.time()
        analyzer.perform_pca()
        pca_time = time.time() - start_time
        
        # Train model
        start_time = time.time()
        analyzer.train_speed_prediction_model()
        model_training_time = time.time() - start_time
        
        # Generate report
        report_path = os.path.join(self.output_dir, f"analysis_test_report.html")
        analyzer.generate_analysis_report(report_path)
        
        # Store results
        results["analysis_time"] = feature_extraction_time + clustering_time + pca_time + model_training_time
        results["feature_extraction_time"] = feature_extraction_time
        results["clustering_time"] = clustering_time
        results["pca_time"] = pca_time
        results["model_training_time"] = model_training_time
        results["model_score"] = analyzer.model_score if hasattr(analyzer, 'model_score') else 0
        results["num_features"] = len(analyzer.features) if hasattr(analyzer, 'features') else 0
        results["num_clusters"] = analyzer.n_clusters if hasattr(analyzer, 'n_clusters') else 0
        results["pca_variance_explained"] = analyzer.pca_variance_explained if hasattr(analyzer, 'pca_variance_explained') else 0
        results["report_path"] = report_path
        
        logger.info(f"Data analysis test completed in {results['analysis_time']:.4f}s")
        logger.info(f"Feature extraction: {feature_extraction_time:.4f}s, {results['num_features']} features")
        logger.info(f"Clustering: {clustering_time:.4f}s, {results['num_clusters']} clusters")
        logger.info(f"PCA: {pca_time:.4f}s, {results['pca_variance_explained']*100:.2f}% variance explained")
        logger.info(f"Model training: {model_training_time:.4f}s, R² score: {results['model_score']:.4f}")
        
        # Store results
        self.test_results["data_analysis"] = results
        
        return results
    
    def test_visualization(self, tracking_data=None, speed_data=None):
        """
        Test the visualization module.
        
        Args:
            tracking_data (DataFrame, optional): Tracking data
            speed_data (DataFrame, optional): Speed data
            
        Returns:
            dict: Test results
        """
        logger.info("Testing visualization module")
        
        # Use tracking data from tracking test if not provided
        if tracking_data is None:
            # Check if tracking test has been run
            if not self.test_results["tracking"] or not self.test_results["tracking"]["videos"]:
                logger.warning("No tracking data available. Running tracking test first.")
                self.test_tracking()
            
            # Get tracking data from the first video
            tracking_data = pd.DataFrame(self.test_results["tracking"]["videos"][0]["tracking_data"])
        
        # Use speed data from speed calculation test if not provided
        if speed_data is None:
            # Check if speed calculation test has been run
            if not self.test_results["speed_calculation"] or not self.test_results["speed_calculation"]["speed_data"]:
                logger.warning("No speed data available. Running speed calculation test first.")
                self.test_speed_calculation()
            
            # Load speed data
            speed_data = pd.read_csv(self.test_results["speed_calculation"]["speed_data"])
        
        results = {
            "visualization_time": 0,
            "frames_processed": 0,
            "average_visualization_time": 0,
            "video_creation_time": 0,
            "report_creation_time": 0,
            "visualization_video": None,
            "interactive_report": None
        }
        
        # Initialize visualizer
        visualizer = GloveVisualizer(fps=30)
        
        # Set frame size
        if 'center_x' in tracking_data.columns and 'center_y' in tracking_data.columns:
            max_x = tracking_data['center_x'].max() + 100  # Add margin
            max_y = tracking_data['center_y'].max() + 100
            visualizer.set_frame_size(int(max_x), int(max_y))
        
        # Set tracking and speed data
        visualizer.set_tracking_data(tracking_data)
        visualizer.set_speed_data(speed_data)
        
        # Create sample visualization frame
        middle_frame = int((tracking_data['frame'].min() + tracking_data['frame'].max()) / 2)
        
        start_time = time.time()
        sample_frame = visualizer.create_visualization_frame(middle_frame)
        frame_time = time.time() - start_time
        
        # Save sample frame
        if sample_frame is not None:
            sample_path = os.path.join(self.output_dir, f"visualization_test_sample.png")
            cv2.imwrite(sample_path, sample_frame)
        
        # Create visualization video
        start_time = time.time()
        video_path = os.path.join(self.output_dir, f"visualization_test_video.mp4")
        visualizer.create_visualization_video(video_path)
        video_time = time.time() - start_time
        
        # Create interactive report
        start_time = time.time()
        report_path = os.path.join(self.output_dir, f"visualization_test_report.html")
        visualizer.create_interactive_report(report_path)
        report_time = time.time() - start_time
        
        # Store results
        results["visualization_time"] = frame_time
        results["frames_processed"] = 1
        results["average_visualization_time"] = frame_time
        results["video_creation_time"] = video_time
        results["report_creation_time"] = report_time
        results["visualization_video"] = video_path
        results["interactive_report"] = report_path
        
        logger.info(f"Visualization test completed")
        logger.info(f"Single frame visualization: {frame_time*1000:.2f} ms")
        logger.info(f"Video creation: {video_time:.2f}s")
        logger.info(f"Report creation: {report_time:.2f}s")
        
        # Store results
        self.test_results["visualization"] = results
        
        return results
    
    def test_end_to_end(self, video_path=None):
        """
        Test the end-to-end pipeline.
        
        Args:
            video_path (str, optional): Path to test video
            
        Returns:
            dict: Test results
        """
        logger.info("Testing end-to-end pipeline")
        
        # Create test video if not provided
        if not video_path:
            video_paths = self.create_test_videos(num_videos=1)
            video_path = video_paths[0]
        
        results = {
            "video_path": video_path,
            "total_processing_time": 0,
            "video_processing_time": 0,
            "detection_tracking_time": 0,
            "speed_calculation_time": 0,
            "data_analysis_time": 0,
            "visualization_time": 0,
            "frames_processed": 0,
            "fps": 0,
            "tracking_success_rate": 0,
            "max_speed": 0,
            "avg_speed": 0,
            "visualization_video": None,
            "interactive_report": None
        }
        
        total_start_time = time.time()
        
        # Initialize components
        processor = VideoProcessor()
        detector = GloveDetector()
        tracker = GloveTracker()
        
        # Load video
        video_start_time = time.time()
        processor.load_video(video_path)
        fps = processor.get_fps()
        frame_count = processor.get_frame_count()
        frame_width = processor.get_frame_width()
        frame_height = processor.get_frame_height()
        video_load_time = time.time() - video_start_time
        
        # Initialize tracking data storage
        tracking_data = []
        
        # Process video frames
        detection_tracking_start_time = time.time()
        frame_idx = 0
        bbox = None
        successful_tracks = 0
        
        while True:
            # Read frame
            success, frame = processor.read_frame()
            if not success:
                break
            
            # Detect glove in first frame or if tracking is lost
            if frame_idx == 0 or bbox is None:
                detections = detector.detect(frame)
                if detections:
                    bbox = detections[0]
                    tracker.init(frame, bbox)
            else:
                # Track glove
                success, bbox = tracker.update(frame)
                if success:
                    successful_tracks += 1
                else:
                    # If tracking fails, try detection again
                    detections = detector.detect(frame)
                    if detections:
                        bbox = detections[0]
                        tracker.init(frame, bbox)
            
            # Store tracking data if bbox is valid
            if bbox is not None:
                x, y, w, h = bbox
                center_x = x + w/2
                center_y = y + h/2
                tracking_data.append({
                    'frame': frame_idx,
                    'x': x,
                    'y': y,
                    'width': w,
                    'height': h,
                    'center_x': center_x,
                    'center_y': center_y
                })
            
            frame_idx += 1
        
        detection_tracking_time = time.time() - detection_tracking_start_time
        
        # Convert tracking data to DataFrame
        tracking_df = pd.DataFrame(tracking_data)
        
        # Save tracking data
        tracking_csv_path = os.path.join(self.output_dir, f"e2e_tracking_data.csv")
        tracking_df.to_csv(tracking_csv_path, index=False)
        
        # Calculate speed
        speed_start_time = time.time()
        calculator = SpeedCalculator(fps=fps)
        calculator.set_tracking_data(tracking_df)
        speed_data = calculator.calculate_speed(smoothing=True)
        calculator.calculate_acceleration(smoothing=True)
        
        # Get statistics
        max_speed = calculator.get_max_speed()
        avg_speed = calculator.get_average_speed()
        
        # Save speed data
        speed_csv_path = os.path.join(self.output_dir, f"e2e_speed_data.csv")
        calculator.save_speed_data(speed_csv_path)
        speed_calculation_time = time.time() - speed_start_time
        
        # Perform data analysis
        analysis_start_time = time.time()
        analyzer = GloveDataAnalyzer()
        analyzer.set_speed_data(speed_data)
        analyzer.extract_features()
        analyzer.cluster_movements()
        analyzer.perform_pca()
        analyzer.train_speed_prediction_model()
        
        # Generate analysis report
        analysis_report_path = os.path.join(self.output_dir, f"e2e_analysis_report.html")
        analyzer.generate_analysis_report(analysis_report_path)
        data_analysis_time = time.time() - analysis_start_time
        
        # Create visualization
        visualization_start_time = time.time()
        visualizer = GloveVisualizer(fps=fps)
        visualizer.set_frame_size(frame_width, frame_height)
        visualizer.set_tracking_data(tracking_df)
        visualizer.set_speed_data(speed_data)
        
        # Generate visualization video
        vis_video_path = os.path.join(self.output_dir, f"e2e_visualization.mp4")
        visualizer.create_visualization_video(vis_video_path)
        
        # Generate interactive report
        interactive_report_path = os.path.join(self.output_dir, f"e2e_interactive_report.html")
        visualizer.create_interactive_report(interactive_report_path)
        visualization_time = time.time() - visualization_start_time
        
        total_time = time.time() - total_start_time
        
        # Store results
        results["total_processing_time"] = total_time
        results["video_processing_time"] = video_load_time
        results["detection_tracking_time"] = detection_tracking_time
        results["speed_calculation_time"] = speed_calculation_time
        results["data_analysis_time"] = data_analysis_time
        results["visualization_time"] = visualization_time
        results["frames_processed"] = frame_idx
        results["fps"] = frame_idx / total_time if total_time > 0 else 0
        results["tracking_success_rate"] = successful_tracks / (frame_idx - 1) if frame_idx > 1 else 0
        results["max_speed"] = {
            "mps": max_speed[0],
            "mph": max_speed[1],
            "frame": int(max_speed[2])
        }
        results["avg_speed"] = {
            "mps": avg_speed[0],
            "mph": avg_speed[1]
        }
        results["visualization_video"] = vis_video_path
        results["interactive_report"] = interactive_report_path
        
        logger.info(f"End-to-end test completed in {total_time:.2f}s ({results['fps']:.2f} fps)")
        logger.info(f"Video processing: {video_load_time:.2f}s")
        logger.info(f"Detection & tracking: {detection_tracking_time:.2f}s, success rate: {results['tracking_success_rate']*100:.2f}%")
        logger.info(f"Speed calculation: {speed_calculation_time:.2f}s")
        logger.info(f"Data analysis: {data_analysis_time:.2f}s")
        logger.info(f"Visualization: {visualization_time:.2f}s")
        
        # Store results
        self.test_results["end_to_end"] = results
        
        return results
    
    def run_all_tests(self):
        """
        Run all performance tests.
        
        Returns:
            dict: All test results
        """
        logger.info("Running all performance tests")
        
        # Create test videos
        video_paths = self.create_test_videos()
        
        # Run individual tests
        self.test_video_processing(video_paths)
        self.test_detection(video_paths)
        self.test_tracking(video_paths[:1])  # Use only first video for tracking
        self.test_speed_calculation()
        self.test_data_analysis()
        self.test_visualization()
        
        # Run end-to-end test
        self.test_end_to_end(video_paths[0])
        
        # Save all results
        self.save_results()
        
        return self.test_results
    
    def save_results(self, output_path=None):
        """
        Save test results to a JSON file.
        
        Args:
            output_path (str, optional): Path to save the results
            
        Returns:
            str: Path to the saved results
        """
        if output_path is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_path = os.path.join(self.output_dir, f"performance_test_results_{timestamp}.json")
        
        # Convert results to JSON-serializable format
        serializable_results = self.test_results.copy()
        
        # Remove non-serializable objects
        for section in serializable_results:
            if isinstance(serializable_results[section], dict):
                for key, value in list(serializable_results[section].items()):
                    if isinstance(value, pd.DataFrame):
                        serializable_results[section][key] = value.to_dict(orient='records')
        
        # Save results
        with open(output_path, 'w') as f:
            json.dump(serializable_results, f, indent=2)
        
        logger.info(f"Test results saved to {output_path}")
        
        return output_path
    
    def generate_performance_report(self, output_path=None):
        """
        Generate a performance report.
        
        Args:
            output_path (str, optional): Path to save the report
            
        Returns:
            str: Path to the saved report
        """
        if output_path is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_path = os.path.join(self.output_dir, f"performance_report_{timestamp}.html")
        
        # Check if tests have been run
        if not self.test_results["end_to_end"]:
            logger.warning("No end-to-end test results available. Running all tests first.")
            self.run_all_tests()
        
        # Create HTML report
        with open(output_path, 'w') as f:
            f.write(f"""
            <!DOCTYPE html>
            <html>
            <head>
                <title>Glove Speed Tracker - Performance Report</title>
                <style>
                    body {{ font-family: Arial, sans-serif; margin: 20px; }}
                    h1, h2, h3 {{ color: #2c3e50; }}
                    .container {{ max-width: 1200px; margin: 0 auto; }}
                    .section {{ background-color: #f8f9fa; padding: 15px; border-radius: 5px; margin-bottom: 20px; }}
                    .stats {{ display: flex; flex-wrap: wrap; }}
                    .stat-box {{ flex: 1; min-width: 200px; margin: 10px; padding: 15px; background-color: #e9ecef; border-radius: 5px; text-align: center; }}
                    .stat-value {{ font-size: 24px; font-weight: bold; margin: 10px 0; }}
                    table {{ border-collapse: collapse; width: 100%; }}
                    th, td {{ border: 1px solid #ddd; padding: 8px; text-align: left; }}
                    th {{ background-color: #f2f2f2; }}
                    tr:nth-child(even) {{ background-color: #f9f9f9; }}
                    .chart-container {{ height: 400px; margin-bottom: 30px; }}
                </style>
                <script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
            </head>
            <body>
                <div class="container">
                    <h1>Glove Speed Tracker - Performance Report</h1>
                    <p>Generated on {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}</p>
                    
                    <div class="section">
                        <h2>End-to-End Performance</h2>
                        <div class="stats">
                            <div class="stat-box">
                                <h3>Total Processing Time</h3>
                                <div class="stat-value">{self.test_results["end_to_end"]["total_processing_time"]:.2f}s</div>
                                <p>For {self.test_results["end_to_end"]["frames_processed"]} frames</p>
                            </div>
                            <div class="stat-box">
                                <h3>Processing Speed</h3>
                                <div class="stat-value">{self.test_results["end_to_end"]["fps"]:.2f} fps</div>
                                <p>Frames per second</p>
                            </div>
                            <div class="stat-box">
                                <h3>Tracking Success Rate</h3>
                                <div class="stat-value">{self.test_results["end_to_end"]["tracking_success_rate"]*100:.1f}%</div>
                                <p>Successful tracking</p>
                            </div>
                        </div>
                        
                        <h3>Processing Time Breakdown</h3>
                        <div class="chart-container">
                            <canvas id="timeBreakdownChart"></canvas>
                        </div>
                        
                        <table>
                            <tr>
                                <th>Component</th>
                                <th>Time (s)</th>
                                <th>Percentage</th>
                            </tr>
                            <tr>
                                <td>Video Processing</td>
                                <td>{self.test_results["end_to_end"]["video_processing_time"]:.2f}</td>
                                <td>{self.test_results["end_to_end"]["video_processing_time"]/self.test_results["end_to_end"]["total_processing_time"]*100:.1f}%</td>
                            </tr>
                            <tr>
                                <td>Detection & Tracking</td>
                                <td>{self.test_results["end_to_end"]["detection_tracking_time"]:.2f}</td>
                                <td>{self.test_results["end_to_end"]["detection_tracking_time"]/self.test_results["end_to_end"]["total_processing_time"]*100:.1f}%</td>
                            </tr>
                            <tr>
                                <td>Speed Calculation</td>
                                <td>{self.test_results["end_to_end"]["speed_calculation_time"]:.2f}</td>
                                <td>{self.test_results["end_to_end"]["speed_calculation_time"]/self.test_results["end_to_end"]["total_processing_time"]*100:.1f}%</td>
                            </tr>
                            <tr>
                                <td>Data Analysis</td>
                                <td>{self.test_results["end_to_end"]["data_analysis_time"]:.2f}</td>
                                <td>{self.test_results["end_to_end"]["data_analysis_time"]/self.test_results["end_to_end"]["total_processing_time"]*100:.1f}%</td>
                            </tr>
                            <tr>
                                <td>Visualization</td>
                                <td>{self.test_results["end_to_end"]["visualization_time"]:.2f}</td>
                                <td>{self.test_results["end_to_end"]["visualization_time"]/self.test_results["end_to_end"]["total_processing_time"]*100:.1f}%</td>
                            </tr>
                        </table>
                    </div>
                    
                    <div class="section">
                        <h2>Component Performance</h2>
                        
                        <h3>Video Processing</h3>
                        <p>Average processing speed: {self.test_results["video_processing"]["average_fps"]:.2f} fps</p>
                        <table>
                            <tr>
                                <th>Video</th>
                                <th>Resolution</th>
                                <th>Frames</th>
                                <th>Processing Time (s)</th>
                                <th>FPS</th>
                            </tr>
            """)
            
            # Add video processing results
            for video in self.test_results["video_processing"]["videos"]:
                f.write(f"""
                            <tr>
                                <td>{os.path.basename(video["path"])}</td>
                                <td>{video["resolution"]}</td>
                                <td>{video["frame_count"]}</td>
                                <td>{video["processing_time"]:.2f}</td>
                                <td>{video["fps"]:.2f}</td>
                            </tr>
                """)
            
            f.write(f"""
                        </table>
                        
                        <h3>Detection</h3>
                        <p>Average detection time: {self.test_results["detection"]["average_detection_time"]*1000:.2f} ms/frame</p>
                        <p>Detection rate: {self.test_results["detection"]["detection_rate"]*100:.1f}%</p>
                        
                        <h3>Tracking</h3>
                        <p>Average tracking time: {self.test_results["tracking"]["average_tracking_time"]*1000:.2f} ms/frame</p>
                        <p>Tracking success rate: {self.test_results["tracking"]["tracking_success_rate"]*100:.1f}%</p>
                        
                        <h3>Speed Calculation</h3>
                        <p>Calculation time: {self.test_results["speed_calculation"]["calculation_time"]:.4f}s for {self.test_results["speed_calculation"]["frames_processed"]} frames</p>
                        <p>Average calculation time: {self.test_results["speed_calculation"]["average_calculation_time"]*1000:.2f} ms/frame</p>
                        
                        <h3>Data Analysis</h3>
                        <p>Total analysis time: {self.test_results["data_analysis"]["analysis_time"]:.4f}s</p>
                        <table>
                            <tr>
                                <th>Component</th>
                                <th>Time (s)</th>
                                <th>Details</th>
                            </tr>
                            <tr>
                                <td>Feature Extraction</td>
                                <td>{self.test_results["data_analysis"]["feature_extraction_time"]:.4f}</td>
                                <td>{self.test_results["data_analysis"]["num_features"]} features</td>
                            </tr>
                            <tr>
                                <td>Clustering</td>
                                <td>{self.test_results["data_analysis"]["clustering_time"]:.4f}</td>
                                <td>{self.test_results["data_analysis"]["num_clusters"]} clusters</td>
                            </tr>
                            <tr>
                                <td>PCA</td>
                                <td>{self.test_results["data_analysis"]["pca_time"]:.4f}</td>
                                <td>{self.test_results["data_analysis"]["pca_variance_explained"]*100:.1f}% variance explained</td>
                            </tr>
                            <tr>
                                <td>Model Training</td>
                                <td>{self.test_results["data_analysis"]["model_training_time"]:.4f}</td>
                                <td>R² score: {self.test_results["data_analysis"]["model_score"]:.4f}</td>
                            </tr>
                        </table>
                        
                        <h3>Visualization</h3>
                        <p>Single frame visualization time: {self.test_results["visualization"]["average_visualization_time"]*1000:.2f} ms</p>
                        <p>Video creation time: {self.test_results["visualization"]["video_creation_time"]:.2f}s</p>
                        <p>Report creation time: {self.test_results["visualization"]["report_creation_time"]:.2f}s</p>
                    </div>
                    
                    <div class="section">
                        <h2>Optimization Recommendations</h2>
                        <ul>
            """)
            
            # Add optimization recommendations based on test results
            recommendations = []
            
            # Check detection performance
            if self.test_results["detection"]["average_detection_time"] > 0.05:  # More than 50ms per frame
                recommendations.append("Consider using a lighter detection model or optimizing the current one for faster inference.")
            
            # Check tracking performance
            if self.test_results["tracking"]["tracking_success_rate"] < 0.8:  # Less than 80% success rate
                recommendations.append("Improve tracking algorithm or parameters to increase tracking success rate.")
            
            # Check end-to-end performance
            if self.test_results["end_to_end"]["fps"] < 15:  # Less than 15 fps
                recommendations.append("Overall processing speed is below real-time (15 fps). Consider optimizing the most time-consuming components.")
            
            # Check component time distribution
            total_time = self.test_results["end_to_end"]["total_processing_time"]
            if self.test_results["end_to_end"]["detection_tracking_time"] / total_time > 0.5:  # More than 50% of time
                recommendations.append("Detection and tracking are taking more than 50% of processing time. Consider optimizing these components first.")
            
            if self.test_results["end_to_end"]["visualization_time"] / total_time > 0.3:  # More than 30% of time
                recommendations.append("Visualization is taking more than 30% of processing time. Consider optimizing rendering or reducing visual elements.")
            
            # Add recommendations to report
            if recommendations:
                for recommendation in recommendations:
                    f.write(f"<li>{recommendation}</li>")
            else:
                f.write("<li>No specific optimization recommendations. The application is performing well.</li>")
            
            f.write(f"""
                        </ul>
                    </div>
                </div>
                
                <script>
                    // Create time breakdown chart
                    var ctxTime = document.getElementById('timeBreakdownChart').getContext('2d');
                    var timeBreakdownChart = new Chart(ctxTime, {{
                        type: 'bar',
                        data: {{
                            labels: ['Video Processing', 'Detection & Tracking', 'Speed Calculation', 'Data Analysis', 'Visualization'],
                            datasets: [{{
                                label: 'Processing Time (seconds)',
                                data: [
                                    {self.test_results["end_to_end"]["video_processing_time"]:.2f},
                                    {self.test_results["end_to_end"]["detection_tracking_time"]:.2f},
                                    {self.test_results["end_to_end"]["speed_calculation_time"]:.2f},
                                    {self.test_results["end_to_end"]["data_analysis_time"]:.2f},
                                    {self.test_results["end_to_end"]["visualization_time"]:.2f}
                                ],
                                backgroundColor: [
                                    'rgba(54, 162, 235, 0.5)',
                                    'rgba(255, 99, 132, 0.5)',
                                    'rgba(255, 206, 86, 0.5)',
                                    'rgba(75, 192, 192, 0.5)',
                                    'rgba(153, 102, 255, 0.5)'
                                ],
                                borderColor: [
                                    'rgba(54, 162, 235, 1)',
                                    'rgba(255, 99, 132, 1)',
                                    'rgba(255, 206, 86, 1)',
                                    'rgba(75, 192, 192, 1)',
                                    'rgba(153, 102, 255, 1)'
                                ],
                                borderWidth: 1
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
                                        text: 'Time (seconds)'
                                    }}
                                }}
                            }}
                        }}
                    }});
                </script>
            </body>
            </html>
            """)
        
        logger.info(f"Performance report generated and saved to {output_path}")
        return output_path

def optimize_application():
    """
    Optimize the application based on performance test results.
    
    Returns:
        dict: Optimization results
    """
    logger.info("Optimizing application")
    
    # Run performance tests
    tester = PerformanceTester()
    test_results = tester.run_all_tests()
    
    # Generate performance report
    report_path = tester.generate_performance_report()
    
    # Identify bottlenecks
    bottlenecks = []
    
    # Check component time distribution
    total_time = test_results["end_to_end"]["total_processing_time"]
    component_times = {
        "video_processing": test_results["end_to_end"]["video_processing_time"] / total_time,
        "detection_tracking": test_results["end_to_end"]["detection_tracking_time"] / total_time,
        "speed_calculation": test_results["end_to_end"]["speed_calculation_time"] / total_time,
        "data_analysis": test_results["end_to_end"]["data_analysis_time"] / total_time,
        "visualization": test_results["end_to_end"]["visualization_time"] / total_time
    }
    
    # Sort components by time percentage
    sorted_components = sorted(component_times.items(), key=lambda x: x[1], reverse=True)
    
    # Identify top bottlenecks
    for component, time_percentage in sorted_components[:2]:
        if time_percentage > 0.2:  # More than 20% of total time
            bottlenecks.append({
                "component": component,
                "time_percentage": time_percentage,
                "time_seconds": total_time * time_percentage
            })
    
    # Optimization strategies
    optimization_strategies = []
    
    for bottleneck in bottlenecks:
        if bottleneck["component"] == "detection_tracking":
            optimization_strategies.append({
                "component": "detection_tracking",
                "strategies": [
                    "Use a lighter detection model or optimize model parameters",
                    "Implement region of interest (ROI) tracking to reduce detection frequency",
                    "Use frame skipping for detection (e.g., detect every 5 frames)",
                    "Implement parallel processing for detection and tracking"
                ]
            })
        elif bottleneck["component"] == "visualization":
            optimization_strategies.append({
                "component": "visualization",
                "strategies": [
                    "Reduce rendering complexity by simplifying visualizations",
                    "Implement frame skipping for visualization (e.g., visualize every 2-3 frames)",
                    "Optimize drawing functions to reduce redundant operations",
                    "Pre-compute visualization elements where possible"
                ]
            })
        elif bottleneck["component"] == "data_analysis":
            optimization_strategies.append({
                "component": "data_analysis",
                "strategies": [
                    "Reduce feature extraction complexity",
                    "Implement incremental analysis instead of processing all data at once",
                    "Use simpler models or algorithms for clustering and prediction",
                    "Implement caching for intermediate results"
                ]
            })
    
    # Optimization results
    optimization_results = {
        "test_results": test_results,
        "report_path": report_path,
        "bottlenecks": bottlenecks,
        "optimization_strategies": optimization_strategies
    }
    
    logger.info(f"Optimization analysis completed")
    logger.info(f"Identified {len(bottlenecks)} major bottlenecks")
    for bottleneck in bottlenecks:
        logger.info(f"Bottleneck: {bottleneck['component']} ({bottleneck['time_percentage']*100:.1f}% of total time)")
    
    return optimization_results

def main():
    """
    Main function to run tests and optimization.
    """
    import argparse
    
    parser = argparse.ArgumentParser(description="Test and optimize the Glove Speed Tracker application")
    parser.add_argument("--test", choices=["video", "detection", "tracking", "speed", "analysis", "visualization", "end-to-end", "all"], 
                        default="all", help="Test to run (default: all)")
    parser.add_argument("--optimize", action="store_true", help="Run optimization after tests")
    parser.add_argument("--output-dir", help="Output directory for test results")
    parser.add_argument("--data-dir", help="Data directory for test videos")
    
    args = parser.parse_args()
    
    # Create tester
    tester = PerformanceTester(
        data_dir=args.data_dir,
        output_dir=args.output_dir
    )
    
    # Run tests
    if args.test == "video":
        tester.test_video_processing()
    elif args.test == "detection":
        tester.test_detection()
    elif args.test == "tracking":
        tester.test_tracking()
    elif args.test == "speed":
        tester.test_speed_calculation()
    elif args.test == "analysis":
        tester.test_data_analysis()
    elif args.test == "visualization":
        tester.test_visualization()
    elif args.test == "end-to-end":
        tester.test_end_to_end()
    else:  # all
        tester.run_all_tests()
    
    # Generate report
    report_path = tester.generate_performance_report()
    print(f"Performance report generated: {report_path}")
    
    # Run optimization if requested
    if args.optimize:
        optimization_results = optimize_application()
        print("Optimization analysis completed")
        print(f"Identified {len(optimization_results['bottlenecks'])} major bottlenecks")
        for bottleneck in optimization_results["bottlenecks"]:
            print(f"Bottleneck: {bottleneck['component']} ({bottleneck['time_percentage']*100:.1f}% of total time)")

if __name__ == "__main__":
    main()
