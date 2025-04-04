import numpy as np

class SpeedCalculator:
    """
    Class for calculating glove speed from tracking data
    """
    
    def __init__(self, fps, frame_width, frame_height, real_world_width=2.0, real_world_height=2.0):
        """
        Initialize the speed calculator
        
        Args:
            fps (float): Frames per second of the video
            frame_width (int): Width of the video frame in pixels
            frame_height (int): Height of the video frame in pixels
            real_world_width (float): Real-world width of the frame in meters
            real_world_height (float): Real-world height of the frame in meters
        """
        self.fps = fps
        self.frame_width = frame_width
        self.frame_height = frame_height
        self.real_world_width = real_world_width
        self.real_world_height = real_world_height
        self.pixel_to_meter_x = real_world_width / frame_width
        self.pixel_to_meter_y = real_world_height / frame_height
        self.speeds = []
        self.accelerations = []
        self.movement_patterns = []
    
    def pixel_to_meter(self, pixel_coords):
        """
        Convert pixel coordinates to real-world coordinates in meters
        
        Args:
            pixel_coords (list): [x, y] coordinates in pixels
        
        Returns:
            list: [x, y] coordinates in meters
        """
        x_meters = pixel_coords[0] * self.pixel_to_meter_x
        y_meters = pixel_coords[1] * self.pixel_to_meter_y
        return [x_meters, y_meters]
    
    def calculate_speeds(self, tracks):
        """
        Calculate speeds from tracking data
        
        Args:
            tracks (dict): Dictionary of tracks {ID: [positions]}
        
        Returns:
            list: List of speeds in mph
        """
        # Process each track
        for track_id, positions in tracks.items():
            # Need at least 2 positions to calculate speed
            if len(positions) < 2:
                continue
            
            # Calculate speeds between consecutive positions
            track_speeds = []
            for i in range(1, len(positions)):
                # Convert pixel positions to meters
                pos1_meters = self.pixel_to_meter(positions[i-1])
                pos2_meters = self.pixel_to_meter(positions[i])
                
                # Calculate displacement in meters
                dx = pos2_meters[0] - pos1_meters[0]
                dy = pos2_meters[1] - pos1_meters[1]
                displacement = np.sqrt(dx**2 + dy**2)
                
                # Calculate time between frames in seconds
                time_diff = 1.0 / self.fps
                
                # Calculate speed in meters per second
                speed_mps = displacement / time_diff
                
                # Convert to mph (1 m/s = 2.23694 mph)
                speed_mph = speed_mps * 2.23694
                
                track_speeds.append(speed_mph)
                
                # Calculate acceleration if we have at least 2 speeds
                if len(track_speeds) >= 2:
                    acceleration = (track_speeds[-1] - track_speeds[-2]) / time_diff
                    self.accelerations.append(acceleration)
            
            # Add track speeds to overall speeds
            self.speeds.extend(track_speeds)
            
            # Analyze movement patterns
            self.analyze_track_patterns(track_id, positions, track_speeds)
        
        return self.speeds
    
    def get_max_speed(self):
        """
        Get the maximum speed
        
        Returns:
            float: Maximum speed in mph
        """
        if not self.speeds:
            return 0.0
        return max(self.speeds)
    
    def get_avg_speed(self):
        """
        Get the average speed
        
        Returns:
            float: Average speed in mph
        """
        if not self.speeds:
            return 0.0
        return sum(self.speeds) / len(self.speeds)
    
    def get_max_acceleration(self):
        """
        Get the maximum acceleration
        
        Returns:
            float: Maximum acceleration in m/s²
        """
        if not self.accelerations:
            return 0.0
        return max(self.accelerations)
    
    def analyze_track_patterns(self, track_id, positions, speeds):
        """
        Analyze movement patterns in a track
        
        Args:
            track_id (int): ID of the track
            positions (list): List of positions
            speeds (list): List of speeds
        """
        # Need at least 3 positions to analyze patterns
        if len(positions) < 3:
            return
        
        # Calculate direction changes
        direction_changes = 0
        prev_direction = None
        
        for i in range(1, len(positions)):
            dx = positions[i][0] - positions[i-1][0]
            dy = positions[i][1] - positions[i-1][1]
            
            # Calculate direction angle
            angle = np.arctan2(dy, dx)
            
            # Quantize direction to 8 cardinal directions
            direction = int((angle + np.pi) / (np.pi/4)) % 8
            
            if prev_direction is not None and direction != prev_direction:
                direction_changes += 1
            
            prev_direction = direction
        
        # Identify quick movements (high speed)
        quick_movements = sum(1 for s in speeds if s > 30)
        
        # Identify lateral movements (horizontal movement)
        lateral_movements = 0
        for i in range(1, len(positions)):
            dx = abs(positions[i][0] - positions[i-1][0])
            dy = abs(positions[i][1] - positions[i-1][1])
            if dx > dy * 2:  # Significantly more horizontal than vertical
                lateral_movements += 1
        
        # Identify vertical movements (vertical movement)
        vertical_movements = 0
        for i in range(1, len(positions)):
            dx = abs(positions[i][0] - positions[i-1][0])
            dy = abs(positions[i][1] - positions[i-1][1])
            if dy > dx * 2:  # Significantly more vertical than horizontal
                vertical_movements += 1
        
        # Add to movement patterns
        if quick_movements > 0:
            self.movement_patterns.append({
                'type': 'Quick Snap',
                'count': quick_movements,
                'avgSpeed': sum([s for s in speeds if s > 30]) / quick_movements if quick_movements > 0 else 0
            })
        
        if lateral_movements > 0:
            self.movement_patterns.append({
                'type': 'Lateral Movement',
                'count': lateral_movements,
                'avgSpeed': sum(speeds) / len(speeds) if speeds else 0
            })
        
        if vertical_movements > 0:
            self.movement_patterns.append({
                'type': 'Vertical Reach',
                'count': vertical_movements,
                'avgSpeed': sum(speeds) / len(speeds) if speeds else 0
            })
    
    def analyze_movement_patterns(self):
        """
        Get movement pattern analysis
        
        Returns:
            list: List of movement patterns
        """
        # Combine similar patterns
        pattern_dict = {}
        for pattern in self.movement_patterns:
            pattern_type = pattern['type']
            if pattern_type in pattern_dict:
                pattern_dict[pattern_type]['count'] += pattern['count']
                pattern_dict[pattern_type]['totalSpeed'] += pattern['avgSpeed'] * pattern['count']
                pattern_dict[pattern_type]['instances'] += 1
            else:
                pattern_dict[pattern_type] = {
                    'count': pattern['count'],
                    'totalSpeed': pattern['avgSpeed'] * pattern['count'],
                    'instances': 1
                }
        
        # Calculate average speeds
        result = []
        for pattern_type, data in pattern_dict.items():
            result.append({
                'type': pattern_type,
                'count': data['count'],
                'avgSpeed': data['totalSpeed'] / data['count'] if data['count'] > 0 else 0
            })
        
        return result
