import numpy as np

class GloveTracker:
    """
    Class for tracking baseball catcher's glove across video frames
    """
    
    def __init__(self, max_disappeared=10, min_distance=50):
        """
        Initialize the glove tracker
        
        Args:
            max_disappeared (int): Maximum number of frames an object can be missing before deregistering
            min_distance (float): Minimum distance between centroids to consider them the same object
        """
        self.max_disappeared = max_disappeared
        self.min_distance = min_distance
        self.next_object_id = 0
        self.objects = {}  # Dictionary of tracked objects {ID: centroid}
        self.disappeared = {}  # Dictionary of disappeared counts {ID: count}
        self.tracks = {}  # Dictionary of tracks {ID: [positions]}
    
    def register(self, centroid):
        """
        Register a new object with the next available ID
        
        Args:
            centroid (list): [x, y] coordinates of the object centroid
        """
        self.objects[self.next_object_id] = centroid
        self.disappeared[self.next_object_id] = 0
        self.tracks[self.next_object_id] = [centroid]
        self.next_object_id += 1
    
    def deregister(self, object_id):
        """
        Deregister an object by deleting it from our dictionaries
        
        Args:
            object_id (int): ID of the object to deregister
        """
        del self.objects[object_id]
        del self.disappeared[object_id]
        # Keep the track for later analysis
    
    def update(self, detection):
        """
        Update the tracker with a new detection
        
        Args:
            detection (dict): Detection result with bounding box and confidence
        
        Returns:
            int: ID of the tracked object
        """
        # If we have no objects, register the detection
        if len(self.objects) == 0:
            self.register(detection['center'])
            return 0
        
        # Get the centroid of the detection
        centroid = detection['center']
        
        # Calculate distances between the new centroid and existing objects
        distances = []
        for object_id, existing_centroid in self.objects.items():
            d = np.sqrt((centroid[0] - existing_centroid[0])**2 + 
                        (centroid[1] - existing_centroid[1])**2)
            distances.append((object_id, d))
        
        # Sort distances
        distances.sort(key=lambda x: x[1])
        
        # Get the closest object
        closest_id, closest_distance = distances[0]
        
        # If the distance is less than the minimum distance, update the object
        if closest_distance < self.min_distance:
            # Update the object's centroid
            self.objects[closest_id] = centroid
            # Reset the disappeared counter
            self.disappeared[closest_id] = 0
            # Add to the track
            self.tracks[closest_id].append(centroid)
            return closest_id
        else:
            # Register a new object
            self.register(centroid)
            return self.next_object_id - 1
    
    def get_tracks(self):
        """
        Get all tracks
        
        Returns:
            dict: Dictionary of tracks {ID: [positions]}
        """
        return self.tracks
