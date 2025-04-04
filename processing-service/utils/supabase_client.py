import os
import requests
import json

class SupabaseClient:
    """
    Client for interacting with Supabase from the processing service
    """
    
    def __init__(self, url, key):
        """
        Initialize the Supabase client
        
        Args:
            url (str): Supabase project URL
            key (str): Supabase API key (service role key for admin operations)
        """
        self.url = url
        self.key = key
        self.headers = {
            'apikey': key,
            'Authorization': f'Bearer {key}',
            'Content-Type': 'application/json'
        }
    
    def download_file(self, bucket, path, destination):
        """
        Download a file from Supabase storage
        
        Args:
            bucket (str): Storage bucket name
            path (str): Path to the file in the bucket
            destination (str): Local path to save the file
        
        Returns:
            bool: True if successful, False otherwise
        """
        try:
            # Get public URL for the file
            url = f"{self.url}/storage/v1/object/public/{bucket}/{path}"
            
            # Download the file
            response = requests.get(url, headers=self.headers)
            response.raise_for_status()
            
            # Save to destination
            with open(destination, 'wb') as f:
                f.write(response.content)
            
            return True
        except Exception as e:
            print(f"Error downloading file: {str(e)}")
            return False
    
    def update_video_status(self, video_id, status, error_message=None):
        """
        Update the status of a video in the database
        
        Args:
            video_id (str): ID of the video
            status (str): New status ('processing', 'completed', 'error')
            error_message (str, optional): Error message if status is 'error'
        
        Returns:
            bool: True if successful, False otherwise
        """
        try:
            data = {'status': status}
            if error_message:
                data['error_message'] = error_message
            
            url = f"{self.url}/rest/v1/videos?id=eq.{video_id}"
            response = requests.patch(url, headers=self.headers, json=data)
            response.raise_for_status()
            
            return True
        except Exception as e:
            print(f"Error updating video status: {str(e)}")
            return False
    
    def store_analysis_results(self, video_id, results):
        """
        Store analysis results in the database
        
        Args:
            video_id (str): ID of the video
            results (dict): Analysis results
        
        Returns:
            bool: True if successful, False otherwise
        """
        try:
            # Format the results for storage
            data = {
                'video_id': video_id,
                'max_speed': results['max_speed'],
                'avg_speed': results['avg_speed'],
                'top_acceleration': results['top_acceleration'],
                'movement_patterns': json.dumps(results['movement_patterns'])
            }
            
            # Check if results already exist
            check_url = f"{self.url}/rest/v1/analysis_results?video_id=eq.{video_id}"
            check_response = requests.get(check_url, headers=self.headers)
            check_response.raise_for_status()
            
            if check_response.json():
                # Update existing results
                url = f"{self.url}/rest/v1/analysis_results?video_id=eq.{video_id}"
                response = requests.patch(url, headers=self.headers, json=data)
            else:
                # Insert new results
                url = f"{self.url}/rest/v1/analysis_results"
                response = requests.post(url, headers=self.headers, json=data)
            
            response.raise_for_status()
            return True
        except Exception as e:
            print(f"Error storing analysis results: {str(e)}")
            return False
