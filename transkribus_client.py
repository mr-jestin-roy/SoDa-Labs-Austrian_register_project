"""
Transkribus API Client for HTR (Handwritten Text Recognition)

This module provides a Python client for interacting with the Transkribus API
based on the OpenID Connect authentication protocol.

Documentation: https://www.transkribus.org/metagrapho/documentation
"""

import os
import requests
import json
import time
from typing import Optional, Dict, Any
from pathlib import Path
from dotenv import load_dotenv
load_dotenv()


class TranskribusClient:
    """
    Client for interacting with the Transkribus API for HTR (Handwritten Text Recognition).
    Based on the OpenID Connect authentication protocol.
    
    Usage:
        client = TranskribusClient()
        result = client.process_image_sync('path/to/image.jpg')
        
    Environment Variables:
        TRANSKRIBUS_USERNAME: Your Transkribus username/email
        TRANSKRIBUS_PASSWORD: Your Transkribus password
    """
    
    def __init__(self):
        """
        Initialize the Transkribus client.
        
        Args:
            username: Transkribus username (optional, will use env var if not provided)
            password: Transkribus password (optional, will use env var if not provided)
        """
        self.base_url = "https://account.readcoop.eu/auth/realms/readcoop/protocol/openid-connect"
        self.api_base_url = "https://api.transkribus.org/processing/v1"  # Processing API endpoint
        self.client_id = "processing-api-client"
        
        # Get credentials from environment variables or parameters
        self.username = os.getenv('TRANSKRIBUS_USERNAME')
        self.password = os.getenv('TRANSKRIBUS_PASSWORD')
        
        if not self.username or not self.password:
            raise ValueError(
                "Transkribus credentials not provided. "
                "Set TRANSKRIBUS_USERNAME and TRANSKRIBUS_PASSWORD environment variables "
                "or pass them as parameters."
            )
        
        self.access_token = None
        self.refresh_token = None
        self.session = requests.Session()
    
    def authenticate(self) -> bool:
        """
        Authenticate with Transkribus API and obtain access token.
        
        Returns:
            True if successful, False otherwise.
        """
        auth_url = f"{self.base_url}/token"
        
        auth_data = {
            'grant_type': 'password',
            'username': self.username,
            'password': self.password,
            'client_id': self.client_id
        }
        
        headers = {
            'Content-Type': 'application/x-www-form-urlencoded'
        }
        
        try:
            response = requests.post(auth_url, data=auth_data, headers=headers)
            response.raise_for_status()
            
            token_data = response.json()
            self.access_token = token_data.get('access_token')
            self.refresh_token = token_data.get('refresh_token')
            
            if self.access_token:
                # Set authorization header for future requests
                self.session.headers.update({
                    'Authorization': f'Bearer {self.access_token}',
                    'Content-Type': 'application/json'
                })
                print("Successfully authenticated with Transkribus API")
                return True
            else:
                print("Failed to obtain access token")
                return False
                
        except requests.exceptions.RequestException as e:
            print(f"Authentication failed: {e}")
            return False
    
    def refresh_access_token(self) -> bool:
        """
        Refresh the access token using the refresh token.
        
        Returns:
            True if successful, False otherwise.
        """
        if not self.refresh_token:
            print("No refresh token available")
            return False
        
        refresh_url = f"{self.base_url}/token"
        
        refresh_data = {
            'grant_type': 'refresh_token',
            'client_id': self.client_id,
            'refresh_token': self.refresh_token
        }
        
        headers = {
            'Content-Type': 'application/x-www-form-urlencoded'
        }
        
        try:
            response = requests.post(refresh_url, data=refresh_data, headers=headers)
            response.raise_for_status()
            
            token_data = response.json()
            self.access_token = token_data.get('access_token')
            
            # Update refresh token if a new one is provided
            if 'refresh_token' in token_data:
                self.refresh_token = token_data['refresh_token']
            
            if self.access_token:
                self.session.headers.update({
                    'Authorization': f'Bearer {self.access_token}'
                })
                print("Access token refreshed successfully")
                return True
            else:
                print("Failed to refresh access token")
                return False
                
        except requests.exceptions.RequestException as e:
            print(f"Token refresh failed: {e}")
            return False
    
    def submit_image_for_processing(self, image_path: str, model_id: Optional[str] = None) -> Optional[str]:
        """
        Submit an image for HTR processing.
        
        Args:
            image_path: Path to the image file
            model_id: Optional model ID for processing
            
        Returns:
            Process ID if successful, None otherwise.
        """
        if not self.access_token:
            if not self.authenticate():
                return None
        
        # Convert to Path object for easier handling
        image_file = Path(image_path)
        if not image_file.exists():
            print(f"Image file not found: {image_path}")
            return None
        
        submit_url = f"{self.api_base_url}/submit"
        
        try:
            # Prepare the file for upload
            with open(image_file, 'rb') as f:
                files = {
                    'image': (image_file.name, f, 'image/jpeg')
                }
                
                # Prepare additional data
                data = {}
                if model_id:
                    data['modelId'] = model_id
                
                response = self.session.post(submit_url, files=files, data=data)
                
                # Handle token expiration
                if response.status_code == 401:
                    if self.refresh_access_token():
                        response = self.session.post(submit_url, files=files, data=data)
                    else:
                        print("Failed to refresh token for image submission")
                        return None
                
                response.raise_for_status()
                result = response.json()
                
                process_id = result.get('processId')
                if process_id:
                    print(f"Image submitted successfully. Process ID: {process_id}")
                    return process_id
                else:
                    print("No process ID returned from submission")
                    return None
                    
        except requests.exceptions.RequestException as e:
            print(f"Image submission failed: {e}")
            return None
    
    def get_processing_status(self, process_id: str) -> Optional[Dict[str, Any]]:
        """
        Get the status of a processing job.
        
        Args:
            process_id: The process ID to check
            
        Returns:
            Status information if successful, None otherwise.
        """
        if not self.access_token:
            if not self.authenticate():
                return None
        
        status_url = f"{self.api_base_url}/status/{process_id}"
        
        try:
            response = self.session.get(status_url)
            
            # Handle token expiration
            if response.status_code == 401:
                if self.refresh_access_token():
                    response = self.session.get(status_url)
                else:
                    print("Failed to refresh token for status check")
                    return None
            
            response.raise_for_status()
            return response.json()
            
        except requests.exceptions.RequestException as e:
            print(f"Status check failed: {e}")
            return None
    
    def get_processing_result(self, process_id: str) -> Optional[str]:
        """
        Get the HTR result for a completed processing job.
        
        Args:
            process_id: The process ID to get results for
            
        Returns:
            The recognized text if successful, None otherwise.
        """
        if not self.access_token:
            if not self.authenticate():
                return None
        
        result_url = f"{self.api_base_url}/result/{process_id}"
        
        try:
            response = self.session.get(result_url)
            
            # Handle token expiration
            if response.status_code == 401:
                if self.refresh_access_token():
                    response = self.session.get(result_url)
                else:
                    print("Failed to refresh token for result retrieval")
                    return None
            
            response.raise_for_status()
            result = response.json()
            
            # Extract text from the result
            text = result.get('text', '')
            if text:
                print(f"HTR processing completed for process {process_id}")
                return text
            else:
                print(f"No text found in result for process {process_id}")
                return None
                
        except requests.exceptions.RequestException as e:
            print(f"Result retrieval failed: {e}")
            return None
    
    def process_image_sync(self, image_path: str, model_id: Optional[str] = None, 
                          max_wait_time: int = 300, poll_interval: int = 5) -> Optional[str]:
        """
        Submit an image and wait for the HTR result synchronously.
        
        Args:
            image_path: Path to the image file
            model_id: Optional model ID to use for processing
            max_wait_time: Maximum time to wait for completion (seconds)
            poll_interval: Time between status checks (seconds)
        
        Returns:
            The recognized text if successful, None otherwise.
        """
        print(f"Starting HTR processing for: {image_path}")
        
        # Submit the image
        process_id = self.submit_image_for_processing(image_path, model_id)
        if not process_id:
            return None
        
        # Wait for completion
        start_time = time.time()
        while time.time() - start_time < max_wait_time:
            status = self.get_processing_status(process_id)
            if not status:
                print("Failed to get processing status")
                return None
            
            status_value = status.get('status', '').lower()
            print(f"Processing status: {status_value}")
            
            if status_value in ['completed', 'finished', 'done']:
                # Get the result
                result = self.get_processing_result(process_id)
                return result
            elif status_value in ['failed', 'error']:
                print(f"Processing failed for process {process_id}")
                return None
            
            # Wait before next check
            time.sleep(poll_interval)
        
        print(f"Processing timed out after {max_wait_time} seconds")
        return None
    
    def logout(self) -> bool:
        """
        Log out and invalidate the refresh token.
        
        Returns:
            True if successful, False otherwise.
        """
        if not self.refresh_token:
            print("No refresh token to invalidate")
            return True
        
        logout_url = f"{self.base_url}/logout"
        
        logout_data = {
            'refresh_token': self.refresh_token,
            'client_id': self.client_id
        }
        
        headers = {
            'Content-Type': 'application/x-www-form-urlencoded'
        }
        
        try:
            response = requests.post(logout_url, data=logout_data, headers=headers)
            response.raise_for_status()
            
            # Clear tokens
            self.access_token = None
            self.refresh_token = None
            self.session.headers.pop('Authorization', None)
            
            print("Successfully logged out from Transkribus")
            return True
            
        except requests.exceptions.RequestException as e:
            print(f"Logout failed: {e}")
            return False 
        
# This is the call method that should be used to process the image:
# def call_transkribus_api(image_path):
#     """
#     Process an image using the Transkribus HTR API.
    
#     Args:
#         image_path: Path to the image file to process
    
#     Returns:
#         The recognized text from the image, or an error message if processing fails
#     """
#     try:
#         client = get_transkribus_client()
        
#         # Process the image synchronously
#         result = client.process_image_sync(
#             image_path=image_path,
#             model_id=None,  # Use default model, can be configured
#             max_wait_time=300,  # 5 minutes timeout
#             poll_interval=5  # Check status every 5 seconds
#         )
        
#         if result:
#             return result
#         else:
#             return "Error: Failed to process image with Transkribus API"
            
#     except Exception as e:
#         print(f"Transkribus API error: {e}")
#         return f"Error: Transkribus API processing failed - {str(e)}"