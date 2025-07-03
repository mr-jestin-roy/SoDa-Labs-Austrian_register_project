#!/usr/bin/env python3
"""
Example usage of the TranskribusClient for HTR processing.

This script demonstrates how to use the Transkribus API to process images
and extract handwritten text.
"""

import os
from transkribus_client import TranskribusClient

def main():
    """Main example function."""
    print("🔍 Transkribus API Client Example")
    print("=" * 40)
    
    # Method 1: Using environment variables (recommended)
    # Set TRANSKRIBUS_USERNAME and TRANSKRIBUS_PASSWORD in your environment
    try:
        print("🔑 Authenticating with environment variables...")
        client = TranskribusClient()
        
        # Example: Process a single image
        image_path = "input_images/sample.jpg"  # Replace with your image path
        
        if os.path.exists(image_path):
            print(f"📄 Processing image: {image_path}")
            
            # Process the image synchronously
            result = client.process_image_sync(
                image_path=image_path,
                model_id=None,  # Use default model
                max_wait_time=300,  # 5 minutes timeout
                poll_interval=10   # Check every 10 seconds
            )
            
            if result:
                print("✅ Processing successful!")
                print(f"📝 Recognized text:\n{result}")
            else:
                print("❌ Processing failed!")
            
            # Always logout when done
            client.logout()
        else:
            print(f"❌ Image file not found: {image_path}")
            print("Please update the image_path variable with a valid image file.")
    
    except ValueError as e:
        print(f"❌ Configuration error: {e}")
        print("\n📋 Setup Instructions:")
        print("1. Set environment variables:")
        print("   export TRANSKRIBUS_USERNAME=your_email@example.com")
        print("   export TRANSKRIBUS_PASSWORD=your_password")
        print("\n2. Or create a .env file with:")
        print("   TRANSKRIBUS_USERNAME=your_email@example.com")
        print("   TRANSKRIBUS_PASSWORD=your_password")
    
    except Exception as e:
        print(f"❌ Error: {e}")

def method2_direct_credentials():
    """Alternative method: Passing credentials directly (less secure)."""
    print("\n🔑 Method 2: Direct credentials (for testing only)")
    print("-" * 50)
    
    # WARNING: Don't hardcode credentials in production code!
    username = "your_email@example.com"
    password = "your_password"
    
    try:
        client = TranskribusClient(username=username, password=password)
        
        # Test authentication only
        if client.authenticate():
            print("✅ Authentication successful with direct credentials!")
            client.logout()
        else:
            print("❌ Authentication failed!")
            
    except Exception as e:
        print(f"❌ Error with direct credentials: {e}")

def batch_processing_example():
    """Example of processing multiple images."""
    print("\n📦 Batch Processing Example")
    print("-" * 30)
    
    try:
        client = TranskribusClient()
        
        # List of images to process
        image_paths = [
            "input_images/image1.jpg",
            "input_images/image2.jpg",
            "input_images/image3.jpg"
        ]
        
        results = []
        
        for i, image_path in enumerate(image_paths, 1):
            if os.path.exists(image_path):
                print(f"Processing image {i}/{len(image_paths)}: {image_path}")
                
                result = client.process_image_sync(
                    image_path=image_path,
                    max_wait_time=180,  # 3 minutes per image
                    poll_interval=5
                )
                
                results.append({
                    'image': image_path,
                    'text': result if result else "Processing failed"
                })
            else:
                print(f"⚠️  Image not found: {image_path}")
                results.append({
                    'image': image_path,
                    'text': "File not found"
                })
        
        # Display results
        print("\n📊 Batch Processing Results:")
        print("=" * 40)
        for i, result in enumerate(results, 1):
            print(f"{i}. {result['image']}")
            print(f"   Text: {result['text'][:100]}..." if len(result['text']) > 100 else f"   Text: {result['text']}")
            print()
        
        client.logout()
        
    except Exception as e:
        print(f"❌ Batch processing error: {e}")

if __name__ == "__main__":
    main()
    
    # Uncomment to try alternative methods:
    # method2_direct_credentials()
    # batch_processing_example() 