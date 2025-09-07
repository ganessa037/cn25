#!/usr/bin/env python3
"""
Interactive License Plate Detection Model Tester
Allows manual testing of the license plate detection functionality through image uploads.
"""

import sys
import os
import time
from pathlib import Path
import cv2
import numpy as np
from PIL import Image
import ssl

# Configure SSL for EasyOCR downloads
ssl._create_default_https_context = ssl._create_unverified_context

# Add the project root directory to the Python path
sys.path.append(os.path.dirname(__file__))

try:
    from src.license_plate.core.processor import LicensePlateProcessor
    from src.license_plate.utils.image_utils import ImageProcessor
except ImportError as e:
    print(f"❌ Import error: {e}")
    print("Make sure you're running this from the project root directory.")
    sys.exit(1)

def display_results(results):
    """Display detection results in a formatted way."""
    print(f"\n📊 Detection Results:")
    print(f"   Processing Time: {results['processing_time']:.3f}s")
    print(f"   Total Detections: {results['total_detections']}")
    print(f"   Valid Plates: {results['valid_plates']}")
    print(f"   Image Dimensions: {results['image_dimensions']['width']}x{results['image_dimensions']['height']}")
    
    if results['results']:
        print("\n🔍 Detected License Plates:")
        for i, result in enumerate(results['results'], 1):
            print(f"\n   Plate {i}:")
            print(f"     • Bounding Box: {result['bbox']}")
            print(f"     • Detection Confidence: {result['detection_confidence']:.3f}")
            print(f"     • Plate Text: '{result['plate_text']}'")
            print(f"     • Valid Plate: {result['is_valid_plate']}")
            print(f"     • Plate Type: {result['plate_type']}")
            print(f"     • State: {result['state']}")
            
            if result['ocr_results']:
                print(f"     • OCR Results:")
                for ocr in result['ocr_results']:
                    print(f"       - Text: '{ocr['text']}' (confidence: {ocr['confidence']:.3f})")
    else:
        print("\n❌ No license plates detected in the image.")

def test_model_loading():
    """Test if the model can be loaded successfully."""
    print("🔧 Testing model loading...")
    
    # Check if model file exists
    model_path = Path("models/license_plate_detection/trained_models/best.pt")
    if not model_path.exists():
        print(f"❌ Model file not found at: {model_path}")
        return False
    
    try:
        # Test loading with the correct model path
        processor = LicensePlateProcessor(
            model_path=str(model_path),
            use_gpu=False,  # Use CPU for testing
            confidence_threshold=0.5
        )
        print("✅ Model loaded successfully!")
        return processor
    except Exception as e:
        print(f"❌ Failed to load model: {e}")
        return False

def get_sample_images():
    """Get list of sample images for testing."""
    sample_dirs = [
        "data/license_plate_detection/test",
        "data/license_plate_detection/sample",
        "../models/license_plate_detection/data/test_images",
        "../models/license_plate_detection/test_images"
    ]
    
    sample_images = []
    for sample_dir in sample_dirs:
        if Path(sample_dir).exists():
            for ext in ['*.jpg', '*.jpeg', '*.png', '*.bmp']:
                sample_images.extend(Path(sample_dir).glob(ext))
    
    return sample_images[:5]  # Return first 5 found

def main():
    print("🚗 License Plate Detection Model Tester")
    print("=" * 45)
    print("Test the license plate detection model with your own images.")
    print("Type 'quit' or 'exit' to stop.\n")
    
    # Test model loading first
    processor = test_model_loading()
    if not processor:
        print("\n❌ Cannot proceed without a working model.")
        print("\nMake sure you have:")
        print("1. Trained model at models/license_plate_detection/trained_models/best.pt")
        print("2. Required dependencies installed")
        print("3. Run this script from the project root directory")
        return 1
    
    # Show sample images if available
    sample_images = get_sample_images()
    if sample_images:
        print("\n📁 Sample images found:")
        for i, img_path in enumerate(sample_images, 1):
            print(f"   {i}. {img_path}")
        print("\n💡 You can test with these sample images or provide your own path.")
    
    print("\n🔍 Enter image path to test (or 'quit' to exit):")
    
    while True:
        try:
            user_input = input("\nImage path: ").strip()
            
            if user_input.lower() in ['quit', 'exit', 'q']:
                print("\n👋 Thanks for testing! Goodbye!")
                break
            
            if not user_input:
                print("Please enter an image path or 'quit' to exit.")
                continue
            
            # Check if it's a sample image number
            if user_input.isdigit() and sample_images:
                img_num = int(user_input) - 1
                if 0 <= img_num < len(sample_images):
                    image_path = sample_images[img_num]
                else:
                    print(f"Invalid sample number. Choose 1-{len(sample_images)}")
                    continue
            else:
                image_path = Path(user_input)
            
            # Check if image exists
            if not image_path.exists():
                print(f"❌ Image not found: {image_path}")
                continue
            
            # Load and process image
            print(f"\n📸 Processing image: {image_path.name}")
            
            try:
                # Load image
                image = cv2.imread(str(image_path))
                if image is None:
                    print("❌ Failed to load image. Please check the file format.")
                    continue
                
                # Process image
                start_time = time.time()
                results = processor.process_image(
                    image,
                    return_annotated=False,
                    preprocess_ocr=True,
                    extract_regions=True
                )
                
                # Display results
                display_results(results)
                
                # Ask if user wants to save annotated image
                if results['total_detections'] > 0:
                    save_annotated = input("\n💾 Save annotated image? (y/n): ").strip().lower()
                    if save_annotated in ['y', 'yes']:
                        annotated_results = processor.process_image(
                            image,
                            return_annotated=True,
                            preprocess_ocr=True,
                            extract_regions=True
                        )
                        
                        if 'annotated_image' in annotated_results:
                            output_path = f"output_annotated_{image_path.stem}.jpg"
                            cv2.imwrite(output_path, annotated_results['annotated_image'])
                            print(f"✅ Annotated image saved as: {output_path}")
                
                print("-" * 50)
                
            except Exception as e:
                print(f"❌ Error processing image: {str(e)}")
                print("-" * 50)
                
        except KeyboardInterrupt:
            print("\n\n👋 Interrupted by user. Goodbye!")
            break
        except Exception as e:
            print(f"❌ Unexpected error: {str(e)}")
            continue
    
    return 0

if __name__ == "__main__":
    exit(main())