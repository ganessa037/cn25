#!/usr/bin/env python3
"""
Interactive Document Parser Model Tester
Allows manual testing of the document parsing functionality through image uploads.
"""

import sys
import os
import time
import json
import statistics
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Any, Tuple
import cv2
import numpy as np
from PIL import Image
import ssl
import matplotlib.pyplot as plt
import seaborn as sns

# Configure SSL for downloads
ssl._create_default_https_context = ssl._create_unverified_context

# Add the project root directory to the Python path
sys.path.append(os.path.join(os.path.dirname(__file__), '.'))

try:
    from src.document_parser.document_classifier import DocumentClassifier
    from src.document_parser.ocr_service import OCRService
    from src.document_parser.field_extractor import FieldExtractor
    from src.document_parser.validator import DocumentValidator
    from src.document_parser.utils import ImagePreprocessor
except ImportError as e:
    print(f"❌ Import error: {e}")
    print("Make sure you're running this from the project root directory.")
    sys.exit(1)

def display_results(results):
    """Display document parsing results in a formatted way."""
    print(f"\n📊 Document Parsing Results:")
    print(f"   Processing Time: {results['processing_time']:.3f}s")
    print(f"   Document Type: {results['document_type']}")
    print(f"   Classification Confidence: {results['classification_confidence']:.3f}")
    print(f"   Image Dimensions: {results['image_dimensions']['width']}x{results['image_dimensions']['height']}")
    print(f"   Total Fields Extracted: {len(results['extracted_fields'])}")
    print(f"   Validation Status: {'✅ PASSED' if results['validation_passed'] else '❌ FAILED'}")
    
    if results['extracted_fields']:
        print("\n🔍 Extracted Fields:")
        for field_name, field_data in results['extracted_fields'].items():
            print(f"\n   {field_name.upper()}:")
            print(f"     • Value: '{field_data['value']}'")
            print(f"     • Confidence: {field_data['confidence']:.3f}")
            if 'coordinates' in field_data:
                print(f"     • Coordinates: {field_data['coordinates']}")
            print(f"     • Valid: {field_data['is_valid']}")
            
            if field_data.get('ocr_results'):
                print(f"     • OCR Results:")
                for ocr in field_data['ocr_results']:
                    print(f"       - Text: '{ocr['text']}' (confidence: {ocr['confidence']:.3f})")
    
    if results.get('validation_errors'):
        print("\n⚠️ Validation Errors:")
        for error in results['validation_errors']:
            print(f"   • {error['field']}: {error['message']}")
    
    if not results['extracted_fields']:
        print("\n❌ No fields could be extracted from the document.")

def test_model_loading():
    """Test if the document parser components can be loaded successfully."""
    print("🔧 Testing model loading...")
    
    # Check if template files exist
    template_dir = Path("src/document_parser/templates")
    if not template_dir.exists():
        print(f"❌ Template directory not found at: {template_dir}")
        return False
    
    try:
        # Test loading document classifier
        classifier = DocumentClassifier(device="cpu")
        print("✅ Document classifier loaded successfully!")
        
        # Test loading OCR service
        ocr_service = OCRService(engines=['tesseract'], languages=['en', 'ms'])
        print("✅ OCR service loaded successfully!")
        
        # Test loading field extractor
        field_extractor = FieldExtractor()
        print("✅ Field extractor loaded successfully!")
        
        # Test loading validator
        validator = DocumentValidator()
        print("✅ Document validator loaded successfully!")
        
        return {
            'classifier': classifier,
            'ocr_service': ocr_service,
            'field_extractor': field_extractor,
            'validator': validator
        }
    except Exception as e:
        print(f"❌ Failed to load components: {e}")
        return False

def get_sample_images():
    """Get list of sample images for testing."""
    sample_dirs = [
        "data/document_parser/test",
        "data/document_parser/sample",
        "models/document_parser/test_images",
        "test_images"
    ]
    
    sample_images = []
    for sample_dir in sample_dirs:
        if Path(sample_dir).exists():
            for ext in ['*.jpg', '*.jpeg', '*.png', '*.bmp', '*.pdf']:
                sample_images.extend(Path(sample_dir).glob(ext))
    
    return sample_images[:5]  # Return first 5 found

def get_supported_document_types():
    """Get list of supported document types."""
    return [
        "mykad",
        "spk",
        "auto"  # Auto-detect
    ]

def test_model_accuracy(components, test_dataset_path: str = None) -> Dict[str, Any]:
    """Test model accuracy on a labeled dataset.
    
    Args:
        components: Loaded model components
        test_dataset_path: Path to test dataset with ground truth labels
        
    Returns:
        Dictionary containing accuracy metrics
    """
    print("\n🎯 Testing Model Accuracy...")
    print("-" * 50)
    
    # Use document_parser_dataset if available, otherwise use sample images
    if test_dataset_path and Path(test_dataset_path).exists():
        test_dir = Path(test_dataset_path)
    else:
        test_dir = Path("document_parser_dataset/test/images")
        if not test_dir.exists():
            test_dir = Path("document_parser_dataset/valid/images")
    
    if not test_dir.exists():
        print("❌ No test dataset found. Using sample images for basic testing.")
        return test_basic_accuracy(components)
    
    # Get test images
    test_images = list(test_dir.glob("*.jpg")) + list(test_dir.glob("*.jpeg")) + list(test_dir.glob("*.png"))
    
    if not test_images:
        print("❌ No test images found in dataset.")
        return {"error": "No test images available"}
    
    print(f"📊 Testing on {len(test_images)} images...")
    
    results = {
        "total_images": len(test_images),
        "successful_detections": 0,
        "failed_detections": 0,
        "processing_times": [],
        "confidence_scores": [],
        "accuracy_by_type": {},
        "detailed_results": []
    }
    
    for i, img_path in enumerate(test_images[:20]):  # Test on first 20 images
        print(f"\r   Processing image {i+1}/{min(20, len(test_images))}...", end="")
        
        try:
            # Load and process image
            image = cv2.imread(str(img_path))
            if image is None:
                continue
            
            start_time = time.time()
            
            # Classify document
            classification_result = components['classifier'].classify_document(image_array=image)
            detected_type = classification_result.get('document_type', 'unknown')
            confidence = classification_result.get('confidence', 0.0)
            
            processing_time = time.time() - start_time
            
            # Record results
            results["processing_times"].append(processing_time)
            results["confidence_scores"].append(confidence)
            
            if detected_type != 'unknown' and confidence > 0.5:
                results["successful_detections"] += 1
            else:
                results["failed_detections"] += 1
            
            # Track by document type
            if detected_type not in results["accuracy_by_type"]:
                results["accuracy_by_type"][detected_type] = {"count": 0, "avg_confidence": 0}
            
            results["accuracy_by_type"][detected_type]["count"] += 1
            results["accuracy_by_type"][detected_type]["avg_confidence"] = (
                (results["accuracy_by_type"][detected_type]["avg_confidence"] * 
                 (results["accuracy_by_type"][detected_type]["count"] - 1) + confidence) /
                results["accuracy_by_type"][detected_type]["count"]
            )
            
            results["detailed_results"].append({
                "image": img_path.name,
                "detected_type": detected_type,
                "confidence": confidence,
                "processing_time": processing_time
            })
            
        except Exception as e:
            results["failed_detections"] += 1
            print(f"\n❌ Error processing {img_path.name}: {e}")
    
    print("\n")
    
    # Calculate final metrics
    if results["processing_times"]:
        results["avg_processing_time"] = statistics.mean(results["processing_times"])
        results["min_processing_time"] = min(results["processing_times"])
        results["max_processing_time"] = max(results["processing_times"])
    
    if results["confidence_scores"]:
        results["avg_confidence"] = statistics.mean(results["confidence_scores"])
        results["min_confidence"] = min(results["confidence_scores"])
        results["max_confidence"] = max(results["confidence_scores"])
    
    results["detection_rate"] = results["successful_detections"] / results["total_images"] if results["total_images"] > 0 else 0
    
    # Display results
    print("📊 Accuracy Test Results:")
    print(f"   Total Images: {results['total_images']}")
    print(f"   Successful Detections: {results['successful_detections']}")
    print(f"   Failed Detections: {results['failed_detections']}")
    print(f"   Detection Rate: {results['detection_rate']:.2%}")
    
    if results.get("avg_confidence"):
        print(f"   Average Confidence: {results['avg_confidence']:.3f}")
        print(f"   Confidence Range: {results['min_confidence']:.3f} - {results['max_confidence']:.3f}")
    
    if results.get("avg_processing_time"):
        print(f"   Average Processing Time: {results['avg_processing_time']:.3f}s")
        print(f"   Processing Time Range: {results['min_processing_time']:.3f}s - {results['max_processing_time']:.3f}s")
    
    print("\n📋 Detection by Document Type:")
    for doc_type, stats in results["accuracy_by_type"].items():
        print(f"   {doc_type}: {stats['count']} images (avg confidence: {stats['avg_confidence']:.3f})")
    
    return results

def test_basic_accuracy(components) -> Dict[str, Any]:
    """Basic accuracy test using sample images."""
    sample_images = get_sample_images()
    
    if not sample_images:
        return {"error": "No sample images available for testing"}
    
    print(f"📊 Basic accuracy test on {len(sample_images)} sample images...")
    
    results = {
        "total_images": len(sample_images),
        "successful_detections": 0,
        "processing_times": [],
        "confidence_scores": []
    }
    
    for img_path in sample_images:
        try:
            image = cv2.imread(str(img_path))
            if image is None:
                continue
            
            start_time = time.time()
            classification_result = components['classifier'].classify_document(image_array=image)
            processing_time = time.time() - start_time
            
            detected_type = classification_result.get('document_type', 'unknown')
            confidence = classification_result.get('confidence', 0.0)
            
            results["processing_times"].append(processing_time)
            results["confidence_scores"].append(confidence)
            
            if detected_type != 'unknown' and confidence > 0.5:
                results["successful_detections"] += 1
                
        except Exception as e:
            print(f"❌ Error processing {img_path.name}: {e}")
    
    results["detection_rate"] = results["successful_detections"] / results["total_images"] if results["total_images"] > 0 else 0
    
    if results["processing_times"]:
        results["avg_processing_time"] = statistics.mean(results["processing_times"])
    if results["confidence_scores"]:
        results["avg_confidence"] = statistics.mean(results["confidence_scores"])
    
    print(f"   Detection Rate: {results['detection_rate']:.2%}")
    if results.get("avg_confidence"):
        print(f"   Average Confidence: {results['avg_confidence']:.3f}")
    if results.get("avg_processing_time"):
        print(f"   Average Processing Time: {results['avg_processing_time']:.3f}s")
    
    return results

def test_performance_benchmark(components, num_iterations: int = 50) -> Dict[str, Any]:
    """Benchmark model performance with multiple iterations.
    
    Args:
        components: Loaded model components
        num_iterations: Number of test iterations
        
    Returns:
        Dictionary containing performance metrics
    """
    print(f"\n⚡ Performance Benchmark ({num_iterations} iterations)...")
    print("-" * 50)
    
    # Get a sample image for testing
    sample_images = get_sample_images()
    if not sample_images:
        return {"error": "No sample images available for benchmarking"}
    
    test_image_path = sample_images[0]
    image = cv2.imread(str(test_image_path))
    
    if image is None:
        return {"error": "Could not load test image"}
    
    print(f"📸 Using test image: {test_image_path.name}")
    print(f"🔄 Running {num_iterations} iterations...")
    
    # Warm-up runs
    print("   Warming up...")
    for _ in range(3):
        try:
            components['classifier'].classify_document(image_array=image)
        except:
            pass
    
    # Benchmark runs
    processing_times = []
    memory_usage = []
    successful_runs = 0
    
    for i in range(num_iterations):
        print(f"\r   Progress: {i+1}/{num_iterations}", end="")
        
        try:
            start_time = time.time()
            result = components['classifier'].classify_document(image_array=image)
            end_time = time.time()
            
            processing_time = end_time - start_time
            processing_times.append(processing_time)
            successful_runs += 1
            
        except Exception as e:
            print(f"\n❌ Error in iteration {i+1}: {e}")
    
    print("\n")
    
    if not processing_times:
        return {"error": "No successful benchmark runs"}
    
    # Calculate statistics
    results = {
        "total_iterations": num_iterations,
        "successful_runs": successful_runs,
        "failed_runs": num_iterations - successful_runs,
        "success_rate": successful_runs / num_iterations,
        "avg_processing_time": statistics.mean(processing_times),
        "min_processing_time": min(processing_times),
        "max_processing_time": max(processing_times),
        "median_processing_time": statistics.median(processing_times),
        "std_processing_time": statistics.stdev(processing_times) if len(processing_times) > 1 else 0,
        "throughput_per_second": 1 / statistics.mean(processing_times) if processing_times else 0,
        "processing_times": processing_times
    }
    
    # Display results
    print("⚡ Performance Benchmark Results:")
    print(f"   Total Iterations: {results['total_iterations']}")
    print(f"   Successful Runs: {results['successful_runs']}")
    print(f"   Success Rate: {results['success_rate']:.2%}")
    print(f"   Average Processing Time: {results['avg_processing_time']:.4f}s")
    print(f"   Median Processing Time: {results['median_processing_time']:.4f}s")
    print(f"   Min/Max Processing Time: {results['min_processing_time']:.4f}s / {results['max_processing_time']:.4f}s")
    print(f"   Standard Deviation: {results['std_processing_time']:.4f}s")
    print(f"   Throughput: {results['throughput_per_second']:.2f} images/second")
    
    return results

def test_edge_cases(components) -> Dict[str, Any]:
    """Test model with edge cases and challenging scenarios.
    
    Args:
        components: Loaded model components
        
    Returns:
        Dictionary containing edge case test results
    """
    print("\n🧪 Edge Case Testing...")
    print("-" * 50)
    
    edge_case_results = {
        "total_tests": 0,
        "passed_tests": 0,
        "failed_tests": 0,
        "test_details": []
    }
    
    # Test 1: Empty/None image
    print("   Testing empty image...")
    try:
        result = components['classifier'].classify_document(image_array=None)
        edge_case_results["test_details"].append({
            "test": "empty_image",
            "status": "handled" if result.get('document_type') == 'unknown' else "unexpected",
            "result": result
        })
        edge_case_results["passed_tests"] += 1
    except Exception as e:
        edge_case_results["test_details"].append({
            "test": "empty_image",
            "status": "error",
            "error": str(e)
        })
        edge_case_results["failed_tests"] += 1
    edge_case_results["total_tests"] += 1
    
    # Test 2: Very small image
    print("   Testing very small image...")
    try:
        small_image = np.ones((10, 10, 3), dtype=np.uint8) * 255
        result = components['classifier'].classify_document(image_array=small_image)
        edge_case_results["test_details"].append({
            "test": "small_image",
            "status": "handled",
            "result": result
        })
        edge_case_results["passed_tests"] += 1
    except Exception as e:
        edge_case_results["test_details"].append({
            "test": "small_image",
            "status": "error",
            "error": str(e)
        })
        edge_case_results["failed_tests"] += 1
    edge_case_results["total_tests"] += 1
    
    # Test 3: Very large image
    print("   Testing very large image...")
    try:
        large_image = np.ones((4000, 4000, 3), dtype=np.uint8) * 255
        start_time = time.time()
        result = components['classifier'].classify_document(image_array=large_image)
        processing_time = time.time() - start_time
        edge_case_results["test_details"].append({
            "test": "large_image",
            "status": "handled",
            "processing_time": processing_time,
            "result": result
        })
        edge_case_results["passed_tests"] += 1
    except Exception as e:
        edge_case_results["test_details"].append({
            "test": "large_image",
            "status": "error",
            "error": str(e)
        })
        edge_case_results["failed_tests"] += 1
    edge_case_results["total_tests"] += 1
    
    # Test 4: Corrupted/noise image
    print("   Testing noise image...")
    try:
        noise_image = np.random.randint(0, 256, (500, 500, 3), dtype=np.uint8)
        result = components['classifier'].classify_document(image_array=noise_image)
        edge_case_results["test_details"].append({
            "test": "noise_image",
            "status": "handled",
            "result": result
        })
        edge_case_results["passed_tests"] += 1
    except Exception as e:
        edge_case_results["test_details"].append({
            "test": "noise_image",
            "status": "error",
            "error": str(e)
        })
        edge_case_results["failed_tests"] += 1
    edge_case_results["total_tests"] += 1
    
    # Test 5: Grayscale image
    print("   Testing grayscale image...")
    try:
        gray_image = np.ones((500, 500), dtype=np.uint8) * 128
        # Convert to 3-channel
        gray_image_3ch = cv2.cvtColor(gray_image, cv2.COLOR_GRAY2BGR)
        result = components['classifier'].classify_document(image_array=gray_image_3ch)
        edge_case_results["test_details"].append({
            "test": "grayscale_image",
            "status": "handled",
            "result": result
        })
        edge_case_results["passed_tests"] += 1
    except Exception as e:
        edge_case_results["test_details"].append({
            "test": "grayscale_image",
            "status": "error",
            "error": str(e)
        })
        edge_case_results["failed_tests"] += 1
    edge_case_results["total_tests"] += 1
    
    # Test 6: Rotated image (if sample available)
    sample_images = get_sample_images()
    if sample_images:
        print("   Testing rotated image...")
        try:
            original_image = cv2.imread(str(sample_images[0]))
            if original_image is not None:
                # Rotate image 90 degrees
                rotated_image = cv2.rotate(original_image, cv2.ROTATE_90_CLOCKWISE)
                result = components['classifier'].classify_document(image_array=rotated_image)
                edge_case_results["test_details"].append({
                    "test": "rotated_image",
                    "status": "handled",
                    "result": result
                })
                edge_case_results["passed_tests"] += 1
            else:
                edge_case_results["failed_tests"] += 1
        except Exception as e:
            edge_case_results["test_details"].append({
                "test": "rotated_image",
                "status": "error",
                "error": str(e)
            })
            edge_case_results["failed_tests"] += 1
        edge_case_results["total_tests"] += 1
    
    # Calculate success rate
    edge_case_results["success_rate"] = edge_case_results["passed_tests"] / edge_case_results["total_tests"] if edge_case_results["total_tests"] > 0 else 0
    
    # Display results
    print("\n🧪 Edge Case Test Results:")
    print(f"   Total Tests: {edge_case_results['total_tests']}")
    print(f"   Passed Tests: {edge_case_results['passed_tests']}")
    print(f"   Failed Tests: {edge_case_results['failed_tests']}")
    print(f"   Success Rate: {edge_case_results['success_rate']:.2%}")
    
    print("\n📋 Detailed Results:")
    for test_detail in edge_case_results["test_details"]:
        test_name = test_detail["test"].replace("_", " ").title()
        status = test_detail["status"]
        if status == "handled":
            print(f"   ✅ {test_name}: Handled successfully")
        elif status == "error":
            print(f"   ❌ {test_name}: Error - {test_detail.get('error', 'Unknown error')}")
        else:
            print(f"   ⚠️  {test_name}: Unexpected result")
    
    return edge_case_results

def generate_test_report(accuracy_results: Dict, performance_results: Dict, edge_case_results: Dict) -> str:
    """Generate a comprehensive test report.
    
    Args:
        accuracy_results: Results from accuracy testing
        performance_results: Results from performance benchmarking
        edge_case_results: Results from edge case testing
        
    Returns:
        Formatted test report string
    """
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    
    report = f"""
📊 DOCUMENT PARSER MODEL TEST REPORT
{'='*60}
Generated: {timestamp}

🎯 ACCURACY METRICS:
{'-'*30}
"""
    
    if "error" not in accuracy_results:
        report += f"Detection Rate: {accuracy_results.get('detection_rate', 0):.2%}\n"
        report += f"Average Confidence: {accuracy_results.get('avg_confidence', 0):.3f}\n"
        report += f"Total Images Tested: {accuracy_results.get('total_images', 0)}\n"
        report += f"Successful Detections: {accuracy_results.get('successful_detections', 0)}\n"
    else:
        report += f"Error: {accuracy_results['error']}\n"
    
    report += f"\n⚡ PERFORMANCE METRICS:\n{'-'*30}\n"
    
    if "error" not in performance_results:
        report += f"Average Processing Time: {performance_results.get('avg_processing_time', 0):.4f}s\n"
        report += f"Throughput: {performance_results.get('throughput_per_second', 0):.2f} images/second\n"
        report += f"Success Rate: {performance_results.get('success_rate', 0):.2%}\n"
        report += f"Standard Deviation: {performance_results.get('std_processing_time', 0):.4f}s\n"
    else:
        report += f"Error: {performance_results['error']}\n"
    
    report += f"\n🧪 EDGE CASE TESTING:\n{'-'*30}\n"
    report += f"Success Rate: {edge_case_results.get('success_rate', 0):.2%}\n"
    report += f"Tests Passed: {edge_case_results.get('passed_tests', 0)}/{edge_case_results.get('total_tests', 0)}\n"
    
    report += f"\n📋 RECOMMENDATIONS:\n{'-'*30}\n"
    
    # Generate recommendations based on results
    recommendations = []
    
    if accuracy_results.get('detection_rate', 0) < 0.8:
        recommendations.append("• Consider retraining with more diverse dataset")
    
    if performance_results.get('avg_processing_time', 0) > 2.0:
        recommendations.append("• Optimize model for faster inference")
    
    if edge_case_results.get('success_rate', 0) < 0.8:
        recommendations.append("• Improve error handling for edge cases")
    
    if accuracy_results.get('avg_confidence', 0) < 0.7:
        recommendations.append("• Review model confidence thresholds")
    
    if not recommendations:
        recommendations.append("• Model performance is satisfactory")
    
    for rec in recommendations:
        report += f"{rec}\n"
    
    report += f"\n{'='*60}\n"
    
    return report

def main():
    print("📄 Interactive Document Parser Model Tester")
    print("=" * 40)
    print("Manual testing interface for document parsing functionality")
    print("Supported documents: MyKad, SPK (Sijil Pelajaran Malaysia)\n")
    
    # Test model loading first
    components = test_model_loading()
    if not components:
        print("\n❌ Cannot proceed without working components.")
        print("\nMake sure you have:")
        print("1. Template files in src/document_parser/templates/")
        print("2. Required dependencies installed")
        print("3. Run this script from the project root directory")
        return 1
    
    # Get sample images for quick testing
    sample_images = get_sample_images()
    supported_types = get_supported_document_types()
    
    while True:
        print("\n" + "=" * 50)
        print("📋 DOCUMENT PARSER TESTING MENU")
        print("=" * 50)
        print("1. 📁 Upload custom image file")
        print("2. 🖼️  Test with sample images")
        print("3. 📊 View supported document types")
        print("4. 🎯 Run accuracy validation test")
        print("5. ⚡ Run performance benchmark")
        print("6. 🧪 Run edge case testing")
        print("7. 📊 Run comprehensive test suite")
        print("8. ❌ Exit")
        
        choice = input("\n👉 Select an option (1-8): ").strip()
        
        if choice == '1':
            # Custom image upload
            image_path = input("\n📂 Enter the path to your image file: ").strip()
            if not image_path:
                print("❌ No path provided.")
                continue
                
            image_path = Path(image_path)
            if not image_path.exists():
                print(f"❌ File not found: {image_path}")
                continue
                
            if image_path.suffix.lower() not in ['.jpg', '.jpeg', '.png', '.bmp']:
                print(f"❌ Unsupported file format: {image_path.suffix}")
                continue
            
            # Ask for document type
            print("\n📋 Document Type Options:")
            for i, doc_type in enumerate(supported_types, 1):
                print(f"   {i}. {doc_type}")
            
            type_choice = input("\n👉 Select document type (1-3, default: auto): ").strip()
            if type_choice == '1':
                doc_type = 'mykad'
            elif type_choice == '2':
                doc_type = 'spk'
            else:
                doc_type = 'auto'
            
            # Process the image
            process_single_image(image_path, doc_type, components)
            
        elif choice == '2':
            # Sample images
            if not sample_images:
                print("\n❌ No sample images found.")
                print("Place some test images in one of these directories:")
                print("   • data/document_parser/test/")
                print("   • data/document_parser/sample/")
                print("   • models/document_parser/test_images/")
                print("   • test_images/")
                continue
            
            print("\n📁 Available Sample Images:")
            for i, img_path in enumerate(sample_images, 1):
                print(f"   {i}. {img_path.name}")
            
            img_choice = input(f"\n👉 Select image (1-{len(sample_images)}): ").strip()
            try:
                img_index = int(img_choice) - 1
                if 0 <= img_index < len(sample_images):
                    selected_image = sample_images[img_index]
                    process_single_image(selected_image, 'auto', components)
                else:
                    print("❌ Invalid selection.")
            except ValueError:
                print("❌ Please enter a valid number.")
                
        elif choice == '3':
            # Show supported document types
            print("\n📋 Supported Document Types:")
            for doc_type in supported_types:
                print(f"   • {doc_type}")
            
        elif choice == '4':
            # Accuracy validation test
            test_dataset_path = input("\nEnter test dataset path (or press Enter for default): ").strip()
            if not test_dataset_path:
                test_dataset_path = None
            accuracy_results = test_model_accuracy(components, test_dataset_path)
        
        elif choice == '5':
            # Performance benchmark
            try:
                iterations = int(input("\nEnter number of iterations (default 50): ").strip() or "50")
                performance_results = test_performance_benchmark(components, iterations)
            except ValueError:
                print("❌ Invalid number. Using default 50 iterations.")
                performance_results = test_performance_benchmark(components, 50)
        
        elif choice == '6':
            # Edge case testing
            edge_case_results = test_edge_cases(components)
        
        elif choice == '7':
            # Comprehensive test suite
            print("\n🚀 Running Comprehensive Test Suite...")
            print("This may take a few minutes...\n")
            
            # Run all tests
            accuracy_results = test_model_accuracy(components)
            performance_results = test_performance_benchmark(components, 30)
            edge_case_results = test_edge_cases(components)
            
            # Generate and display report
            report = generate_test_report(accuracy_results, performance_results, edge_case_results)
            print(report)
            
            # Save report to file
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            report_filename = f"test_report_{timestamp}.txt"
            try:
                with open(report_filename, 'w') as f:
                    f.write(report)
                print(f"📄 Test report saved to: {report_filename}")
            except Exception as e:
                print(f"❌ Could not save report: {e}")
        
        elif choice == '8':
            print("\n👋 Goodbye!")
            break
            
        else:
            print("❌ Invalid choice. Please select 1-8.")
    
    return 0

def process_single_image(image_path, document_type, components):
    """Process a single image and display detailed results."""
    print(f"\n🔄 Processing: {image_path.name}")
    print("-" * 50)
    
    try:
        # Load image
        image = cv2.imread(str(image_path))
        if image is None:
            print(f"❌ Failed to load image: {image_path}")
            return
        
        # Process document using individual components
        start_time = time.time()
        
        # Step 1: Classify document
        if document_type == 'auto':
            classification_result = components['classifier'].classify_document(image_array=image)
            detected_type = classification_result.get('document_type', 'unknown')
        else:
            detected_type = document_type
            classification_result = {'document_type': document_type, 'confidence': 1.0}
        
        # Step 2: Extract text using OCR
        ocr_result = components['ocr_service'].extract_text(image)
        
        # Step 3: Extract fields
        field_result = components['field_extractor'].extract_fields(
            ocr_result.get('text', ''),
            document_type=detected_type
        )
        
        # Step 4: Validate results
        validation_result = components['validator'].validate_document(
            field_result,
            document_type=detected_type
        )
        
        processing_time = time.time() - start_time
        
        # Combine results
        results = {
            'processing_time': processing_time,
            'document_type': detected_type,
            'classification_confidence': classification_result.get('confidence', 0.0),
            'image_dimensions': {'width': image.shape[1], 'height': image.shape[0]},
            'extracted_fields': field_result.get('fields', {}),
            'validation_passed': validation_result.get('is_valid', False),
            'validation_errors': validation_result.get('errors', []),
            'ocr_text': ocr_result.get('text', ''),
            'ocr_confidence': ocr_result.get('confidence', 0.0)
        }
        
        # Display detailed results
        display_results(results)
        
        # Ask if user wants to save results
        save_choice = input("\n💾 Save results to file? (y/n): ").strip().lower()
        if save_choice == 'y':
            output_path = Path(f"result_{image_path.stem}_{int(time.time())}.json")
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(results, f, indent=2, ensure_ascii=False)
            print(f"✅ Results saved to: {output_path.absolute()}")
        
    except Exception as e:
        print(f"❌ Error processing image: {str(e)}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    exit(main())