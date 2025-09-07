#!/usr/bin/env python3
"""
License Plate Detection Testing Script

This script provides comprehensive testing capabilities for license plate detection
models including single image testing, batch processing, and performance metrics.
"""

import argparse
import os
import sys
import logging
import time
import json
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Any, Optional, Tuple
import cv2
import numpy as np
from collections import defaultdict

# Add src to path for imports
sys.path.append(str(Path(__file__).parent.parent / 'src'))

from license_plate.core.processor import LicensePlateProcessor
from license_plate.core.detector import LicensePlateDetector
from license_plate.core.ocr_engine import LicensePlateOCR
from license_plate.utils.image_utils import ImageProcessor
from license_plate.config.settings import (
    TRAINED_MODELS_DIR, OUTPUTS_DIR, LOGGING_CONFIG,
    DETECTION_CONFIG, OCR_CONFIG
)

# Setup logging
logging.basicConfig(
    level=getattr(logging, LOGGING_CONFIG['level']),
    format=LOGGING_CONFIG['format'],
    handlers=[
        logging.FileHandler(LOGGING_CONFIG['log_file']),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class LicensePlateTester:
    """Comprehensive testing suite for license plate detection."""
    
    def __init__(self, model_path: Optional[str] = None):
        """Initialize tester with optional custom model.
        
        Args:
            model_path: Path to custom trained model
        """
        self.processor = LicensePlateProcessor(model_path=model_path)
        self.image_processor = ImageProcessor()
        self.results = []
        self.metrics = defaultdict(list)
        
    def test_single_image(self, image_path: str, save_output: bool = True, 
                         output_dir: Optional[str] = None) -> Dict[str, Any]:
        """Test license plate detection on a single image.
        
        Args:
            image_path: Path to input image
            save_output: Whether to save annotated output
            output_dir: Directory to save output (optional)
            
        Returns:
            dict: Detection results with metrics
        """
        logger.info(f"Testing single image: {image_path}")
        
        if not Path(image_path).exists():
            raise FileNotFoundError(f"Image not found: {image_path}")
            
        start_time = time.time()
        
        try:
            # Process image
            result = self.processor.process_image(
                image_path=image_path,
                save_results=save_output,
                output_dir=output_dir
            )
            
            processing_time = time.time() - start_time
            
            # Add timing information
            result['processing_time'] = processing_time
            result['image_path'] = image_path
            result['timestamp'] = datetime.now().isoformat()
            
            # Log results
            logger.info(f"Processing completed in {processing_time:.3f}s")
            logger.info(f"Detected {len(result.get('detections', []))} license plates")
            
            for i, detection in enumerate(result.get('detections', [])):
                logger.info(f"Plate {i+1}: {detection.get('text', 'N/A')} (confidence: {detection.get('confidence', 0):.3f})")
                
            return result
            
        except Exception as e:
            logger.error(f"Error processing image {image_path}: {e}")
            return {
                'image_path': image_path,
                'error': str(e),
                'processing_time': time.time() - start_time,
                'timestamp': datetime.now().isoformat()
            }
            
    def test_batch_directory(self, input_dir: str, output_dir: Optional[str] = None,
                           file_extensions: List[str] = None) -> Dict[str, Any]:
        """Test license plate detection on all images in a directory.
        
        Args:
            input_dir: Directory containing input images
            output_dir: Directory to save outputs (optional)
            file_extensions: List of file extensions to process
            
        Returns:
            dict: Batch processing results with metrics
        """
        if file_extensions is None:
            file_extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.tiff']
            
        input_path = Path(input_dir)
        if not input_path.exists():
            raise FileNotFoundError(f"Input directory not found: {input_dir}")
            
        # Find all image files
        image_files = []
        for ext in file_extensions:
            image_files.extend(input_path.glob(f"*{ext}"))
            image_files.extend(input_path.glob(f"*{ext.upper()}"))
            
        if not image_files:
            logger.warning(f"No image files found in {input_dir}")
            return {'error': 'No image files found', 'processed_count': 0}
            
        logger.info(f"Found {len(image_files)} images to process")
        
        # Process images
        batch_results = []
        successful_count = 0
        failed_count = 0
        total_processing_time = 0
        
        for i, image_file in enumerate(image_files, 1):
            logger.info(f"Processing {i}/{len(image_files)}: {image_file.name}")
            
            try:
                result = self.test_single_image(
                    str(image_file),
                    save_output=output_dir is not None,
                    output_dir=output_dir
                )
                
                if 'error' not in result:
                    successful_count += 1
                    total_processing_time += result.get('processing_time', 0)
                else:
                    failed_count += 1
                    
                batch_results.append(result)
                
            except Exception as e:
                logger.error(f"Failed to process {image_file}: {e}")
                failed_count += 1
                batch_results.append({
                    'image_path': str(image_file),
                    'error': str(e),
                    'timestamp': datetime.now().isoformat()
                })
                
        # Calculate batch metrics
        avg_processing_time = total_processing_time / successful_count if successful_count > 0 else 0
        
        batch_summary = {
            'input_directory': str(input_path),
            'output_directory': output_dir,
            'total_images': len(image_files),
            'successful_count': successful_count,
            'failed_count': failed_count,
            'success_rate': successful_count / len(image_files) if image_files else 0,
            'total_processing_time': total_processing_time,
            'average_processing_time': avg_processing_time,
            'timestamp': datetime.now().isoformat(),
            'results': batch_results
        }
        
        logger.info(f"Batch processing completed:")
        logger.info(f"  Total images: {len(image_files)}")
        logger.info(f"  Successful: {successful_count}")
        logger.info(f"  Failed: {failed_count}")
        logger.info(f"  Success rate: {batch_summary['success_rate']:.1%}")
        logger.info(f"  Average processing time: {avg_processing_time:.3f}s")
        
        return batch_summary
        
    def calculate_performance_metrics(self, results: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Calculate comprehensive performance metrics.
        
        Args:
            results: List of detection results
            
        Returns:
            dict: Performance metrics
        """
        if not results:
            return {'error': 'No results to analyze'}
            
        # Filter successful results
        successful_results = [r for r in results if 'error' not in r]
        
        if not successful_results:
            return {'error': 'No successful results to analyze'}
            
        # Processing time metrics
        processing_times = [r.get('processing_time', 0) for r in successful_results]
        
        # Detection metrics
        total_detections = 0
        confidence_scores = []
        text_lengths = []
        
        for result in successful_results:
            detections = result.get('detections', [])
            total_detections += len(detections)
            
            for detection in detections:
                if 'confidence' in detection:
                    confidence_scores.append(detection['confidence'])
                if 'text' in detection and detection['text']:
                    text_lengths.append(len(detection['text']))
                    
        # Calculate metrics
        metrics = {
            'total_images_processed': len(successful_results),
            'total_detections': total_detections,
            'average_detections_per_image': total_detections / len(successful_results),
            'processing_time': {
                'min': min(processing_times) if processing_times else 0,
                'max': max(processing_times) if processing_times else 0,
                'mean': np.mean(processing_times) if processing_times else 0,
                'median': np.median(processing_times) if processing_times else 0,
                'std': np.std(processing_times) if processing_times else 0
            },
            'confidence_scores': {
                'min': min(confidence_scores) if confidence_scores else 0,
                'max': max(confidence_scores) if confidence_scores else 0,
                'mean': np.mean(confidence_scores) if confidence_scores else 0,
                'median': np.median(confidence_scores) if confidence_scores else 0,
                'std': np.std(confidence_scores) if confidence_scores else 0
            },
            'text_analysis': {
                'total_texts_extracted': len(text_lengths),
                'average_text_length': np.mean(text_lengths) if text_lengths else 0,
                'text_extraction_rate': len(text_lengths) / total_detections if total_detections > 0 else 0
            },
            'throughput': {
                'images_per_second': len(successful_results) / sum(processing_times) if sum(processing_times) > 0 else 0,
                'detections_per_second': total_detections / sum(processing_times) if sum(processing_times) > 0 else 0
            }
        }
        
        return metrics
        
    def save_test_report(self, results: Dict[str, Any], output_path: str):
        """Save comprehensive test report to file.
        
        Args:
            results: Test results to save
            output_path: Path to save report
        """
        report = {
            'test_info': {
                'timestamp': datetime.now().isoformat(),
                'model_info': self.processor.get_model_info(),
                'configuration': {
                    'detection_config': DETECTION_CONFIG,
                    'ocr_config': OCR_CONFIG
                }
            },
            'results': results
        }
        
        # Add performance metrics if results contain multiple images
        if isinstance(results.get('results'), list):
            report['performance_metrics'] = self.calculate_performance_metrics(results['results'])
            
        with open(output_path, 'w') as f:
            json.dump(report, f, indent=2, default=str)
            
        logger.info(f"Test report saved to: {output_path}")
        
    def benchmark_model(self, test_images: List[str], iterations: int = 3) -> Dict[str, Any]:
        """Benchmark model performance with multiple iterations.
        
        Args:
            test_images: List of test image paths
            iterations: Number of iterations per image
            
        Returns:
            dict: Benchmark results
        """
        logger.info(f"Benchmarking model with {len(test_images)} images, {iterations} iterations each")
        
        benchmark_results = []
        
        for image_path in test_images:
            image_results = []
            
            for i in range(iterations):
                result = self.test_single_image(image_path, save_output=False)
                if 'error' not in result:
                    image_results.append(result['processing_time'])
                    
            if image_results:
                benchmark_results.append({
                    'image_path': image_path,
                    'iterations': len(image_results),
                    'min_time': min(image_results),
                    'max_time': max(image_results),
                    'mean_time': np.mean(image_results),
                    'std_time': np.std(image_results)
                })
                
        # Overall benchmark metrics
        all_times = []
        for result in benchmark_results:
            # Approximate all iteration times (we only have summary stats)
            all_times.extend([result['mean_time']] * result['iterations'])
            
        benchmark_summary = {
            'total_tests': len(all_times),
            'total_images': len(test_images),
            'iterations_per_image': iterations,
            'overall_performance': {
                'min_time': min(all_times) if all_times else 0,
                'max_time': max(all_times) if all_times else 0,
                'mean_time': np.mean(all_times) if all_times else 0,
                'median_time': np.median(all_times) if all_times else 0,
                'std_time': np.std(all_times) if all_times else 0
            },
            'per_image_results': benchmark_results,
            'timestamp': datetime.now().isoformat()
        }
        
        logger.info(f"Benchmark completed: {benchmark_summary['overall_performance']['mean_time']:.3f}s average")
        
        return benchmark_summary
        
def main():
    """Main testing function with CLI interface."""
    parser = argparse.ArgumentParser(
        description='Test license plate detection model'
    )
    
    # Mode selection
    parser.add_argument(
        'mode',
        choices=['single', 'batch', 'benchmark'],
        help='Testing mode: single image, batch directory, or benchmark'
    )
    
    # Input arguments
    parser.add_argument(
        'input_path',
        help='Path to input image or directory'
    )
    
    # Optional arguments
    parser.add_argument(
        '--model', '-m',
        help='Path to custom trained model'
    )
    
    parser.add_argument(
        '--output-dir', '-o',
        help='Output directory for results and annotations'
    )
    
    parser.add_argument(
        '--report', '-r',
        help='Path to save test report (JSON format)'
    )
    
    parser.add_argument(
        '--extensions', '-ext',
        nargs='+',
        default=['.jpg', '.jpeg', '.png', '.bmp', '.tiff'],
        help='File extensions to process (for batch mode)'
    )
    
    parser.add_argument(
        '--iterations', '-i',
        type=int,
        default=3,
        help='Number of iterations for benchmark mode (default: 3)'
    )
    
    parser.add_argument(
        '--no-save',
        action='store_true',
        help='Do not save annotated output images'
    )
    
    parser.add_argument(
        '--verbose', '-v',
        action='store_true',
        help='Enable verbose logging'
    )
    
    args = parser.parse_args()
    
    # Set logging level
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
        
    # Create output directory if specified
    if args.output_dir:
        Path(args.output_dir).mkdir(parents=True, exist_ok=True)
        
    try:
        # Initialize tester
        tester = LicensePlateTester(model_path=args.model)
        
        # Execute based on mode
        if args.mode == 'single':
            logger.info("Running single image test")
            result = tester.test_single_image(
                image_path=args.input_path,
                save_output=not args.no_save,
                output_dir=args.output_dir
            )
            
        elif args.mode == 'batch':
            logger.info("Running batch directory test")
            result = tester.test_batch_directory(
                input_dir=args.input_path,
                output_dir=args.output_dir,
                file_extensions=args.extensions
            )
            
        elif args.mode == 'benchmark':
            logger.info("Running benchmark test")
            # For benchmark, input_path should be a directory
            input_path = Path(args.input_path)
            if input_path.is_file():
                test_images = [str(input_path)]
            else:
                test_images = []
                for ext in args.extensions:
                    test_images.extend([str(p) for p in input_path.glob(f"*{ext}")])
                    test_images.extend([str(p) for p in input_path.glob(f"*{ext.upper()}")])
                    
            if not test_images:
                logger.error("No test images found for benchmark")
                sys.exit(1)
                
            result = tester.benchmark_model(
                test_images=test_images[:10],  # Limit to 10 images for benchmark
                iterations=args.iterations
            )
            
        # Save report if requested
        if args.report:
            tester.save_test_report(result, args.report)
            
        # Print summary
        if args.mode == 'single':
            if 'error' in result:
                logger.error(f"Test failed: {result['error']}")
            else:
                logger.info(f"Test completed successfully in {result['processing_time']:.3f}s")
                logger.info(f"Detected {len(result.get('detections', []))} license plates")
                
        elif args.mode == 'batch':
            if 'error' in result:
                logger.error(f"Batch test failed: {result['error']}")
            else:
                logger.info(f"Batch test completed: {result['success_rate']:.1%} success rate")
                
        elif args.mode == 'benchmark':
            perf = result['overall_performance']
            logger.info(f"Benchmark completed: {perf['mean_time']:.3f}±{perf['std_time']:.3f}s")
            
    except KeyboardInterrupt:
        logger.info("Testing interrupted by user")
        sys.exit(1)
    except Exception as e:
        logger.error(f"Testing failed with error: {e}")
        sys.exit(1)
        
if __name__ == '__main__':
    main()