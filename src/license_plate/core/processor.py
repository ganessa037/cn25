"""Main License Plate Processor

This module combines YOLOv8 detection and EasyOCR capabilities to provide
a complete license plate recognition pipeline with batch processing and
result formatting for Malaysian license plates.
"""

import logging
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import cv2
import numpy as np
from PIL import Image

from .detector import LicensePlateDetector
from .ocr_engine import LicensePlateOCR
from ..config.settings import (
    DETECTION_CONFIG,
    OCR_CONFIG,
    IMAGE_CONFIG,
    validate_malaysian_plate,
    get_state_from_plate
)

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class LicensePlateProcessor:
    """Complete license plate recognition processor combining detection and OCR."""
    
    def __init__(
        self,
        model_path: Optional[str] = None,
        ocr_languages: Optional[List[str]] = None,
        use_gpu: bool = False,
        confidence_threshold: Optional[float] = None
    ):
        """
        Initialize the license plate processor.
        
        Args:
            model_path: Path to YOLOv8 model file
            ocr_languages: Languages for OCR (default: ['en'])
            use_gpu: Whether to use GPU acceleration
            confidence_threshold: Detection confidence threshold
        """
        self.use_gpu = use_gpu
        self.confidence_threshold = confidence_threshold or DETECTION_CONFIG["confidence_threshold"]
        
        # Initialize detector
        self.detector = LicensePlateDetector(
            model_path=model_path
        )
        
        # Initialize OCR engine
        self.ocr = LicensePlateOCR(
            languages=ocr_languages,
            gpu=use_gpu
        )
        
        logger.info("LicensePlateProcessor initialized successfully")
    
    def process_image(
        self,
        image: Union[np.ndarray, str, Path],
        return_annotated: bool = False,
        preprocess_ocr: bool = True,
        extract_regions: bool = True
    ) -> Dict:
        """
        Process a single image for license plate detection and recognition.
        
        Args:
            image: Input image (numpy array, file path, or Path object)
            return_annotated: Whether to return annotated image
            preprocess_ocr: Whether to preprocess images for OCR
            extract_regions: Whether to extract plate regions
            
        Returns:
            Dictionary containing detection and OCR results
        """
        start_time = time.time()
        
        # Load image if path is provided
        if isinstance(image, (str, Path)):
            img = cv2.imread(str(image))
            if img is None:
                raise ValueError(f"Could not load image: {image}")
            image_path = str(image)
        else:
            img = image.copy()
            image_path = "memory"
        
        # Get image dimensions
        height, width = img.shape[:2]
        
        # Detect license plates
        detection_results = self.detector.detect(img)
        
        # Process each detection
        processed_results = []
        plate_regions = []
        
        for i, detection in enumerate(detection_results):
            # Extract plate region
            if extract_regions:
                plate_regions = self.detector.extract_plate_regions(img, [detection])
                plate_region = plate_regions[0] if plate_regions else None
                plate_regions.append(plate_region)
            else:
                plate_region = None
            
            # Perform OCR on the plate region
            ocr_results = []
            best_text = None
            
            if plate_region is not None:
                try:
                    ocr_detections = self.ocr.extract_text(
                        plate_region,
                        preprocess=preprocess_ocr
                    )
                    ocr_results = ocr_detections
                    
                    # Get best text
                    best_detection = self.ocr.extract_best_text(
                        plate_region,
                        preprocess=preprocess_ocr
                    )
                    best_text = best_detection["text"] if best_detection else None
                    
                except Exception as e:
                    logger.error(f"OCR failed for detection {i}: {e}")
                    ocr_results = []
                    best_text = None
            
            # Combine detection and OCR results
            processed_result = {
                "detection_id": i,
                "bbox": detection["bbox"],
                "detection_confidence": detection["confidence"],
                "class_name": detection["class_name"],
                "plate_text": best_text,
                "ocr_results": ocr_results,
                "is_valid_plate": False,
                "plate_type": "unknown",
                "state": "Unknown",
                "region_extracted": plate_region is not None
            }
            
            # Validate plate text
            if best_text:
                is_valid, plate_type = validate_malaysian_plate(best_text)
                processed_result["is_valid_plate"] = is_valid
                processed_result["plate_type"] = plate_type
                
                if is_valid:
                    processed_result["state"] = get_state_from_plate(best_text)
            
            processed_results.append(processed_result)
        
        # Create final result
        result = {
            "image_path": image_path,
            "image_dimensions": {"width": width, "height": height},
            "processing_time": time.time() - start_time,
            "total_detections": len(detection_results),
            "valid_plates": len([r for r in processed_results if r["is_valid_plate"]]),
            "results": processed_results,
            "metadata": {
                "detector_model": self.detector.get_model_info(),
                "ocr_engine": self.ocr.get_ocr_info(),
                "confidence_threshold": self.confidence_threshold,
                "preprocessing_enabled": preprocess_ocr
            }
        }
        
        # Add annotated image if requested
        if return_annotated:
            annotated_img = self._create_annotated_image(img, processed_results)
            result["annotated_image"] = annotated_img
        
        # Add extracted regions if requested
        if extract_regions and plate_regions:
            result["plate_regions"] = plate_regions
        
        logger.info(
            f"Processed image: {len(detection_results)} detections, "
            f"{result['valid_plates']} valid plates in {result['processing_time']:.2f}s"
        )
        
        return result
    
    def process_batch(
        self,
        images: List[Union[np.ndarray, str, Path]],
        return_annotated: bool = False,
        preprocess_ocr: bool = True,
        extract_regions: bool = True,
        progress_callback: Optional[callable] = None
    ) -> List[Dict]:
        """
        Process multiple images in batch.
        
        Args:
            images: List of images to process
            return_annotated: Whether to return annotated images
            preprocess_ocr: Whether to preprocess images for OCR
            extract_regions: Whether to extract plate regions
            progress_callback: Optional callback function for progress updates
            
        Returns:
            List of processing results
        """
        results = []
        total_images = len(images)
        
        logger.info(f"Starting batch processing of {total_images} images")
        
        for i, image in enumerate(images):
            try:
                result = self.process_image(
                    image,
                    return_annotated=return_annotated,
                    preprocess_ocr=preprocess_ocr,
                    extract_regions=extract_regions
                )
                
                # Add batch information
                result["batch_index"] = i
                result["batch_total"] = total_images
                
                results.append(result)
                
                # Call progress callback if provided
                if progress_callback:
                    progress_callback(i + 1, total_images, result)
                
                logger.info(f"Processed image {i+1}/{total_images}")
                
            except Exception as e:
                logger.error(f"Failed to process image {i+1}: {e}")
                error_result = {
                    "batch_index": i,
                    "batch_total": total_images,
                    "error": str(e),
                    "image_path": str(image) if isinstance(image, (str, Path)) else "memory",
                    "processing_time": 0,
                    "total_detections": 0,
                    "valid_plates": 0,
                    "results": []
                }
                results.append(error_result)
        
        logger.info(f"Batch processing completed: {len(results)} results")
        return results
    
    def process_directory(
        self,
        directory_path: Union[str, Path],
        output_dir: Optional[Union[str, Path]] = None,
        image_extensions: Optional[List[str]] = None,
        save_annotated: bool = False,
        save_regions: bool = False,
        recursive: bool = False
    ) -> Dict:
        """
        Process all images in a directory.
        
        Args:
            directory_path: Path to directory containing images
            output_dir: Directory to save results (optional)
            image_extensions: List of image file extensions to process
            save_annotated: Whether to save annotated images
            save_regions: Whether to save extracted plate regions
            recursive: Whether to search subdirectories
            
        Returns:
            Summary of processing results
        """
        directory_path = Path(directory_path)
        if not directory_path.exists():
            raise ValueError(f"Directory does not exist: {directory_path}")
        
        # Default image extensions
        if image_extensions is None:
            image_extensions = IMAGE_CONFIG["supported_formats"]
        
        # Find all image files
        image_files = []
        for ext in image_extensions:
            if recursive:
                pattern = f"**/*{ext}"
            else:
                pattern = f"*{ext}"
            image_files.extend(directory_path.glob(pattern))
        
        if not image_files:
            logger.warning(f"No image files found in {directory_path}")
            return {"total_files": 0, "processed_files": 0, "results": []}
        
        logger.info(f"Found {len(image_files)} image files to process")
        
        # Process images
        results = self.process_batch(
            image_files,
            return_annotated=save_annotated,
            extract_regions=save_regions
        )
        
        # Save results if output directory is specified
        if output_dir:
            output_dir = Path(output_dir)
            output_dir.mkdir(parents=True, exist_ok=True)
            
            self._save_batch_results(
                results,
                output_dir,
                save_annotated=save_annotated,
                save_regions=save_regions
            )
        
        # Create summary
        summary = self._create_batch_summary(results)
        summary["source_directory"] = str(directory_path)
        summary["output_directory"] = str(output_dir) if output_dir else None
        
        return summary
    
    def _create_annotated_image(
        self,
        image: np.ndarray,
        results: List[Dict]
    ) -> np.ndarray:
        """
        Create annotated image with detection and OCR results.
        
        Args:
            image: Original image
            results: Processing results
            
        Returns:
            Annotated image
        """
        annotated = image.copy()
        
        for result in results:
            bbox = result["bbox"]
            plate_text = result["plate_text"]
            confidence = result["detection_confidence"]
            is_valid = result["is_valid_plate"]
            
            # Draw bounding box
            color = (0, 255, 0) if is_valid else (0, 165, 255)  # Green for valid, orange for invalid
            cv2.rectangle(
                annotated,
                (int(bbox["x1"]), int(bbox["y1"])),
                (int(bbox["x2"]), int(bbox["y2"])),
                color,
                2
            )
            
            # Add text label
            label_parts = [f"Conf: {confidence:.2f}"]
            if plate_text:
                label_parts.append(f"Text: {plate_text}")
            if is_valid:
                label_parts.append(f"State: {result['state']}")
            
            label = " | ".join(label_parts)
            
            # Calculate text position
            text_y = max(int(bbox["y1"]) - 10, 20)
            
            # Draw text background
            (text_width, text_height), _ = cv2.getTextSize(
                label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1
            )
            cv2.rectangle(
                annotated,
                (int(bbox["x1"]), text_y - text_height - 5),
                (int(bbox["x1"]) + text_width, text_y + 5),
                color,
                -1
            )
            
            # Draw text
            cv2.putText(
                annotated,
                label,
                (int(bbox["x1"]), text_y),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (255, 255, 255),
                1
            )
        
        return annotated
    
    def _save_batch_results(
        self,
        results: List[Dict],
        output_dir: Path,
        save_annotated: bool = False,
        save_regions: bool = False
    ) -> None:
        """
        Save batch processing results to files.
        
        Args:
            results: Processing results
            output_dir: Output directory
            save_annotated: Whether to save annotated images
            save_regions: Whether to save plate regions
        """
        import json
        
        # Save JSON results
        json_path = output_dir / "results.json"
        
        # Prepare results for JSON serialization
        json_results = []
        for result in results:
            json_result = result.copy()
            
            # Remove non-serializable items
            json_result.pop("annotated_image", None)
            json_result.pop("plate_regions", None)
            
            json_results.append(json_result)
        
        with open(json_path, 'w') as f:
            json.dump(json_results, f, indent=2, default=str)
        
        logger.info(f"Saved results to {json_path}")
        
        # Save annotated images
        if save_annotated:
            annotated_dir = output_dir / "annotated"
            annotated_dir.mkdir(exist_ok=True)
            
            for i, result in enumerate(results):
                if "annotated_image" in result:
                    filename = f"annotated_{i:04d}.jpg"
                    cv2.imwrite(
                        str(annotated_dir / filename),
                        result["annotated_image"]
                    )
        
        # Save plate regions
        if save_regions:
            regions_dir = output_dir / "regions"
            regions_dir.mkdir(exist_ok=True)
            
            for i, result in enumerate(results):
                if "plate_regions" in result:
                    for j, region in enumerate(result["plate_regions"]):
                        filename = f"region_{i:04d}_{j:02d}.jpg"
                        cv2.imwrite(
                            str(regions_dir / filename),
                            region
                        )
    
    def _create_batch_summary(self, results: List[Dict]) -> Dict:
        """
        Create summary statistics for batch processing results.
        
        Args:
            results: Processing results
            
        Returns:
            Summary dictionary
        """
        total_files = len(results)
        successful_files = len([r for r in results if "error" not in r])
        total_detections = sum(r.get("total_detections", 0) for r in results)
        total_valid_plates = sum(r.get("valid_plates", 0) for r in results)
        total_processing_time = sum(r.get("processing_time", 0) for r in results)
        
        # Calculate average processing time
        avg_processing_time = total_processing_time / successful_files if successful_files > 0 else 0
        
        # Count by plate types and states
        plate_types = {}
        states = {}
        
        for result in results:
            for detection_result in result.get("results", []):
                if detection_result["is_valid_plate"]:
                    plate_type = detection_result["plate_type"]
                    state = detection_result["state"]
                    
                    plate_types[plate_type] = plate_types.get(plate_type, 0) + 1
                    states[state] = states.get(state, 0) + 1
        
        return {
            "total_files": total_files,
            "processed_files": successful_files,
            "failed_files": total_files - successful_files,
            "total_detections": total_detections,
            "total_valid_plates": total_valid_plates,
            "total_processing_time": total_processing_time,
            "average_processing_time": avg_processing_time,
            "detection_rate": total_detections / successful_files if successful_files > 0 else 0,
            "validation_rate": total_valid_plates / total_detections if total_detections > 0 else 0,
            "plate_types": plate_types,
            "states": states,
            "results": results
        }
    
    def get_processor_info(self) -> Dict:
        """
        Get information about the processor components.
        
        Returns:
            Dictionary with processor information
        """
        return {
            "detector": self.detector.get_model_info(),
            "ocr": self.ocr.get_ocr_info(),
            "gpu_enabled": self.use_gpu,
            "confidence_threshold": self.confidence_threshold
        }

# Convenience functions for quick processing
def process_single_image(
    image: Union[np.ndarray, str, Path],
    model_path: Optional[str] = None,
    use_gpu: bool = False
) -> Dict:
    """
    Quick function to process a single image.
    
    Args:
        image: Input image
        model_path: Path to YOLOv8 model
        use_gpu: Whether to use GPU
        
    Returns:
        Processing result
    """
    processor = LicensePlateProcessor(
        model_path=model_path,
        use_gpu=use_gpu
    )
    return processor.process_image(image)

def extract_plate_info(
    image: Union[np.ndarray, str, Path],
    model_path: Optional[str] = None
) -> List[str]:
    """
    Quick function to extract just the plate text from an image.
    
    Args:
        image: Input image
        model_path: Path to YOLOv8 model
        
    Returns:
        List of detected plate texts
    """
    result = process_single_image(image, model_path)
    return [
        r["plate_text"] for r in result["results"]
        if r["plate_text"] and r["is_valid_plate"]
    ]