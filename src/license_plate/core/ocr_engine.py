"""EasyOCR-based OCR Engine for Malaysian License Plate Text Recognition

This module provides comprehensive OCR capabilities specifically optimized for
Malaysian license plates, including preprocessing, text extraction, and validation.
"""

import logging
import re
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import cv2
import easyocr
import numpy as np
from PIL import Image, ImageEnhance, ImageFilter

from ..config.settings import (
    OCR_CONFIG,
    OCR_POSTPROCESS_CONFIG,
    MALAYSIAN_PLATE_PATTERNS,
    validate_malaysian_plate,
    get_state_from_plate
)

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class LicensePlateOCR:
    """EasyOCR-based OCR engine optimized for Malaysian license plates."""
    
    def __init__(self, languages: Optional[List[str]] = None, gpu: bool = False):
        """
        Initialize the OCR engine.
        
        Args:
            languages: List of languages for OCR (default: ['en'])
            gpu: Whether to use GPU acceleration
        """
        self.languages = languages or OCR_CONFIG["languages"]
        self.gpu = gpu if gpu else OCR_CONFIG["gpu"]
        self.reader = None
        
        # Initialize EasyOCR reader
        self._initialize_reader()
        
        logger.info(f"LicensePlateOCR initialized with languages: {self.languages}, GPU: {self.gpu}")
    
    def _initialize_reader(self) -> None:
        """Initialize the EasyOCR reader with error handling."""
        try:
            self.reader = easyocr.Reader(
                self.languages,
                gpu=self.gpu,
                verbose=False
            )
            logger.info("EasyOCR reader initialized successfully")
            
        except Exception as e:
            logger.error(f"Failed to initialize EasyOCR reader: {e}")
            raise RuntimeError(f"OCR initialization failed: {e}")
    
    def preprocess_image(self, image: np.ndarray, enhance: bool = True) -> np.ndarray:
        """
        Preprocess image for better OCR accuracy.
        
        Args:
            image: Input image as numpy array
            enhance: Whether to apply image enhancement
            
        Returns:
            Preprocessed image
        """
        try:
            # Convert to PIL Image for better processing
            if len(image.shape) == 3:
                pil_image = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
            else:
                pil_image = Image.fromarray(image)
            
            # Resize if too small
            width, height = pil_image.size
            if width < 200 or height < 50:
                scale_factor = max(200 / width, 50 / height)
                new_width = int(width * scale_factor)
                new_height = int(height * scale_factor)
                pil_image = pil_image.resize((new_width, new_height), Image.LANCZOS)
            
            if enhance:
                # Enhance contrast
                enhancer = ImageEnhance.Contrast(pil_image)
                pil_image = enhancer.enhance(1.5)
                
                # Enhance sharpness
                enhancer = ImageEnhance.Sharpness(pil_image)
                pil_image = enhancer.enhance(1.2)
                
                # Apply slight blur to reduce noise
                pil_image = pil_image.filter(ImageFilter.GaussianBlur(radius=0.5))
            
            # Convert back to numpy array
            processed_image = np.array(pil_image)
            
            # Convert to grayscale if needed
            if len(processed_image.shape) == 3:
                processed_image = cv2.cvtColor(processed_image, cv2.COLOR_RGB2GRAY)
            
            # Apply adaptive thresholding for better text contrast
            processed_image = cv2.adaptiveThreshold(
                processed_image,
                255,
                cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                cv2.THRESH_BINARY,
                11,
                2
            )
            
            return processed_image
            
        except Exception as e:
            logger.error(f"Image preprocessing failed: {e}")
            return image
    
    def extract_text(
        self,
        image: Union[np.ndarray, str, Path],
        preprocess: bool = True,
        confidence_threshold: Optional[float] = None
    ) -> List[Dict]:
        """
        Extract text from license plate image.
        
        Args:
            image: Input image (numpy array, file path, or Path object)
            preprocess: Whether to preprocess the image
            confidence_threshold: Minimum confidence for text detection
            
        Returns:
            List of text detection dictionaries
        """
        if self.reader is None:
            raise RuntimeError("OCR reader not initialized")
        
        # Load image if path is provided
        if isinstance(image, (str, Path)):
            img = cv2.imread(str(image))
            if img is None:
                raise ValueError(f"Could not load image: {image}")
        else:
            img = image.copy()
        
        # Preprocess image if requested
        if preprocess:
            img = self.preprocess_image(img)
        
        # Set confidence threshold
        conf_thresh = confidence_threshold or OCR_CONFIG["confidence_threshold"]
        
        try:
            # Perform OCR
            results = self.reader.readtext(
                img,
                width_ths=OCR_CONFIG["width_ths"],
                height_ths=OCR_CONFIG["height_ths"],
                paragraph=OCR_CONFIG["paragraph"]
            )
            
            # Process results
            text_detections = []
            for detection in results:
                bbox, text, confidence = detection
                
                if confidence >= conf_thresh:
                    # Clean and validate text
                    cleaned_text = self._clean_text(text)
                    
                    if cleaned_text:  # Only add non-empty cleaned text
                        text_detection = {
                            "bbox": self._normalize_bbox(bbox),
                            "text": cleaned_text,
                            "raw_text": text,
                            "confidence": float(confidence),
                            "is_valid_plate": False,
                            "plate_type": "unknown",
                            "state": "Unknown"
                        }
                        
                        # Validate as Malaysian plate
                        is_valid, plate_type = validate_malaysian_plate(cleaned_text)
                        text_detection["is_valid_plate"] = is_valid
                        text_detection["plate_type"] = plate_type
                        
                        if is_valid:
                            text_detection["state"] = get_state_from_plate(cleaned_text)
                        
                        text_detections.append(text_detection)
            
            logger.info(f"Extracted {len(text_detections)} text regions")
            return text_detections
            
        except Exception as e:
            logger.error(f"Text extraction failed: {e}")
            raise RuntimeError(f"OCR failed: {e}")
    
    def _normalize_bbox(self, bbox: List[List[int]]) -> Dict:
        """Normalize bounding box coordinates."""
        # EasyOCR returns bbox as [[x1,y1], [x2,y1], [x2,y2], [x1,y2]]
        x_coords = [point[0] for point in bbox]
        y_coords = [point[1] for point in bbox]
        
        x1, x2 = min(x_coords), max(x_coords)
        y1, y2 = min(y_coords), max(y_coords)
        
        return {
            "x1": x1,
            "y1": y1,
            "x2": x2,
            "y2": y2,
            "width": x2 - x1,
            "height": y2 - y1
        }
    
    def _clean_text(self, text: str) -> str:
        """
        Clean and normalize extracted text for Malaysian license plates.
        
        Args:
            text: Raw OCR text
            
        Returns:
            Cleaned text
        """
        if not text:
            return ""
        
        # Convert to uppercase
        if OCR_POSTPROCESS_CONFIG["convert_to_uppercase"]:
            text = text.upper()
        
        # Remove special characters if configured
        if OCR_POSTPROCESS_CONFIG["remove_special_chars"]:
            allowed_chars = OCR_POSTPROCESS_CONFIG["allowed_chars"]
            text = ''.join(char for char in text if char in allowed_chars)
        
        # Clean common OCR errors for license plates
        text = self._fix_common_ocr_errors(text)
        
        # Remove extra spaces and normalize
        text = ' '.join(text.split())
        
        # Check length constraints
        min_len = OCR_POSTPROCESS_CONFIG["min_text_length"]
        max_len = OCR_POSTPROCESS_CONFIG["max_text_length"]
        
        if len(text) < min_len or len(text) > max_len:
            logger.debug(f"Text length out of range: '{text}' (length: {len(text)})")
            return ""
        
        return text
    
    def _fix_common_ocr_errors(self, text: str) -> str:
        """
        Fix common OCR errors specific to license plates.
        
        Args:
            text: Input text with potential OCR errors
            
        Returns:
            Text with common errors corrected
        """
        # Common character substitutions
        substitutions = {
            # Numbers that look like letters
            '0': 'O',  # Zero to O (context-dependent)
            '1': 'I',  # One to I (context-dependent)
            '5': 'S',  # Five to S (context-dependent)
            '8': 'B',  # Eight to B (context-dependent)
            
            # Letters that look like numbers
            'O': '0',  # O to zero (context-dependent)
            'I': '1',  # I to one (context-dependent)
            'S': '5',  # S to five (context-dependent)
            'B': '8',  # B to eight (context-dependent)
            
            # Common misreads
            'Q': 'O',
            'D': '0',
            'Z': '2',
            'G': '6',
        }
        
        # Apply intelligent substitutions based on context
        # This is a simplified version - in practice, you'd use more sophisticated logic
        corrected_text = text
        
        # Pattern-based corrections for Malaysian plates
        # Example: If we see a pattern like "ABC 123O", the O is likely a 0
        patterns = [
            (r'([A-Z]+\s*\d+)[O]$', r'\g<1>0'),  # O at end after numbers -> 0
            (r'^([A-Z]*)[0]([A-Z]+)', r'\g<1>O\g<2>'),  # 0 between letters -> O
            (r'([A-Z]+\s*\d+)[I]$', r'\g<1>1'),  # I at end after numbers -> 1
            (r'^([A-Z]*)[1]([A-Z]+)', r'\g<1>I\g<2>'),  # 1 between letters -> I
        ]
        
        for pattern, replacement in patterns:
            corrected_text = re.sub(pattern, replacement, corrected_text)
        
        return corrected_text
    
    def extract_best_text(
        self,
        image: Union[np.ndarray, str, Path],
        preprocess: bool = True
    ) -> Optional[Dict]:
        """
        Extract the best (highest confidence valid) text from license plate image.
        
        Args:
            image: Input image
            preprocess: Whether to preprocess the image
            
        Returns:
            Best text detection dictionary or None
        """
        text_detections = self.extract_text(image, preprocess=preprocess)
        
        if not text_detections:
            return None
        
        # Filter valid plates first
        valid_plates = [det for det in text_detections if det["is_valid_plate"]]
        
        if valid_plates:
            # Return highest confidence valid plate
            return max(valid_plates, key=lambda x: x["confidence"])
        else:
            # Return highest confidence detection even if not valid
            return max(text_detections, key=lambda x: x["confidence"])
    
    def process_multiple_regions(
        self,
        plate_regions: List[np.ndarray],
        preprocess: bool = True
    ) -> List[List[Dict]]:
        """
        Process multiple license plate regions for OCR.
        
        Args:
            plate_regions: List of plate region images
            preprocess: Whether to preprocess images
            
        Returns:
            List of text detection lists for each region
        """
        results = []
        
        for i, region in enumerate(plate_regions):
            try:
                text_detections = self.extract_text(region, preprocess=preprocess)
                results.append(text_detections)
                logger.info(f"Processed region {i+1}/{len(plate_regions)}")
                
            except Exception as e:
                logger.error(f"Failed to process region {i+1}: {e}")
                results.append([])
        
        return results
    
    def get_ocr_info(self) -> Dict:
        """
        Get information about the OCR engine.
        
        Returns:
            Dictionary with OCR engine information
        """
        return {
            "engine": "EasyOCR",
            "languages": self.languages,
            "gpu_enabled": self.gpu,
            "confidence_threshold": OCR_CONFIG["confidence_threshold"],
            "supported_patterns": list(MALAYSIAN_PLATE_PATTERNS.keys())
        }
    
    def benchmark_preprocessing(
        self,
        image: np.ndarray,
        methods: Optional[List[str]] = None
    ) -> Dict[str, Dict]:
        """
        Benchmark different preprocessing methods on an image.
        
        Args:
            image: Input image
            methods: List of preprocessing methods to test
            
        Returns:
            Dictionary with results for each method
        """
        if methods is None:
            methods = ["none", "basic", "enhanced"]
        
        results = {}
        
        for method in methods:
            try:
                if method == "none":
                    processed_img = image
                elif method == "basic":
                    processed_img = self.preprocess_image(image, enhance=False)
                elif method == "enhanced":
                    processed_img = self.preprocess_image(image, enhance=True)
                else:
                    continue
                
                # Extract text with this preprocessing
                text_detections = self.extract_text(processed_img, preprocess=False)
                
                # Calculate metrics
                valid_detections = [d for d in text_detections if d["is_valid_plate"]]
                avg_confidence = np.mean([d["confidence"] for d in text_detections]) if text_detections else 0
                
                results[method] = {
                    "total_detections": len(text_detections),
                    "valid_detections": len(valid_detections),
                    "average_confidence": float(avg_confidence),
                    "best_text": text_detections[0]["text"] if text_detections else None
                }
                
            except Exception as e:
                logger.error(f"Benchmark failed for method {method}: {e}")
                results[method] = {"error": str(e)}
        
        return results

# Convenience function for quick OCR
def extract_plate_text(
    image: Union[np.ndarray, str, Path],
    preprocess: bool = True,
    gpu: bool = False
) -> Optional[str]:
    """
    Quick function to extract license plate text from an image.
    
    Args:
        image: Input image
        preprocess: Whether to preprocess the image
        gpu: Whether to use GPU acceleration
        
    Returns:
        Extracted plate text or None
    """
    ocr = LicensePlateOCR(gpu=gpu)
    best_detection = ocr.extract_best_text(image, preprocess=preprocess)
    
    return best_detection["text"] if best_detection else None