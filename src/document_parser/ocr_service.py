#!/usr/bin/env python3
"""
OCR Service Module

Provides optical character recognition capabilities using multiple OCR engines
with preprocessing and multilingual support for Malaysian documents.
"""

import logging
import cv2
import numpy as np
from typing import Dict, List, Optional, Tuple, Union
from pathlib import Path
import pytesseract
import easyocr
from PIL import Image, ImageEnhance, ImageFilter
import re

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class OCRService:
    """
    OCR service providing text extraction from document images.
    
    Features:
    - Multiple OCR engines (Tesseract, EasyOCR)
    - Image preprocessing for better accuracy
    - Multilingual support (English, Malay, Chinese)
    - Confidence scoring and result validation
    """
    
    def __init__(self, 
                 engines: List[str] = None,
                 languages: List[str] = None,
                 tesseract_path: Optional[str] = None):
        """
        Initialize the OCR service.
        
        Args:
            engines: List of OCR engines to use ['tesseract', 'easyocr']
            languages: List of language codes ['en', 'ms', 'zh']
            tesseract_path: Path to tesseract executable
        """
        self.engines = engines or ['tesseract', 'easyocr']
        self.languages = languages or ['en', 'ms']
        
        # Set tesseract path if provided
        if tesseract_path:
            pytesseract.pytesseract.tesseract_cmd = tesseract_path
        
        # Initialize EasyOCR reader
        self.easyocr_reader = None
        if 'easyocr' in self.engines:
            try:
                self.easyocr_reader = easyocr.Reader(self.languages, gpu=False)
                logger.info(f"EasyOCR initialized with languages: {self.languages}")
            except Exception as e:
                logger.warning(f"Failed to initialize EasyOCR: {e}")
                self.engines = [eng for eng in self.engines if eng != 'easyocr']
        
        # Tesseract language mapping
        self.tesseract_lang_map = {
            'en': 'eng',
            'ms': 'msa',
            'zh': 'chi_sim'
        }
        
        logger.info(f"OCRService initialized with engines: {self.engines}")
    
    def extract_text(self, 
                    image_input: Union[str, np.ndarray, Image.Image],
                    preprocess: bool = True,
                    engine: Optional[str] = None) -> Dict:
        """
        Extract text from document image.
        
        Args:
            image_input: Image file path, numpy array, or PIL Image
            preprocess: Whether to apply image preprocessing
            engine: Specific OCR engine to use
            
        Returns:
            Dict: Extraction results with text, confidence, and metadata
        """
        try:
            # Load and preprocess image
            image = self._load_image(image_input)
            if preprocess:
                image = self._preprocess_image(image)
            
            # Extract text using specified or all engines
            engines_to_use = [engine] if engine else self.engines
            results = {}
            
            for eng in engines_to_use:
                if eng == 'tesseract':
                    results['tesseract'] = self._extract_with_tesseract(image)
                elif eng == 'easyocr' and self.easyocr_reader:
                    results['easyocr'] = self._extract_with_easyocr(image)
            
            # Combine results
            final_result = self._combine_results(results)
            
            return {
                "text": final_result["text"],
                "confidence": final_result["confidence"],
                "engine_results": results,
                "word_count": len(final_result["text"].split()),
                "character_count": len(final_result["text"]),
                "preprocessing_applied": preprocess
            }
            
        except Exception as e:
            logger.error(f"OCR extraction error: {e}")
            return {
                "text": "",
                "confidence": 0.0,
                "error": str(e),
                "engine_results": {},
                "word_count": 0,
                "character_count": 0,
                "preprocessing_applied": preprocess
            }
    
    def extract_text_with_coordinates(self, 
                                    image_input: Union[str, np.ndarray, Image.Image],
                                    preprocess: bool = True) -> Dict:
        """
        Extract text with bounding box coordinates.
        
        Args:
            image_input: Image file path, numpy array, or PIL Image
            preprocess: Whether to apply image preprocessing
            
        Returns:
            Dict: Text extraction with coordinate information
        """
        try:
            image = self._load_image(image_input)
            if preprocess:
                image = self._preprocess_image(image)
            
            # Convert to format suitable for coordinate extraction
            if isinstance(image, Image.Image):
                cv_image = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
            else:
                cv_image = image
            
            results = {
                "words": [],
                "lines": [],
                "blocks": [],
                "full_text": ""
            }
            
            # Tesseract with coordinates
            if 'tesseract' in self.engines:
                tesseract_data = pytesseract.image_to_data(
                    cv_image, 
                    lang='+'.join([self.tesseract_lang_map.get(lang, lang) for lang in self.languages]),
                    output_type=pytesseract.Output.DICT
                )
                
                words = []
                for i in range(len(tesseract_data['text'])):
                    if int(tesseract_data['conf'][i]) > 30:  # Confidence threshold
                        word_info = {
                            "text": tesseract_data['text'][i].strip(),
                            "confidence": int(tesseract_data['conf'][i]),
                            "bbox": {
                                "x": tesseract_data['left'][i],
                                "y": tesseract_data['top'][i],
                                "width": tesseract_data['width'][i],
                                "height": tesseract_data['height'][i]
                            },
                            "block_num": tesseract_data['block_num'][i],
                            "line_num": tesseract_data['line_num'][i]
                        }
                        if word_info["text"]:  # Only add non-empty text
                            words.append(word_info)
                
                results["words"] = words
                results["full_text"] = " ".join([w["text"] for w in words])
            
            # EasyOCR with coordinates
            if 'easyocr' in self.engines and self.easyocr_reader:
                easyocr_results = self.easyocr_reader.readtext(cv_image)
                
                easyocr_words = []
                for (bbox, text, confidence) in easyocr_results:
                    if confidence > 0.3:  # Confidence threshold
                        # Convert bbox format
                        x_coords = [point[0] for point in bbox]
                        y_coords = [point[1] for point in bbox]
                        
                        word_info = {
                            "text": text.strip(),
                            "confidence": int(confidence * 100),
                            "bbox": {
                                "x": int(min(x_coords)),
                                "y": int(min(y_coords)),
                                "width": int(max(x_coords) - min(x_coords)),
                                "height": int(max(y_coords) - min(y_coords))
                            },
                            "engine": "easyocr"
                        }
                        easyocr_words.append(word_info)
                
                results["easyocr_words"] = easyocr_words
                if not results["full_text"]:  # Use EasyOCR if Tesseract failed
                    results["full_text"] = " ".join([w["text"] for w in easyocr_words])
            
            return results
            
        except Exception as e:
            logger.error(f"Coordinate extraction error: {e}")
            return {
                "words": [],
                "lines": [],
                "blocks": [],
                "full_text": "",
                "error": str(e)
            }
    
    def _load_image(self, image_input: Union[str, np.ndarray, Image.Image]) -> Image.Image:
        """
        Load image from various input formats.
        
        Args:
            image_input: Image in various formats
            
        Returns:
            Image.Image: PIL Image object
        """
        if isinstance(image_input, str):
            return Image.open(image_input)
        elif isinstance(image_input, np.ndarray):
            return Image.fromarray(cv2.cvtColor(image_input, cv2.COLOR_BGR2RGB))
        elif isinstance(image_input, Image.Image):
            return image_input
        else:
            raise ValueError(f"Unsupported image input type: {type(image_input)}")
    
    def _preprocess_image(self, image: Image.Image) -> Image.Image:
        """
        Apply preprocessing to improve OCR accuracy.
        
        Args:
            image: Input PIL Image
            
        Returns:
            Image.Image: Preprocessed image
        """
        try:
            # Convert to grayscale
            if image.mode != 'L':
                image = image.convert('L')
            
            # Enhance contrast
            enhancer = ImageEnhance.Contrast(image)
            image = enhancer.enhance(1.5)
            
            # Enhance sharpness
            enhancer = ImageEnhance.Sharpness(image)
            image = enhancer.enhance(2.0)
            
            # Apply slight blur to reduce noise
            image = image.filter(ImageFilter.MedianFilter(size=3))
            
            # Convert to numpy for OpenCV operations
            img_array = np.array(image)
            
            # Apply adaptive thresholding
            img_array = cv2.adaptiveThreshold(
                img_array, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2
            )
            
            # Morphological operations to clean up
            kernel = np.ones((1, 1), np.uint8)
            img_array = cv2.morphologyEx(img_array, cv2.MORPH_CLOSE, kernel)
            
            # Convert back to PIL
            return Image.fromarray(img_array)
            
        except Exception as e:
            logger.warning(f"Preprocessing failed: {e}, using original image")
            return image
    
    def _extract_with_tesseract(self, image: Image.Image) -> Dict:
        """
        Extract text using Tesseract OCR.
        
        Args:
            image: PIL Image object
            
        Returns:
            Dict: Tesseract extraction results
        """
        try:
            # Configure Tesseract
            lang_string = '+'.join([self.tesseract_lang_map.get(lang, lang) for lang in self.languages])
            
            # Extract text
            text = pytesseract.image_to_string(image, lang=lang_string)
            
            # Get confidence score
            data = pytesseract.image_to_data(image, lang=lang_string, output_type=pytesseract.Output.DICT)
            confidences = [int(conf) for conf in data['conf'] if int(conf) > 0]
            avg_confidence = np.mean(confidences) if confidences else 0
            
            return {
                "text": text.strip(),
                "confidence": float(avg_confidence),
                "engine": "tesseract",
                "language": lang_string
            }
            
        except Exception as e:
            logger.error(f"Tesseract extraction error: {e}")
            return {
                "text": "",
                "confidence": 0.0,
                "engine": "tesseract",
                "error": str(e)
            }
    
    def _extract_with_easyocr(self, image: Image.Image) -> Dict:
        """
        Extract text using EasyOCR.
        
        Args:
            image: PIL Image object
            
        Returns:
            Dict: EasyOCR extraction results
        """
        try:
            # Convert PIL to numpy array
            img_array = np.array(image)
            
            # Extract text
            results = self.easyocr_reader.readtext(img_array)
            
            # Combine all text
            text_parts = []
            confidences = []
            
            for (bbox, text, confidence) in results:
                if confidence > 0.1:  # Filter low confidence
                    text_parts.append(text)
                    confidences.append(confidence)
            
            combined_text = " ".join(text_parts)
            avg_confidence = np.mean(confidences) if confidences else 0
            
            return {
                "text": combined_text.strip(),
                "confidence": float(avg_confidence * 100),  # Convert to percentage
                "engine": "easyocr",
                "detections": len(results)
            }
            
        except Exception as e:
            logger.error(f"EasyOCR extraction error: {e}")
            return {
                "text": "",
                "confidence": 0.0,
                "engine": "easyocr",
                "error": str(e)
            }
    
    def _combine_results(self, results: Dict) -> Dict:
        """
        Combine results from multiple OCR engines.
        
        Args:
            results: Dictionary of engine results
            
        Returns:
            Dict: Combined result
        """
        if not results:
            return {"text": "", "confidence": 0.0}
        
        # Find the result with highest confidence
        best_result = None
        best_confidence = 0
        
        for engine, result in results.items():
            if result.get("confidence", 0) > best_confidence:
                best_confidence = result["confidence"]
                best_result = result
        
        if best_result:
            return {
                "text": best_result["text"],
                "confidence": best_result["confidence"],
                "best_engine": best_result["engine"]
            }
        
        # Fallback: combine all text
        all_text = " ".join([r.get("text", "") for r in results.values()])
        avg_confidence = np.mean([r.get("confidence", 0) for r in results.values()])
        
        return {
            "text": all_text.strip(),
            "confidence": float(avg_confidence),
            "best_engine": "combined"
        }
    
    def clean_text(self, text: str) -> str:
        """
        Clean extracted text by removing noise and formatting.
        
        Args:
            text: Raw extracted text
            
        Returns:
            str: Cleaned text
        """
        # Remove extra whitespace
        text = re.sub(r'\s+', ' ', text)
        
        # Remove special characters that are likely OCR errors
        text = re.sub(r'[^\w\s\-\.\,\:\/\(\)\@]', '', text)
        
        # Fix common OCR mistakes
        replacements = {
            '0': 'O',  # In text contexts
            '1': 'I',  # In text contexts
            '5': 'S',  # In text contexts
        }
        
        # Apply replacements contextually
        words = text.split()
        cleaned_words = []
        
        for word in words:
            # Only apply replacements if word looks like text (not numbers)
            if not word.isdigit() and len(word) > 1:
                for old, new in replacements.items():
                    if old in word and not any(c.isdigit() for c in word.replace(old, '')):
                        word = word.replace(old, new)
            cleaned_words.append(word)
        
        return ' '.join(cleaned_words).strip()
    
    def get_engine_info(self) -> Dict:
        """
        Get information about available OCR engines.
        
        Returns:
            Dict: Engine information
        """
        info = {
            "available_engines": self.engines,
            "languages": self.languages,
            "tesseract_available": 'tesseract' in self.engines,
            "easyocr_available": self.easyocr_reader is not None
        }
        
        # Test Tesseract
        if 'tesseract' in self.engines:
            try:
                version = pytesseract.get_tesseract_version()
                info["tesseract_version"] = str(version)
            except Exception as e:
                info["tesseract_error"] = str(e)
        
        return info