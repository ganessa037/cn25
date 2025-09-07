#!/usr/bin/env python3
"""
OCR Engines for Document Parser

This module provides multi-language OCR capabilities using Tesseract, PaddleOCR,
EasyOCR, and cloud APIs with support for Malaysian languages (Malay, English,
Chinese, Tamil), following the organizational patterns established by the
autocorrect model.

Author: AI Assistant
Date: 2024
"""

import os
import sys
import json
import cv2
import numpy as np
from PIL import Image
from pathlib import Path
from typing import Dict, List, Tuple, Any, Optional, Union
from datetime import datetime
import logging
from dataclasses import dataclass, asdict
from abc import ABC, abstractmethod
import base64
import io
from concurrent.futures import ThreadPoolExecutor, as_completed
import time

# Add src directory to path for imports
sys.path.append(str(Path(__file__).parent.parent.parent))

@dataclass
class OCRConfig:
    """Configuration for OCR engines"""
    engine_type: str = "tesseract"  # tesseract, paddleocr, easyocr, google_vision, aws_textract, azure_cognitive
    languages: List[str] = None  # ['en', 'ms', 'chi_sim', 'tam']
    confidence_threshold: float = 0.6
    preprocessing: bool = True
    deskew: bool = True
    denoise: bool = True
    enhance_contrast: bool = True
    resize_factor: float = 2.0
    psm_mode: int = 6  # Tesseract Page Segmentation Mode
    oem_mode: int = 3  # Tesseract OCR Engine Mode
    parallel_processing: bool = True
    max_workers: int = 4
    
    def __post_init__(self):
        if self.languages is None:
            self.languages = ['en', 'ms', 'chi_sim', 'tam']  # Malaysian languages
    
    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'OCRConfig':
        return cls(**data)

@dataclass
class OCRResult:
    """OCR result data structure"""
    text: str
    confidence: float
    bbox: Tuple[int, int, int, int]  # (x1, y1, x2, y2)
    language: str
    engine: str
    processing_time: float
    
    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

class BaseOCREngine(ABC):
    """Abstract base class for OCR engines"""
    
    def __init__(self, config: OCRConfig):
        self.config = config
        self.logger = self._setup_logging()
    
    def _setup_logging(self) -> logging.Logger:
        """Setup logging configuration"""
        logger = logging.getLogger(f'{self.__class__.__name__}')
        logger.setLevel(logging.INFO)
        
        # Create console handler
        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.INFO)
        
        # Create formatter
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        console_handler.setFormatter(formatter)
        
        # Add handler to logger
        logger.addHandler(console_handler)
        
        return logger
    
    def preprocess_image(self, image: np.ndarray) -> np.ndarray:
        """Preprocess image for better OCR results"""
        if not self.config.preprocessing:
            return image
        
        processed = image.copy()
        
        # Convert to grayscale if needed
        if len(processed.shape) == 3:
            processed = cv2.cvtColor(processed, cv2.COLOR_BGR2GRAY)
        
        # Resize image
        if self.config.resize_factor != 1.0:
            height, width = processed.shape[:2]
            new_width = int(width * self.config.resize_factor)
            new_height = int(height * self.config.resize_factor)
            processed = cv2.resize(processed, (new_width, new_height), interpolation=cv2.INTER_CUBIC)
        
        # Denoise
        if self.config.denoise:
            processed = cv2.fastNlMeansDenoising(processed)
        
        # Enhance contrast
        if self.config.enhance_contrast:
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
            processed = clahe.apply(processed)
        
        # Deskew
        if self.config.deskew:
            processed = self._deskew_image(processed)
        
        # Binarization
        processed = cv2.threshold(processed, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]
        
        return processed
    
    def _deskew_image(self, image: np.ndarray) -> np.ndarray:
        """Deskew image using Hough line detection"""
        try:
            # Edge detection
            edges = cv2.Canny(image, 50, 150, apertureSize=3)
            
            # Hough line detection
            lines = cv2.HoughLines(edges, 1, np.pi/180, threshold=100)
            
            if lines is not None:
                # Calculate average angle
                angles = []
                for rho, theta in lines[:, 0]:
                    angle = theta * 180 / np.pi
                    if angle < 45:
                        angles.append(angle)
                    elif angle > 135:
                        angles.append(angle - 180)
                
                if angles:
                    median_angle = np.median(angles)
                    
                    # Rotate image
                    if abs(median_angle) > 0.5:  # Only rotate if significant skew
                        height, width = image.shape[:2]
                        center = (width // 2, height // 2)
                        rotation_matrix = cv2.getRotationMatrix2D(center, median_angle, 1.0)
                        image = cv2.warpAffine(image, rotation_matrix, (width, height), 
                                             flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)
        except Exception as e:
            self.logger.warning(f"Deskewing failed: {e}")
        
        return image
    
    @abstractmethod
    def extract_text(self, image: Union[str, np.ndarray, Image.Image], 
                    bbox: Optional[Tuple[int, int, int, int]] = None) -> List[OCRResult]:
        """Extract text from image"""
        pass
    
    @abstractmethod
    def is_available(self) -> bool:
        """Check if OCR engine is available"""
        pass

class TesseractEngine(BaseOCREngine):
    """Tesseract OCR Engine"""
    
    def __init__(self, config: OCRConfig):
        super().__init__(config)
        self.engine_name = "tesseract"
        
        try:
            import pytesseract
            self.pytesseract = pytesseract
            
            # Language mapping for Tesseract
            self.lang_map = {
                'en': 'eng',
                'ms': 'msa',  # Malay
                'chi_sim': 'chi_sim',  # Simplified Chinese
                'tam': 'tam'  # Tamil
            }
            
            # Configure Tesseract
            self.tesseract_config = f'--psm {config.psm_mode} --oem {config.oem_mode}'
            
            self.logger.info("Tesseract engine initialized")
        except ImportError:
            self.logger.error("pytesseract not installed. Install with: pip install pytesseract")
            self.pytesseract = None
    
    def is_available(self) -> bool:
        """Check if Tesseract is available"""
        return self.pytesseract is not None
    
    def extract_text(self, image: Union[str, np.ndarray, Image.Image], 
                    bbox: Optional[Tuple[int, int, int, int]] = None) -> List[OCRResult]:
        """Extract text using Tesseract"""
        if not self.is_available():
            return []
        
        start_time = time.time()
        
        # Load and preprocess image
        if isinstance(image, str):
            img = cv2.imread(image)
        elif isinstance(image, Image.Image):
            img = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
        else:
            img = image.copy()
        
        # Crop to bbox if provided
        if bbox:
            x1, y1, x2, y2 = bbox
            img = img[y1:y2, x1:x2]
        
        # Preprocess
        processed_img = self.preprocess_image(img)
        
        results = []
        
        # Try each language
        for lang_code in self.config.languages:
            if lang_code not in self.lang_map:
                continue
            
            tesseract_lang = self.lang_map[lang_code]
            
            try:
                # Extract text with confidence
                data = self.pytesseract.image_to_data(
                    processed_img, 
                    lang=tesseract_lang, 
                    config=self.tesseract_config,
                    output_type=self.pytesseract.Output.DICT
                )
                
                # Process results
                for i in range(len(data['text'])):
                    text = data['text'][i].strip()
                    confidence = float(data['conf'][i]) / 100.0
                    
                    if text and confidence >= self.config.confidence_threshold:
                        x, y, w, h = data['left'][i], data['top'][i], data['width'][i], data['height'][i]
                        
                        # Adjust coordinates if bbox was provided
                        if bbox:
                            x += bbox[0]
                            y += bbox[1]
                        
                        result = OCRResult(
                            text=text,
                            confidence=confidence,
                            bbox=(x, y, x + w, y + h),
                            language=lang_code,
                            engine=self.engine_name,
                            processing_time=time.time() - start_time
                        )
                        results.append(result)
                        
            except Exception as e:
                self.logger.warning(f"Tesseract extraction failed for {lang_code}: {e}")
        
        return results

class PaddleOCREngine(BaseOCREngine):
    """PaddleOCR Engine"""
    
    def __init__(self, config: OCRConfig):
        super().__init__(config)
        self.engine_name = "paddleocr"
        
        try:
            from paddleocr import PaddleOCR
            
            # Language mapping for PaddleOCR
            self.lang_map = {
                'en': 'en',
                'ms': 'en',  # Use English for Malay (similar script)
                'chi_sim': 'ch',
                'tam': 'ta'
            }
            
            # Initialize PaddleOCR for each language
            self.ocr_engines = {}
            for lang_code in self.config.languages:
                if lang_code in self.lang_map:
                    paddle_lang = self.lang_map[lang_code]
                    try:
                        self.ocr_engines[lang_code] = PaddleOCR(
                            use_angle_cls=True, 
                            lang=paddle_lang,
                            show_log=False
                        )
                    except Exception as e:
                        self.logger.warning(f"Failed to initialize PaddleOCR for {lang_code}: {e}")
            
            self.logger.info(f"PaddleOCR engine initialized for {list(self.ocr_engines.keys())}")
        except ImportError:
            self.logger.error("PaddleOCR not installed. Install with: pip install paddlepaddle paddleocr")
            self.ocr_engines = {}
    
    def is_available(self) -> bool:
        """Check if PaddleOCR is available"""
        return len(self.ocr_engines) > 0
    
    def extract_text(self, image: Union[str, np.ndarray, Image.Image], 
                    bbox: Optional[Tuple[int, int, int, int]] = None) -> List[OCRResult]:
        """Extract text using PaddleOCR"""
        if not self.is_available():
            return []
        
        start_time = time.time()
        
        # Load image
        if isinstance(image, str):
            img = cv2.imread(image)
        elif isinstance(image, Image.Image):
            img = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
        else:
            img = image.copy()
        
        # Crop to bbox if provided
        if bbox:
            x1, y1, x2, y2 = bbox
            img = img[y1:y2, x1:x2]
        
        results = []
        
        # Try each language
        for lang_code, ocr_engine in self.ocr_engines.items():
            try:
                # Extract text
                paddle_results = ocr_engine.ocr(img, cls=True)
                
                if paddle_results and paddle_results[0]:
                    for line in paddle_results[0]:
                        if line:
                            bbox_coords, (text, confidence) = line
                            
                            if confidence >= self.config.confidence_threshold:
                                # Convert bbox format
                                x_coords = [point[0] for point in bbox_coords]
                                y_coords = [point[1] for point in bbox_coords]
                                x1, y1 = int(min(x_coords)), int(min(y_coords))
                                x2, y2 = int(max(x_coords)), int(max(y_coords))
                                
                                # Adjust coordinates if bbox was provided
                                if bbox:
                                    x1 += bbox[0]
                                    y1 += bbox[1]
                                    x2 += bbox[0]
                                    y2 += bbox[1]
                                
                                result = OCRResult(
                                    text=text,
                                    confidence=confidence,
                                    bbox=(x1, y1, x2, y2),
                                    language=lang_code,
                                    engine=self.engine_name,
                                    processing_time=time.time() - start_time
                                )
                                results.append(result)
                                
            except Exception as e:
                self.logger.warning(f"PaddleOCR extraction failed for {lang_code}: {e}")
        
        return results

class EasyOCREngine(BaseOCREngine):
    """EasyOCR Engine"""
    
    def __init__(self, config: OCRConfig):
        super().__init__(config)
        self.engine_name = "easyocr"
        
        try:
            import easyocr
            
            # Language mapping for EasyOCR
            self.lang_map = {
                'en': 'en',
                'ms': 'ms',  # Malay
                'chi_sim': 'ch_sim',
                'tam': 'ta'
            }
            
            # Get available languages
            available_langs = [self.lang_map[lang] for lang in self.config.languages 
                             if lang in self.lang_map]
            
            if available_langs:
                self.reader = easyocr.Reader(available_langs, gpu=True)
                self.logger.info(f"EasyOCR engine initialized for {available_langs}")
            else:
                self.reader = None
                self.logger.warning("No supported languages found for EasyOCR")
                
        except ImportError:
            self.logger.error("EasyOCR not installed. Install with: pip install easyocr")
            self.reader = None
    
    def is_available(self) -> bool:
        """Check if EasyOCR is available"""
        return self.reader is not None
    
    def extract_text(self, image: Union[str, np.ndarray, Image.Image], 
                    bbox: Optional[Tuple[int, int, int, int]] = None) -> List[OCRResult]:
        """Extract text using EasyOCR"""
        if not self.is_available():
            return []
        
        start_time = time.time()
        
        # Load image
        if isinstance(image, str):
            img = cv2.imread(image)
        elif isinstance(image, Image.Image):
            img = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
        else:
            img = image.copy()
        
        # Crop to bbox if provided
        if bbox:
            x1, y1, x2, y2 = bbox
            img = img[y1:y2, x1:x2]
        
        results = []
        
        try:
            # Extract text
            easyocr_results = self.reader.readtext(img)
            
            for bbox_coords, text, confidence in easyocr_results:
                if confidence >= self.config.confidence_threshold:
                    # Convert bbox format
                    x_coords = [point[0] for point in bbox_coords]
                    y_coords = [point[1] for point in bbox_coords]
                    x1, y1 = int(min(x_coords)), int(min(y_coords))
                    x2, y2 = int(max(x_coords)), int(max(y_coords))
                    
                    # Adjust coordinates if bbox was provided
                    if bbox:
                        x1 += bbox[0]
                        y1 += bbox[1]
                        x2 += bbox[0]
                        y2 += bbox[1]
                    
                    # Detect language (simplified)
                    detected_lang = self._detect_language(text)
                    
                    result = OCRResult(
                        text=text,
                        confidence=confidence,
                        bbox=(x1, y1, x2, y2),
                        language=detected_lang,
                        engine=self.engine_name,
                        processing_time=time.time() - start_time
                    )
                    results.append(result)
                    
        except Exception as e:
            self.logger.warning(f"EasyOCR extraction failed: {e}")
        
        return results
    
    def _detect_language(self, text: str) -> str:
        """Simple language detection based on character sets"""
        # Check for Chinese characters
        if any('\u4e00' <= char <= '\u9fff' for char in text):
            return 'chi_sim'
        
        # Check for Tamil characters
        if any('\u0b80' <= char <= '\u0bff' for char in text):
            return 'tam'
        
        # Default to English/Malay
        return 'en'

class GoogleVisionEngine(BaseOCREngine):
    """Google Cloud Vision API Engine"""
    
    def __init__(self, config: OCRConfig, api_key: Optional[str] = None):
        super().__init__(config)
        self.engine_name = "google_vision"
        self.api_key = api_key or os.getenv('GOOGLE_CLOUD_API_KEY')
        
        try:
            from google.cloud import vision
            self.client = vision.ImageAnnotatorClient()
            self.logger.info("Google Vision API engine initialized")
        except ImportError:
            self.logger.error("Google Cloud Vision not installed. Install with: pip install google-cloud-vision")
            self.client = None
        except Exception as e:
            self.logger.error(f"Failed to initialize Google Vision API: {e}")
            self.client = None
    
    def is_available(self) -> bool:
        """Check if Google Vision API is available"""
        return self.client is not None
    
    def extract_text(self, image: Union[str, np.ndarray, Image.Image], 
                    bbox: Optional[Tuple[int, int, int, int]] = None) -> List[OCRResult]:
        """Extract text using Google Vision API"""
        if not self.is_available():
            return []
        
        start_time = time.time()
        
        try:
            from google.cloud import vision
            
            # Prepare image
            if isinstance(image, str):
                with open(image, 'rb') as image_file:
                    content = image_file.read()
            else:
                # Convert to bytes
                if isinstance(image, Image.Image):
                    img_array = np.array(image)
                else:
                    img_array = image
                
                # Crop if bbox provided
                if bbox:
                    x1, y1, x2, y2 = bbox
                    img_array = img_array[y1:y2, x1:x2]
                
                # Convert to PIL and then to bytes
                pil_image = Image.fromarray(cv2.cvtColor(img_array, cv2.COLOR_BGR2RGB))
                img_byte_arr = io.BytesIO()
                pil_image.save(img_byte_arr, format='PNG')
                content = img_byte_arr.getvalue()
            
            # Create Vision API image object
            vision_image = vision.Image(content=content)
            
            # Perform text detection
            response = self.client.text_detection(image=vision_image)
            texts = response.text_annotations
            
            results = []
            
            for text in texts[1:]:  # Skip the first one (full text)
                confidence = 0.9  # Google doesn't provide confidence scores
                
                if confidence >= self.config.confidence_threshold:
                    # Get bounding box
                    vertices = text.bounding_poly.vertices
                    x_coords = [vertex.x for vertex in vertices]
                    y_coords = [vertex.y for vertex in vertices]
                    x1, y1 = min(x_coords), min(y_coords)
                    x2, y2 = max(x_coords), max(y_coords)
                    
                    # Adjust coordinates if bbox was provided
                    if bbox:
                        x1 += bbox[0]
                        y1 += bbox[1]
                        x2 += bbox[0]
                        y2 += bbox[1]
                    
                    result = OCRResult(
                        text=text.description,
                        confidence=confidence,
                        bbox=(x1, y1, x2, y2),
                        language='auto',
                        engine=self.engine_name,
                        processing_time=time.time() - start_time
                    )
                    results.append(result)
            
            return results
            
        except Exception as e:
            self.logger.error(f"Google Vision API extraction failed: {e}")
            return []

class OCRManager:
    """Main OCR manager with multiple engine support"""
    
    def __init__(self, config: OCRConfig = None):
        self.config = config or OCRConfig()
        self.logger = self._setup_logging()
        
        # Initialize engines
        self.engines = {}
        self._initialize_engines()
        
        self.logger.info(f"OCR Manager initialized with engines: {list(self.engines.keys())}")
    
    def _setup_logging(self) -> logging.Logger:
        """Setup logging configuration"""
        logger = logging.getLogger('OCRManager')
        logger.setLevel(logging.INFO)
        
        # Create console handler
        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.INFO)
        
        # Create formatter
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        console_handler.setFormatter(formatter)
        
        # Add handler to logger
        logger.addHandler(console_handler)
        
        return logger
    
    def _initialize_engines(self):
        """Initialize available OCR engines"""
        # Tesseract
        tesseract = TesseractEngine(self.config)
        if tesseract.is_available():
            self.engines['tesseract'] = tesseract
        
        # PaddleOCR
        paddleocr = PaddleOCREngine(self.config)
        if paddleocr.is_available():
            self.engines['paddleocr'] = paddleocr
        
        # EasyOCR
        easyocr = EasyOCREngine(self.config)
        if easyocr.is_available():
            self.engines['easyocr'] = easyocr
        
        # Google Vision (if API key available)
        if os.getenv('GOOGLE_CLOUD_API_KEY'):
            google_vision = GoogleVisionEngine(self.config)
            if google_vision.is_available():
                self.engines['google_vision'] = google_vision
    
    def extract_text(self, image: Union[str, np.ndarray, Image.Image], 
                    engine: Optional[str] = None,
                    bbox: Optional[Tuple[int, int, int, int]] = None) -> Dict[str, Any]:
        """Extract text using specified engine or all available engines"""
        start_time = time.time()
        
        if engine and engine in self.engines:
            # Use specific engine
            results = self.engines[engine].extract_text(image, bbox)
            engine_results = {engine: results}
        else:
            # Use all available engines
            engine_results = {}
            
            if self.config.parallel_processing and len(self.engines) > 1:
                # Parallel processing
                with ThreadPoolExecutor(max_workers=self.config.max_workers) as executor:
                    future_to_engine = {
                        executor.submit(eng.extract_text, image, bbox): name 
                        for name, eng in self.engines.items()
                    }
                    
                    for future in as_completed(future_to_engine):
                        engine_name = future_to_engine[future]
                        try:
                            results = future.result()
                            engine_results[engine_name] = results
                        except Exception as e:
                            self.logger.error(f"Engine {engine_name} failed: {e}")
                            engine_results[engine_name] = []
            else:
                # Sequential processing
                for engine_name, engine_obj in self.engines.items():
                    try:
                        results = engine_obj.extract_text(image, bbox)
                        engine_results[engine_name] = results
                    except Exception as e:
                        self.logger.error(f"Engine {engine_name} failed: {e}")
                        engine_results[engine_name] = []
        
        # Combine and rank results
        combined_results = self._combine_results(engine_results)
        
        return {
            'combined_results': combined_results,
            'engine_results': {k: [r.to_dict() for r in v] for k, v in engine_results.items()},
            'processing_time': time.time() - start_time,
            'engines_used': list(engine_results.keys()),
            'extraction_date': datetime.now().isoformat()
        }
    
    def _combine_results(self, engine_results: Dict[str, List[OCRResult]]) -> List[Dict[str, Any]]:
        """Combine results from multiple engines"""
        # Simple combination: take best confidence for each text region
        all_results = []
        for engine_name, results in engine_results.items():
            for result in results:
                result_dict = result.to_dict()
                result_dict['engine'] = engine_name
                all_results.append(result_dict)
        
        # Sort by confidence
        all_results.sort(key=lambda x: x['confidence'], reverse=True)
        
        return all_results
    
    def get_available_engines(self) -> List[str]:
        """Get list of available engines"""
        return list(self.engines.keys())
    
    def get_supported_languages(self) -> List[str]:
        """Get list of supported languages"""
        return self.config.languages

def main():
    """Main function for standalone execution"""
    print("🔤 OCR Engines for Malaysian Documents")
    print("=" * 40)
    
    # Initialize OCR manager
    config = OCRConfig(
        languages=['en', 'ms', 'chi_sim', 'tam'],
        confidence_threshold=0.6,
        preprocessing=True
    )
    
    ocr_manager = OCRManager(config)
    
    # Display configuration
    print(f"\n⚙️ Configuration:")
    print(f"   Languages: {config.languages}")
    print(f"   Confidence threshold: {config.confidence_threshold}")
    print(f"   Available engines: {ocr_manager.get_available_engines()}")
    print(f"   Preprocessing: {config.preprocessing}")
    
    # Example usage
    print("\n📋 Usage Examples:")
    print("1. result = ocr_manager.extract_text('/path/to/image.jpg')")
    print("2. result = ocr_manager.extract_text(image, engine='tesseract')")
    print("3. result = ocr_manager.extract_text(image, bbox=(x1, y1, x2, y2))")
    
    return 0

if __name__ == "__main__":
    exit(main())