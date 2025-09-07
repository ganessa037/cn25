#!/usr/bin/env python3
"""
Utility Functions

Common utility functions for image preprocessing, text processing,
coordinate mapping, and other helper functions.
"""

import re
import cv2
import numpy as np
import logging
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any, Union
from PIL import Image, ImageEnhance, ImageFilter
import base64
import io
from datetime import datetime
import hashlib
import unicodedata

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ImagePreprocessor:
    """Image preprocessing utilities for OCR optimization."""
    
    def __init__(self, target_size: Tuple[int, int] = (2048, 2048)):
        self.target_size = target_size
        
    def preprocess_image(self, image: Union[np.ndarray, Image.Image, str, Path], 
                        operations: List[str] = None) -> np.ndarray:
        """Apply preprocessing operations to image."""
        if operations is None:
            operations = ['resize', 'denoise', 'enhance_contrast', 'binarize']
        
        # Load image
        img = self._load_image(image)
        
        # Apply operations
        for operation in operations:
            if operation == 'resize':
                img = self.resize_image(img)
            elif operation == 'denoise':
                img = self.denoise_image(img)
            elif operation == 'enhance_contrast':
                img = self.enhance_contrast(img)
            elif operation == 'binarize':
                img = self.binarize_image(img)
            elif operation == 'deskew':
                img = self.deskew_image(img)
            elif operation == 'remove_shadows':
                img = self.remove_shadows(img)
            else:
                logger.warning(f"Unknown preprocessing operation: {operation}")
        
        return img
    
    def _load_image(self, image: Union[np.ndarray, Image.Image, str, Path]) -> np.ndarray:
        """Load image from various input types."""
        if isinstance(image, np.ndarray):
            return image
        elif isinstance(image, Image.Image):
            return np.array(image)
        elif isinstance(image, (str, Path)):
            img = cv2.imread(str(image))
            if img is None:
                raise ValueError(f"Could not load image from {image}")
            return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        else:
            raise ValueError(f"Unsupported image type: {type(image)}")
    
    def resize_image(self, image: np.ndarray, target_size: Tuple[int, int] = None) -> np.ndarray:
        """Resize image while maintaining aspect ratio."""
        if target_size is None:
            target_size = self.target_size
        
        h, w = image.shape[:2]
        target_w, target_h = target_size
        
        # Calculate scaling factor
        scale = min(target_w / w, target_h / h)
        
        if scale < 1.0:
            new_w = int(w * scale)
            new_h = int(h * scale)
            image = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_AREA)
        
        return image
    
    def denoise_image(self, image: np.ndarray) -> np.ndarray:
        """Remove noise from image."""
        if len(image.shape) == 3:
            return cv2.fastNlMeansDenoisingColored(image, None, 10, 10, 7, 21)
        else:
            return cv2.fastNlMeansDenoising(image, None, 10, 7, 21)
    
    def enhance_contrast(self, image: np.ndarray) -> np.ndarray:
        """Enhance image contrast."""
        # Convert to LAB color space
        if len(image.shape) == 3:
            lab = cv2.cvtColor(image, cv2.COLOR_RGB2LAB)
            l, a, b = cv2.split(lab)
            
            # Apply CLAHE to L channel
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
            l = clahe.apply(l)
            
            # Merge channels and convert back
            lab = cv2.merge([l, a, b])
            return cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)
        else:
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
            return clahe.apply(image)
    
    def binarize_image(self, image: np.ndarray) -> np.ndarray:
        """Convert image to binary (black and white)."""
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        else:
            gray = image
        
        # Apply adaptive thresholding
        binary = cv2.adaptiveThreshold(
            gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2
        )
        
        return binary
    
    def deskew_image(self, image: np.ndarray) -> np.ndarray:
        """Correct image skew."""
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        else:
            gray = image
        
        # Find edges
        edges = cv2.Canny(gray, 50, 150, apertureSize=3)
        
        # Find lines using Hough transform
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
                h, w = image.shape[:2]
                center = (w // 2, h // 2)
                rotation_matrix = cv2.getRotationMatrix2D(center, median_angle, 1.0)
                
                return cv2.warpAffine(image, rotation_matrix, (w, h), 
                                    flags=cv2.INTER_CUBIC, 
                                    borderMode=cv2.BORDER_REPLICATE)
        
        return image
    
    def remove_shadows(self, image: np.ndarray) -> np.ndarray:
        """Remove shadows from image."""
        if len(image.shape) != 3:
            return image
        
        # Convert to LAB
        lab = cv2.cvtColor(image, cv2.COLOR_RGB2LAB)
        l, a, b = cv2.split(lab)
        
        # Apply morphological operations to L channel
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (20, 20))
        background = cv2.morphologyEx(l, cv2.MORPH_CLOSE, kernel)
        
        # Subtract background
        l_corrected = cv2.subtract(l, background)
        l_corrected = cv2.add(l_corrected, 128)
        
        # Merge and convert back
        lab_corrected = cv2.merge([l_corrected, a, b])
        return cv2.cvtColor(lab_corrected, cv2.COLOR_LAB2RGB)
    
    def get_image_quality_score(self, image: np.ndarray) -> float:
        """Calculate image quality score (0-1)."""
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        else:
            gray = image
        
        # Calculate Laplacian variance (focus measure)
        laplacian_var = cv2.Laplacian(gray, cv2.CV_64F).var()
        
        # Normalize to 0-1 range
        quality_score = min(laplacian_var / 1000.0, 1.0)
        
        return quality_score

class TextProcessor:
    """Text processing utilities for cleaning and normalizing extracted text."""
    
    def __init__(self):
        # Common OCR error patterns
        self.ocr_corrections = {
            r'\b0\b': 'O',  # Zero to O
            r'\b1\b': 'I',  # One to I
            r'\b5\b': 'S',  # Five to S
            r'\b8\b': 'B',  # Eight to B
            r'rn': 'm',     # rn to m
            r'vv': 'w',     # vv to w
            r'\|': 'l',     # | to l
        }
        
        # Malaysian-specific patterns
        self.malaysian_patterns = {
            'ic_number': r'\b(\d{6}[-\s]?\d{2}[-\s]?\d{4})\b',
            'phone_mobile': r'\b(\+?6?0?1[0-9][-\s]?\d{7,8})\b',
            'phone_landline': r'\b(\+?6?0?[2-9]\d{1}[-\s]?\d{7,8})\b',
            'postcode': r'\b(\d{5})\b',
            'email': r'\b([a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,})\b'
        }
    
    def clean_text(self, text: str) -> str:
        """Clean and normalize extracted text."""
        if not text:
            return ""
        
        # Remove extra whitespace
        text = re.sub(r'\s+', ' ', text.strip())
        
        # Remove special characters but keep important punctuation
        text = re.sub(r'[^\w\s@.-]', '', text)
        
        # Apply OCR corrections
        for pattern, replacement in self.ocr_corrections.items():
            text = re.sub(pattern, replacement, text)
        
        # Normalize unicode
        text = unicodedata.normalize('NFKD', text)
        
        return text
    
    def extract_ic_number(self, text: str) -> Optional[str]:
        """Extract Malaysian IC number from text."""
        pattern = self.malaysian_patterns['ic_number']
        matches = re.findall(pattern, text)
        
        if matches:
            # Clean and format IC number
            ic = re.sub(r'[^\d]', '', matches[0])
            if len(ic) == 12:
                return f"{ic[:6]}-{ic[6:8]}-{ic[8:]}"
        
        return None
    
    def extract_phone_numbers(self, text: str) -> List[str]:
        """Extract phone numbers from text."""
        phones = []
        
        # Mobile numbers
        mobile_matches = re.findall(self.malaysian_patterns['phone_mobile'], text)
        phones.extend(mobile_matches)
        
        # Landline numbers
        landline_matches = re.findall(self.malaysian_patterns['phone_landline'], text)
        phones.extend(landline_matches)
        
        # Clean and format phone numbers
        cleaned_phones = []
        for phone in phones:
            cleaned = re.sub(r'[^\d+]', '', phone)
            if cleaned.startswith('60'):
                cleaned = '+' + cleaned
            elif cleaned.startswith('0'):
                cleaned = '+6' + cleaned
            cleaned_phones.append(cleaned)
        
        return list(set(cleaned_phones))  # Remove duplicates
    
    def extract_email(self, text: str) -> Optional[str]:
        """Extract email address from text."""
        pattern = self.malaysian_patterns['email']
        matches = re.findall(pattern, text, re.IGNORECASE)
        
        if matches:
            return matches[0].lower()
        
        return None
    
    def extract_dates(self, text: str) -> List[str]:
        """Extract dates from text."""
        date_patterns = [
            r'\b(\d{1,2}[/-]\d{1,2}[/-]\d{4})\b',  # DD/MM/YYYY or DD-MM-YYYY
            r'\b(\d{4}[/-]\d{1,2}[/-]\d{1,2})\b',  # YYYY/MM/DD or YYYY-MM-DD
            r'\b(\d{1,2}\s+\w+\s+\d{4})\b',        # DD Month YYYY
        ]
        
        dates = []
        for pattern in date_patterns:
            matches = re.findall(pattern, text)
            dates.extend(matches)
        
        return dates
    
    def normalize_name(self, name: str) -> str:
        """Normalize Malaysian names."""
        if not name:
            return ""
        
        # Convert to uppercase
        name = name.upper().strip()
        
        # Remove extra spaces
        name = re.sub(r'\s+', ' ', name)
        
        # Common Malaysian name prefixes/suffixes
        prefixes = ['BIN', 'BINTI', 'A/L', 'A/P', 'S/O', 'D/O']
        
        # Ensure proper spacing around prefixes
        for prefix in prefixes:
            name = re.sub(f'\s*{prefix}\s*', f' {prefix} ', name)
        
        return name.strip()
    
    def calculate_text_similarity(self, text1: str, text2: str) -> float:
        """Calculate similarity between two text strings."""
        from difflib import SequenceMatcher
        
        # Clean both texts
        clean_text1 = self.clean_text(text1).lower()
        clean_text2 = self.clean_text(text2).lower()
        
        # Calculate similarity
        similarity = SequenceMatcher(None, clean_text1, clean_text2).ratio()
        
        return similarity
    
    def extract_addresses(self, text: str) -> List[str]:
        """Extract Malaysian addresses from text."""
        # Malaysian address patterns
        address_patterns = [
            r'\b\d+[A-Z]?[,\s]+[^,\n]+[,\s]+\d{5}[,\s]+[^,\n]+\b',  # Street, postcode, state
            r'\b[^,\n]+[,\s]+\d{5}[,\s]+[^,\n]+\b',                    # Area, postcode, state
        ]
        
        addresses = []
        for pattern in address_patterns:
            matches = re.findall(pattern, text, re.IGNORECASE)
            addresses.extend(matches)
        
        # Clean addresses
        cleaned_addresses = []
        for addr in addresses:
            cleaned = re.sub(r'\s+', ' ', addr.strip())
            if len(cleaned) > 10:  # Minimum address length
                cleaned_addresses.append(cleaned)
        
        return list(set(cleaned_addresses))

class CoordinateMapper:
    """Coordinate mapping utilities for OCR bounding boxes."""
    
    def __init__(self):
        self.field_regions = {
            'mykad': {
                'ic_number': {'x': 0.3, 'y': 0.4, 'w': 0.4, 'h': 0.1},
                'name': {'x': 0.3, 'y': 0.5, 'w': 0.6, 'h': 0.1},
                'address': {'x': 0.3, 'y': 0.6, 'w': 0.6, 'h': 0.2},
            },
            'spk': {
                'name': {'x': 0.2, 'y': 0.3, 'w': 0.6, 'h': 0.1},
                'school': {'x': 0.2, 'y': 0.4, 'w': 0.6, 'h': 0.1},
                'year': {'x': 0.2, 'y': 0.5, 'w': 0.3, 'h': 0.1},
            },
            'passport': {
                'passport_number': {'x': 0.5, 'y': 0.2, 'w': 0.3, 'h': 0.1},
                'name': {'x': 0.2, 'y': 0.4, 'w': 0.6, 'h': 0.1},
                'nationality': {'x': 0.2, 'y': 0.5, 'w': 0.4, 'h': 0.1},
            }
        }
    
    def get_field_region(self, document_type: str, field_name: str, 
                        image_width: int, image_height: int) -> Optional[Tuple[int, int, int, int]]:
        """Get expected region for a field in a document type."""
        if document_type not in self.field_regions:
            return None
        
        if field_name not in self.field_regions[document_type]:
            return None
        
        region = self.field_regions[document_type][field_name]
        
        x = int(region['x'] * image_width)
        y = int(region['y'] * image_height)
        w = int(region['w'] * image_width)
        h = int(region['h'] * image_height)
        
        return (x, y, w, h)
    
    def is_in_expected_region(self, bbox: Tuple[int, int, int, int], 
                             expected_region: Tuple[int, int, int, int], 
                             tolerance: float = 0.2) -> bool:
        """Check if bounding box is in expected region."""
        x1, y1, w1, h1 = bbox
        x2, y2, w2, h2 = expected_region
        
        # Calculate centers
        center1_x = x1 + w1 / 2
        center1_y = y1 + h1 / 2
        center2_x = x2 + w2 / 2
        center2_y = y2 + h2 / 2
        
        # Calculate distance
        distance = np.sqrt((center1_x - center2_x)**2 + (center1_y - center2_y)**2)
        
        # Calculate tolerance distance
        tolerance_distance = tolerance * min(w2, h2)
        
        return distance <= tolerance_distance
    
    def calculate_overlap(self, bbox1: Tuple[int, int, int, int], 
                         bbox2: Tuple[int, int, int, int]) -> float:
        """Calculate overlap ratio between two bounding boxes."""
        x1, y1, w1, h1 = bbox1
        x2, y2, w2, h2 = bbox2
        
        # Calculate intersection
        left = max(x1, x2)
        top = max(y1, y2)
        right = min(x1 + w1, x2 + w2)
        bottom = min(y1 + h1, y2 + h2)
        
        if left < right and top < bottom:
            intersection = (right - left) * (bottom - top)
            union = w1 * h1 + w2 * h2 - intersection
            return intersection / union if union > 0 else 0
        
        return 0
    
    def merge_nearby_boxes(self, boxes: List[Tuple[int, int, int, int]], 
                          threshold: float = 0.1) -> List[Tuple[int, int, int, int]]:
        """Merge nearby bounding boxes."""
        if not boxes:
            return []
        
        merged = []
        used = set()
        
        for i, box1 in enumerate(boxes):
            if i in used:
                continue
            
            current_box = box1
            used.add(i)
            
            # Find boxes to merge
            for j, box2 in enumerate(boxes[i+1:], i+1):
                if j in used:
                    continue
                
                if self.calculate_overlap(current_box, box2) > threshold:
                    # Merge boxes
                    x1, y1, w1, h1 = current_box
                    x2, y2, w2, h2 = box2
                    
                    left = min(x1, x2)
                    top = min(y1, y2)
                    right = max(x1 + w1, x2 + w2)
                    bottom = max(y1 + h1, y2 + h2)
                    
                    current_box = (left, top, right - left, bottom - top)
                    used.add(j)
            
            merged.append(current_box)
        
        return merged

class DocumentUtils:
    """General document processing utilities."""
    
    @staticmethod
    def generate_document_id() -> str:
        """Generate unique document ID."""
        timestamp = datetime.now().strftime('%Y%m%d%H%M%S')
        random_part = hashlib.md5(str(datetime.now().microsecond).encode()).hexdigest()[:8]
        return f"doc_{timestamp}_{random_part}"
    
    @staticmethod
    def encode_image_to_base64(image: Union[np.ndarray, Image.Image, str, Path]) -> str:
        """Encode image to base64 string."""
        if isinstance(image, (str, Path)):
            with open(image, 'rb') as f:
                return base64.b64encode(f.read()).decode('utf-8')
        
        elif isinstance(image, np.ndarray):
            # Convert numpy array to PIL Image
            if len(image.shape) == 3:
                pil_image = Image.fromarray(image)
            else:
                pil_image = Image.fromarray(image, mode='L')
        
        elif isinstance(image, Image.Image):
            pil_image = image
        
        else:
            raise ValueError(f"Unsupported image type: {type(image)}")
        
        # Convert to base64
        buffer = io.BytesIO()
        pil_image.save(buffer, format='PNG')
        return base64.b64encode(buffer.getvalue()).decode('utf-8')
    
    @staticmethod
    def decode_base64_to_image(base64_string: str) -> Image.Image:
        """Decode base64 string to PIL Image."""
        image_data = base64.b64decode(base64_string)
        return Image.open(io.BytesIO(image_data))
    
    @staticmethod
    def calculate_file_hash(file_path: Union[str, Path]) -> str:
        """Calculate MD5 hash of file."""
        hash_md5 = hashlib.md5()
        with open(file_path, 'rb') as f:
            for chunk in iter(lambda: f.read(4096), b""):
                hash_md5.update(chunk)
        return hash_md5.hexdigest()
    
    @staticmethod
    def validate_file_type(file_path: Union[str, Path], allowed_types: List[str] = None) -> bool:
        """Validate file type based on extension."""
        if allowed_types is None:
            allowed_types = ['jpg', 'jpeg', 'png', 'tiff', 'bmp', 'pdf']
        
        file_path = Path(file_path)
        extension = file_path.suffix.lower().lstrip('.')
        
        return extension in allowed_types
    
    @staticmethod
    def get_file_size_mb(file_path: Union[str, Path]) -> float:
        """Get file size in megabytes."""
        file_path = Path(file_path)
        if file_path.exists():
            return file_path.stat().st_size / (1024 * 1024)
        return 0.0
    
    @staticmethod
    def create_thumbnail(image: Union[np.ndarray, Image.Image], size: Tuple[int, int] = (200, 200)) -> Image.Image:
        """Create thumbnail of image."""
        if isinstance(image, np.ndarray):
            if len(image.shape) == 3:
                pil_image = Image.fromarray(image)
            else:
                pil_image = Image.fromarray(image, mode='L')
        else:
            pil_image = image
        
        pil_image.thumbnail(size, Image.Resampling.LANCZOS)
        return pil_image
    
    @staticmethod
    def format_confidence_score(confidence: float) -> str:
        """Format confidence score as percentage."""
        return f"{confidence * 100:.1f}%"
    
    @staticmethod
    def validate_malaysian_ic(ic_number: str) -> bool:
        """Validate Malaysian IC number format and checksum."""
        # Remove non-digits
        ic_clean = re.sub(r'[^\d]', '', ic_number)
        
        if len(ic_clean) != 12:
            return False
        
        # Basic format validation
        birth_date = ic_clean[:6]
        birth_place = ic_clean[6:8]
        sequence = ic_clean[8:]
        
        # Validate birth date (YYMMDD)
        try:
            year = int(birth_date[:2])
            month = int(birth_date[2:4])
            day = int(birth_date[4:6])
            
            if month < 1 or month > 12:
                return False
            if day < 1 or day > 31:
                return False
        except ValueError:
            return False
        
        # Validate birth place code (01-59)
        try:
            place_code = int(birth_place)
            if place_code < 1 or place_code > 59:
                return False
        except ValueError:
            return False
        
        return True
    
    @staticmethod
    def format_malaysian_phone(phone: str) -> str:
        """Format Malaysian phone number."""
        # Remove all non-digits except +
        clean_phone = re.sub(r'[^\d+]', '', phone)
        
        # Handle different formats
        if clean_phone.startswith('+60'):
            return clean_phone
        elif clean_phone.startswith('60'):
            return '+' + clean_phone
        elif clean_phone.startswith('0'):
            return '+6' + clean_phone
        else:
            return '+60' + clean_phone

# Utility instances
image_preprocessor = ImagePreprocessor()
text_processor = TextProcessor()
coordinate_mapper = CoordinateMapper()
document_utils = DocumentUtils()

# Convenience functions
def preprocess_image(image: Union[np.ndarray, Image.Image, str, Path], 
                   operations: List[str] = None) -> np.ndarray:
    """Preprocess image for OCR."""
    return image_preprocessor.preprocess_image(image, operations)

def clean_text(text: str) -> str:
    """Clean extracted text."""
    return text_processor.clean_text(text)

def extract_ic_number(text: str) -> Optional[str]:
    """Extract IC number from text."""
    return text_processor.extract_ic_number(text)

def extract_phone_numbers(text: str) -> List[str]:
    """Extract phone numbers from text."""
    return text_processor.extract_phone_numbers(text)

def generate_document_id() -> str:
    """Generate unique document ID."""
    return document_utils.generate_document_id()

def validate_malaysian_ic(ic_number: str) -> bool:
    """Validate Malaysian IC number."""
    return document_utils.validate_malaysian_ic(ic_number)