#!/usr/bin/env python3
"""
Unit Test Configuration

Configuration settings, mock data, and utilities for unit testing
the document parser components.
"""

import os
import tempfile
from pathlib import Path
from dataclasses import dataclass
from typing import Dict, List, Any
import numpy as np
import cv2

# Test configuration constants
TEST_CONFIDENCE_THRESHOLD = 0.7
TEST_TIMEOUT = 30
TEST_MAX_FILE_SIZE = 10 * 1024 * 1024  # 10MB
TEST_SUPPORTED_FORMATS = ["jpg", "jpeg", "png", "bmp", "pdf"]

# Test directories
TEST_ROOT = Path(__file__).parent.parent
TEST_DATA_DIR = TEST_ROOT / "data"
TEST_OUTPUT_DIR = TEST_ROOT / "output"
TEST_TEMP_DIR = TEST_ROOT / "temp"
TEST_FIXTURES_DIR = TEST_ROOT / "fixtures"

# Ensure test directories exist
for test_dir in [TEST_DATA_DIR, TEST_OUTPUT_DIR, TEST_TEMP_DIR, TEST_FIXTURES_DIR]:
    test_dir.mkdir(parents=True, exist_ok=True)


@dataclass
class TestConfiguration:
    """Test configuration data class."""
    confidence_threshold: float = TEST_CONFIDENCE_THRESHOLD
    timeout: int = TEST_TIMEOUT
    max_file_size: int = TEST_MAX_FILE_SIZE
    supported_formats: List[str] = None
    use_gpu: bool = False
    strict_mode: bool = True
    
    def __post_init__(self):
        if self.supported_formats is None:
            self.supported_formats = TEST_SUPPORTED_FORMATS.copy()


class MockDataGenerator:
    """Generate mock data for testing."""
    
    @staticmethod
    def create_sample_image(width: int = 800, height: int = 600, 
                          text: str = "SAMPLE TEXT") -> np.ndarray:
        """Create a sample image with text."""
        image = np.ones((height, width, 3), dtype=np.uint8) * 255
        
        # Add border
        cv2.rectangle(image, (10, 10), (width-10, height-10), (0, 0, 0), 2)
        
        # Add text
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 1.0
        color = (0, 0, 0)
        thickness = 2
        
        # Calculate text size and position
        text_size = cv2.getTextSize(text, font, font_scale, thickness)[0]
        text_x = (width - text_size[0]) // 2
        text_y = (height + text_size[1]) // 2
        
        cv2.putText(image, text, (text_x, text_y), font, font_scale, color, thickness)
        
        return image
    
    @staticmethod
    def create_mykad_image() -> np.ndarray:
        """Create a MyKad-like sample image."""
        image = np.ones((600, 950, 3), dtype=np.uint8) * 255
        
        # Add MyKad border (red)
        cv2.rectangle(image, (50, 50), (900, 550), (0, 0, 255), 3)
        
        # Add header
        cv2.putText(image, "MALAYSIA", (300, 100), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 0), 2)
        cv2.putText(image, "MYKAD", (350, 140), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 0), 2)
        
        # Add sample data
        cv2.putText(image, "AHMAD BIN ALI", (100, 200), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 2)
        cv2.putText(image, "123456-78-9012", (100, 240), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 2)
        cv2.putText(image, "LELAKI", (100, 280), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 2)
        cv2.putText(image, "KUALA LUMPUR", (100, 320), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 2)
        cv2.putText(image, "ISLAM", (100, 360), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 2)
        
        return image
    
    @staticmethod
    def create_spk_image() -> np.ndarray:
        """Create an SPK-like sample image."""
        image = np.ones((800, 600, 3), dtype=np.uint8) * 255
        
        # Add border
        cv2.rectangle(image, (50, 50), (550, 750), (0, 0, 0), 2)
        
        # Add header
        cv2.putText(image, "SURAT PERAKUAN", (150, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 2)
        cv2.putText(image, "KELAHIRAN", (200, 140), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 2)
        
        # Add sample data
        cv2.putText(image, "NAMA: SITI AMINAH", (80, 200), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)
        cv2.putText(image, "BINTI HASSAN", (80, 230), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)
        cv2.putText(image, "NO. SIJIL: SPK123456789", (80, 280), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)
        cv2.putText(image, "TARIKH LAHIR: 15/03/2020", (80, 320), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)
        cv2.putText(image, "TEMPAT LAHIR:", (80, 360), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)
        cv2.putText(image, "HOSPITAL KL", (80, 390), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)
        cv2.putText(image, "JANTINA: PEREMPUAN", (80, 440), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)
        
        return image
    
    @staticmethod
    def create_noisy_image(base_image: np.ndarray, noise_level: float = 0.1) -> np.ndarray:
        """Add noise to an image for testing robustness."""
        noise = np.random.normal(0, noise_level * 255, base_image.shape)
        noisy_image = base_image.astype(np.float32) + noise
        noisy_image = np.clip(noisy_image, 0, 255).astype(np.uint8)
        return noisy_image
    
    @staticmethod
    def create_blurred_image(base_image: np.ndarray, blur_kernel: int = 5) -> np.ndarray:
        """Create a blurred version of an image."""
        return cv2.GaussianBlur(base_image, (blur_kernel, blur_kernel), 0)
    
    @staticmethod
    def create_rotated_image(base_image: np.ndarray, angle: float = 5.0) -> np.ndarray:
        """Create a rotated version of an image."""
        height, width = base_image.shape[:2]
        center = (width // 2, height // 2)
        
        rotation_matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
        rotated_image = cv2.warpAffine(base_image, rotation_matrix, (width, height), 
                                     borderValue=(255, 255, 255))
        return rotated_image


class MockResponses:
    """Mock responses for external services."""
    
    MOCK_OCR_RESPONSES = {
        'mykad': {
            'text': "MALAYSIA\nMYKAD\nAHMAD BIN ALI\n123456-78-9012\nLELAKI\nKUALA LUMPUR\nISLAM",
            'confidence': 0.92,
            'bounding_boxes': [
                (300, 80, 400, 120),  # MALAYSIA
                (350, 120, 420, 160), # MYKAD
                (100, 180, 280, 220), # AHMAD BIN ALI
                (100, 220, 250, 260), # 123456-78-9012
                (100, 260, 180, 300), # LELAKI
                (100, 300, 280, 340), # KUALA LUMPUR
                (100, 340, 160, 380)  # ISLAM
            ]
        },
        'spk': {
            'text': "SURAT PERAKUAN KELAHIRAN\nNAMA: SITI AMINAH BINTI HASSAN\nNO. SIJIL: SPK123456789\nTARIKH LAHIR: 15/03/2020\nTEMPAT LAHIR: HOSPITAL KL\nJANTINA: PEREMPUAN",
            'confidence': 0.89,
            'bounding_boxes': [
                (150, 80, 450, 160),  # Header
                (80, 180, 350, 250),  # Name
                (80, 260, 300, 300),  # Certificate number
                (80, 300, 320, 340),  # Birth date
                (80, 340, 280, 410),  # Birth place
                (80, 420, 280, 460)   # Gender
            ]
        },
        'empty': {
            'text': "",
            'confidence': 0.0,
            'bounding_boxes': []
        },
        'low_quality': {
            'text': "UNCLEAR TEXT WITH ERRORS",
            'confidence': 0.3,
            'bounding_boxes': [(50, 50, 200, 100)]
        }
    }
    
    MOCK_CLASSIFICATION_RESPONSES = {
        'mykad_high_confidence': {
            'document_type': 'MYKAD',
            'confidence': 0.95,
            'confidence_level': 'HIGH'
        },
        'spk_high_confidence': {
            'document_type': 'SPK',
            'confidence': 0.88,
            'confidence_level': 'HIGH'
        },
        'unknown_low_confidence': {
            'document_type': 'UNKNOWN',
            'confidence': 0.4,
            'confidence_level': 'LOW'
        }
    }
    
    MOCK_EXTRACTION_RESPONSES = {
        'mykad_fields': {
            'name': {'value': 'AHMAD BIN ALI', 'confidence': 0.95},
            'ic_number': {'value': '123456-78-9012', 'confidence': 0.98},
            'gender': {'value': 'LELAKI', 'confidence': 0.92},
            'address': {'value': 'KUALA LUMPUR', 'confidence': 0.85},
            'religion': {'value': 'ISLAM', 'confidence': 0.88}
        },
        'spk_fields': {
            'name': {'value': 'SITI AMINAH BINTI HASSAN', 'confidence': 0.94},
            'certificate_number': {'value': 'SPK123456789', 'confidence': 0.96},
            'birth_date': {'value': '15/03/2020', 'confidence': 0.91},
            'birth_place': {'value': 'HOSPITAL KL', 'confidence': 0.87},
            'gender': {'value': 'PEREMPUAN', 'confidence': 0.93}
        }
    }
    
    MOCK_VALIDATION_RESPONSES = {
        'valid_mykad': {
            'is_valid': True,
            'errors': [],
            'warnings': []
        },
        'invalid_ic_format': {
            'is_valid': False,
            'errors': [
                {
                    'field': 'ic_number',
                    'error_type': 'format_error',
                    'message': 'Invalid IC number format'
                }
            ],
            'warnings': []
        },
        'low_confidence': {
            'is_valid': False,
            'errors': [
                {
                    'field': 'name',
                    'error_type': 'low_confidence',
                    'message': 'Field confidence below threshold'
                }
            ],
            'warnings': []
        }
    }


class TestUtilities:
    """Utility functions for testing."""
    
    @staticmethod
    def create_temp_file(content: str = "", suffix: str = ".txt") -> str:
        """Create a temporary file for testing."""
        with tempfile.NamedTemporaryFile(mode='w', suffix=suffix, delete=False) as f:
            f.write(content)
            return f.name
    
    @staticmethod
    def create_temp_image(image: np.ndarray, suffix: str = ".jpg") -> str:
        """Save an image to a temporary file."""
        temp_path = tempfile.mktemp(suffix=suffix)
        cv2.imwrite(temp_path, image)
        return temp_path
    
    @staticmethod
    def cleanup_temp_files(file_paths: List[str]):
        """Clean up temporary files."""
        for file_path in file_paths:
            try:
                if os.path.exists(file_path):
                    os.unlink(file_path)
            except Exception:
                pass  # Ignore cleanup errors
    
    @staticmethod
    def assert_image_similarity(img1: np.ndarray, img2: np.ndarray, 
                              threshold: float = 0.95) -> bool:
        """Assert that two images are similar."""
        if img1.shape != img2.shape:
            return False
        
        # Calculate structural similarity
        diff = cv2.absdiff(img1, img2)
        similarity = 1.0 - (np.sum(diff) / (img1.size * 255))
        
        return similarity >= threshold
    
    @staticmethod
    def measure_processing_time(func, *args, **kwargs) -> tuple:
        """Measure function processing time."""
        import time
        
        start_time = time.time()
        result = func(*args, **kwargs)
        end_time = time.time()
        
        processing_time = end_time - start_time
        return result, processing_time
    
    @staticmethod
    def measure_memory_usage(func, *args, **kwargs) -> tuple:
        """Measure function memory usage."""
        import psutil
        import os
        
        process = psutil.Process(os.getpid())
        initial_memory = process.memory_info().rss
        
        result = func(*args, **kwargs)
        
        final_memory = process.memory_info().rss
        memory_increase = final_memory - initial_memory
        
        return result, memory_increase


# Test data constants
TEST_SAMPLE_TEXTS = {
    'mykad': "MALAYSIA\nMYKAD\nAHMAD BIN ALI\n123456-78-9012\nLELAKI\nKUALA LUMPUR\nISLAM",
    'spk': "SURAT PERAKUAN KELAHIRAN\nNAMA: SITI AMINAH BINTI HASSAN\nNO. SIJIL: SPK123456789\nTARIKH LAHIR: 15/03/2020\nTEMPAT LAHIR: HOSPITAL KL\nJANTINA: PEREMPUAN",
    'invalid': "RANDOM TEXT WITH NO STRUCTURE",
    'empty': "",
    'noisy': "UNCLEAR T3XT W1TH N01S3 4ND 3RR0RS"
}

TEST_VALIDATION_CASES = {
    'valid_ic_numbers': [
        "123456-78-9012",
        "987654-32-1098",
        "111111-11-1111"
    ],
    'invalid_ic_numbers': [
        "123456789012",
        "12345-78-9012",
        "123456-7-9012",
        "123456-78-901",
        "abcdef-78-9012"
    ],
    'valid_dates': [
        "15/03/2020",
        "01/01/2000",
        "31/12/1999"
    ],
    'invalid_dates': [
        "32/01/2020",
        "15/13/2020",
        "15/03/20",
        "2020/03/15"
    ],
    'valid_names': [
        "AHMAD BIN ALI",
        "SITI AMINAH BINTI HASSAN",
        "JOHN DOE"
    ],
    'invalid_names': [
        "ahmad bin ali",
        "AHMAD123",
        "AHMAD@ALI",
        "A"
    ]
}

# Export main configuration
DEFAULT_TEST_CONFIG = TestConfiguration()