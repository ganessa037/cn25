"""Configuration settings for Malaysian License Plate Detection System"""

import os
from pathlib import Path
from typing import Dict, List, Tuple

# Base paths
BASE_DIR = Path(__file__).parent.parent.parent.parent
MODELS_DIR = BASE_DIR / "models" / "license_plate"
SRC_DIR = BASE_DIR / "src" / "license_plate"

# Model paths
TRAINED_MODELS_DIR = MODELS_DIR / "trained_models"
DATA_DIR = MODELS_DIR / "data"
OUTPUTS_DIR = MODELS_DIR / "outputs"

# YOLOv8 Configuration
YOLO_CONFIG = {
    "model_name": "yolov8n.pt",  # nano model for faster inference
    "confidence_threshold": 0.5,
    "iou_threshold": 0.45,
    "max_detections": 10,
    "image_size": 640,
    "device": "cpu",  # Change to "cuda" if GPU available
}

# EasyOCR Configuration
OCR_CONFIG = {
    "languages": ["en"],  # English for Malaysian plates
    "gpu": False,  # Set to True if GPU available
    "confidence_threshold": 0.6,
    "width_ths": 0.7,
    "height_ths": 0.7,
    "paragraph": False,
}

# Malaysian License Plate Patterns
MALAYSIAN_PLATE_PATTERNS = {
    # Standard format: ABC 1234 or AB 1234 C
    "standard": [
        r"^[A-Z]{1,3}\s*\d{1,4}[A-Z]?$",  # ABC 1234 or AB 1234 C
        r"^[A-Z]{2,3}\s*\d{1,4}$",        # ABC 1234
    ],
    
    # Special formats
    "government": [
        r"^1\s*MALAYSIA\s*\d{1,4}$",       # Government vehicles
        r"^MALAYSIA\s*\d{1,4}$",          # Malaysia prefix
    ],
    
    # Commercial vehicles
    "commercial": [
        r"^[A-Z]{1,2}\s*\d{1,4}\s*[A-Z]$", # Commercial format
    ],
    
    # Motorcycle format
    "motorcycle": [
        r"^[A-Z]{1,3}\s*\d{1,4}$",        # Motorcycle plates
    ]
}

# Image Processing Configuration
IMAGE_CONFIG = {
    "supported_formats": [".jpg", ".jpeg", ".png", ".bmp", ".tiff"],
    "max_image_size": (1920, 1080),  # Max resolution
    "min_image_size": (320, 240),    # Min resolution
    "preprocessing": {
        "resize": True,
        "normalize": True,
        "enhance_contrast": True,
        "denoise": False,
    }
}

# Detection Configuration
DETECTION_CONFIG = {
    "min_plate_area": 1000,          # Minimum plate area in pixels
    "max_plate_area": 50000,         # Maximum plate area in pixels
    "aspect_ratio_range": (2.0, 6.0), # Width/height ratio range
    "confidence_threshold": 0.5,      # Minimum detection confidence
    "nms_threshold": 0.4,            # Non-maximum suppression threshold
}

# OCR Post-processing Configuration
OCR_POSTPROCESS_CONFIG = {
    "remove_special_chars": True,
    "convert_to_uppercase": True,
    "min_text_length": 3,
    "max_text_length": 10,
    "allowed_chars": "ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789 ",
}

# Logging Configuration
LOGGING_CONFIG = {
    "level": "INFO",
    "format": "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    "log_file": OUTPUTS_DIR / "detection.log",
    "max_file_size": 10 * 1024 * 1024,  # 10MB
    "backup_count": 5,
}

# API Configuration
API_CONFIG = {
    "max_file_size": 10 * 1024 * 1024,  # 10MB
    "allowed_extensions": {".jpg", ".jpeg", ".png", ".bmp"},
    "response_timeout": 30,  # seconds
    "max_concurrent_requests": 10,
}

# Paths Configuration
PATHS_CONFIG = {
    "base_dir": BASE_DIR,
    "models_dir": MODELS_DIR,
    "trained_models_dir": TRAINED_MODELS_DIR,
    "data_dir": DATA_DIR,
    "outputs_dir": OUTPUTS_DIR,
    "src_dir": SRC_DIR,
}

# Training Configuration (for future use)
TRAINING_CONFIG = {
    "epochs": 100,
    "batch_size": 16,
    "learning_rate": 0.001,
    "image_size": 640,
    "patience": 10,  # Early stopping patience
    "save_best_only": True,
    "validation_split": 0.2,
    "workers": 8,
}

# State codes for Malaysian license plates
MALAYSIAN_STATE_CODES = {
    "A": "Perak",
    "B": "Selangor",
    "C": "Pahang",
    "D": "Kelantan",
    "F": "Putrajaya",
    "G": "Pahang",
    "H": "Selangor",
    "J": "Johor",
    "K": "Kedah",
    "L": "Labuan",
    "M": "Malacca",
    "N": "Negeri Sembilan",
    "P": "Penang",
    "Q": "Sarawak",
    "R": "Perlis",
    "S": "Sabah",
    "T": "Terengganu",
    "U": "Sabah",
    "V": "Kuala Lumpur",
    "W": "Kuala Lumpur",
    "X": "Sabah",
    "Y": "Sabah",
    "Z": "Sabah",
}

def get_model_path(model_name: str) -> Path:
    """Get the full path to a trained model."""
    return TRAINED_MODELS_DIR / model_name

def get_output_path(filename: str) -> Path:
    """Get the full path for output files."""
    return OUTPUTS_DIR / filename

def validate_malaysian_plate(plate_text: str) -> Tuple[bool, str]:
    """Validate if the detected text matches Malaysian plate patterns.
    
    Args:
        plate_text: The detected plate text
        
    Returns:
        Tuple of (is_valid, plate_type)
    """
    import re
    
    # Clean the text
    clean_text = plate_text.strip().upper().replace(" ", "")
    
    # Check against all patterns
    for plate_type, patterns in MALAYSIAN_PLATE_PATTERNS.items():
        for pattern in patterns:
            if re.match(pattern, clean_text):
                return True, plate_type
    
    return False, "unknown"

def get_state_from_plate(plate_text: str) -> str:
    """Extract state information from Malaysian license plate.
    
    Args:
        plate_text: The detected plate text
        
    Returns:
        State name or 'Unknown'
    """
    clean_text = plate_text.strip().upper()
    
    # Extract first letter(s) for state identification
    if len(clean_text) > 0:
        first_char = clean_text[0]
        return MALAYSIAN_STATE_CODES.get(first_char, "Unknown")
    
    return "Unknown"