#!/usr/bin/env python3
"""
Test Fixtures

Provides sample data, mock responses, and test configurations
for the document parser test suite.
"""

import json
from datetime import datetime, timedelta
from typing import Dict, Any, List
from pathlib import Path

import pytest

from src.document_parser.models.document_models import (
    DocumentType, ProcessingStatus, ConfidenceLevel,
    ExtractedField, DocumentMetadata, ExtractionResult
)

# Sample document data

@pytest.fixture
def sample_mykad_data():
    """Sample MyKad document data."""
    return {
        "ic_number": "123456-78-9012",
        "full_name": "AHMAD BIN ALI",
        "address": "123 JALAN MERDEKA\nTAMAN DESA\n50480 KUALA LUMPUR",
        "postcode": "50480",
        "state": "KUALA LUMPUR",
        "nationality": "WARGANEGARA",
        "religion": "ISLAM",
        "race": "MELAYU",
        "date_of_birth": "1978-12-34",
        "place_of_birth": "KUALA LUMPUR",
        "gender": "LELAKI"
    }

@pytest.fixture
def sample_spk_data():
    """Sample SPK certificate data."""
    return {
        "student_name": "SITI AMINAH BINTI HASSAN",
        "ic_number": "987654-32-1098",
        "school_name": "SMK TAMAN DESA",
        "school_code": "ABC1234",
        "certificate_number": "SPM2023001234",
        "examination_year": "2023",
        "examination_session": "NOVEMBER",
        "date_of_birth": "2005-03-15",
        "subjects": [
            {"subject": "BAHASA MELAYU", "grade": "A"},
            {"subject": "BAHASA INGGERIS", "grade": "B+"},
            {"subject": "MATEMATIK", "grade": "A-"},
            {"subject": "SEJARAH", "grade": "B"},
            {"subject": "FIZIK", "grade": "A"},
            {"subject": "KIMIA", "grade": "B+"},
            {"subject": "BIOLOGI", "grade": "A-"}
        ],
        "overall_grade": "DISTINCTION",
        "issue_date": "2024-01-15"
    }

@pytest.fixture
def sample_documents():
    """Collection of sample documents for testing."""
    return {
        "mykad": {
            "valid": [
                {
                    "ic_number": "123456-78-9012",
                    "full_name": "AHMAD BIN ALI",
                    "address": "123 JALAN MERDEKA\nKUALA LUMPUR",
                    "postcode": "50480",
                    "state": "KUALA LUMPUR"
                },
                {
                    "ic_number": "987654-32-1098",
                    "full_name": "SITI FATIMAH BINTI IBRAHIM",
                    "address": "456 JALAN RAJA\nPETALING JAYA",
                    "postcode": "47400",
                    "state": "SELANGOR"
                }
            ],
            "invalid": [
                {
                    "ic_number": "invalid-ic",
                    "full_name": "INVALID NAME 123",
                    "address": "",
                    "postcode": "invalid",
                    "state": ""
                }
            ]
        },
        "spk": {
            "valid": [
                {
                    "student_name": "MUHAMMAD HAFIZ BIN RAHMAN",
                    "ic_number": "050315-14-5678",
                    "school_name": "SMK BANDAR BARU",
                    "certificate_number": "SPM2023001234",
                    "examination_year": "2023"
                },
                {
                    "student_name": "NUR AISYAH BINTI AHMAD",
                    "ic_number": "040820-08-9876",
                    "school_name": "SMK TAMAN MELATI",
                    "certificate_number": "SPM2023005678",
                    "examination_year": "2023"
                }
            ],
            "invalid": [
                {
                    "student_name": "",
                    "ic_number": "invalid",
                    "school_name": "123 INVALID SCHOOL",
                    "certificate_number": "",
                    "examination_year": "invalid"
                }
            ]
        }
    }

# Mock OCR responses

@pytest.fixture
def mock_ocr_responses():
    """Mock OCR service responses."""
    return {
        "mykad_success": {
            "text": "MALAYSIA MyKad No. K/P: 123456-78-9012 Nama: AHMAD BIN ALI Alamat: 123 JALAN MERDEKA KUALA LUMPUR 50480",
            "confidence": 0.95,
            "processing_time": 1.2,
            "engine": "tesseract",
            "language": "eng+msa",
            "words": [
                {"text": "MALAYSIA", "confidence": 0.98, "bbox": [60, 70, 160, 95]},
                {"text": "MyKad", "confidence": 0.97, "bbox": [60, 100, 120, 125]},
                {"text": "123456-78-9012", "confidence": 0.96, "bbox": [150, 150, 300, 175]},
                {"text": "AHMAD", "confidence": 0.95, "bbox": [150, 200, 200, 225]},
                {"text": "BIN", "confidence": 0.94, "bbox": [210, 200, 240, 225]},
                {"text": "ALI", "confidence": 0.95, "bbox": [250, 200, 280, 225]}
            ]
        },
        "spk_success": {
            "text": "SIJIL PELAJARAN MALAYSIA CERTIFICATE Nama Pelajar: SITI AMINAH BINTI HASSAN No. Kad Pengenalan: 987654-32-1098 Sekolah: SMK TAMAN DESA Tahun: 2023",
            "confidence": 0.93,
            "processing_time": 1.5,
            "engine": "easyocr",
            "language": "en+ms",
            "words": [
                {"text": "SIJIL", "confidence": 0.96, "bbox": [100, 80, 150, 105]},
                {"text": "PELAJARAN", "confidence": 0.95, "bbox": [160, 80, 250, 105]},
                {"text": "MALAYSIA", "confidence": 0.97, "bbox": [260, 80, 350, 105]},
                {"text": "SITI", "confidence": 0.94, "bbox": [100, 230, 130, 255]},
                {"text": "AMINAH", "confidence": 0.93, "bbox": [140, 230, 200, 255]},
                {"text": "987654-32-1098", "confidence": 0.92, "bbox": [100, 310, 250, 335]}
            ]
        },
        "low_quality": {
            "text": "CORRUPTED TEXT WITH LOW CONFIDENCE",
            "confidence": 0.45,
            "processing_time": 2.8,
            "engine": "tesseract",
            "language": "eng",
            "words": [
                {"text": "CORRUPTED", "confidence": 0.50, "bbox": [10, 10, 100, 35]},
                {"text": "TEXT", "confidence": 0.40, "bbox": [110, 10, 150, 35]}
            ]
        },
        "empty_result": {
            "text": "",
            "confidence": 0.0,
            "processing_time": 0.5,
            "engine": "tesseract",
            "language": "eng",
            "words": []
        }
    }

# Test configurations

@pytest.fixture
def test_configurations():
    """Test configuration settings."""
    return {
        "ocr_config": {
            "tesseract": {
                "language": "eng+msa",
                "config": "--psm 6 --oem 3",
                "timeout": 30,
                "dpi": 300
            },
            "easyocr": {
                "languages": ["en", "ms"],
                "gpu": False,
                "confidence_threshold": 0.5
            }
        },
        "preprocessing_config": {
            "resize": {
                "enabled": True,
                "max_width": 2000,
                "max_height": 2000
            },
            "denoise": {
                "enabled": True,
                "method": "gaussian",
                "kernel_size": 3
            },
            "contrast": {
                "enabled": True,
                "factor": 1.2
            },
            "rotation_correction": {
                "enabled": True,
                "max_angle": 10
            }
        },
        "validation_config": {
            "ic_number": {
                "pattern": r"^\d{6}-\d{2}-\d{4}$",
                "checksum_validation": True
            },
            "name": {
                "min_length": 2,
                "max_length": 100,
                "allowed_chars": "ABCDEFGHIJKLMNOPQRSTUVWXYZ @/.-'"
            },
            "postcode": {
                "pattern": r"^\d{5}$",
                "state_validation": True
            }
        },
        "performance_thresholds": {
            "ocr_processing_time": 5.0,
            "classification_time": 1.0,
            "field_extraction_time": 2.0,
            "validation_time": 0.5,
            "total_processing_time": 10.0
        }
    }

# Validation test cases

@pytest.fixture
def validation_test_cases():
    """Test cases for validation functions."""
    return {
        "ic_number": {
            "valid": [
                "123456-78-9012",
                "987654-32-1098",
                "050315-14-5678",
                "040820-08-9876"
            ],
            "invalid": [
                "invalid-ic",
                "123456789012",
                "12-34-56",
                "123456-78-90123",
                "abcdef-gh-ijkl",
                "",
                "123456-78-",
                "123456--9012"
            ]
        },
        "name": {
            "valid": [
                "AHMAD BIN ALI",
                "SITI FATIMAH BINTI IBRAHIM",
                "MUHAMMAD HAFIZ BIN RAHMAN",
                "NUR AISYAH BINTI AHMAD",
                "TAN WEI MING",
                "RAJESH A/L KUMAR",
                "MARY D/O ANTHONY"
            ],
            "invalid": [
                "INVALID NAME 123",
                "name with numbers 456",
                "special@characters#here",
                "A",  # Too short
                "A" * 101,  # Too long
                "",  # Empty
                "lowercase name"
            ]
        },
        "postcode": {
            "valid": [
                "50480",
                "47400",
                "10200",
                "80000",
                "93350"
            ],
            "invalid": [
                "invalid",
                "1234",  # Too short
                "123456",  # Too long
                "abcde",
                "",
                "12 34"
            ]
        },
        "date": {
            "valid": [
                "1978-12-34",
                "2005-03-15",
                "1990-01-01",
                "2000-12-31"
            ],
            "invalid": [
                "invalid-date",
                "1978/12/34",
                "34-12-1978",
                "2005-13-15",  # Invalid month
                "2005-03-32",  # Invalid day
                "",
                "1900-01-01",  # Too old
                "2030-01-01"   # Future date
            ]
        }
    }

# API test data

@pytest.fixture
def api_test_data():
    """Test data for API endpoints."""
    return {
        "upload_requests": {
            "valid": {
                "document_type": "mykad",
                "extract_fields": ["ic_number", "full_name", "address"],
                "validation_level": "standard",
                "return_coordinates": True,
                "return_confidence": True,
                "preprocessing_options": {
                    "denoise": True,
                    "contrast_enhancement": True
                }
            },
            "minimal": {
                "document_type": None
            },
            "invalid": {
                "document_type": "invalid_type",
                "extract_fields": ["invalid_field"],
                "validation_level": "invalid_level"
            }
        },
        "batch_requests": {
            "valid": {
                "batch_name": "Test Batch",
                "document_type": "spk",
                "processing_options": {
                    "validation_level": "strict",
                    "return_confidence": True
                },
                "priority": 5
            },
            "minimal": {
                "batch_name": "Minimal Batch"
            }
        },
        "expected_responses": {
            "upload_success": {
                "success": True,
                "message": "Document uploaded successfully",
                "document_id": "doc_123456789",
                "status": "processing"
            },
            "upload_error": {
                "success": False,
                "message": "Document upload failed",
                "error": {
                    "error_type": "validation_error",
                    "error_code": "INVALID_FILE_FORMAT",
                    "message": "Unsupported file format"
                }
            },
            "processing_complete": {
                "success": True,
                "message": "Document processed successfully",
                "document_id": "doc_123456789",
                "status": "completed",
                "result": {
                    "document_type": "mykad",
                    "classification_confidence": 0.95,
                    "extracted_fields": [
                        {
                            "field_name": "ic_number",
                            "value": "123456-78-9012",
                            "confidence": 0.96,
                            "confidence_level": "high"
                        }
                    ]
                }
            }
        }
    }

# Performance test data

@pytest.fixture
def performance_test_data():
    """Performance testing scenarios."""
    return {
        "load_test_scenarios": [
            {
                "name": "single_document",
                "concurrent_requests": 1,
                "total_requests": 10,
                "document_type": "mykad",
                "expected_avg_response_time": 3.0
            },
            {
                "name": "moderate_load",
                "concurrent_requests": 5,
                "total_requests": 50,
                "document_type": "spk",
                "expected_avg_response_time": 5.0
            },
            {
                "name": "high_load",
                "concurrent_requests": 10,
                "total_requests": 100,
                "document_type": "mixed",
                "expected_avg_response_time": 8.0
            }
        ],
        "stress_test_scenarios": [
            {
                "name": "memory_stress",
                "large_file_size_mb": 50,
                "concurrent_uploads": 3,
                "max_memory_usage_mb": 1000
            },
            {
                "name": "cpu_stress",
                "complex_documents": 20,
                "concurrent_processing": 8,
                "max_cpu_usage_percent": 80
            }
        ]
    }

# Error simulation data

@pytest.fixture
def error_simulation_data():
    """Data for testing error handling."""
    return {
        "ocr_errors": [
            {
                "error_type": "timeout",
                "message": "OCR processing timeout",
                "should_retry": True
            },
            {
                "error_type": "engine_failure",
                "message": "Tesseract engine not found",
                "should_retry": False
            },
            {
                "error_type": "low_quality",
                "message": "Image quality too low for OCR",
                "should_retry": False
            }
        ],
        "validation_errors": [
            {
                "field": "ic_number",
                "error_type": "format_error",
                "message": "Invalid IC number format"
            },
            {
                "field": "name",
                "error_type": "content_error",
                "message": "Name contains invalid characters"
            }
        ],
        "system_errors": [
            {
                "error_type": "database_connection",
                "message": "Database connection failed",
                "should_retry": True
            },
            {
                "error_type": "redis_connection",
                "message": "Redis connection failed",
                "should_retry": True
            },
            {
                "error_type": "disk_space",
                "message": "Insufficient disk space",
                "should_retry": False
            }
        ]
    }

# Integration test data

@pytest.fixture
def integration_test_data():
    """Data for integration testing."""
    return {
        "end_to_end_scenarios": [
            {
                "name": "mykad_complete_flow",
                "document_type": "mykad",
                "input_data": {
                    "ic_number": "123456-78-9012",
                    "full_name": "AHMAD BIN ALI",
                    "address": "123 JALAN MERDEKA\nKUALA LUMPUR"
                },
                "expected_output": {
                    "document_type": "mykad",
                    "status": "completed",
                    "field_count": 11,
                    "validation_passed": True
                }
            },
            {
                "name": "spk_complete_flow",
                "document_type": "spk",
                "input_data": {
                    "student_name": "SITI AMINAH BINTI HASSAN",
                    "ic_number": "987654-32-1098",
                    "school_name": "SMK TAMAN DESA"
                },
                "expected_output": {
                    "document_type": "spk",
                    "status": "completed",
                    "field_count": 9,
                    "validation_passed": True
                }
            }
        ],
        "batch_processing_scenarios": [
            {
                "name": "mixed_document_batch",
                "documents": [
                    {"type": "mykad", "count": 3},
                    {"type": "spk", "count": 2}
                ],
                "expected_results": {
                    "total_documents": 5,
                    "successful_processing": 5,
                    "failed_processing": 0
                }
            }
        ]
    }

# Export all fixtures
__all__ = [
    "sample_mykad_data",
    "sample_spk_data",
    "sample_documents",
    "mock_ocr_responses",
    "test_configurations",
    "validation_test_cases",
    "api_test_data",
    "performance_test_data",
    "error_simulation_data",
    "integration_test_data"
]