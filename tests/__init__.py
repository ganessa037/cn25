#!/usr/bin/env python3
"""
Document Parser Tests

Test suite for the document parser system including unit tests,
integration tests, and performance tests.
"""

__version__ = "1.0.0"
__author__ = "Document Parser Team"

# Test configuration
TEST_CONFIG = {
    "test_data_dir": "tests/data",
    "temp_dir": "tests/temp",
    "fixtures_dir": "tests/fixtures",
    "mock_responses_dir": "tests/mocks",
    "performance_thresholds": {
        "ocr_processing_time": 5.0,  # seconds
        "classification_time": 1.0,  # seconds
        "field_extraction_time": 2.0,  # seconds
        "validation_time": 0.5,  # seconds
        "api_response_time": 3.0  # seconds
    },
    "test_documents": {
        "spk_samples": 5,
        "mykad_samples": 5,
        "invalid_samples": 3
    }
}

# Test utilities
from .utils import (
    TestDataManager,
    MockServices,
    PerformanceProfiler,
    TestDocumentGenerator
)

# Test fixtures
from .fixtures import (
    sample_documents,
    mock_ocr_responses,
    test_configurations,
    validation_test_cases
)

__all__ = [
    "TEST_CONFIG",
    "TestDataManager",
    "MockServices", 
    "PerformanceProfiler",
    "TestDocumentGenerator",
    "sample_documents",
    "mock_ocr_responses",
    "test_configurations",
    "validation_test_cases"
]