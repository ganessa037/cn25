#!/usr/bin/env python3
"""
Global Test Configuration and Fixtures

Centralized pytest configuration and fixtures for all test suites including
unit tests, integration tests, and user acceptance tests for the Malaysian document parser.
"""

import os
import sys
import json
import tempfile
import shutil
import logging
from pathlib import Path
from typing import Dict, List, Any, Optional, Generator
from datetime import datetime
from unittest.mock import Mock, MagicMock, patch

import pytest
import requests
from PIL import Image
import numpy as np
import cv2

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

# Configure logging for tests
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

# Test configuration directories
TEST_DATA_DIR = project_root / "tests" / "data"
TEST_OUTPUT_DIR = project_root / "tests" / "output"
TEST_TEMP_DIR = project_root / "tests" / "temp"

# Ensure test directories exist
for test_dir in [TEST_DATA_DIR, TEST_OUTPUT_DIR, TEST_TEMP_DIR]:
    test_dir.mkdir(parents=True, exist_ok=True)


# ============================================================================
# Test Configuration
# ============================================================================

class TestConfig:
    """Global test configuration."""
    
    # Test environment settings
    TEST_ENV = os.getenv('TEST_ENV', 'test')
    API_BASE_URL = os.getenv('TEST_API_URL', 'http://localhost:8000')
    API_TIMEOUT = int(os.getenv('TEST_API_TIMEOUT', '30'))
    
    # Database settings
    TEST_DATABASE_URL = os.getenv('TEST_DATABASE_URL', 'sqlite:///test.db')
    
    # File system settings
    TEST_DATA_DIR = TEST_DATA_DIR
    TEMP_DIR = Path(tempfile.gettempdir()) / 'document_parser_tests'
    UPLOAD_DIR = TEMP_DIR / 'uploads'
    OUTPUT_DIR = TEMP_DIR / 'outputs'
    
    # Test execution settings
    PARALLEL_WORKERS = int(os.getenv('TEST_PARALLEL_WORKERS', '4'))
    TEST_TIMEOUT = int(os.getenv('TEST_TIMEOUT', '300'))
    
    # Performance thresholds
    MAX_PROCESSING_TIME = float(os.getenv('TEST_MAX_PROCESSING_TIME', '10.0'))
    MAX_MEMORY_USAGE = float(os.getenv('TEST_MAX_MEMORY_USAGE', '512.0'))
    
    # Accuracy thresholds
    MIN_ACCURACY = float(os.getenv('TEST_MIN_ACCURACY', '0.85'))
    MIN_CONFIDENCE = float(os.getenv('TEST_MIN_CONFIDENCE', '0.7'))
    
    # Mock settings
    ENABLE_MOCKS = os.getenv('TEST_ENABLE_MOCKS', 'true').lower() == 'true'
    MOCK_EXTERNAL_APIS = os.getenv('TEST_MOCK_EXTERNAL_APIS', 'true').lower() == 'true'
    
    # Cleanup settings
    CLEANUP_TEMP_FILES = os.getenv('TEST_CLEANUP_TEMP_FILES', 'true').lower() == 'true'
    PRESERVE_FAILED_TEST_DATA = os.getenv('TEST_PRESERVE_FAILED_DATA', 'false').lower() == 'true'
    
    # Legacy compatibility
    @property
    def test_data_dir(self):
        return self.TEST_DATA_DIR
    
    @property
    def test_output_dir(self):
        return TEST_OUTPUT_DIR
    
    @property
    def test_temp_dir(self):
        return TEST_TEMP_DIR
    
    @property
    def project_root(self):
        return project_root
    
    @property
    def use_gpu(self):
        return False
    
    @property
    def confidence_threshold(self):
        return 0.5
    
    @property
    def timeout(self):
        return self.API_TIMEOUT
    
    @property
    def max_file_size(self):
        return 10 * 1024 * 1024  # 10MB
    
    @property
    def supported_formats(self):
        return ["jpg", "jpeg", "png", "bmp", "pdf"]


# ============================================================================
# Session-level Fixtures
# ============================================================================

@pytest.fixture(scope='session', autouse=True)
def setup_test_environment():
    """Setup test environment before all tests."""
    # Create test directories
    TestConfig.TEMP_DIR.mkdir(parents=True, exist_ok=True)
    TestConfig.UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
    TestConfig.OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    TestConfig.TEST_DATA_DIR.mkdir(parents=True, exist_ok=True)
    
    # Setup logging
    logging.getLogger('requests').setLevel(logging.WARNING)
    logging.getLogger('urllib3').setLevel(logging.WARNING)
    
    yield
    
    # Cleanup after all tests
    if TestConfig.CLEANUP_TEMP_FILES:
        if TestConfig.TEMP_DIR.exists():
            shutil.rmtree(TestConfig.TEMP_DIR, ignore_errors=True)


@pytest.fixture(scope="session")
def test_config():
    """Global test configuration."""
    return TestConfig()


@pytest.fixture(scope='session')
def api_client():
    """Create API client for testing."""
    class APIClient:
        def __init__(self, base_url: str, timeout: int = 30):
            self.base_url = base_url.rstrip('/')
            self.timeout = timeout
            self.session = requests.Session()
        
        def get(self, endpoint: str, **kwargs):
            url = f"{self.base_url}/{endpoint.lstrip('/')}"
            return self.session.get(url, timeout=self.timeout, **kwargs)
        
        def post(self, endpoint: str, **kwargs):
            url = f"{self.base_url}/{endpoint.lstrip('/')}"
            return self.session.post(url, timeout=self.timeout, **kwargs)
        
        def put(self, endpoint: str, **kwargs):
            url = f"{self.base_url}/{endpoint.lstrip('/')}"
            return self.session.put(url, timeout=self.timeout, **kwargs)
        
        def delete(self, endpoint: str, **kwargs):
            url = f"{self.base_url}/{endpoint.lstrip('/')}"
            return self.session.delete(url, timeout=self.timeout, **kwargs)
        
        def upload_file(self, endpoint: str, file_path: str, field_name: str = 'file'):
            url = f"{self.base_url}/{endpoint.lstrip('/')}"
            with open(file_path, 'rb') as f:
                files = {field_name: f}
                return self.session.post(url, files=files, timeout=self.timeout)
        
        def health_check(self):
            try:
                response = self.get('/health')
                return response.status_code == 200
            except Exception:
                return False
    
    return APIClient(TestConfig.API_BASE_URL, TestConfig.API_TIMEOUT)


@pytest.fixture(scope="session")
def test_database():
    """Create a test database for testing."""
    # This would set up a test database
    # For now, we'll use a mock
    db_mock = Mock()
    db_mock.connect.return_value = True
    db_mock.disconnect.return_value = True
    return db_mock


@pytest.fixture(scope="session")
def test_redis():
    """Create a test Redis instance for testing."""
    # This would set up a test Redis instance
    # For now, we'll use a mock
    redis_mock = Mock()
    redis_mock.ping.return_value = True
    redis_mock.set.return_value = True
    redis_mock.get.return_value = None
    redis_mock.delete.return_value = True
    redis_mock.flushdb.return_value = True
    return redis_mock


# ============================================================================
# Function-level Fixtures
# ============================================================================

@pytest.fixture
def temp_dir():
    """Create temporary directory for test."""
    temp_path = TestConfig.TEMP_DIR / f"test_{datetime.now().strftime('%Y%m%d_%H%M%S_%f')}"
    temp_path.mkdir(parents=True, exist_ok=True)
    
    yield temp_path
    
    # Cleanup
    if TestConfig.CLEANUP_TEMP_FILES and temp_path.exists():
        shutil.rmtree(temp_path, ignore_errors=True)


@pytest.fixture
def sample_images():
    """Generate sample images for testing."""
    class SampleImageGenerator:
        @staticmethod
        def create_mykad_image(width: int = 800, height: int = 500, quality: str = 'high') -> Image.Image:
            """Create sample MyKad image."""
            # Create base image
            if quality == 'high':
                img = Image.new('RGB', (width, height), color='lightblue')
            elif quality == 'medium':
                img = Image.new('RGB', (width, height), color='lightgray')
            else:  # low quality
                img = Image.new('RGB', (width//2, height//2), color='gray')
                img = img.resize((width, height))
            
            return img
        
        @staticmethod
        def create_spk_image(width: int = 600, height: int = 800, quality: str = 'high') -> Image.Image:
            """Create sample SPK certificate image."""
            # Create base image
            if quality == 'high':
                img = Image.new('RGB', (width, height), color='white')
            elif quality == 'medium':
                img = Image.new('RGB', (width, height), color='lightgray')
            else:  # low quality
                img = Image.new('RGB', (width//2, height//2), color='gray')
                img = img.resize((width, height))
            
            return img
        
        @staticmethod
        def create_damaged_image(base_image: Image.Image, damage_type: str = 'noise') -> Image.Image:
            """Create damaged version of image."""
            img_array = np.array(base_image)
            
            if damage_type == 'noise':
                noise = np.random.randint(0, 50, img_array.shape, dtype=np.uint8)
                img_array = np.clip(img_array.astype(np.int16) + noise, 0, 255).astype(np.uint8)
            elif damage_type == 'blur':
                # Simple blur simulation
                from scipy import ndimage
                img_array = ndimage.gaussian_filter(img_array, sigma=2)
            elif damage_type == 'dark':
                img_array = (img_array * 0.5).astype(np.uint8)
            elif damage_type == 'bright':
                img_array = np.clip(img_array * 1.5, 0, 255).astype(np.uint8)
            
            return Image.fromarray(img_array)
        
        @staticmethod
        def save_image(image: Image.Image, file_path: str, format: str = 'JPEG'):
            """Save image to file."""
            image.save(file_path, format=format)
    
    return SampleImageGenerator


@pytest.fixture
def mock_data():
    """Generate mock data for testing."""
    class MockDataGenerator:
        @staticmethod
        def mykad_data(quality: str = 'complete') -> Dict[str, Any]:
            """Generate mock MyKad data."""
            base_data = {
                'ic_number': '123456-12-1234',
                'name': 'AHMAD BIN ALI',
                'date_of_birth': '12/06/1990',
                'place_of_birth': 'KUALA LUMPUR',
                'address': 'NO. 123, JALAN MERDEKA, 50000 KUALA LUMPUR',
                'religion': 'ISLAM',
                'race': 'MELAYU',
                'gender': 'LELAKI',
                'citizenship': 'WARGANEGARA'
            }
            
            if quality == 'incomplete':
                # Remove some fields
                del base_data['place_of_birth']
                del base_data['citizenship']
            elif quality == 'corrupted':
                # Corrupt some data
                base_data['ic_number'] = '123456-XX-1234'
                base_data['date_of_birth'] = '32/13/1990'
            
            return base_data
        
        @staticmethod
        def spk_data(quality: str = 'complete') -> Dict[str, Any]:
            """Generate mock SPK data."""
            base_data = {
                'certificate_number': 'SPK123456789',
                'student_name': 'SITI AMINAH BINTI HASSAN',
                'ic_number': '987654-32-9876',
                'school_name': 'SMK TAMAN DESA',
                'year': '2020',
                'subjects': {
                    'BAHASA MELAYU': 'A',
                    'BAHASA INGGERIS': 'B+',
                    'MATEMATIK': 'A-',
                    'SEJARAH': 'B',
                    'SAINS': 'A'
                },
                'grades': ['A', 'B+', 'A-', 'B', 'A']
            }
            
            if quality == 'incomplete':
                # Remove some subjects
                del base_data['subjects']['SAINS']
                base_data['grades'] = base_data['grades'][:-1]
            elif quality == 'corrupted':
                # Corrupt some data
                base_data['year'] = '202X'
                base_data['subjects']['MATEMATIK'] = 'Z+'
            
            return base_data
        
        @staticmethod
        def api_response(success: bool = True, data: Optional[Dict] = None, error: Optional[str] = None) -> Dict[str, Any]:
            """Generate mock API response."""
            if success:
                return {
                    'success': True,
                    'data': data or {},
                    'message': 'Operation completed successfully',
                    'timestamp': datetime.now().isoformat()
                }
            else:
                return {
                    'success': False,
                    'error': error or 'Unknown error occurred',
                    'message': 'Operation failed',
                    'timestamp': datetime.now().isoformat()
                }
        
        @staticmethod
        def processing_result(document_type: str = 'mykad', accuracy: float = 0.95) -> Dict[str, Any]:
            """Generate mock processing result."""
            if document_type == 'mykad':
                extracted_data = MockDataGenerator.mykad_data()
            else:
                extracted_data = MockDataGenerator.spk_data()
            
            return {
                'document_type': document_type,
                'confidence_score': accuracy,
                'extracted_data': extracted_data,
                'processing_time': 2.5,
                'validation_status': 'passed' if accuracy > 0.8 else 'failed',
                'field_confidences': {field: accuracy + np.random.uniform(-0.1, 0.1) for field in extracted_data.keys()}
            }
    
    return MockDataGenerator


@pytest.fixture
def performance_monitor():
    """Monitor test performance."""
    import psutil
    import time
    
    class PerformanceMonitor:
        def __init__(self):
            self.start_time = None
            self.end_time = None
            self.start_memory = None
            self.peak_memory = None
            self.process = psutil.Process()
        
        def start(self):
            """Start monitoring."""
            self.start_time = time.time()
            self.start_memory = self.process.memory_info().rss / 1024 / 1024  # MB
            self.peak_memory = self.start_memory
        
        def update(self):
            """Update peak memory usage."""
            current_memory = self.process.memory_info().rss / 1024 / 1024  # MB
            self.peak_memory = max(self.peak_memory, current_memory)
        
        def stop(self) -> Dict[str, float]:
            """Stop monitoring and return metrics."""
            self.end_time = time.time()
            self.update()
            
            return {
                'execution_time': self.end_time - self.start_time,
                'memory_usage': self.peak_memory - self.start_memory,
                'peak_memory': self.peak_memory
            }
    
    return PerformanceMonitor()


@pytest.fixture
def sample_image():
    """Create a sample test image."""
    # Create a realistic document-like image
    image = np.ones((800, 600, 3), dtype=np.uint8) * 255
    
    # Add document-like elements
    # Header
    cv2.rectangle(image, (50, 50), (550, 100), (200, 200, 200), -1)
    cv2.putText(image, "GOVERNMENT OF MALAYSIA", (60, 80), 
               cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)
    
    # ID Number field
    cv2.rectangle(image, (50, 120), (550, 160), (240, 240, 240), -1)
    cv2.putText(image, "IC Number: 123456-78-9012", (60, 145), 
               cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 1)
    
    # Name field
    cv2.rectangle(image, (50, 180), (550, 220), (240, 240, 240), -1)
    cv2.putText(image, "Name: AHMAD BIN ALI", (60, 205), 
               cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 1)
    
    # Address field
    cv2.rectangle(image, (50, 240), (550, 320), (240, 240, 240), -1)
    cv2.putText(image, "Address: 123 JALAN MERDEKA", (60, 265), 
               cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
    cv2.putText(image, "TAMAN DESA", (60, 285), 
               cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
    cv2.putText(image, "50480 KUALA LUMPUR", (60, 305), 
               cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
    
    return image


@pytest.fixture
def sample_mykad_image():
    """Create a sample MyKad-like image."""
    image = np.ones((600, 900, 3), dtype=np.uint8) * 255
    
    # MyKad header
    cv2.rectangle(image, (50, 30), (850, 80), (0, 100, 200), -1)
    cv2.putText(image, "MALAYSIA", (350, 60), 
               cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255, 255, 255), 3)
    
    # IC Number
    cv2.putText(image, "123456-78-9012", (500, 150), 
               cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 0), 2)
    
    # Name
    cv2.putText(image, "AHMAD BIN ALI", (500, 200), 
               cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 2)
    
    # Address
    cv2.putText(image, "123 JALAN MERDEKA", (500, 250), 
               cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 1)
    cv2.putText(image, "TAMAN DESA", (500, 280), 
               cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 1)
    cv2.putText(image, "50480 KUALA LUMPUR", (500, 310), 
               cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 1)
    
    # Photo placeholder
    cv2.rectangle(image, (100, 150), (300, 350), (200, 200, 200), -1)
    cv2.putText(image, "PHOTO", (150, 250), 
               cv2.FONT_HERSHEY_SIMPLEX, 0.8, (100, 100, 100), 2)
    
    return image


@pytest.fixture
def sample_spk_image():
    """Create a sample SPK certificate image."""
    image = np.ones((800, 1000, 3), dtype=np.uint8) * 255
    
    # Certificate header
    cv2.rectangle(image, (50, 30), (950, 100), (0, 50, 150), -1)
    cv2.putText(image, "SIJIL PELAJARAN MALAYSIA", (200, 70), 
               cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2)
    
    # Student details
    cv2.putText(image, "Name: SITI AMINAH BINTI HASSAN", (100, 200), 
               cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)
    
    cv2.putText(image, "IC Number: 987654-32-1098", (100, 240), 
               cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)
    
    cv2.putText(image, "School: SMK TAMAN DESA", (100, 280), 
               cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)
    
    cv2.putText(image, "Certificate No: SPM2023001234", (100, 320), 
               cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)
    
    cv2.putText(image, "Year: 2023", (100, 360), 
               cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)
    
    # Grades table header
    cv2.rectangle(image, (100, 420), (900, 460), (200, 200, 200), -1)
    cv2.putText(image, "SUBJECT", (150, 445), 
               cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)
    cv2.putText(image, "GRADE", (500, 445), 
               cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)
    
    # Sample grades
    subjects = [("BAHASA MALAYSIA", "A"), ("ENGLISH", "B+"), ("MATHEMATICS", "A-")]
    for i, (subject, grade) in enumerate(subjects):
        y_pos = 490 + (i * 30)
        cv2.putText(image, subject, (150, y_pos), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
        cv2.putText(image, grade, (500, y_pos), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
    
    return image


@pytest.fixture
def mock_ocr_service():
    """Create a mock OCR service."""
    mock_service = Mock()
    mock_service.extract_text.return_value = {
        'text_blocks': [
            {
                'text': 'GOVERNMENT OF MALAYSIA',
                'confidence': 0.95,
                'coordinates': [50, 50, 550, 100]
            },
            {
                'text': '123456-78-9012',
                'confidence': 0.92,
                'coordinates': [500, 130, 700, 160]
            },
            {
                'text': 'AHMAD BIN ALI',
                'confidence': 0.88,
                'coordinates': [500, 180, 700, 210]
            }
        ],
        'processing_time': 1.23
    }
    return mock_service


@pytest.fixture
def mock_classifier():
    """Create a mock document classifier."""
    mock_classifier = Mock()
    mock_classifier.classify.return_value = {
        'document_type': 'mykad',
        'confidence': 0.95,
        'probabilities': {
            'mykad': 0.95,
            'spk': 0.03,
            'unknown': 0.02
        }
    }
    return mock_classifier


@pytest.fixture
def mock_field_extractor():
    """Create a mock field extractor."""
    mock_extractor = Mock()
    mock_extractor.extract_fields.return_value = {
        'ic_number': {
            'value': '123456-78-9012',
            'confidence': 0.92,
            'coordinates': [500, 130, 700, 160],
            'is_valid': True
        },
        'full_name': {
            'value': 'AHMAD BIN ALI',
            'confidence': 0.88,
            'coordinates': [500, 180, 700, 210],
            'is_valid': True
        }
    }
    return mock_extractor


@pytest.fixture
def mock_validator():
    """Create a mock document validator."""
    mock_validator = Mock()
    mock_validator.validate.return_value = {
        'is_valid': True,
        'errors': [],
        'warnings': [],
        'field_validations': {
            'ic_number': {'is_valid': True, 'message': 'Valid IC format'},
            'full_name': {'is_valid': True, 'message': 'Valid name format'}
        }
    }
    return mock_validator


@pytest.fixture(autouse=True)
def setup_test_environment(test_config):
    """Set up test environment before each test."""
    # Set environment variables for testing
    os.environ['TESTING'] = 'true'
    os.environ['LOG_LEVEL'] = 'DEBUG'
    os.environ['USE_GPU'] = 'false'
    
    yield
    
    # Cleanup after test
    # Remove test environment variables
    os.environ.pop('TESTING', None)
    os.environ.pop('LOG_LEVEL', None)
    os.environ.pop('USE_GPU', None)


@pytest.fixture(autouse=True)
def cleanup_test_files(test_config):
    """Clean up test files after each test."""
    yield
    
    # Clean up temporary files
    temp_patterns = [
        "output_*.jpg",
        "output_*.json",
        "test_*.tmp",
        "*.log"
    ]
    
    for pattern in temp_patterns:
        for file_path in test_config["test_output_dir"].glob(pattern):
            try:
                file_path.unlink()
            except OSError:
                pass


# Pytest configuration
def pytest_configure(config):
    """Configure pytest settings."""
    # Add custom markers
    config.addinivalue_line(
        "markers", "slow: marks tests as slow (deselect with '-m \"not slow\"')"
    )
    config.addinivalue_line(
        "markers", "integration: marks tests as integration tests"
    )
    config.addinivalue_line(
        "markers", "gpu: marks tests that require GPU"
    )
    config.addinivalue_line(
        "markers", "api: marks tests for API endpoints"
    )
    config.addinivalue_line(
        "markers", "unit: marks tests as unit tests"
    )


def pytest_collection_modifyitems(config, items):
    """Modify test collection to add markers automatically."""
    for item in items:
        # Add 'unit' marker to all tests by default
        if not any(marker.name in ['integration', 'api', 'slow'] for marker in item.iter_markers()):
            item.add_marker(pytest.mark.unit)
        
        # Add 'slow' marker to tests that might be slow
        if 'integration' in item.name or 'end_to_end' in item.name:
            item.add_marker(pytest.mark.slow)
        
        # Add 'api' marker to API tests
        if 'api' in str(item.fspath) or 'routes' in str(item.fspath):
            item.add_marker(pytest.mark.api)


# Skip GPU tests if no GPU available
def pytest_runtest_setup(item):
    """Setup for individual test runs."""
    if 'gpu' in [marker.name for marker in item.iter_markers()]:
        # Check if GPU is available
        try:
            import torch
            if not torch.cuda.is_available():
                pytest.skip("GPU not available")
        except ImportError:
            pytest.skip("PyTorch not available")