#!/usr/bin/env python3
"""
Test Utilities

Utility classes and functions for testing the document parser system.
Provides mock services, test data management, and performance profiling.
"""

import os
import json
import time
import tempfile
import shutil
from pathlib import Path
from typing import Dict, Any, List, Optional, Union
from datetime import datetime, timedelta
from unittest.mock import Mock, AsyncMock, patch
import asyncio
from contextlib import asynccontextmanager

import pytest
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import redis.asyncio as redis
from sqlalchemy.ext.asyncio import AsyncSession, create_async_engine
from sqlalchemy.pool import StaticPool

from src.document_parser.models.document_models import (
    DocumentType, ProcessingStatus, ConfidenceLevel,
    ExtractedField, DocumentMetadata, ExtractionResult
)
from src.document_parser.config import DocumentParserConfig

class TestDataManager:
    """Manages test data files and temporary directories."""
    
    def __init__(self, base_dir: str = "tests"):
        self.base_dir = Path(base_dir)
        self.data_dir = self.base_dir / "data"
        self.temp_dir = self.base_dir / "temp"
        self.fixtures_dir = self.base_dir / "fixtures"
        
        # Create directories
        for directory in [self.data_dir, self.temp_dir, self.fixtures_dir]:
            directory.mkdir(parents=True, exist_ok=True)
    
    def get_test_image_path(self, document_type: DocumentType, sample_id: int = 1) -> Path:
        """Get path to test image file."""
        filename = f"{document_type.value}_sample_{sample_id}.jpg"
        return self.data_dir / filename
    
    def get_temp_file(self, suffix: str = ".tmp") -> Path:
        """Create a temporary file and return its path."""
        temp_file = tempfile.NamedTemporaryFile(
            dir=self.temp_dir,
            suffix=suffix,
            delete=False
        )
        temp_file.close()
        return Path(temp_file.name)
    
    def get_temp_dir(self) -> Path:
        """Create a temporary directory and return its path."""
        return Path(tempfile.mkdtemp(dir=self.temp_dir))
    
    def cleanup_temp_files(self):
        """Clean up all temporary files and directories."""
        if self.temp_dir.exists():
            shutil.rmtree(self.temp_dir)
            self.temp_dir.mkdir(parents=True, exist_ok=True)
    
    def load_test_config(self, config_name: str) -> Dict[str, Any]:
        """Load test configuration from JSON file."""
        config_path = self.fixtures_dir / f"{config_name}.json"
        
        if not config_path.exists():
            raise FileNotFoundError(f"Test config not found: {config_path}")
        
        with open(config_path, 'r') as f:
            return json.load(f)
    
    def save_test_result(self, test_name: str, result: Dict[str, Any]):
        """Save test result for analysis."""
        result_path = self.temp_dir / f"{test_name}_result.json"
        
        with open(result_path, 'w') as f:
            json.dump(result, f, indent=2, default=str)

class TestDocumentGenerator:
    """Generates synthetic test documents for testing."""
    
    def __init__(self, output_dir: str = "tests/data"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
    
    def generate_mykad_image(self, 
                           ic_number: str = "123456-78-9012",
                           name: str = "AHMAD BIN ALI",
                           address: str = "123 JALAN MERDEKA\nKUALA LUMPUR",
                           width: int = 800,
                           height: int = 500) -> Path:
        """Generate a synthetic MyKad image for testing."""
        
        # Create image
        img = Image.new('RGB', (width, height), color='lightblue')
        draw = ImageDraw.Draw(img)
        
        try:
            # Try to use a system font
            font_large = ImageFont.truetype("/System/Library/Fonts/Arial.ttf", 24)
            font_medium = ImageFont.truetype("/System/Library/Fonts/Arial.ttf", 18)
            font_small = ImageFont.truetype("/System/Library/Fonts/Arial.ttf", 14)
        except (OSError, IOError):
            # Fallback to default font
            font_large = ImageFont.load_default()
            font_medium = ImageFont.load_default()
            font_small = ImageFont.load_default()
        
        # Draw MyKad layout
        draw.rectangle([50, 50, width-50, height-50], outline='black', width=3)
        
        # Title
        draw.text((60, 70), "MALAYSIA", fill='black', font=font_large)
        draw.text((60, 100), "MyKad", fill='black', font=font_medium)
        
        # IC Number
        draw.text((60, 150), "No. K/P:", fill='black', font=font_medium)
        draw.text((150, 150), ic_number, fill='black', font=font_large)
        
        # Name
        draw.text((60, 200), "Nama:", fill='black', font=font_medium)
        draw.text((150, 200), name, fill='black', font=font_large)
        
        # Address
        draw.text((60, 250), "Alamat:", fill='black', font=font_medium)
        address_lines = address.split('\n')
        for i, line in enumerate(address_lines):
            draw.text((150, 250 + i*25), line, fill='black', font=font_medium)
        
        # Save image
        output_path = self.output_dir / f"mykad_test_{int(time.time())}.jpg"
        img.save(output_path, 'JPEG')
        
        return output_path
    
    def generate_spk_image(self,
                          student_name: str = "SITI AMINAH BINTI HASSAN",
                          ic_number: str = "987654-32-1098",
                          school: str = "SMK TAMAN DESA",
                          year: str = "2023",
                          width: int = 800,
                          height: int = 600) -> Path:
        """Generate a synthetic SPK certificate image for testing."""
        
        # Create image
        img = Image.new('RGB', (width, height), color='white')
        draw = ImageDraw.Draw(img)
        
        try:
            font_title = ImageFont.truetype("/System/Library/Fonts/Arial.ttf", 28)
            font_large = ImageFont.truetype("/System/Library/Fonts/Arial.ttf", 20)
            font_medium = ImageFont.truetype("/System/Library/Fonts/Arial.ttf", 16)
        except (OSError, IOError):
            font_title = ImageFont.load_default()
            font_large = ImageFont.load_default()
            font_medium = ImageFont.load_default()
        
        # Draw certificate layout
        draw.rectangle([30, 30, width-30, height-30], outline='navy', width=5)
        draw.rectangle([50, 50, width-50, height-50], outline='navy', width=2)
        
        # Title
        draw.text((width//2-150, 80), "SIJIL PELAJARAN MALAYSIA", fill='navy', font=font_title)
        draw.text((width//2-100, 120), "CERTIFICATE", fill='navy', font=font_large)
        
        # Student details
        draw.text((100, 200), "Nama Pelajar / Student Name:", fill='black', font=font_medium)
        draw.text((100, 230), student_name, fill='black', font=font_large)
        
        draw.text((100, 280), "No. Kad Pengenalan / Identity Card No.:", fill='black', font=font_medium)
        draw.text((100, 310), ic_number, fill='black', font=font_large)
        
        draw.text((100, 360), "Sekolah / School:", fill='black', font=font_medium)
        draw.text((100, 390), school, fill='black', font=font_large)
        
        draw.text((100, 440), "Tahun / Year:", fill='black', font=font_medium)
        draw.text((200, 440), year, fill='black', font=font_large)
        
        # Save image
        output_path = self.output_dir / f"spk_test_{int(time.time())}.jpg"
        img.save(output_path, 'JPEG')
        
        return output_path
    
    def generate_corrupted_image(self, width: int = 400, height: int = 300) -> Path:
        """Generate a corrupted/low quality image for testing error handling."""
        
        # Create noisy image
        noise = np.random.randint(0, 256, (height, width, 3), dtype=np.uint8)
        img = Image.fromarray(noise)
        
        # Add some text that's barely readable
        draw = ImageDraw.Draw(img)
        try:
            font = ImageFont.truetype("/System/Library/Fonts/Arial.ttf", 12)
        except (OSError, IOError):
            font = ImageFont.load_default()
        
        draw.text((10, 10), "CORRUPTED DOCUMENT", fill='white', font=font)
        
        # Save with low quality
        output_path = self.output_dir / f"corrupted_test_{int(time.time())}.jpg"
        img.save(output_path, 'JPEG', quality=10)
        
        return output_path

class MockServices:
    """Provides mock implementations of external services."""
    
    def __init__(self):
        self.ocr_responses = {}
        self.classification_responses = {}
        self.validation_responses = {}
    
    def mock_ocr_service(self, image_path: str, expected_text: str) -> Mock:
        """Create a mock OCR service that returns expected text."""
        mock_ocr = Mock()
        mock_ocr.extract_text = AsyncMock(return_value={
            "text": expected_text,
            "confidence": 0.95,
            "processing_time": 1.2,
            "engine": "tesseract",
            "language": "eng+msa"
        })
        
        mock_ocr.extract_text_with_coordinates = AsyncMock(return_value={
            "text": expected_text,
            "words": [
                {
                    "text": word,
                    "confidence": 0.9 + (i * 0.01),
                    "bbox": [i*50, 100, (i+1)*50, 120]
                }
                for i, word in enumerate(expected_text.split())
            ],
            "processing_time": 1.5
        })
        
        return mock_ocr
    
    def mock_document_classifier(self, expected_type: DocumentType) -> Mock:
        """Create a mock document classifier."""
        mock_classifier = Mock()
        mock_classifier.classify_document = AsyncMock(return_value={
            "document_type": expected_type,
            "confidence": 0.92,
            "alternatives": [
                {"type": DocumentType.UNKNOWN, "confidence": 0.08}
            ],
            "processing_time": 0.5
        })
        
        return mock_classifier
    
    def mock_field_extractor(self, expected_fields: Dict[str, str]) -> Mock:
        """Create a mock field extractor."""
        mock_extractor = Mock()
        
        extracted_fields = []
        for field_name, value in expected_fields.items():
            extracted_fields.append(ExtractedField(
                field_name=field_name,
                value=value,
                confidence=0.9,
                confidence_level=ConfidenceLevel.HIGH,
                coordinates={"x": 100, "y": 200, "width": 150, "height": 25},
                validation_status="valid",
                raw_text=value
            ))
        
        mock_extractor.extract_fields = AsyncMock(return_value={
            "fields": extracted_fields,
            "processing_time": 2.1,
            "extraction_method": "hybrid"
        })
        
        return mock_extractor
    
    def mock_validator(self, validation_results: Dict[str, bool]) -> Mock:
        """Create a mock document validator."""
        mock_validator = Mock()
        
        mock_validator.validate_document = AsyncMock(return_value={
            "is_valid": all(validation_results.values()),
            "field_validations": validation_results,
            "errors": [f"{field} validation failed" for field, valid in validation_results.items() if not valid],
            "warnings": [],
            "processing_time": 0.3
        })
        
        return mock_validator
    
    @asynccontextmanager
    async def mock_redis_client(self):
        """Create a mock Redis client for testing."""
        mock_redis = Mock()
        
        # Mock Redis operations
        mock_redis.hset = AsyncMock(return_value=True)
        mock_redis.hgetall = AsyncMock(return_value={})
        mock_redis.delete = AsyncMock(return_value=1)
        mock_redis.exists = AsyncMock(return_value=True)
        mock_redis.expire = AsyncMock(return_value=True)
        mock_redis.keys = AsyncMock(return_value=[])
        mock_redis.flushdb = AsyncMock(return_value=True)
        mock_redis.info = AsyncMock(return_value={
            "redis_version": "6.2.0",
            "connected_clients": 1,
            "used_memory_human": "1.5M"
        })
        
        yield mock_redis
    
    @asynccontextmanager
    async def mock_database_session(self):
        """Create a mock database session for testing."""
        # Create in-memory SQLite database for testing
        engine = create_async_engine(
            "sqlite+aiosqlite:///:memory:",
            poolclass=StaticPool,
            connect_args={"check_same_thread": False}
        )
        
        # Create mock session
        mock_session = Mock(spec=AsyncSession)
        mock_session.execute = AsyncMock()
        mock_session.commit = AsyncMock()
        mock_session.rollback = AsyncMock()
        mock_session.close = AsyncMock()
        
        yield mock_session

class PerformanceProfiler:
    """Profiles performance of document processing operations."""
    
    def __init__(self):
        self.measurements = []
        self.thresholds = {
            "ocr_processing": 5.0,
            "classification": 1.0,
            "field_extraction": 2.0,
            "validation": 0.5,
            "total_processing": 10.0
        }
    
    def start_measurement(self, operation_name: str) -> str:
        """Start timing an operation."""
        measurement_id = f"{operation_name}_{int(time.time() * 1000)}"
        
        self.measurements.append({
            "id": measurement_id,
            "operation": operation_name,
            "start_time": time.time(),
            "end_time": None,
            "duration": None,
            "memory_before": self._get_memory_usage(),
            "memory_after": None,
            "memory_delta": None
        })
        
        return measurement_id
    
    def end_measurement(self, measurement_id: str) -> Dict[str, Any]:
        """End timing an operation and return results."""
        measurement = next(
            (m for m in self.measurements if m["id"] == measurement_id),
            None
        )
        
        if not measurement:
            raise ValueError(f"Measurement not found: {measurement_id}")
        
        end_time = time.time()
        memory_after = self._get_memory_usage()
        
        measurement.update({
            "end_time": end_time,
            "duration": end_time - measurement["start_time"],
            "memory_after": memory_after,
            "memory_delta": memory_after - measurement["memory_before"]
        })
        
        return measurement
    
    def check_performance_threshold(self, operation: str, duration: float) -> bool:
        """Check if operation duration is within acceptable threshold."""
        threshold = self.thresholds.get(operation, float('inf'))
        return duration <= threshold
    
    def get_performance_report(self) -> Dict[str, Any]:
        """Generate a performance report."""
        completed_measurements = [m for m in self.measurements if m["duration"] is not None]
        
        if not completed_measurements:
            return {"message": "No completed measurements"}
        
        # Group by operation
        operations = {}
        for measurement in completed_measurements:
            op_name = measurement["operation"]
            if op_name not in operations:
                operations[op_name] = []
            operations[op_name].append(measurement)
        
        # Calculate statistics
        report = {
            "total_measurements": len(completed_measurements),
            "operations": {},
            "overall_stats": {
                "total_duration": sum(m["duration"] for m in completed_measurements),
                "average_duration": sum(m["duration"] for m in completed_measurements) / len(completed_measurements),
                "total_memory_delta": sum(m["memory_delta"] for m in completed_measurements if m["memory_delta"])
            }
        }
        
        for op_name, measurements in operations.items():
            durations = [m["duration"] for m in measurements]
            memory_deltas = [m["memory_delta"] for m in measurements if m["memory_delta"]]
            
            report["operations"][op_name] = {
                "count": len(measurements),
                "total_duration": sum(durations),
                "average_duration": sum(durations) / len(durations),
                "min_duration": min(durations),
                "max_duration": max(durations),
                "threshold": self.thresholds.get(op_name),
                "within_threshold": all(
                    self.check_performance_threshold(op_name, d) for d in durations
                ),
                "average_memory_delta": sum(memory_deltas) / len(memory_deltas) if memory_deltas else 0
            }
        
        return report
    
    def _get_memory_usage(self) -> float:
        """Get current memory usage in MB."""
        try:
            import psutil
            process = psutil.Process()
            return process.memory_info().rss / 1024 / 1024  # Convert to MB
        except ImportError:
            return 0.0
    
    def reset(self):
        """Reset all measurements."""
        self.measurements.clear()

# Test decorators and context managers

def performance_test(operation_name: str, threshold: Optional[float] = None):
    """Decorator to measure performance of test functions."""
    def decorator(func):
        async def async_wrapper(*args, **kwargs):
            profiler = PerformanceProfiler()
            if threshold:
                profiler.thresholds[operation_name] = threshold
            
            measurement_id = profiler.start_measurement(operation_name)
            
            try:
                result = await func(*args, **kwargs)
                measurement = profiler.end_measurement(measurement_id)
                
                # Check threshold
                if not profiler.check_performance_threshold(operation_name, measurement["duration"]):
                    pytest.fail(
                        f"Performance threshold exceeded for {operation_name}: "
                        f"{measurement['duration']:.2f}s > {profiler.thresholds[operation_name]}s"
                    )
                
                return result
                
            except Exception as e:
                profiler.end_measurement(measurement_id)
                raise e
        
        def sync_wrapper(*args, **kwargs):
            profiler = PerformanceProfiler()
            if threshold:
                profiler.thresholds[operation_name] = threshold
            
            measurement_id = profiler.start_measurement(operation_name)
            
            try:
                result = func(*args, **kwargs)
                measurement = profiler.end_measurement(measurement_id)
                
                # Check threshold
                if not profiler.check_performance_threshold(operation_name, measurement["duration"]):
                    pytest.fail(
                        f"Performance threshold exceeded for {operation_name}: "
                        f"{measurement['duration']:.2f}s > {profiler.thresholds[operation_name]}s"
                    )
                
                return result
                
            except Exception as e:
                profiler.end_measurement(measurement_id)
                raise e
        
        return async_wrapper if asyncio.iscoroutinefunction(func) else sync_wrapper
    
    return decorator

@asynccontextmanager
async def temporary_config(config_overrides: Dict[str, Any]):
    """Context manager to temporarily override configuration for testing."""
    original_config = {}
    
    try:
        # Store original values and apply overrides
        config = DocumentParserConfig()
        
        for key, value in config_overrides.items():
            if hasattr(config, key):
                original_config[key] = getattr(config, key)
                setattr(config, key, value)
        
        yield config
        
    finally:
        # Restore original values
        for key, value in original_config.items():
            setattr(config, key, value)

# Export all utilities
__all__ = [
    "TestDataManager",
    "TestDocumentGenerator", 
    "MockServices",
    "PerformanceProfiler",
    "performance_test",
    "temporary_config"
]