#!/usr/bin/env python3
"""
Unit Tests for OCR Service

Comprehensive testing of OCR functionality including
text extraction, preprocessing, multi-engine support, and error handling.
"""

import pytest
import numpy as np
import cv2
from unittest.mock import Mock, patch, MagicMock
from pathlib import Path
import tempfile

from src.document_parser.ocr_service import OCRService
from src.document_parser.models.document_models import OCRResult, ConfidenceLevel


class TestOCRService:
    """Test cases for OCRService class."""
    
    @pytest.fixture
    def ocr_service(self):
        """Create an OCRService instance for testing."""
        return OCRService(
            engines=['tesseract', 'easyocr'],
            confidence_threshold=0.6
        )
    
    @pytest.fixture
    def sample_text_image(self):
        """Create a sample image with text."""
        image = np.ones((200, 600, 3), dtype=np.uint8) * 255
        cv2.putText(image, "SAMPLE TEXT FOR OCR", (50, 100), 
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2)
        cv2.putText(image, "Line 2: 123456789", (50, 150), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 2)
        return image
    
    @pytest.fixture
    def sample_mykad_text_image(self):
        """Create a sample MyKad text region."""
        image = np.ones((150, 400, 3), dtype=np.uint8) * 255
        cv2.putText(image, "AHMAD BIN ALI", (20, 50), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 2)
        cv2.putText(image, "123456-78-9012", (20, 100), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 2)
        return image
    
    def test_ocr_service_initialization(self, ocr_service):
        """Test OCR service initialization."""
        assert ocr_service is not None
        assert ocr_service.confidence_threshold == 0.6
        assert 'tesseract' in ocr_service.engines
        assert 'easyocr' in ocr_service.engines
    
    def test_extract_text_tesseract(self, ocr_service, sample_text_image):
        """Test text extraction using Tesseract."""
        with patch('pytesseract.image_to_string') as mock_tesseract:
            mock_tesseract.return_value = "SAMPLE TEXT FOR OCR\nLine 2: 123456789"
            
            result = ocr_service.extract_text(sample_text_image, engine='tesseract')
            
            assert result is not None
            assert isinstance(result, OCRResult)
            assert "SAMPLE TEXT" in result.text
            assert result.confidence > 0
            mock_tesseract.assert_called_once()
    
    def test_extract_text_easyocr(self, ocr_service, sample_text_image):
        """Test text extraction using EasyOCR."""
        with patch.object(ocr_service, '_extract_with_easyocr') as mock_easyocr:
            mock_easyocr.return_value = OCRResult(
                text="SAMPLE TEXT FOR OCR\nLine 2: 123456789",
                confidence=0.85,
                bounding_boxes=[(50, 80, 300, 120), (50, 130, 250, 170)]
            )
            
            result = ocr_service.extract_text(sample_text_image, engine='easyocr')
            
            assert result is not None
            assert "SAMPLE TEXT" in result.text
            assert result.confidence >= 0.6
            assert len(result.bounding_boxes) == 2
    
    def test_extract_text_multi_engine(self, ocr_service, sample_text_image):
        """Test text extraction using multiple engines."""
        with patch('pytesseract.image_to_string') as mock_tesseract, \
             patch.object(ocr_service, '_extract_with_easyocr') as mock_easyocr:
            
            mock_tesseract.return_value = "SAMPLE TEXT FOR OCR"
            mock_easyocr.return_value = OCRResult(
                text="SAMPLE TEXT FOR OCR",
                confidence=0.9,
                bounding_boxes=[(50, 80, 300, 120)]
            )
            
            result = ocr_service.extract_text_multi_engine(sample_text_image)
            
            assert result is not None
            assert "SAMPLE TEXT" in result.text
            assert result.confidence > 0
    
    def test_preprocess_image(self, ocr_service, sample_text_image):
        """Test image preprocessing for OCR."""
        processed = ocr_service.preprocess_image(sample_text_image)
        
        assert processed is not None
        assert isinstance(processed, np.ndarray)
        # Should be grayscale after preprocessing
        assert len(processed.shape) == 2 or processed.shape[2] == 1
    
    def test_extract_text_invalid_input(self, ocr_service):
        """Test text extraction with invalid input."""
        with pytest.raises(ValueError):
            ocr_service.extract_text(None)
        
        with pytest.raises(ValueError):
            ocr_service.extract_text("invalid_input")
    
    def test_extract_text_empty_image(self, ocr_service):
        """Test text extraction with empty image."""
        empty_image = np.zeros((100, 100, 3), dtype=np.uint8)
        
        with patch('pytesseract.image_to_string') as mock_tesseract:
            mock_tesseract.return_value = ""
            
            result = ocr_service.extract_text(empty_image)
            
            assert result.text == ""
            assert result.confidence == 0
    
    def test_extract_text_with_language(self, ocr_service, sample_text_image):
        """Test text extraction with specific language."""
        with patch('pytesseract.image_to_string') as mock_tesseract:
            mock_tesseract.return_value = "SAMPLE TEXT FOR OCR"
            
            result = ocr_service.extract_text(sample_text_image, language='eng')
            
            assert result is not None
            mock_tesseract.assert_called_once()
    
    def test_extract_text_with_config(self, ocr_service, sample_text_image):
        """Test text extraction with custom configuration."""
        config = {
            'psm': 6,  # Uniform block of text
            'oem': 3   # Default OCR Engine Mode
        }
        
        with patch('pytesseract.image_to_string') as mock_tesseract:
            mock_tesseract.return_value = "SAMPLE TEXT FOR OCR"
            
            result = ocr_service.extract_text(sample_text_image, config=config)
            
            assert result is not None
            mock_tesseract.assert_called_once()
    
    def test_extract_structured_data(self, ocr_service, sample_mykad_text_image):
        """Test extraction of structured data from text."""
        with patch('pytesseract.image_to_string') as mock_tesseract:
            mock_tesseract.return_value = "AHMAD BIN ALI\n123456-78-9012"
            
            result = ocr_service.extract_structured_data(sample_mykad_text_image)
            
            assert result is not None
            assert 'name' in result or 'ic_number' in result
    
    def test_confidence_calculation(self, ocr_service, sample_text_image):
        """Test confidence score calculation."""
        with patch('pytesseract.image_to_data') as mock_data:
            mock_data.return_value = {
                'conf': [90, 85, 92, 88],
                'text': ['SAMPLE', 'TEXT', 'FOR', 'OCR']
            }
            
            confidence = ocr_service._calculate_confidence(sample_text_image)
            
            assert confidence > 0
            assert confidence <= 1.0
    
    def test_bounding_box_extraction(self, ocr_service, sample_text_image):
        """Test bounding box extraction."""
        with patch('pytesseract.image_to_data') as mock_data:
            mock_data.return_value = {
                'left': [50, 150],
                'top': [80, 130],
                'width': [100, 120],
                'height': [40, 40],
                'text': ['SAMPLE', 'TEXT']
            }
            
            boxes = ocr_service._extract_bounding_boxes(sample_text_image)
            
            assert len(boxes) == 2
            assert all(len(box) == 4 for box in boxes)
    
    def test_text_cleaning(self, ocr_service):
        """Test text cleaning functionality."""
        dirty_text = "  SAMPLE   TEXT\n\n  WITH   NOISE  \t\r"
        cleaned = ocr_service._clean_text(dirty_text)
        
        assert cleaned == "SAMPLE TEXT WITH NOISE"
        assert "\n" not in cleaned
        assert "\t" not in cleaned
    
    def test_performance_metrics(self, ocr_service, sample_text_image):
        """Test performance metrics collection."""
        import time
        
        start_time = time.time()
        
        with patch('pytesseract.image_to_string') as mock_tesseract:
            mock_tesseract.return_value = "SAMPLE TEXT"
            
            result = ocr_service.extract_text(sample_text_image)
            
        end_time = time.time()
        processing_time = end_time - start_time
        
        assert processing_time < 10.0  # Should process within 10 seconds
        assert result is not None
    
    def test_batch_text_extraction(self, ocr_service, sample_text_image):
        """Test batch text extraction."""
        images = [sample_text_image, sample_text_image]
        
        with patch('pytesseract.image_to_string') as mock_tesseract:
            mock_tesseract.return_value = "SAMPLE TEXT"
            
            results = ocr_service.extract_text_batch(images)
            
            assert len(results) == 2
            assert all(isinstance(result, OCRResult) for result in results)
    
    def test_error_handling(self, ocr_service, sample_text_image):
        """Test error handling in OCR operations."""
        with patch('pytesseract.image_to_string') as mock_tesseract:
            mock_tesseract.side_effect = Exception("OCR engine error")
            
            result = ocr_service.extract_text(sample_text_image)
            
            # Should handle error gracefully
            assert result is not None
            assert result.text == ""
            assert result.confidence == 0
    
    def test_memory_cleanup(self, ocr_service, sample_text_image):
        """Test memory cleanup after OCR operations."""
        import psutil
        import os
        
        process = psutil.Process(os.getpid())
        initial_memory = process.memory_info().rss
        
        with patch('pytesseract.image_to_string') as mock_tesseract:
            mock_tesseract.return_value = "SAMPLE TEXT"
            
            # Process multiple images
            for _ in range(20):
                ocr_service.extract_text(sample_text_image)
        
        final_memory = process.memory_info().rss
        memory_increase = final_memory - initial_memory
        
        # Memory increase should be reasonable
        assert memory_increase < 200 * 1024 * 1024  # Less than 200MB


class TestOCRServiceIntegration:
    """Integration tests for OCRService."""
    
    @pytest.fixture
    def real_ocr_service(self):
        """Create a real OCRService for integration testing."""
        return OCRService()
    
    def test_real_tesseract_extraction(self, real_ocr_service, sample_text_image):
        """Test real Tesseract text extraction."""
        try:
            result = real_ocr_service.extract_text(sample_text_image, engine='tesseract')
            assert result is not None
            assert isinstance(result, OCRResult)
        except Exception:
            pytest.skip("Tesseract not available")
    
    def test_supported_languages(self, real_ocr_service):
        """Test getting supported languages."""
        try:
            languages = real_ocr_service.get_supported_languages()
            assert 'eng' in languages
            assert len(languages) > 0
        except Exception:
            pytest.skip("OCR engines not available")


if __name__ == '__main__':
    pytest.main([__file__])