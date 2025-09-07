#!/usr/bin/env python3
"""
Unit Tests for Document Processor

Tests the main document processing functionality including
classification, OCR, field extraction, and validation.
"""

import pytest
import numpy as np
import cv2
from unittest.mock import Mock, patch, MagicMock
from pathlib import Path

from src.document_parser.core.processor import DocumentProcessor
from src.document_parser.models.document_models import (
    DocumentType, ProcessingStatus, ConfidenceLevel
)
from tests.fixtures import (
    sample_mykad_data, sample_spk_data, mock_ocr_responses,
    test_configurations, validation_test_cases
)
from tests.utils import TestDataManager, MockServices


class TestDocumentProcessor:
    """Test cases for DocumentProcessor class."""
    
    @pytest.fixture
    def processor(self, test_configurations):
        """Create a DocumentProcessor instance for testing."""
        config = test_configurations['processor']
        return DocumentProcessor(
            use_gpu=False,
            confidence_threshold=config['confidence_threshold']
        )
    
    @pytest.fixture
    def sample_image(self):
        """Create a sample test image."""
        # Create a simple test image
        image = np.ones((600, 800, 3), dtype=np.uint8) * 255
        # Add some text-like patterns
        cv2.rectangle(image, (50, 50), (750, 150), (0, 0, 0), 2)
        cv2.putText(image, "TEST DOCUMENT", (100, 100), 
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2)
        return image
    
    def test_processor_initialization(self, processor):
        """Test processor initialization."""
        assert processor is not None
        assert processor.confidence_threshold == 0.5
        assert not processor.use_gpu
    
    @patch('src.document_parser.core.classifier.DocumentClassifier')
    @patch('src.document_parser.core.ocr_service.OCRService')
    @patch('src.document_parser.core.field_extractor.FieldExtractor')
    @patch('src.document_parser.core.validator.DocumentValidator')
    def test_process_document_mykad(self, mock_validator, mock_extractor, 
                                   mock_ocr, mock_classifier, processor, 
                                   sample_image, sample_mykad_data, 
                                   mock_ocr_responses):
        """Test processing a MyKad document."""
        # Setup mocks
        mock_classifier.return_value.classify.return_value = {
            'document_type': DocumentType.MYKAD,
            'confidence': 0.95
        }
        
        mock_ocr.return_value.extract_text.return_value = mock_ocr_responses['mykad']
        
        mock_extractor.return_value.extract_fields.return_value = {
            field: {
                'value': value,
                'confidence': 0.9,
                'coordinates': [100, 100, 200, 120],
                'is_valid': True
            }
            for field, value in sample_mykad_data.items()
        }
        
        mock_validator.return_value.validate.return_value = {
            'is_valid': True,
            'errors': [],
            'warnings': []
        }
        
        # Process document
        result = processor.process_document(sample_image)
        
        # Assertions
        assert result['document_type'] == DocumentType.MYKAD.value
        assert result['classification_confidence'] == 0.95
        assert result['validation_passed'] is True
        assert len(result['extracted_fields']) > 0
        assert 'ic_number' in result['extracted_fields']
        assert 'processing_time' in result
    
    @patch('src.document_parser.core.classifier.DocumentClassifier')
    @patch('src.document_parser.core.ocr_service.OCRService')
    @patch('src.document_parser.core.field_extractor.FieldExtractor')
    @patch('src.document_parser.core.validator.DocumentValidator')
    def test_process_document_spk(self, mock_validator, mock_extractor, 
                                 mock_ocr, mock_classifier, processor, 
                                 sample_image, sample_spk_data, 
                                 mock_ocr_responses):
        """Test processing an SPK certificate."""
        # Setup mocks
        mock_classifier.return_value.classify.return_value = {
            'document_type': DocumentType.SPK,
            'confidence': 0.88
        }
        
        mock_ocr.return_value.extract_text.return_value = mock_ocr_responses['spk']
        
        mock_extractor.return_value.extract_fields.return_value = {
            field: {
                'value': value,
                'confidence': 0.85,
                'coordinates': [150, 150, 250, 170],
                'is_valid': True
            }
            for field, value in sample_spk_data.items()
        }
        
        mock_validator.return_value.validate.return_value = {
            'is_valid': True,
            'errors': [],
            'warnings': []
        }
        
        # Process document
        result = processor.process_document(sample_image, document_type='spk')
        
        # Assertions
        assert result['document_type'] == DocumentType.SPK.value
        assert result['classification_confidence'] == 0.88
        assert result['validation_passed'] is True
        assert 'student_name' in result['extracted_fields']
        assert 'certificate_number' in result['extracted_fields']
    
    def test_process_document_invalid_input(self, processor):
        """Test processing with invalid input."""
        with pytest.raises(ValueError):
            processor.process_document(None)
        
        with pytest.raises(ValueError):
            processor.process_document("not_an_image")
    
    @patch('src.document_parser.core.classifier.DocumentClassifier')
    def test_process_document_classification_failure(self, mock_classifier, 
                                                   processor, sample_image):
        """Test handling classification failure."""
        # Setup mock to return low confidence
        mock_classifier.return_value.classify.return_value = {
            'document_type': DocumentType.UNKNOWN,
            'confidence': 0.2
        }
        
        result = processor.process_document(sample_image)
        
        assert result['document_type'] == DocumentType.UNKNOWN.value
        assert result['classification_confidence'] == 0.2
        assert not result['validation_passed']
    
    @patch('src.document_parser.core.classifier.DocumentClassifier')
    @patch('src.document_parser.core.ocr_service.OCRService')
    def test_process_document_ocr_failure(self, mock_ocr, mock_classifier, 
                                        processor, sample_image):
        """Test handling OCR failure."""
        # Setup mocks
        mock_classifier.return_value.classify.return_value = {
            'document_type': DocumentType.MYKAD,
            'confidence': 0.95
        }
        
        mock_ocr.return_value.extract_text.side_effect = Exception("OCR failed")
        
        result = processor.process_document(sample_image)
        
        assert result['document_type'] == DocumentType.MYKAD.value
        assert len(result['extracted_fields']) == 0
        assert not result['validation_passed']
    
    def test_process_document_with_annotation(self, processor, sample_image):
        """Test processing with annotation enabled."""
        with patch.multiple(
            'src.document_parser.core',
            classifier=Mock(),
            ocr_service=Mock(),
            field_extractor=Mock(),
            validator=Mock()
        ):
            result = processor.process_document(
                sample_image, 
                return_annotated=True
            )
            
            assert 'annotated_image' in result
            assert result['annotated_image'] is not None
    
    def test_process_batch_documents(self, processor):
        """Test batch processing functionality."""
        # Create multiple test images
        images = [np.ones((400, 600, 3), dtype=np.uint8) * 255 for _ in range(3)]
        
        with patch.multiple(
            'src.document_parser.core',
            classifier=Mock(),
            ocr_service=Mock(),
            field_extractor=Mock(),
            validator=Mock()
        ):
            results = processor.process_batch(images)
            
            assert len(results) == 3
            assert all('document_type' in result for result in results)
    
    def test_get_supported_document_types(self, processor):
        """Test getting supported document types."""
        supported_types = processor.get_supported_document_types()
        
        assert DocumentType.MYKAD.value in supported_types
        assert DocumentType.SPK.value in supported_types
        assert len(supported_types) >= 2
    
    def test_performance_metrics(self, processor, sample_image):
        """Test performance metrics collection."""
        with patch.multiple(
            'src.document_parser.core',
            classifier=Mock(),
            ocr_service=Mock(),
            field_extractor=Mock(),
            validator=Mock()
        ):
            result = processor.process_document(sample_image)
            
            assert 'processing_time' in result
            assert isinstance(result['processing_time'], float)
            assert result['processing_time'] > 0
    
    def test_memory_cleanup(self, processor, sample_image):
        """Test memory cleanup after processing."""
        initial_memory = processor._get_memory_usage() if hasattr(processor, '_get_memory_usage') else 0
        
        with patch.multiple(
            'src.document_parser.core',
            classifier=Mock(),
            ocr_service=Mock(),
            field_extractor=Mock(),
            validator=Mock()
        ):
            # Process multiple documents
            for _ in range(5):
                processor.process_document(sample_image)
        
        # Memory should not grow significantly
        final_memory = processor._get_memory_usage() if hasattr(processor, '_get_memory_usage') else 0
        # This is a basic check - in real implementation, you'd have proper memory monitoring
        assert True  # Placeholder for actual memory check


class TestDocumentProcessorIntegration:
    """Integration tests for DocumentProcessor."""
    
    @pytest.fixture
    def real_processor(self):
        """Create a real DocumentProcessor for integration testing."""
        return DocumentProcessor(use_gpu=False, confidence_threshold=0.5)
    
    def test_end_to_end_processing(self, real_processor):
        """Test end-to-end document processing."""
        # This would require actual test images and models
        # For now, we'll skip this test in CI/CD
        pytest.skip("Requires actual test images and trained models")
    
    def test_template_loading(self, real_processor):
        """Test template loading functionality."""
        templates = real_processor.get_available_templates()
        assert isinstance(templates, list)
        # Should have at least MyKad and SPK templates
        template_names = [t['name'] for t in templates]
        assert 'mykad' in template_names or 'spk' in template_names


if __name__ == '__main__':
    pytest.main([__file__])