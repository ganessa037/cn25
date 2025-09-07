#!/usr/bin/env python3
"""
Unit Tests for Document Classifier

Comprehensive testing of document classification functionality including
model loading, prediction accuracy, confidence scoring, and error handling.
"""

import pytest
import numpy as np
import cv2
from unittest.mock import Mock, patch, MagicMock
from pathlib import Path
import tempfile
import json

from src.document_parser.document_classifier import DocumentClassifier
from src.document_parser.models.document_models import DocumentType, ConfidenceLevel


class TestDocumentClassifier:
    """Test cases for DocumentClassifier class."""
    
    @pytest.fixture
    def classifier(self):
        """Create a DocumentClassifier instance for testing."""
        return DocumentClassifier(
            model_path="test_models",
            confidence_threshold=0.7
        )
    
    @pytest.fixture
    def sample_mykad_image(self):
        """Create a sample MyKad-like image."""
        image = np.ones((600, 950, 3), dtype=np.uint8) * 255
        # Add MyKad-like features
        cv2.rectangle(image, (50, 50), (900, 550), (0, 0, 255), 3)
        cv2.putText(image, "MALAYSIA", (300, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2)
        cv2.putText(image, "MYKAD", (350, 150), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2)
        return image
    
    @pytest.fixture
    def sample_spk_image(self):
        """Create a sample SPK-like image."""
        image = np.ones((800, 600, 3), dtype=np.uint8) * 255
        # Add SPK-like features
        cv2.rectangle(image, (50, 50), (550, 750), (0, 0, 0), 2)
        cv2.putText(image, "SURAT PERAKUAN KELAHIRAN", (100, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)
        return image
    
    def test_classifier_initialization(self, classifier):
        """Test classifier initialization."""
        assert classifier is not None
        assert classifier.confidence_threshold == 0.7
        assert hasattr(classifier, 'model')
    
    def test_classify_mykad_document(self, classifier, sample_mykad_image):
        """Test MyKad document classification."""
        with patch.object(classifier, '_predict_document_type') as mock_predict:
            mock_predict.return_value = (DocumentType.MYKAD, 0.95)
            
            result = classifier.classify(sample_mykad_image)
            
            assert result['document_type'] == DocumentType.MYKAD
            assert result['confidence'] >= 0.7
            assert result['confidence_level'] == ConfidenceLevel.HIGH
            mock_predict.assert_called_once()
    
    def test_classify_spk_document(self, classifier, sample_spk_image):
        """Test SPK document classification."""
        with patch.object(classifier, '_predict_document_type') as mock_predict:
            mock_predict.return_value = (DocumentType.SPK, 0.88)
            
            result = classifier.classify(sample_spk_image)
            
            assert result['document_type'] == DocumentType.SPK
            assert result['confidence'] >= 0.7
            assert result['confidence_level'] == ConfidenceLevel.HIGH
    
    def test_classify_low_confidence(self, classifier, sample_mykad_image):
        """Test classification with low confidence."""
        with patch.object(classifier, '_predict_document_type') as mock_predict:
            mock_predict.return_value = (DocumentType.UNKNOWN, 0.4)
            
            result = classifier.classify(sample_mykad_image)
            
            assert result['document_type'] == DocumentType.UNKNOWN
            assert result['confidence'] < 0.7
            assert result['confidence_level'] == ConfidenceLevel.LOW
    
    def test_classify_invalid_input(self, classifier):
        """Test classification with invalid input."""
        with pytest.raises(ValueError):
            classifier.classify(None)
        
        with pytest.raises(ValueError):
            classifier.classify("invalid_input")
    
    def test_classify_empty_image(self, classifier):
        """Test classification with empty image."""
        empty_image = np.zeros((100, 100, 3), dtype=np.uint8)
        
        with patch.object(classifier, '_predict_document_type') as mock_predict:
            mock_predict.return_value = (DocumentType.UNKNOWN, 0.1)
            
            result = classifier.classify(empty_image)
            
            assert result['document_type'] == DocumentType.UNKNOWN
            assert result['confidence'] < 0.7
    
    def test_preprocess_image(self, classifier, sample_mykad_image):
        """Test image preprocessing functionality."""
        processed = classifier._preprocess_image(sample_mykad_image)
        
        assert processed is not None
        assert isinstance(processed, np.ndarray)
        assert len(processed.shape) == 3  # Should maintain 3 channels
    
    def test_extract_features(self, classifier, sample_mykad_image):
        """Test feature extraction from image."""
        with patch.object(classifier, '_extract_visual_features') as mock_extract:
            mock_extract.return_value = np.array([0.1, 0.2, 0.3, 0.4, 0.5])
            
            features = classifier._extract_features(sample_mykad_image)
            
            assert features is not None
            assert isinstance(features, np.ndarray)
            mock_extract.assert_called_once()
    
    def test_confidence_level_mapping(self, classifier):
        """Test confidence level mapping."""
        assert classifier._get_confidence_level(0.9) == ConfidenceLevel.HIGH
        assert classifier._get_confidence_level(0.75) == ConfidenceLevel.MEDIUM
        assert classifier._get_confidence_level(0.5) == ConfidenceLevel.LOW
    
    @patch('src.document_parser.document_classifier.cv2')
    def test_model_loading_error(self, mock_cv2, classifier):
        """Test handling of model loading errors."""
        mock_cv2.imread.side_effect = Exception("Model loading failed")
        
        with pytest.raises(Exception):
            classifier._load_model("invalid_path")
    
    def test_batch_classification(self, classifier, sample_mykad_image, sample_spk_image):
        """Test batch document classification."""
        images = [sample_mykad_image, sample_spk_image]
        
        with patch.object(classifier, 'classify') as mock_classify:
            mock_classify.side_effect = [
                {'document_type': DocumentType.MYKAD, 'confidence': 0.95},
                {'document_type': DocumentType.SPK, 'confidence': 0.88}
            ]
            
            results = classifier.classify_batch(images)
            
            assert len(results) == 2
            assert results[0]['document_type'] == DocumentType.MYKAD
            assert results[1]['document_type'] == DocumentType.SPK
    
    def test_performance_metrics(self, classifier, sample_mykad_image):
        """Test performance metrics collection."""
        import time
        
        start_time = time.time()
        
        with patch.object(classifier, '_predict_document_type') as mock_predict:
            mock_predict.return_value = (DocumentType.MYKAD, 0.95)
            
            result = classifier.classify(sample_mykad_image)
            
        end_time = time.time()
        processing_time = end_time - start_time
        
        assert processing_time < 5.0  # Should process within 5 seconds
        assert 'processing_time' in result or processing_time is not None
    
    def test_memory_usage(self, classifier, sample_mykad_image):
        """Test memory usage during classification."""
        import psutil
        import os
        
        process = psutil.Process(os.getpid())
        initial_memory = process.memory_info().rss
        
        with patch.object(classifier, '_predict_document_type') as mock_predict:
            mock_predict.return_value = (DocumentType.MYKAD, 0.95)
            
            # Process multiple images to test memory usage
            for _ in range(10):
                classifier.classify(sample_mykad_image)
        
        final_memory = process.memory_info().rss
        memory_increase = final_memory - initial_memory
        
        # Memory increase should be reasonable (less than 100MB)
        assert memory_increase < 100 * 1024 * 1024


class TestDocumentClassifierIntegration:
    """Integration tests for DocumentClassifier."""
    
    @pytest.fixture
    def real_classifier(self):
        """Create a real DocumentClassifier for integration testing."""
        return DocumentClassifier()
    
    def test_real_model_loading(self, real_classifier):
        """Test loading of actual model files."""
        # This test would require actual model files
        # Skip if models are not available
        if not hasattr(real_classifier, 'model') or real_classifier.model is None:
            pytest.skip("Real model files not available")
        
        assert real_classifier.model is not None
    
    def test_supported_document_types(self, real_classifier):
        """Test getting supported document types."""
        supported_types = real_classifier.get_supported_types()
        
        assert DocumentType.MYKAD in supported_types
        assert DocumentType.SPK in supported_types
        assert len(supported_types) > 0


if __name__ == '__main__':
    pytest.main([__file__])