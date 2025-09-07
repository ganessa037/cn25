#!/usr/bin/env python3
"""
End-to-End Integration Tests for Document Parser Pipeline

Comprehensive integration tests that validate the complete document processing
pipeline from image input to structured data output.
"""

import os
import sys
import json
import time
import tempfile
from pathlib import Path
from typing import Dict, List, Any, Optional
from unittest.mock import Mock, patch, MagicMock

import pytest
import numpy as np
from PIL import Image

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

try:
    from src.document_parser import (
        DocumentClassifier,
        OCRService,
        FieldExtractor,
        DocumentValidator,
        ImagePreprocessor
    )
    from src.document_parser.schema import ExtractedData
except ImportError as e:
    pytest.skip(f"Document parser modules not available: {e}", allow_module_level=True)


class TestEndToEndPipeline:
    """End-to-end integration tests for the complete document processing pipeline."""
    
    @pytest.fixture(autouse=True)
    def setup_pipeline(self):
        """Setup the complete document processing pipeline."""
        # Initialize all components
        self.classifier = DocumentClassifier()
        self.ocr_service = OCRService()
        self.field_extractor = FieldExtractor()
        self.validator = DocumentValidator()
        self.preprocessor = ImagePreprocessor()
        
        # Pipeline configuration
        self.pipeline_config = {
            'classification_threshold': 0.8,
            'ocr_confidence_threshold': 0.7,
            'extraction_confidence_threshold': 0.6,
            'validation_strict_mode': True,
            'preprocessing_enabled': True
        }
        
        # Test data directory
        self.test_data_dir = Path(__file__).parent / "test_data"
        self.test_data_dir.mkdir(exist_ok=True)
        
        # Performance tracking
        self.performance_metrics = {
            'classification_time': [],
            'ocr_time': [],
            'extraction_time': [],
            'validation_time': [],
            'total_time': []
        }
    
    def create_test_image(self, document_type: str = "mykad", 
                         quality: str = "high") -> np.ndarray:
        """Create a test document image."""
        if document_type == "mykad":
            # Create MyKad-like image
            img = np.ones((400, 600, 3), dtype=np.uint8) * 255
            
            # Add some text-like patterns
            if quality == "high":
                # Clear, high-quality text simulation
                img[50:70, 50:200] = [0, 0, 0]  # Name area
                img[100:120, 50:150] = [0, 0, 0]  # IC number area
                img[150:170, 50:180] = [0, 0, 0]  # Address area
            elif quality == "medium":
                # Slightly blurred text
                img[50:70, 50:200] = [50, 50, 50]
                img[100:120, 50:150] = [50, 50, 50]
                img[150:170, 50:180] = [50, 50, 50]
            else:  # low quality
                # Very blurred/noisy text
                img[50:70, 50:200] = [100, 100, 100]
                img[100:120, 50:150] = [100, 100, 100]
                img[150:170, 50:180] = [100, 100, 100]
                # Add noise
                noise = np.random.randint(0, 50, img.shape, dtype=np.uint8)
                img = np.clip(img.astype(int) + noise, 0, 255).astype(np.uint8)
        
        elif document_type == "spk":
            # Create SPK-like image
            img = np.ones((800, 600, 3), dtype=np.uint8) * 255
            
            # Add certificate-like patterns
            if quality == "high":
                img[100:120, 100:300] = [0, 0, 0]  # Certificate title
                img[200:220, 100:250] = [0, 0, 0]  # Name area
                img[300:320, 100:200] = [0, 0, 0]  # Grade area
            elif quality == "medium":
                img[100:120, 100:300] = [50, 50, 50]
                img[200:220, 100:250] = [50, 50, 50]
                img[300:320, 100:200] = [50, 50, 50]
            else:  # low quality
                img[100:120, 100:300] = [100, 100, 100]
                img[200:220, 100:250] = [100, 100, 100]
                img[300:320, 100:200] = [100, 100, 100]
                noise = np.random.randint(0, 50, img.shape, dtype=np.uint8)
                img = np.clip(img.astype(int) + noise, 0, 255).astype(np.uint8)
        
        return img
    
    def process_document_pipeline(self, image: np.ndarray, 
                                expected_type: str = None) -> Dict[str, Any]:
        """Process a document through the complete pipeline."""
        start_time = time.time()
        results = {
            'success': False,
            'document_type': None,
            'extracted_data': None,
            'validation_results': None,
            'performance': {},
            'errors': []
        }
        
        try:
            # Step 1: Image Preprocessing
            if self.pipeline_config['preprocessing_enabled']:
                preprocessed_image = self.preprocessor.preprocess(image)
            else:
                preprocessed_image = image
            
            # Step 2: Document Classification
            classification_start = time.time()
            classification_result = self.classifier.classify(preprocessed_image)
            classification_time = time.time() - classification_start
            
            if classification_result['confidence'] < self.pipeline_config['classification_threshold']:
                results['errors'].append("Classification confidence too low")
                return results
            
            document_type = classification_result['document_type']
            results['document_type'] = document_type
            
            # Validate expected type if provided
            if expected_type and document_type != expected_type:
                results['errors'].append(f"Expected {expected_type}, got {document_type}")
            
            # Step 3: OCR Processing
            ocr_start = time.time()
            ocr_result = self.ocr_service.extract_text(preprocessed_image)
            ocr_time = time.time() - ocr_start
            
            if ocr_result['confidence'] < self.pipeline_config['ocr_confidence_threshold']:
                results['errors'].append("OCR confidence too low")
                return results
            
            # Step 4: Field Extraction
            extraction_start = time.time()
            extracted_data = self.field_extractor.extract_fields(
                ocr_result['text'], 
                document_type
            )
            extraction_time = time.time() - extraction_start
            
            if extracted_data.confidence < self.pipeline_config['extraction_confidence_threshold']:
                results['errors'].append("Field extraction confidence too low")
                return results
            
            results['extracted_data'] = extracted_data
            
            # Step 5: Validation
            validation_start = time.time()
            validation_results = self.validator.validate(
                extracted_data, 
                strict_mode=self.pipeline_config['validation_strict_mode']
            )
            validation_time = time.time() - validation_start
            
            results['validation_results'] = validation_results
            
            # Calculate performance metrics
            total_time = time.time() - start_time
            results['performance'] = {
                'classification_time': classification_time,
                'ocr_time': ocr_time,
                'extraction_time': extraction_time,
                'validation_time': validation_time,
                'total_time': total_time
            }
            
            # Update performance tracking
            self.performance_metrics['classification_time'].append(classification_time)
            self.performance_metrics['ocr_time'].append(ocr_time)
            self.performance_metrics['extraction_time'].append(extraction_time)
            self.performance_metrics['validation_time'].append(validation_time)
            self.performance_metrics['total_time'].append(total_time)
            
            # Mark as successful if no errors and validation passed
            if not results['errors'] and validation_results['is_valid']:
                results['success'] = True
            
        except Exception as e:
            results['errors'].append(f"Pipeline error: {str(e)}")
        
        return results
    
    @pytest.mark.integration
    def test_mykad_high_quality_pipeline(self):
        """Test complete pipeline with high-quality MyKad image."""
        # Create high-quality MyKad image
        image = self.create_test_image("mykad", "high")
        
        # Process through pipeline
        results = self.process_document_pipeline(image, "mykad")
        
        # Assertions
        assert results['success'], f"Pipeline failed: {results['errors']}"
        assert results['document_type'] == "mykad"
        assert results['extracted_data'] is not None
        assert results['validation_results']['is_valid']
        assert results['performance']['total_time'] < 10.0  # Should complete within 10 seconds
    
    @pytest.mark.integration
    def test_spk_high_quality_pipeline(self):
        """Test complete pipeline with high-quality SPK image."""
        # Create high-quality SPK image
        image = self.create_test_image("spk", "high")
        
        # Process through pipeline
        results = self.process_document_pipeline(image, "spk")
        
        # Assertions
        assert results['success'], f"Pipeline failed: {results['errors']}"
        assert results['document_type'] == "spk"
        assert results['extracted_data'] is not None
        assert results['validation_results']['is_valid']
        assert results['performance']['total_time'] < 10.0
    
    @pytest.mark.integration
    def test_medium_quality_document_pipeline(self):
        """Test pipeline with medium-quality document images."""
        for doc_type in ["mykad", "spk"]:
            image = self.create_test_image(doc_type, "medium")
            results = self.process_document_pipeline(image, doc_type)
            
            # Should still work but may have lower confidence
            assert results['document_type'] == doc_type
            assert results['extracted_data'] is not None
            # May not pass strict validation due to quality
    
    @pytest.mark.integration
    def test_low_quality_document_handling(self):
        """Test pipeline behavior with low-quality documents."""
        for doc_type in ["mykad", "spk"]:
            image = self.create_test_image(doc_type, "low")
            results = self.process_document_pipeline(image, doc_type)
            
            # Should handle gracefully, may fail validation
            assert len(results['errors']) > 0 or not results['validation_results']['is_valid']
    
    @pytest.mark.integration
    def test_batch_processing_pipeline(self):
        """Test pipeline with batch processing of multiple documents."""
        documents = [
            ("mykad", "high"),
            ("spk", "high"),
            ("mykad", "medium"),
            ("spk", "medium")
        ]
        
        results = []
        for doc_type, quality in documents:
            image = self.create_test_image(doc_type, quality)
            result = self.process_document_pipeline(image, doc_type)
            results.append(result)
        
        # Check batch results
        successful_results = [r for r in results if r['success']]
        assert len(successful_results) >= 2  # At least high-quality ones should succeed
        
        # Check performance consistency
        total_times = [r['performance']['total_time'] for r in results]
        avg_time = sum(total_times) / len(total_times)
        assert avg_time < 8.0  # Average processing time should be reasonable
    
    @pytest.mark.integration
    def test_pipeline_error_handling(self):
        """Test pipeline error handling and recovery."""
        # Test with invalid image
        invalid_image = np.zeros((10, 10, 3), dtype=np.uint8)
        results = self.process_document_pipeline(invalid_image)
        
        assert not results['success']
        assert len(results['errors']) > 0
    
    @pytest.mark.integration
    def test_pipeline_configuration_changes(self):
        """Test pipeline behavior with different configurations."""
        image = self.create_test_image("mykad", "medium")
        
        # Test with strict configuration
        self.pipeline_config['classification_threshold'] = 0.9
        self.pipeline_config['ocr_confidence_threshold'] = 0.9
        self.pipeline_config['validation_strict_mode'] = True
        
        strict_results = self.process_document_pipeline(image, "mykad")
        
        # Test with lenient configuration
        self.pipeline_config['classification_threshold'] = 0.5
        self.pipeline_config['ocr_confidence_threshold'] = 0.5
        self.pipeline_config['validation_strict_mode'] = False
        
        lenient_results = self.process_document_pipeline(image, "mykad")
        
        # Lenient should be more likely to succeed
        if not strict_results['success']:
            assert lenient_results['success'] or len(lenient_results['errors']) < len(strict_results['errors'])
    
    @pytest.mark.integration
    def test_pipeline_memory_usage(self):
        """Test pipeline memory usage and cleanup."""
        import psutil
        import gc
        
        process = psutil.Process(os.getpid())
        initial_memory = process.memory_info().rss
        
        # Process multiple documents
        for i in range(10):
            image = self.create_test_image("mykad", "high")
            results = self.process_document_pipeline(image)
            
            # Force garbage collection
            del image, results
            gc.collect()
        
        final_memory = process.memory_info().rss
        memory_increase = final_memory - initial_memory
        
        # Memory increase should be reasonable (less than 100MB)
        assert memory_increase < 100 * 1024 * 1024
    
    @pytest.mark.integration
    def test_pipeline_performance_benchmarks(self):
        """Test pipeline performance benchmarks."""
        # Process several documents to get performance metrics
        for i in range(5):
            for doc_type in ["mykad", "spk"]:
                image = self.create_test_image(doc_type, "high")
                self.process_document_pipeline(image, doc_type)
        
        # Calculate performance statistics
        avg_classification_time = sum(self.performance_metrics['classification_time']) / len(self.performance_metrics['classification_time'])
        avg_ocr_time = sum(self.performance_metrics['ocr_time']) / len(self.performance_metrics['ocr_time'])
        avg_extraction_time = sum(self.performance_metrics['extraction_time']) / len(self.performance_metrics['extraction_time'])
        avg_validation_time = sum(self.performance_metrics['validation_time']) / len(self.performance_metrics['validation_time'])
        avg_total_time = sum(self.performance_metrics['total_time']) / len(self.performance_metrics['total_time'])
        
        # Performance assertions
        assert avg_classification_time < 2.0  # Classification should be fast
        assert avg_ocr_time < 5.0  # OCR is typically the slowest
        assert avg_extraction_time < 1.0  # Field extraction should be fast
        assert avg_validation_time < 0.5  # Validation should be very fast
        assert avg_total_time < 8.0  # Total pipeline should complete quickly
    
    @pytest.mark.integration
    def test_pipeline_concurrent_processing(self):
        """Test pipeline behavior under concurrent processing."""
        import threading
        import queue
        
        results_queue = queue.Queue()
        
        def process_document(doc_type, quality):
            image = self.create_test_image(doc_type, quality)
            result = self.process_document_pipeline(image, doc_type)
            results_queue.put(result)
        
        # Create multiple threads
        threads = []
        for i in range(4):
            doc_type = "mykad" if i % 2 == 0 else "spk"
            thread = threading.Thread(target=process_document, args=(doc_type, "high"))
            threads.append(thread)
            thread.start()
        
        # Wait for all threads to complete
        for thread in threads:
            thread.join()
        
        # Collect results
        results = []
        while not results_queue.empty():
            results.append(results_queue.get())
        
        assert len(results) == 4
        # At least some should succeed
        successful_results = [r for r in results if r['success']]
        assert len(successful_results) >= 2
    
    def teardown_method(self):
        """Cleanup after each test method."""
        # Clean up any temporary files
        import shutil
        temp_dirs = [d for d in Path("/tmp").glob("test_document_*") if d.is_dir()]
        for temp_dir in temp_dirs:
            try:
                shutil.rmtree(temp_dir)
            except:
                pass


if __name__ == "__main__":
    pytest.main([__file__, "-v"])