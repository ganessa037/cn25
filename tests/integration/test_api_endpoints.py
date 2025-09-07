#!/usr/bin/env python3
"""
API Integration Tests for Document Parser

Comprehensive API integration tests that validate all endpoints with various
inputs, error scenarios, and performance requirements.
"""

import os
import sys
import json
import time
import base64
import tempfile
from pathlib import Path
from typing import Dict, List, Any, Optional
from unittest.mock import Mock, patch, MagicMock

import pytest
import requests
import numpy as np
from PIL import Image
import io

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

try:
    from fastapi.testclient import TestClient
    from fastapi import FastAPI
except ImportError:
    pytest.skip("FastAPI not available for API testing", allow_module_level=True)


class TestAPIEndpoints:
    """Integration tests for Document Parser API endpoints."""
    
    @pytest.fixture(autouse=True)
    def setup_api_client(self):
        """Setup API test client and test data."""
        # Create mock FastAPI app for testing
        self.app = FastAPI(title="Document Parser API")
        
        # Setup test client
        self.client = TestClient(self.app)
        
        # API configuration
        self.api_config = {
            'base_url': 'http://localhost:8000',
            'timeout': 30,
            'max_file_size': 10 * 1024 * 1024,  # 10MB
            'supported_formats': ['jpg', 'jpeg', 'png', 'pdf'],
            'rate_limit': 100  # requests per minute
        }
        
        # Test endpoints
        self.endpoints = {
            'health': '/health',
            'classify': '/api/v1/classify',
            'extract': '/api/v1/extract',
            'validate': '/api/v1/validate',
            'process': '/api/v1/process',
            'batch': '/api/v1/batch',
            'status': '/api/v1/status/{job_id}',
            'metrics': '/api/v1/metrics'
        }
        
        # Performance tracking
        self.response_times = []
        self.error_counts = {'4xx': 0, '5xx': 0}
    
    def create_test_image_file(self, doc_type: str = "mykad", 
                              format: str = "jpg") -> bytes:
        """Create a test image file in specified format."""
        # Create test image
        if doc_type == "mykad":
            img = np.ones((400, 600, 3), dtype=np.uint8) * 255
            # Add some text-like patterns
            img[50:70, 50:200] = [0, 0, 0]  # Name area
            img[100:120, 50:150] = [0, 0, 0]  # IC number area
        else:  # spk
            img = np.ones((800, 600, 3), dtype=np.uint8) * 255
            img[100:120, 100:300] = [0, 0, 0]  # Certificate title
            img[200:220, 100:250] = [0, 0, 0]  # Name area
        
        # Convert to PIL Image
        pil_img = Image.fromarray(img)
        
        # Save to bytes
        img_bytes = io.BytesIO()
        pil_img.save(img_bytes, format=format.upper())
        img_bytes.seek(0)
        
        return img_bytes.getvalue()
    
    def encode_image_base64(self, image_bytes: bytes) -> str:
        """Encode image bytes to base64 string."""
        return base64.b64encode(image_bytes).decode('utf-8')
    
    def make_api_request(self, method: str, endpoint: str, 
                        data: Dict = None, files: Dict = None,
                        headers: Dict = None) -> requests.Response:
        """Make API request and track performance."""
        start_time = time.time()
        
        try:
            if method.upper() == 'GET':
                response = self.client.get(endpoint, headers=headers)
            elif method.upper() == 'POST':
                if files:
                    response = self.client.post(endpoint, data=data, files=files, headers=headers)
                else:
                    response = self.client.post(endpoint, json=data, headers=headers)
            elif method.upper() == 'PUT':
                response = self.client.put(endpoint, json=data, headers=headers)
            elif method.upper() == 'DELETE':
                response = self.client.delete(endpoint, headers=headers)
            else:
                raise ValueError(f"Unsupported HTTP method: {method}")
            
            # Track performance
            response_time = time.time() - start_time
            self.response_times.append(response_time)
            
            # Track errors
            if 400 <= response.status_code < 500:
                self.error_counts['4xx'] += 1
            elif response.status_code >= 500:
                self.error_counts['5xx'] += 1
            
            return response
            
        except Exception as e:
            pytest.fail(f"API request failed: {e}")
    
    @pytest.mark.integration
    @pytest.mark.api
    def test_health_endpoint(self):
        """Test health check endpoint."""
        response = self.make_api_request('GET', self.endpoints['health'])
        
        assert response.status_code == 200
        data = response.json()
        assert 'status' in data
        assert data['status'] == 'healthy'
        assert 'timestamp' in data
        assert 'version' in data
    
    @pytest.mark.integration
    @pytest.mark.api
    def test_classify_endpoint_valid_image(self):
        """Test document classification endpoint with valid image."""
        # Test with MyKad image
        image_bytes = self.create_test_image_file("mykad", "jpg")
        files = {'file': ('test_mykad.jpg', image_bytes, 'image/jpeg')}
        
        response = self.make_api_request('POST', self.endpoints['classify'], files=files)
        
        assert response.status_code == 200
        data = response.json()
        assert 'document_type' in data
        assert 'confidence' in data
        assert data['document_type'] in ['mykad', 'spk', 'unknown']
        assert 0 <= data['confidence'] <= 1
    
    @pytest.mark.integration
    @pytest.mark.api
    def test_classify_endpoint_base64_input(self):
        """Test classification endpoint with base64 encoded image."""
        image_bytes = self.create_test_image_file("spk", "png")
        base64_image = self.encode_image_base64(image_bytes)
        
        data = {
            'image': base64_image,
            'format': 'png'
        }
        
        response = self.make_api_request('POST', self.endpoints['classify'], data=data)
        
        assert response.status_code == 200
        result = response.json()
        assert 'document_type' in result
        assert 'confidence' in result
    
    @pytest.mark.integration
    @pytest.mark.api
    def test_extract_endpoint_mykad(self):
        """Test field extraction endpoint with MyKad document."""
        image_bytes = self.create_test_image_file("mykad", "jpg")
        files = {'file': ('test_mykad.jpg', image_bytes, 'image/jpeg')}
        
        data = {'document_type': 'mykad'}
        
        response = self.make_api_request('POST', self.endpoints['extract'], 
                                       data=data, files=files)
        
        assert response.status_code == 200
        result = response.json()
        assert 'extracted_data' in result
        assert 'confidence' in result
        
        extracted_data = result['extracted_data']
        # Check for expected MyKad fields
        expected_fields = ['name', 'ic_number', 'address', 'gender', 'religion']
        for field in expected_fields:
            assert field in extracted_data
    
    @pytest.mark.integration
    @pytest.mark.api
    def test_extract_endpoint_spk(self):
        """Test field extraction endpoint with SPK document."""
        image_bytes = self.create_test_image_file("spk", "jpg")
        files = {'file': ('test_spk.jpg', image_bytes, 'image/jpeg')}
        
        data = {'document_type': 'spk'}
        
        response = self.make_api_request('POST', self.endpoints['extract'], 
                                       data=data, files=files)
        
        assert response.status_code == 200
        result = response.json()
        assert 'extracted_data' in result
        
        extracted_data = result['extracted_data']
        # Check for expected SPK fields
        expected_fields = ['candidate_name', 'certificate_number', 'subject', 'grade']
        for field in expected_fields:
            assert field in extracted_data
    
    @pytest.mark.integration
    @pytest.mark.api
    def test_validate_endpoint(self):
        """Test data validation endpoint."""
        # Test with valid MyKad data
        valid_data = {
            'document_type': 'mykad',
            'extracted_data': {
                'name': 'AHMAD BIN ALI',
                'ic_number': '123456-78-9012',
                'address': 'NO 123, JALAN ABC, 12345 KUALA LUMPUR',
                'gender': 'LELAKI',
                'religion': 'ISLAM'
            }
        }
        
        response = self.make_api_request('POST', self.endpoints['validate'], data=valid_data)
        
        assert response.status_code == 200
        result = response.json()
        assert 'is_valid' in result
        assert 'validation_errors' in result
        assert 'confidence' in result
    
    @pytest.mark.integration
    @pytest.mark.api
    def test_process_endpoint_complete_pipeline(self):
        """Test complete document processing pipeline endpoint."""
        image_bytes = self.create_test_image_file("mykad", "jpg")
        files = {'file': ('test_mykad.jpg', image_bytes, 'image/jpeg')}
        
        data = {
            'include_classification': True,
            'include_extraction': True,
            'include_validation': True,
            'strict_validation': False
        }
        
        response = self.make_api_request('POST', self.endpoints['process'], 
                                       data=data, files=files)
        
        assert response.status_code == 200
        result = response.json()
        
        # Check all pipeline components are included
        assert 'classification' in result
        assert 'extraction' in result
        assert 'validation' in result
        assert 'processing_time' in result
        
        # Check classification results
        classification = result['classification']
        assert 'document_type' in classification
        assert 'confidence' in classification
        
        # Check extraction results
        extraction = result['extraction']
        assert 'extracted_data' in extraction
        assert 'confidence' in extraction
        
        # Check validation results
        validation = result['validation']
        assert 'is_valid' in validation
        assert 'validation_errors' in validation
    
    @pytest.mark.integration
    @pytest.mark.api
    def test_batch_processing_endpoint(self):
        """Test batch processing endpoint."""
        # Create multiple test images
        images = []
        for i, doc_type in enumerate(['mykad', 'spk', 'mykad']):
            image_bytes = self.create_test_image_file(doc_type, "jpg")
            base64_image = self.encode_image_base64(image_bytes)
            images.append({
                'id': f'doc_{i}',
                'image': base64_image,
                'format': 'jpg'
            })
        
        data = {
            'images': images,
            'processing_options': {
                'include_classification': True,
                'include_extraction': True,
                'include_validation': False
            }
        }
        
        response = self.make_api_request('POST', self.endpoints['batch'], data=data)
        
        assert response.status_code == 202  # Accepted for async processing
        result = response.json()
        assert 'job_id' in result
        assert 'status' in result
        assert result['status'] == 'processing'
    
    @pytest.mark.integration
    @pytest.mark.api
    def test_status_endpoint(self):
        """Test job status endpoint."""
        # First create a batch job
        image_bytes = self.create_test_image_file("mykad", "jpg")
        base64_image = self.encode_image_base64(image_bytes)
        
        batch_data = {
            'images': [{'id': 'test_doc', 'image': base64_image, 'format': 'jpg'}],
            'processing_options': {'include_classification': True}
        }
        
        batch_response = self.make_api_request('POST', self.endpoints['batch'], data=batch_data)
        job_id = batch_response.json()['job_id']
        
        # Check status
        status_endpoint = self.endpoints['status'].format(job_id=job_id)
        response = self.make_api_request('GET', status_endpoint)
        
        assert response.status_code == 200
        result = response.json()
        assert 'job_id' in result
        assert 'status' in result
        assert 'progress' in result
        assert result['status'] in ['processing', 'completed', 'failed']
    
    @pytest.mark.integration
    @pytest.mark.api
    def test_metrics_endpoint(self):
        """Test metrics endpoint."""
        response = self.make_api_request('GET', self.endpoints['metrics'])
        
        assert response.status_code == 200
        result = response.json()
        
        # Check for expected metrics
        expected_metrics = [
            'total_requests',
            'successful_requests',
            'failed_requests',
            'average_response_time',
            'documents_processed',
            'classification_accuracy',
            'extraction_accuracy'
        ]
        
        for metric in expected_metrics:
            assert metric in result
    
    @pytest.mark.integration
    @pytest.mark.api
    def test_invalid_file_format_error(self):
        """Test API response to invalid file formats."""
        # Create invalid file (text file)
        invalid_file = b"This is not an image file"
        files = {'file': ('test.txt', invalid_file, 'text/plain')}
        
        response = self.make_api_request('POST', self.endpoints['classify'], files=files)
        
        assert response.status_code == 400
        result = response.json()
        assert 'error' in result
        assert 'invalid format' in result['error'].lower()
    
    @pytest.mark.integration
    @pytest.mark.api
    def test_file_size_limit_error(self):
        """Test API response to oversized files."""
        # Create oversized file (simulate with large data)
        large_data = b"x" * (self.api_config['max_file_size'] + 1000)
        files = {'file': ('large_file.jpg', large_data, 'image/jpeg')}
        
        response = self.make_api_request('POST', self.endpoints['classify'], files=files)
        
        assert response.status_code == 413  # Payload Too Large
        result = response.json()
        assert 'error' in result
        assert 'file size' in result['error'].lower()
    
    @pytest.mark.integration
    @pytest.mark.api
    def test_missing_required_fields_error(self):
        """Test API response to missing required fields."""
        # Test extract endpoint without document_type
        image_bytes = self.create_test_image_file("mykad", "jpg")
        files = {'file': ('test.jpg', image_bytes, 'image/jpeg')}
        
        response = self.make_api_request('POST', self.endpoints['extract'], files=files)
        
        assert response.status_code == 422  # Unprocessable Entity
        result = response.json()
        assert 'error' in result or 'detail' in result
    
    @pytest.mark.integration
    @pytest.mark.api
    def test_invalid_document_type_error(self):
        """Test API response to invalid document types."""
        image_bytes = self.create_test_image_file("mykad", "jpg")
        files = {'file': ('test.jpg', image_bytes, 'image/jpeg')}
        data = {'document_type': 'invalid_type'}
        
        response = self.make_api_request('POST', self.endpoints['extract'], 
                                       data=data, files=files)
        
        assert response.status_code == 400
        result = response.json()
        assert 'error' in result
        assert 'document_type' in result['error'].lower()
    
    @pytest.mark.integration
    @pytest.mark.api
    def test_api_response_time_performance(self):
        """Test API response time performance."""
        # Make multiple requests to test performance
        for i in range(5):
            image_bytes = self.create_test_image_file("mykad", "jpg")
            files = {'file': (f'test_{i}.jpg', image_bytes, 'image/jpeg')}
            
            start_time = time.time()
            response = self.make_api_request('POST', self.endpoints['classify'], files=files)
            response_time = time.time() - start_time
            
            assert response.status_code == 200
            assert response_time < 5.0  # Should respond within 5 seconds
        
        # Check average response time
        avg_response_time = sum(self.response_times) / len(self.response_times)
        assert avg_response_time < 3.0  # Average should be under 3 seconds
    
    @pytest.mark.integration
    @pytest.mark.api
    def test_concurrent_api_requests(self):
        """Test API behavior under concurrent requests."""
        import threading
        import queue
        
        results_queue = queue.Queue()
        
        def make_concurrent_request(request_id):
            image_bytes = self.create_test_image_file("mykad", "jpg")
            files = {'file': (f'test_{request_id}.jpg', image_bytes, 'image/jpeg')}
            
            try:
                response = self.make_api_request('POST', self.endpoints['classify'], files=files)
                results_queue.put({
                    'request_id': request_id,
                    'status_code': response.status_code,
                    'success': response.status_code == 200
                })
            except Exception as e:
                results_queue.put({
                    'request_id': request_id,
                    'error': str(e),
                    'success': False
                })
        
        # Create multiple concurrent threads
        threads = []
        for i in range(10):
            thread = threading.Thread(target=make_concurrent_request, args=(i,))
            threads.append(thread)
            thread.start()
        
        # Wait for all threads to complete
        for thread in threads:
            thread.join()
        
        # Collect results
        results = []
        while not results_queue.empty():
            results.append(results_queue.get())
        
        assert len(results) == 10
        successful_requests = [r for r in results if r['success']]
        
        # At least 80% should succeed under concurrent load
        success_rate = len(successful_requests) / len(results)
        assert success_rate >= 0.8
    
    @pytest.mark.integration
    @pytest.mark.api
    def test_api_rate_limiting(self):
        """Test API rate limiting behavior."""
        # Make requests rapidly to test rate limiting
        responses = []
        
        for i in range(self.api_config['rate_limit'] + 10):
            try:
                response = self.make_api_request('GET', self.endpoints['health'])
                responses.append(response.status_code)
            except Exception:
                responses.append(429)  # Rate limited
        
        # Should eventually get rate limited (429 status)
        rate_limited_responses = [r for r in responses if r == 429]
        assert len(rate_limited_responses) > 0
    
    @pytest.mark.integration
    @pytest.mark.api
    def test_api_error_response_format(self):
        """Test that API error responses follow consistent format."""
        # Test various error scenarios
        error_scenarios = [
            ('POST', self.endpoints['classify'], {'invalid': 'data'}, None, 400),
            ('GET', '/nonexistent/endpoint', None, None, 404),
            ('POST', self.endpoints['extract'], None, None, 422)
        ]
        
        for method, endpoint, data, files, expected_status in error_scenarios:
            response = self.make_api_request(method, endpoint, data=data, files=files)
            
            assert response.status_code == expected_status
            result = response.json()
            
            # Check error response format
            assert 'error' in result or 'detail' in result
            if 'error' in result:
                assert isinstance(result['error'], str)
            assert 'timestamp' in result or 'detail' in result
    
    def teardown_method(self):
        """Cleanup after each test method."""
        # Log performance summary
        if self.response_times:
            avg_time = sum(self.response_times) / len(self.response_times)
            max_time = max(self.response_times)
            print(f"\nAPI Performance Summary:")
            print(f"  Average response time: {avg_time:.3f}s")
            print(f"  Maximum response time: {max_time:.3f}s")
            print(f"  4xx errors: {self.error_counts['4xx']}")
            print(f"  5xx errors: {self.error_counts['5xx']}")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])