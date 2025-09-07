#!/usr/bin/env python3
"""
Unit Tests for API Routes

Tests the FastAPI endpoints for document processing,
health checks, and administrative functions.
"""

import pytest
import json
import io
from unittest.mock import Mock, patch, AsyncMock
from fastapi.testclient import TestClient
from fastapi import status
import numpy as np
from PIL import Image

from src.document_parser.api.main import app
from src.document_parser.models.document_models import DocumentType, ProcessingStatus
from tests.fixtures import (
    sample_mykad_data, sample_spk_data, api_test_data,
    mock_ocr_responses, test_configurations
)
from tests.utils import TestDataManager


class TestDocumentRoutes:
    """Test cases for document processing routes."""
    
    @pytest.fixture
    def client(self):
        """Create a test client for the FastAPI app."""
        return TestClient(app)
    
    @pytest.fixture
    def sample_image_file(self):
        """Create a sample image file for testing."""
        # Create a simple test image
        image = Image.new('RGB', (800, 600), color='white')
        img_byte_arr = io.BytesIO()
        image.save(img_byte_arr, format='JPEG')
        img_byte_arr.seek(0)
        return img_byte_arr
    
    @pytest.fixture
    def auth_headers(self):
        """Create authentication headers for testing."""
        return {"Authorization": "Bearer test_token"}
    
    def test_upload_document_success(self, client, sample_image_file, auth_headers):
        """Test successful document upload."""
        with patch('src.document_parser.core.processor.DocumentProcessor') as mock_processor:
            # Setup mock response
            mock_processor.return_value.process_document.return_value = {
                'document_type': DocumentType.MYKAD.value,
                'classification_confidence': 0.95,
                'extracted_fields': {
                    'ic_number': {
                        'value': '123456-78-9012',
                        'confidence': 0.9,
                        'coordinates': [100, 100, 200, 120],
                        'is_valid': True
                    }
                },
                'validation_passed': True,
                'processing_time': 1.23,
                'image_dimensions': {'width': 800, 'height': 600}
            }
            
            response = client.post(
                "/api/v1/documents/upload",
                files={"file": ("test.jpg", sample_image_file, "image/jpeg")},
                data={"document_type": "mykad"},
                headers=auth_headers
            )
            
            assert response.status_code == status.HTTP_200_OK
            data = response.json()
            assert data['success'] is True
            assert data['document_type'] == DocumentType.MYKAD.value
            assert 'document_id' in data
            assert 'extracted_fields' in data
    
    def test_upload_document_invalid_file(self, client, auth_headers):
        """Test upload with invalid file."""
        # Create invalid file content
        invalid_file = io.BytesIO(b"not an image")
        
        response = client.post(
            "/api/v1/documents/upload",
            files={"file": ("test.txt", invalid_file, "text/plain")},
            headers=auth_headers
        )
        
        assert response.status_code == status.HTTP_400_BAD_REQUEST
        data = response.json()
        assert data['success'] is False
        assert 'error' in data
    
    def test_upload_document_no_auth(self, client, sample_image_file):
        """Test upload without authentication."""
        response = client.post(
            "/api/v1/documents/upload",
            files={"file": ("test.jpg", sample_image_file, "image/jpeg")}
        )
        
        assert response.status_code == status.HTTP_401_UNAUTHORIZED
    
    def test_get_document_status(self, client, auth_headers):
        """Test getting document processing status."""
        document_id = "test_doc_123"
        
        with patch('src.document_parser.api.dependencies.get_database') as mock_db:
            # Setup mock database response
            mock_db.return_value.get_document_status.return_value = {
                'document_id': document_id,
                'status': ProcessingStatus.COMPLETED.value,
                'progress': 100,
                'created_at': '2024-01-01T00:00:00Z',
                'completed_at': '2024-01-01T00:01:00Z'
            }
            
            response = client.get(
                f"/api/v1/documents/{document_id}/status",
                headers=auth_headers
            )
            
            assert response.status_code == status.HTTP_200_OK
            data = response.json()
            assert data['document_id'] == document_id
            assert data['status'] == ProcessingStatus.COMPLETED.value
    
    def test_get_document_status_not_found(self, client, auth_headers):
        """Test getting status for non-existent document."""
        document_id = "nonexistent_doc"
        
        with patch('src.document_parser.api.dependencies.get_database') as mock_db:
            mock_db.return_value.get_document_status.return_value = None
            
            response = client.get(
                f"/api/v1/documents/{document_id}/status",
                headers=auth_headers
            )
            
            assert response.status_code == status.HTTP_404_NOT_FOUND
    
    def test_batch_upload(self, client, sample_image_file, auth_headers):
        """Test batch document upload."""
        # Create multiple files
        files = [
            ("files", ("test1.jpg", sample_image_file, "image/jpeg")),
            ("files", ("test2.jpg", sample_image_file, "image/jpeg"))
        ]
        
        with patch('src.document_parser.core.processor.DocumentProcessor') as mock_processor:
            mock_processor.return_value.process_batch.return_value = [
                {
                    'document_type': DocumentType.MYKAD.value,
                    'classification_confidence': 0.95,
                    'extracted_fields': {},
                    'validation_passed': True,
                    'processing_time': 1.0
                },
                {
                    'document_type': DocumentType.SPK.value,
                    'classification_confidence': 0.88,
                    'extracted_fields': {},
                    'validation_passed': True,
                    'processing_time': 1.2
                }
            ]
            
            response = client.post(
                "/api/v1/documents/batch",
                files=files,
                headers=auth_headers
            )
            
            assert response.status_code == status.HTTP_200_OK
            data = response.json()
            assert data['success'] is True
            assert len(data['results']) == 2
    
    def test_delete_document(self, client, auth_headers):
        """Test document deletion."""
        document_id = "test_doc_123"
        
        with patch('src.document_parser.api.dependencies.get_database') as mock_db:
            mock_db.return_value.delete_document.return_value = True
            
            response = client.delete(
                f"/api/v1/documents/{document_id}",
                headers=auth_headers
            )
            
            assert response.status_code == status.HTTP_200_OK
            data = response.json()
            assert data['success'] is True
    
    def test_get_user_documents(self, client, auth_headers):
        """Test getting user's document history."""
        with patch('src.document_parser.api.dependencies.get_database') as mock_db:
            mock_db.return_value.get_user_documents.return_value = [
                {
                    'document_id': 'doc1',
                    'filename': 'test1.jpg',
                    'document_type': DocumentType.MYKAD.value,
                    'status': ProcessingStatus.COMPLETED.value,
                    'created_at': '2024-01-01T00:00:00Z'
                },
                {
                    'document_id': 'doc2',
                    'filename': 'test2.jpg',
                    'document_type': DocumentType.SPK.value,
                    'status': ProcessingStatus.COMPLETED.value,
                    'created_at': '2024-01-01T01:00:00Z'
                }
            ]
            
            response = client.get(
                "/api/v1/documents/history",
                headers=auth_headers
            )
            
            assert response.status_code == status.HTTP_200_OK
            data = response.json()
            assert len(data['documents']) == 2
            assert data['total'] == 2


class TestHealthRoutes:
    """Test cases for health check routes."""
    
    @pytest.fixture
    def client(self):
        """Create a test client for the FastAPI app."""
        return TestClient(app)
    
    def test_health_check(self, client):
        """Test basic health check endpoint."""
        response = client.get("/api/v1/health")
        
        assert response.status_code == status.HTTP_200_OK
        data = response.json()
        assert data['status'] == 'healthy'
        assert 'timestamp' in data
        assert 'version' in data
    
    def test_readiness_check(self, client):
        """Test readiness check endpoint."""
        with patch('src.document_parser.api.routes.health.check_database_health') as mock_db_check:
            mock_db_check.return_value = {'status': 'healthy', 'response_time': 0.01}
            
            response = client.get("/api/v1/health/ready")
            
            assert response.status_code == status.HTTP_200_OK
            data = response.json()
            assert data['status'] == 'ready'
            assert 'components' in data
    
    def test_liveness_check(self, client):
        """Test liveness check endpoint."""
        response = client.get("/api/v1/health/live")
        
        assert response.status_code == status.HTTP_200_OK
        data = response.json()
        assert data['status'] == 'alive'
    
    def test_metrics_endpoint(self, client):
        """Test metrics endpoint."""
        response = client.get("/api/v1/health/metrics")
        
        assert response.status_code == status.HTTP_200_OK
        data = response.json()
        assert 'system_metrics' in data
        assert 'application_metrics' in data
    
    def test_prometheus_metrics(self, client):
        """Test Prometheus metrics endpoint."""
        response = client.get("/api/v1/health/metrics/prometheus")
        
        assert response.status_code == status.HTTP_200_OK
        assert response.headers['content-type'] == 'text/plain; charset=utf-8'


class TestAdminRoutes:
    """Test cases for admin routes."""
    
    @pytest.fixture
    def client(self):
        """Create a test client for the FastAPI app."""
        return TestClient(app)
    
    @pytest.fixture
    def admin_headers(self):
        """Create admin authentication headers."""
        return {"Authorization": "Bearer admin_token"}
    
    def test_get_system_stats(self, client, admin_headers):
        """Test getting system statistics."""
        with patch('src.document_parser.api.dependencies.get_database') as mock_db:
            mock_db.return_value.get_system_stats.return_value = {
                'total_documents': 1000,
                'documents_today': 50,
                'success_rate': 0.95,
                'average_processing_time': 2.3
            }
            
            response = client.get(
                "/api/v1/admin/stats",
                headers=admin_headers
            )
            
            assert response.status_code == status.HTTP_200_OK
            data = response.json()
            assert 'total_documents' in data
            assert 'success_rate' in data
    
    def test_get_configuration(self, client, admin_headers):
        """Test getting system configuration."""
        response = client.get(
            "/api/v1/admin/config",
            headers=admin_headers
        )
        
        assert response.status_code == status.HTTP_200_OK
        data = response.json()
        assert 'configuration' in data
    
    def test_update_configuration(self, client, admin_headers):
        """Test updating system configuration."""
        new_config = {
            "ocr": {
                "confidence_threshold": 0.8
            }
        }
        
        response = client.put(
            "/api/v1/admin/config",
            json=new_config,
            headers=admin_headers
        )
        
        assert response.status_code == status.HTTP_200_OK
        data = response.json()
        assert data['success'] is True
    
    def test_upload_template(self, client, admin_headers):
        """Test uploading a new document template."""
        template_data = {
            "document_type": "test_doc",
            "fields": {
                "test_field": {
                    "type": "text",
                    "required": True
                }
            }
        }
        
        template_file = io.BytesIO(json.dumps(template_data).encode())
        
        response = client.post(
            "/api/v1/admin/templates",
            files={"template": ("test_template.json", template_file, "application/json")},
            headers=admin_headers
        )
        
        assert response.status_code == status.HTTP_200_OK
        data = response.json()
        assert data['success'] is True
    
    def test_clear_cache(self, client, admin_headers):
        """Test clearing system cache."""
        with patch('src.document_parser.api.dependencies.get_redis_client') as mock_redis:
            mock_redis.return_value.flushdb.return_value = True
            
            response = client.post(
                "/api/v1/admin/cache/clear",
                headers=admin_headers
            )
            
            assert response.status_code == status.HTTP_200_OK
            data = response.json()
            assert data['success'] is True
    
    def test_admin_unauthorized(self, client):
        """Test admin endpoints without proper authorization."""
        response = client.get("/api/v1/admin/stats")
        
        assert response.status_code == status.HTTP_401_UNAUTHORIZED


class TestAPIErrorHandling:
    """Test cases for API error handling."""
    
    @pytest.fixture
    def client(self):
        """Create a test client for the FastAPI app."""
        return TestClient(app)
    
    def test_404_error(self, client):
        """Test 404 error handling."""
        response = client.get("/api/v1/nonexistent")
        
        assert response.status_code == status.HTTP_404_NOT_FOUND
        data = response.json()
        assert 'detail' in data
    
    def test_validation_error(self, client):
        """Test request validation error handling."""
        # Send invalid JSON
        response = client.post(
            "/api/v1/documents/upload",
            json={"invalid": "data"}
        )
        
        assert response.status_code in [status.HTTP_400_BAD_REQUEST, status.HTTP_422_UNPROCESSABLE_ENTITY]
    
    def test_rate_limiting(self, client):
        """Test rate limiting functionality."""
        # This would require actual rate limiting implementation
        # For now, we'll skip this test
        pytest.skip("Rate limiting not implemented yet")
    
    def test_internal_server_error(self, client):
        """Test internal server error handling."""
        with patch('src.document_parser.api.routes.documents.process_document_upload') as mock_process:
            mock_process.side_effect = Exception("Internal error")
            
            # This test would need a valid request that triggers the error
            # Implementation depends on actual error handling setup
            pass


if __name__ == '__main__':
    pytest.main([__file__])