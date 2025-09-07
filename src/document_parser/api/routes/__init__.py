#!/usr/bin/env python3
"""
API Routes

FastAPI route modules for the document parser API.
Includes routes for document processing, health checks, and administration.

Author: Document Parser Team
Version: 1.0.0
Created: 2024-01-15
"""

from . import documents, health, admin

__version__ = "1.0.0"
__author__ = "Document Parser Team"

__all__ = [
    "documents",
    "health", 
    "admin"
]

# Route Information
ROUTE_INFO = {
    "documents": {
        "prefix": "/api/v1/documents",
        "description": "Document processing endpoints",
        "endpoints": [
            "POST /upload - Upload and process documents",
            "GET /{document_id} - Get processing results",
            "POST /batch - Batch document processing",
            "GET /{document_id}/download - Download processed results"
        ]
    },
    "health": {
        "prefix": "/health",
        "description": "Health check and monitoring endpoints",
        "endpoints": [
            "GET / - Basic health check",
            "GET /detailed - Detailed system status",
            "GET /metrics - Prometheus metrics"
        ]
    },
    "admin": {
        "prefix": "/api/v1/admin",
        "description": "Administrative endpoints",
        "endpoints": [
            "GET /stats - System statistics",
            "POST /models/reload - Reload ML models",
            "GET /users - User management",
            "POST /maintenance - Maintenance mode"
        ]
    }
}