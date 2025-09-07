#!/usr/bin/env python3
"""
Document Parser API

FastAPI-based REST API for document parsing and field extraction.
Provides endpoints for document upload, processing, and result retrieval.

Author: Document Parser Team
Version: 1.0.0
Created: 2024-01-15
"""

from .main import app
from .dependencies import get_database, get_redis, get_config
from .routes import documents, health, admin

__version__ = "1.0.0"
__author__ = "Document Parser Team"

__all__ = [
    "app",
    "get_database",
    "get_redis", 
    "get_config",
    "documents",
    "health",
    "admin"
]

# API Information
API_INFO = {
    "title": "Document Parser API",
    "description": "REST API for Malaysian document parsing and field extraction",
    "version": __version__,
    "author": __author__,
    "supported_documents": ["MyKad", "SPK", "Passport", "License"],
    "features": [
        "Document Classification",
        "OCR Text Extraction", 
        "Field Extraction",
        "Data Validation",
        "Confidence Scoring",
        "Batch Processing"
    ]
}