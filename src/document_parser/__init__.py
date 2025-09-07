#!/usr/bin/env python3
"""
Document Parser Module

A comprehensive document parsing system for extracting structured data from various document types
including SPK, MyKad, and other Malaysian identity documents.

Features:
- Multi-format document classification
- OCR with preprocessing (Tesseract + EasyOCR)
- Field extraction using rule-based + ML hybrid approach
- Business logic validation
- RESTful API integration

Author: CN25 Team
Version: 1.0.0
"""

__version__ = "1.0.0"
__author__ = "CN25 Team"

# Core modules
from .document_classifier import DocumentClassifier
from .ocr_service import OCRService
from .field_extractor import FieldExtractor
from .validator import DocumentValidator
from .utils import ImagePreprocessor, TextProcessor

__all__ = [
    "DocumentClassifier",
    "OCRService", 
    "FieldExtractor",
    "DocumentValidator",
    "ImagePreprocessor",
    "TextProcessor"
]