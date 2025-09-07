#!/usr/bin/env python3
"""
Document Parser Models Package

Contains data models, machine learning models, and training infrastructure
for the document parser system.
"""

__version__ = "1.0.0"
__author__ = "Document Parser Team"

# Import core model classes
from .document_models import (
    DocumentType,
    DocumentMetadata,
    ExtractionResult,
    ValidationResult as ModelValidationResult,
    ProcessingStatus
)

from .extraction_models import (
    FieldExtractionModel,
    ClassificationModel,
    OCRModel,
    ModelMetrics
)

# Model registry
__all__ = [
    # Document models
    "DocumentType",
    "DocumentMetadata", 
    "ExtractionResult",
    "ModelValidationResult",
    "ProcessingStatus",
    
    # Extraction models
    "FieldExtractionModel",
    "ClassificationModel",
    "OCRModel",
    "ModelMetrics"
]

# Package information
PACKAGE_INFO = {
    "name": "document_parser_models",
    "version": __version__,
    "description": "Data models and ML models for document parsing",
    "author": __author__,
    "models_included": [
        "Document Classification",
        "Field Extraction", 
        "OCR Processing",
        "Validation"
    ]
}