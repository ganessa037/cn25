#!/usr/bin/env python3
"""
API Models

Pydantic models for request/response validation and documentation.
Defines all data structures used in the FastAPI endpoints.
"""

import logging
from datetime import datetime
from typing import Dict, Any, List, Optional, Union
from enum import Enum

from pydantic import BaseModel, Field, validator, HttpUrl
from fastapi import UploadFile

from ..models.document_models import DocumentType, ProcessingStatus, ConfidenceLevel

# Configure logging
logger = logging.getLogger(__name__)

# Enums for API

class SortOrder(str, Enum):
    """Sort order options."""
    ASC = "asc"
    DESC = "desc"

class BatchStatus(str, Enum):
    """Batch processing status."""
    PENDING = "pending"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"

class ErrorType(str, Enum):
    """Error type classification."""
    VALIDATION_ERROR = "validation_error"
    PROCESSING_ERROR = "processing_error"
    OCR_ERROR = "ocr_error"
    CLASSIFICATION_ERROR = "classification_error"
    SYSTEM_ERROR = "system_error"

# Base Models

class APIResponse(BaseModel):
    """Base API response model."""
    success: bool = Field(..., description="Whether the request was successful")
    message: str = Field(..., description="Response message")
    timestamp: datetime = Field(default_factory=datetime.utcnow, description="Response timestamp")
    request_id: Optional[str] = Field(None, description="Request tracking ID")

class ErrorDetail(BaseModel):
    """Error detail model."""
    error_type: ErrorType = Field(..., description="Type of error")
    error_code: str = Field(..., description="Specific error code")
    message: str = Field(..., description="Human-readable error message")
    field: Optional[str] = Field(None, description="Field that caused the error")
    details: Optional[Dict[str, Any]] = Field(None, description="Additional error details")

class APIError(APIResponse):
    """API error response model."""
    success: bool = Field(False, description="Always false for errors")
    error: ErrorDetail = Field(..., description="Error details")
    trace_id: Optional[str] = Field(None, description="Error trace ID for debugging")

# Document Processing Models

class DocumentUploadRequest(BaseModel):
    """Document upload request model."""
    document_type: Optional[DocumentType] = Field(None, description="Expected document type (auto-detect if not provided)")
    extract_fields: Optional[List[str]] = Field(None, description="Specific fields to extract")
    validation_level: Optional[str] = Field("standard", description="Validation strictness level")
    return_coordinates: bool = Field(False, description="Include field coordinates in response")
    return_confidence: bool = Field(True, description="Include confidence scores")
    preprocessing_options: Optional[Dict[str, Any]] = Field(None, description="Image preprocessing options")
    callback_url: Optional[HttpUrl] = Field(None, description="Webhook URL for async processing")
    metadata: Optional[Dict[str, Any]] = Field(None, description="Additional metadata")

class FieldResult(BaseModel):
    """Individual field extraction result."""
    field_name: str = Field(..., description="Name of the extracted field")
    value: Optional[str] = Field(None, description="Extracted value")
    confidence: float = Field(..., ge=0.0, le=1.0, description="Confidence score")
    confidence_level: ConfidenceLevel = Field(..., description="Confidence level category")
    coordinates: Optional[Dict[str, float]] = Field(None, description="Field coordinates (x, y, width, height)")
    validation_status: str = Field(..., description="Validation result")
    validation_errors: Optional[List[str]] = Field(None, description="Validation error messages")
    raw_text: Optional[str] = Field(None, description="Raw OCR text before processing")
    alternatives: Optional[List[str]] = Field(None, description="Alternative extraction candidates")

class ProcessingMetadata(BaseModel):
    """Processing metadata model."""
    processing_time: float = Field(..., description="Total processing time in seconds")
    ocr_time: float = Field(..., description="OCR processing time in seconds")
    classification_time: float = Field(..., description="Classification time in seconds")
    extraction_time: float = Field(..., description="Field extraction time in seconds")
    validation_time: float = Field(..., description="Validation time in seconds")
    image_preprocessing: Dict[str, Any] = Field(..., description="Preprocessing operations applied")
    ocr_engine: str = Field(..., description="OCR engine used")
    model_versions: Dict[str, str] = Field(..., description="Model versions used")
    quality_metrics: Dict[str, float] = Field(..., description="Image and text quality metrics")

class DocumentProcessingResult(BaseModel):
    """Document processing result model."""
    document_id: str = Field(..., description="Unique document identifier")
    document_type: DocumentType = Field(..., description="Detected document type")
    classification_confidence: float = Field(..., ge=0.0, le=1.0, description="Document classification confidence")
    status: ProcessingStatus = Field(..., description="Processing status")
    extracted_fields: List[FieldResult] = Field(..., description="Extracted field results")
    validation_summary: Dict[str, Any] = Field(..., description="Overall validation summary")
    processing_metadata: ProcessingMetadata = Field(..., description="Processing metadata")
    errors: Optional[List[ErrorDetail]] = Field(None, description="Processing errors")
    warnings: Optional[List[str]] = Field(None, description="Processing warnings")
    created_at: datetime = Field(..., description="Processing start time")
    completed_at: Optional[datetime] = Field(None, description="Processing completion time")

class DocumentUploadResponse(APIResponse):
    """Document upload response model."""
    document_id: str = Field(..., description="Unique document identifier")
    status: ProcessingStatus = Field(..., description="Initial processing status")
    estimated_completion: Optional[datetime] = Field(None, description="Estimated completion time")
    result: Optional[DocumentProcessingResult] = Field(None, description="Processing result (if completed synchronously)")

class DocumentStatusResponse(APIResponse):
    """Document status response model."""
    document_id: str = Field(..., description="Document identifier")
    status: ProcessingStatus = Field(..., description="Current processing status")
    progress: Optional[float] = Field(None, ge=0.0, le=100.0, description="Processing progress percentage")
    result: Optional[DocumentProcessingResult] = Field(None, description="Processing result (if completed)")
    error: Optional[ErrorDetail] = Field(None, description="Error details (if failed)")
    estimated_completion: Optional[datetime] = Field(None, description="Estimated completion time")

# Batch Processing Models

class BatchUploadRequest(BaseModel):
    """Batch upload request model."""
    batch_name: Optional[str] = Field(None, description="Optional batch name")
    document_type: Optional[DocumentType] = Field(None, description="Expected document type for all documents")
    processing_options: Optional[DocumentUploadRequest] = Field(None, description="Common processing options")
    priority: int = Field(1, ge=1, le=10, description="Batch processing priority")
    callback_url: Optional[HttpUrl] = Field(None, description="Webhook URL for batch completion")
    metadata: Optional[Dict[str, Any]] = Field(None, description="Batch metadata")

class BatchDocumentStatus(BaseModel):
    """Individual document status in batch."""
    document_id: str = Field(..., description="Document identifier")
    filename: str = Field(..., description="Original filename")
    status: ProcessingStatus = Field(..., description="Document processing status")
    result: Optional[DocumentProcessingResult] = Field(None, description="Processing result")
    error: Optional[ErrorDetail] = Field(None, description="Error details")
    processing_order: int = Field(..., description="Processing order in batch")

class BatchProcessingResult(BaseModel):
    """Batch processing result model."""
    batch_id: str = Field(..., description="Unique batch identifier")
    batch_name: Optional[str] = Field(None, description="Batch name")
    status: BatchStatus = Field(..., description="Batch processing status")
    total_documents: int = Field(..., description="Total number of documents in batch")
    completed_documents: int = Field(..., description="Number of completed documents")
    failed_documents: int = Field(..., description="Number of failed documents")
    progress: float = Field(..., ge=0.0, le=100.0, description="Batch progress percentage")
    documents: List[BatchDocumentStatus] = Field(..., description="Individual document statuses")
    processing_summary: Dict[str, Any] = Field(..., description="Batch processing summary")
    created_at: datetime = Field(..., description="Batch creation time")
    started_at: Optional[datetime] = Field(None, description="Batch processing start time")
    completed_at: Optional[datetime] = Field(None, description="Batch completion time")
    estimated_completion: Optional[datetime] = Field(None, description="Estimated completion time")

class BatchUploadResponse(APIResponse):
    """Batch upload response model."""
    batch_id: str = Field(..., description="Unique batch identifier")
    total_documents: int = Field(..., description="Number of documents in batch")
    status: BatchStatus = Field(..., description="Initial batch status")
    estimated_completion: Optional[datetime] = Field(None, description="Estimated completion time")

class BatchStatusResponse(APIResponse):
    """Batch status response model."""
    batch_id: str = Field(..., description="Batch identifier")
    result: BatchProcessingResult = Field(..., description="Batch processing result")

# User and History Models

class DocumentHistoryItem(BaseModel):
    """Document history item model."""
    document_id: str = Field(..., description="Document identifier")
    filename: str = Field(..., description="Original filename")
    document_type: DocumentType = Field(..., description="Document type")
    status: ProcessingStatus = Field(..., description="Processing status")
    created_at: datetime = Field(..., description="Upload time")
    completed_at: Optional[datetime] = Field(None, description="Completion time")
    processing_time: Optional[float] = Field(None, description="Processing time in seconds")
    field_count: int = Field(..., description="Number of extracted fields")
    confidence_score: Optional[float] = Field(None, description="Overall confidence score")
    has_errors: bool = Field(..., description="Whether processing had errors")
    batch_id: Optional[str] = Field(None, description="Batch ID if part of batch")

class UserHistoryRequest(BaseModel):
    """User history request model."""
    limit: int = Field(50, ge=1, le=200, description="Maximum number of items to return")
    offset: int = Field(0, ge=0, description="Number of items to skip")
    document_type: Optional[DocumentType] = Field(None, description="Filter by document type")
    status: Optional[ProcessingStatus] = Field(None, description="Filter by status")
    start_date: Optional[datetime] = Field(None, description="Filter by start date")
    end_date: Optional[datetime] = Field(None, description="Filter by end date")
    sort_by: str = Field("created_at", description="Field to sort by")
    sort_order: SortOrder = Field(SortOrder.DESC, description="Sort order")
    include_batches: bool = Field(True, description="Include batch information")

class UserHistoryResponse(APIResponse):
    """User history response model."""
    documents: List[DocumentHistoryItem] = Field(..., description="Document history items")
    total_count: int = Field(..., description="Total number of documents")
    limit: int = Field(..., description="Applied limit")
    offset: int = Field(..., description="Applied offset")
    has_more: bool = Field(..., description="Whether more items are available")
    filters: Dict[str, Any] = Field(..., description="Applied filters")

# Health and Monitoring Models

class ComponentHealth(BaseModel):
    """Individual component health status."""
    name: str = Field(..., description="Component name")
    status: str = Field(..., description="Health status (healthy, degraded, unhealthy)")
    response_time: Optional[float] = Field(None, description="Response time in seconds")
    last_check: datetime = Field(..., description="Last health check time")
    details: Optional[Dict[str, Any]] = Field(None, description="Additional health details")
    error: Optional[str] = Field(None, description="Error message if unhealthy")

class SystemHealth(BaseModel):
    """System health status model."""
    overall_status: str = Field(..., description="Overall system health status")
    components: List[ComponentHealth] = Field(..., description="Individual component health")
    uptime: float = Field(..., description="System uptime in seconds")
    version: str = Field(..., description="System version")
    timestamp: datetime = Field(..., description="Health check timestamp")

class HealthResponse(APIResponse):
    """Health check response model."""
    health: SystemHealth = Field(..., description="System health information")

class MetricsResponse(BaseModel):
    """Metrics response model."""
    system_metrics: Dict[str, float] = Field(..., description="System resource metrics")
    application_metrics: Dict[str, float] = Field(..., description="Application-specific metrics")
    processing_metrics: Dict[str, float] = Field(..., description="Document processing metrics")
    timestamp: datetime = Field(..., description="Metrics timestamp")
    collection_interval: int = Field(..., description="Metrics collection interval in seconds")

# Authentication Models

class LoginRequest(BaseModel):
    """User login request model."""
    username: str = Field(..., description="Username or email")
    password: str = Field(..., description="User password")
    remember_me: bool = Field(False, description="Whether to extend session duration")

class TokenResponse(BaseModel):
    """Authentication token response model."""
    access_token: str = Field(..., description="JWT access token")
    token_type: str = Field("bearer", description="Token type")
    expires_in: int = Field(..., description="Token expiration time in seconds")
    refresh_token: Optional[str] = Field(None, description="Refresh token")
    user_id: str = Field(..., description="User identifier")
    permissions: List[str] = Field(..., description="User permissions")

class RefreshTokenRequest(BaseModel):
    """Token refresh request model."""
    refresh_token: str = Field(..., description="Refresh token")

class UserInfo(BaseModel):
    """User information model."""
    user_id: str = Field(..., description="User identifier")
    username: str = Field(..., description="Username")
    email: str = Field(..., description="User email")
    role: str = Field(..., description="User role")
    permissions: List[str] = Field(..., description="User permissions")
    created_at: datetime = Field(..., description="Account creation time")
    last_login: Optional[datetime] = Field(None, description="Last login time")
    is_active: bool = Field(..., description="Whether account is active")

# Configuration Models

class ConfigurationSection(BaseModel):
    """Configuration section model."""
    section_name: str = Field(..., description="Configuration section name")
    settings: Dict[str, Any] = Field(..., description="Section settings")
    description: Optional[str] = Field(None, description="Section description")
    last_modified: datetime = Field(..., description="Last modification time")
    modified_by: Optional[str] = Field(None, description="User who last modified")

class ConfigurationResponse(APIResponse):
    """Configuration response model."""
    sections: List[ConfigurationSection] = Field(..., description="Configuration sections")
    version: str = Field(..., description="Configuration version")
    environment: str = Field(..., description="Environment name")

# Validation Models

class ValidationRule(BaseModel):
    """Validation rule model."""
    field_name: str = Field(..., description="Field name")
    rule_type: str = Field(..., description="Validation rule type")
    parameters: Dict[str, Any] = Field(..., description="Rule parameters")
    error_message: str = Field(..., description="Error message for validation failure")
    severity: str = Field(..., description="Validation severity (error, warning)")

class ValidationResult(BaseModel):
    """Field validation result model."""
    field_name: str = Field(..., description="Field name")
    is_valid: bool = Field(..., description="Whether field passed validation")
    errors: List[str] = Field(..., description="Validation error messages")
    warnings: List[str] = Field(..., description="Validation warning messages")
    applied_rules: List[str] = Field(..., description="Validation rules applied")

class DocumentValidationResult(BaseModel):
    """Document validation result model."""
    document_id: str = Field(..., description="Document identifier")
    is_valid: bool = Field(..., description="Whether document passed validation")
    field_results: List[ValidationResult] = Field(..., description="Field validation results")
    cross_field_validation: Dict[str, Any] = Field(..., description="Cross-field validation results")
    validation_summary: Dict[str, Any] = Field(..., description="Validation summary")
    validated_at: datetime = Field(..., description="Validation timestamp")

# Utility Models

class PaginationInfo(BaseModel):
    """Pagination information model."""
    total_items: int = Field(..., description="Total number of items")
    items_per_page: int = Field(..., description="Items per page")
    current_page: int = Field(..., description="Current page number")
    total_pages: int = Field(..., description="Total number of pages")
    has_next: bool = Field(..., description="Whether next page exists")
    has_previous: bool = Field(..., description="Whether previous page exists")

class SortInfo(BaseModel):
    """Sort information model."""
    field: str = Field(..., description="Sort field")
    order: SortOrder = Field(..., description="Sort order")
    secondary_sort: Optional['SortInfo'] = Field(None, description="Secondary sort criteria")

class FilterInfo(BaseModel):
    """Filter information model."""
    field: str = Field(..., description="Filter field")
    operator: str = Field(..., description="Filter operator")
    value: Any = Field(..., description="Filter value")
    case_sensitive: bool = Field(False, description="Whether filter is case sensitive")

# Model validators

@validator('confidence', 'classification_confidence')
def validate_confidence(cls, v):
    """Validate confidence scores are between 0 and 1."""
    if not 0.0 <= v <= 1.0:
        raise ValueError('Confidence must be between 0.0 and 1.0')
    return v

@validator('progress')
def validate_progress(cls, v):
    """Validate progress is between 0 and 100."""
    if not 0.0 <= v <= 100.0:
        raise ValueError('Progress must be between 0.0 and 100.0')
    return v

# Update forward references
SortInfo.update_forward_refs()

# Export all models
__all__ = [
    # Enums
    'SortOrder', 'BatchStatus', 'ErrorType',
    
    # Base Models
    'APIResponse', 'ErrorDetail', 'APIError',
    
    # Document Processing
    'DocumentUploadRequest', 'FieldResult', 'ProcessingMetadata',
    'DocumentProcessingResult', 'DocumentUploadResponse', 'DocumentStatusResponse',
    
    # Batch Processing
    'BatchUploadRequest', 'BatchDocumentStatus', 'BatchProcessingResult',
    'BatchUploadResponse', 'BatchStatusResponse',
    
    # User and History
    'DocumentHistoryItem', 'UserHistoryRequest', 'UserHistoryResponse',
    
    # Health and Monitoring
    'ComponentHealth', 'SystemHealth', 'HealthResponse', 'MetricsResponse',
    
    # Authentication
    'LoginRequest', 'TokenResponse', 'RefreshTokenRequest', 'UserInfo',
    
    # Configuration
    'ConfigurationSection', 'ConfigurationResponse',
    
    # Validation
    'ValidationRule', 'ValidationResult', 'DocumentValidationResult',
    
    # Utility
    'PaginationInfo', 'SortInfo', 'FilterInfo'
]