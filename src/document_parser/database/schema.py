#!/usr/bin/env python3
"""
Database Schema Design

Comprehensive database schema for document processing system including:
- Document metadata and processing status
- Extracted data with validation
- Audit logs for all activities
- Performance metrics and analytics
"""

import logging
from datetime import datetime, timezone
from typing import Dict, Any, Optional, List
from enum import Enum
from decimal import Decimal

from sqlalchemy import (
    Column, Integer, String, Text, DateTime, Boolean, Float, JSON,
    ForeignKey, Index, UniqueConstraint, CheckConstraint, Numeric,
    LargeBinary, TIMESTAMP, func
)
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import relationship, backref
from sqlalchemy.dialects.postgresql import UUID, JSONB, ARRAY
from sqlalchemy.ext.hybrid import hybrid_property
import uuid

# Configure logging
logger = logging.getLogger(__name__)

# Base class for all models
Base = declarative_base()

# Enums for database fields
class DocumentStatus(str, Enum):
    """Document processing status."""
    UPLOADED = "uploaded"
    QUEUED = "queued"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"
    ARCHIVED = "archived"

class DocumentType(str, Enum):
    """Supported document types."""
    PDF = "pdf"
    IMAGE = "image"
    WORD = "word"
    EXCEL = "excel"
    POWERPOINT = "powerpoint"
    TEXT = "text"
    HTML = "html"
    XML = "xml"
    JSON = "json"
    CSV = "csv"

class UserRole(str, Enum):
    """User roles in the system."""
    ADMIN = "admin"
    USER = "user"
    VIEWER = "viewer"
    API_USER = "api_user"
    GUEST = "guest"

class AuditAction(str, Enum):
    """Audit log action types."""
    CREATE = "create"
    READ = "read"
    UPDATE = "update"
    DELETE = "delete"
    UPLOAD = "upload"
    DOWNLOAD = "download"
    PROCESS = "process"
    EXPORT = "export"
    LOGIN = "login"
    LOGOUT = "logout"
    API_CALL = "api_call"

class ValidationStatus(str, Enum):
    """Data validation status."""
    PENDING = "pending"
    VALID = "valid"
    INVALID = "invalid"
    WARNING = "warning"
    MANUAL_REVIEW = "manual_review"

# Base Model with Common Fields
class BaseModel(Base):
    """Base model with common fields for all tables."""
    __abstract__ = True
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    created_at = Column(DateTime(timezone=True), default=func.now(), nullable=False)
    updated_at = Column(DateTime(timezone=True), default=func.now(), onupdate=func.now(), nullable=False)
    created_by = Column(UUID(as_uuid=True), ForeignKey('users.id'), nullable=True)
    updated_by = Column(UUID(as_uuid=True), ForeignKey('users.id'), nullable=True)
    is_active = Column(Boolean, default=True, nullable=False)
    version = Column(Integer, default=1, nullable=False)

# User Management Tables
class User(BaseModel):
    """User accounts and authentication."""
    __tablename__ = 'users'
    
    username = Column(String(50), unique=True, nullable=False, index=True)
    email = Column(String(255), unique=True, nullable=False, index=True)
    password_hash = Column(String(255), nullable=False)
    full_name = Column(String(100), nullable=True)
    organization = Column(String(100), nullable=True)
    role = Column(String(20), nullable=False, default=UserRole.USER.value)
    is_verified = Column(Boolean, default=False, nullable=False)
    is_locked = Column(Boolean, default=False, nullable=False)
    last_login = Column(DateTime(timezone=True), nullable=True)
    login_attempts = Column(Integer, default=0, nullable=False)
    password_changed_at = Column(DateTime(timezone=True), nullable=True)
    preferences = Column(JSONB, nullable=True)
    api_quota_limit = Column(Integer, default=1000, nullable=False)
    api_quota_used = Column(Integer, default=0, nullable=False)
    api_quota_reset_date = Column(DateTime(timezone=True), nullable=True)
    
    # Relationships
    documents = relationship("Document", back_populates="owner")
    api_keys = relationship("APIKey", back_populates="user")
    audit_logs = relationship("AuditLog", back_populates="user")
    
    # Indexes
    __table_args__ = (
        Index('idx_users_email_active', 'email', 'is_active'),
        Index('idx_users_username_active', 'username', 'is_active'),
        Index('idx_users_role', 'role'),
        CheckConstraint('api_quota_limit >= 0', name='check_api_quota_limit_positive'),
        CheckConstraint('api_quota_used >= 0', name='check_api_quota_used_positive'),
    )
    
    @hybrid_property
    def api_quota_remaining(self):
        """Calculate remaining API quota."""
        return max(0, self.api_quota_limit - self.api_quota_used)

class APIKey(BaseModel):
    """API keys for programmatic access."""
    __tablename__ = 'api_keys'
    
    user_id = Column(UUID(as_uuid=True), ForeignKey('users.id'), nullable=False)
    key_hash = Column(String(255), unique=True, nullable=False, index=True)
    name = Column(String(100), nullable=False)
    permissions = Column(ARRAY(String), nullable=True)
    rate_limit = Column(Integer, default=1000, nullable=False)
    usage_count = Column(Integer, default=0, nullable=False)
    last_used = Column(DateTime(timezone=True), nullable=True)
    expires_at = Column(DateTime(timezone=True), nullable=True)
    allowed_ips = Column(ARRAY(String), nullable=True)
    is_revoked = Column(Boolean, default=False, nullable=False)
    
    # Relationships
    user = relationship("User", back_populates="api_keys")
    
    # Indexes
    __table_args__ = (
        Index('idx_api_keys_user_active', 'user_id', 'is_active'),
        Index('idx_api_keys_hash', 'key_hash'),
        CheckConstraint('rate_limit > 0', name='check_rate_limit_positive'),
    )

# Document Management Tables
class Document(BaseModel):
    """Document metadata and processing information."""
    __tablename__ = 'documents'
    
    # Basic document information
    filename = Column(String(255), nullable=False)
    original_filename = Column(String(255), nullable=False)
    file_path = Column(String(500), nullable=False)
    file_size = Column(Integer, nullable=False)
    file_hash = Column(String(64), nullable=False, index=True)  # SHA-256
    mime_type = Column(String(100), nullable=False)
    document_type = Column(String(20), nullable=False)
    
    # Processing information
    status = Column(String(20), nullable=False, default=DocumentStatus.UPLOADED.value)
    processing_started_at = Column(DateTime(timezone=True), nullable=True)
    processing_completed_at = Column(DateTime(timezone=True), nullable=True)
    processing_duration = Column(Float, nullable=True)  # seconds
    processing_attempts = Column(Integer, default=0, nullable=False)
    error_message = Column(Text, nullable=True)
    
    # Ownership and access
    owner_id = Column(UUID(as_uuid=True), ForeignKey('users.id'), nullable=False)
    is_public = Column(Boolean, default=False, nullable=False)
    access_level = Column(String(20), default='private', nullable=False)
    
    # Content metadata
    page_count = Column(Integer, nullable=True)
    word_count = Column(Integer, nullable=True)
    language = Column(String(10), nullable=True)
    encoding = Column(String(50), nullable=True)
    
    # Processing configuration
    processing_config = Column(JSONB, nullable=True)
    extraction_rules = Column(JSONB, nullable=True)
    
    # Quality metrics
    confidence_score = Column(Float, nullable=True)
    quality_score = Column(Float, nullable=True)
    
    # Relationships
    owner = relationship("User", back_populates="documents")
    extracted_data = relationship("ExtractedData", back_populates="document", cascade="all, delete-orphan")
    processing_logs = relationship("ProcessingLog", back_populates="document", cascade="all, delete-orphan")
    validations = relationship("DataValidation", back_populates="document", cascade="all, delete-orphan")
    
    # Indexes
    __table_args__ = (
        Index('idx_documents_owner_status', 'owner_id', 'status'),
        Index('idx_documents_hash', 'file_hash'),
        Index('idx_documents_type_status', 'document_type', 'status'),
        Index('idx_documents_created_at', 'created_at'),
        CheckConstraint('file_size > 0', name='check_file_size_positive'),
        CheckConstraint('confidence_score >= 0 AND confidence_score <= 1', name='check_confidence_score_range'),
        CheckConstraint('quality_score >= 0 AND quality_score <= 1', name='check_quality_score_range'),
    )
    
    @hybrid_property
    def processing_time_minutes(self):
        """Get processing time in minutes."""
        if self.processing_duration:
            return self.processing_duration / 60
        return None

class ExtractedData(BaseModel):
    """Extracted data from documents with field-level tracking."""
    __tablename__ = 'extracted_data'
    
    document_id = Column(UUID(as_uuid=True), ForeignKey('documents.id'), nullable=False)
    field_name = Column(String(100), nullable=False)
    field_type = Column(String(50), nullable=False)  # text, number, date, boolean, etc.
    field_value = Column(Text, nullable=True)
    field_value_json = Column(JSONB, nullable=True)  # For complex data types
    confidence_score = Column(Float, nullable=True)
    extraction_method = Column(String(50), nullable=True)  # ocr, nlp, regex, etc.
    bounding_box = Column(JSONB, nullable=True)  # For image coordinates
    page_number = Column(Integer, nullable=True)
    
    # Validation information
    is_validated = Column(Boolean, default=False, nullable=False)
    validation_status = Column(String(20), default=ValidationStatus.PENDING.value)
    validation_message = Column(Text, nullable=True)
    validated_by = Column(UUID(as_uuid=True), ForeignKey('users.id'), nullable=True)
    validated_at = Column(DateTime(timezone=True), nullable=True)
    
    # Data lineage
    source_text = Column(Text, nullable=True)  # Original text that was extracted
    extraction_context = Column(JSONB, nullable=True)  # Additional context
    
    # Relationships
    document = relationship("Document", back_populates="extracted_data")
    validator = relationship("User", foreign_keys=[validated_by])
    
    # Indexes
    __table_args__ = (
        Index('idx_extracted_data_document_field', 'document_id', 'field_name'),
        Index('idx_extracted_data_type', 'field_type'),
        Index('idx_extracted_data_validation', 'validation_status'),
        UniqueConstraint('document_id', 'field_name', name='uq_document_field'),
        CheckConstraint('confidence_score >= 0 AND confidence_score <= 1', name='check_confidence_score_range'),
    )

class DataValidation(BaseModel):
    """Data validation rules and results."""
    __tablename__ = 'data_validations'
    
    document_id = Column(UUID(as_uuid=True), ForeignKey('documents.id'), nullable=False)
    field_name = Column(String(100), nullable=False)
    validation_rule = Column(String(100), nullable=False)  # required, format, range, etc.
    rule_config = Column(JSONB, nullable=True)  # Rule-specific configuration
    validation_result = Column(Boolean, nullable=False)
    error_message = Column(Text, nullable=True)
    severity = Column(String(20), default='error', nullable=False)  # error, warning, info
    
    # Relationships
    document = relationship("Document", back_populates="validations")
    
    # Indexes
    __table_args__ = (
        Index('idx_data_validations_document', 'document_id'),
        Index('idx_data_validations_result', 'validation_result'),
        Index('idx_data_validations_severity', 'severity'),
    )

# Processing and Audit Tables
class ProcessingLog(BaseModel):
    """Detailed processing logs for debugging and monitoring."""
    __tablename__ = 'processing_logs'
    
    document_id = Column(UUID(as_uuid=True), ForeignKey('documents.id'), nullable=True)
    task_id = Column(String(100), nullable=True, index=True)
    process_name = Column(String(100), nullable=False)
    log_level = Column(String(20), nullable=False)  # DEBUG, INFO, WARNING, ERROR
    message = Column(Text, nullable=False)
    details = Column(JSONB, nullable=True)
    execution_time = Column(Float, nullable=True)  # seconds
    memory_usage = Column(Float, nullable=True)  # MB
    cpu_usage = Column(Float, nullable=True)  # percentage
    
    # Error information
    exception_type = Column(String(100), nullable=True)
    stack_trace = Column(Text, nullable=True)
    
    # Relationships
    document = relationship("Document", back_populates="processing_logs")
    
    # Indexes
    __table_args__ = (
        Index('idx_processing_logs_document', 'document_id'),
        Index('idx_processing_logs_task', 'task_id'),
        Index('idx_processing_logs_level', 'log_level'),
        Index('idx_processing_logs_created', 'created_at'),
    )

class AuditLog(BaseModel):
    """Comprehensive audit trail for all system activities."""
    __tablename__ = 'audit_logs'
    
    user_id = Column(UUID(as_uuid=True), ForeignKey('users.id'), nullable=True)
    session_id = Column(String(100), nullable=True)
    action = Column(String(50), nullable=False)
    resource_type = Column(String(50), nullable=False)  # document, user, api_key, etc.
    resource_id = Column(String(100), nullable=True)
    
    # Request information
    ip_address = Column(String(45), nullable=True)  # IPv6 compatible
    user_agent = Column(Text, nullable=True)
    request_method = Column(String(10), nullable=True)
    request_path = Column(String(500), nullable=True)
    request_params = Column(JSONB, nullable=True)
    
    # Response information
    response_status = Column(Integer, nullable=True)
    response_time = Column(Float, nullable=True)  # milliseconds
    
    # Change tracking
    old_values = Column(JSONB, nullable=True)
    new_values = Column(JSONB, nullable=True)
    
    # Additional context
    description = Column(Text, nullable=True)
    metadata = Column(JSONB, nullable=True)
    
    # Relationships
    user = relationship("User", back_populates="audit_logs")
    
    # Indexes
    __table_args__ = (
        Index('idx_audit_logs_user_action', 'user_id', 'action'),
        Index('idx_audit_logs_resource', 'resource_type', 'resource_id'),
        Index('idx_audit_logs_created', 'created_at'),
        Index('idx_audit_logs_ip', 'ip_address'),
        Index('idx_audit_logs_session', 'session_id'),
    )

# Performance and Analytics Tables
class PerformanceMetric(BaseModel):
    """System performance metrics and analytics."""
    __tablename__ = 'performance_metrics'
    
    metric_name = Column(String(100), nullable=False)
    metric_type = Column(String(50), nullable=False)  # counter, gauge, histogram, timer
    metric_value = Column(Numeric(precision=15, scale=6), nullable=False)
    metric_unit = Column(String(20), nullable=True)  # seconds, bytes, count, etc.
    
    # Dimensions for grouping
    dimensions = Column(JSONB, nullable=True)  # {"endpoint": "/api/upload", "status": "success"}
    
    # Time-based partitioning
    timestamp = Column(TIMESTAMP(timezone=True), nullable=False, default=func.now())
    date_partition = Column(String(10), nullable=False)  # YYYY-MM-DD for partitioning
    
    # Aggregation support
    sample_count = Column(Integer, default=1, nullable=False)
    min_value = Column(Numeric(precision=15, scale=6), nullable=True)
    max_value = Column(Numeric(precision=15, scale=6), nullable=True)
    sum_value = Column(Numeric(precision=15, scale=6), nullable=True)
    
    # Indexes
    __table_args__ = (
        Index('idx_performance_metrics_name_time', 'metric_name', 'timestamp'),
        Index('idx_performance_metrics_type', 'metric_type'),
        Index('idx_performance_metrics_partition', 'date_partition'),
        Index('idx_performance_metrics_timestamp', 'timestamp'),
    )

class SystemHealth(BaseModel):
    """System health monitoring and status."""
    __tablename__ = 'system_health'
    
    component_name = Column(String(100), nullable=False)
    component_type = Column(String(50), nullable=False)  # service, database, queue, etc.
    status = Column(String(20), nullable=False)  # healthy, degraded, unhealthy
    
    # Health metrics
    response_time = Column(Float, nullable=True)  # milliseconds
    cpu_usage = Column(Float, nullable=True)  # percentage
    memory_usage = Column(Float, nullable=True)  # percentage
    disk_usage = Column(Float, nullable=True)  # percentage
    
    # Queue-specific metrics
    queue_size = Column(Integer, nullable=True)
    active_workers = Column(Integer, nullable=True)
    
    # Database-specific metrics
    connection_count = Column(Integer, nullable=True)
    query_performance = Column(Float, nullable=True)
    
    # Additional details
    details = Column(JSONB, nullable=True)
    error_message = Column(Text, nullable=True)
    
    # Indexes
    __table_args__ = (
        Index('idx_system_health_component', 'component_name'),
        Index('idx_system_health_status', 'status'),
        Index('idx_system_health_created', 'created_at'),
    )

class UsageStatistics(BaseModel):
    """Usage statistics and analytics."""
    __tablename__ = 'usage_statistics'
    
    user_id = Column(UUID(as_uuid=True), ForeignKey('users.id'), nullable=True)
    statistic_type = Column(String(50), nullable=False)  # daily_uploads, api_calls, etc.
    statistic_date = Column(String(10), nullable=False)  # YYYY-MM-DD
    
    # Counters
    document_uploads = Column(Integer, default=0, nullable=False)
    document_downloads = Column(Integer, default=0, nullable=False)
    api_calls = Column(Integer, default=0, nullable=False)
    processing_time_total = Column(Float, default=0, nullable=False)  # seconds
    
    # Data volumes
    bytes_uploaded = Column(Integer, default=0, nullable=False)
    bytes_downloaded = Column(Integer, default=0, nullable=False)
    
    # Success rates
    successful_processes = Column(Integer, default=0, nullable=False)
    failed_processes = Column(Integer, default=0, nullable=False)
    
    # Additional metrics
    unique_document_types = Column(ARRAY(String), nullable=True)
    peak_concurrent_users = Column(Integer, nullable=True)
    
    # Relationships
    user = relationship("User")
    
    # Indexes
    __table_args__ = (
        Index('idx_usage_statistics_user_date', 'user_id', 'statistic_date'),
        Index('idx_usage_statistics_type_date', 'statistic_type', 'statistic_date'),
        UniqueConstraint('user_id', 'statistic_type', 'statistic_date', name='uq_user_stat_date'),
    )

# Configuration and Settings Tables
class SystemConfiguration(BaseModel):
    """System configuration and settings."""
    __tablename__ = 'system_configurations'
    
    config_key = Column(String(100), unique=True, nullable=False, index=True)
    config_value = Column(Text, nullable=True)
    config_type = Column(String(20), nullable=False)  # string, integer, boolean, json
    description = Column(Text, nullable=True)
    is_sensitive = Column(Boolean, default=False, nullable=False)
    requires_restart = Column(Boolean, default=False, nullable=False)
    
    # Validation
    validation_rule = Column(String(200), nullable=True)
    default_value = Column(Text, nullable=True)
    
    # Indexes
    __table_args__ = (
        Index('idx_system_config_key', 'config_key'),
    )

# Notification and Alert Tables
class NotificationTemplate(BaseModel):
    """Email and notification templates."""
    __tablename__ = 'notification_templates'
    
    template_name = Column(String(100), unique=True, nullable=False, index=True)
    template_type = Column(String(50), nullable=False)  # email, sms, push, webhook
    subject = Column(String(200), nullable=True)
    body_template = Column(Text, nullable=False)
    variables = Column(ARRAY(String), nullable=True)
    is_html = Column(Boolean, default=False, nullable=False)
    
    # Indexes
    __table_args__ = (
        Index('idx_notification_templates_name', 'template_name'),
        Index('idx_notification_templates_type', 'template_type'),
    )

class UserNotification(BaseModel):
    """User notifications and alerts."""
    __tablename__ = 'user_notifications'
    
    user_id = Column(UUID(as_uuid=True), ForeignKey('users.id'), nullable=False)
    notification_type = Column(String(50), nullable=False)
    title = Column(String(200), nullable=False)
    message = Column(Text, nullable=False)
    
    # Status tracking
    is_read = Column(Boolean, default=False, nullable=False)
    read_at = Column(DateTime(timezone=True), nullable=True)
    
    # Delivery tracking
    delivery_method = Column(String(20), nullable=True)  # email, in_app, sms
    delivered_at = Column(DateTime(timezone=True), nullable=True)
    delivery_status = Column(String(20), nullable=True)  # sent, delivered, failed
    
    # Priority and expiration
    priority = Column(String(20), default='normal', nullable=False)
    expires_at = Column(DateTime(timezone=True), nullable=True)
    
    # Additional data
    metadata = Column(JSONB, nullable=True)
    
    # Relationships
    user = relationship("User")
    
    # Indexes
    __table_args__ = (
        Index('idx_user_notifications_user_read', 'user_id', 'is_read'),
        Index('idx_user_notifications_created', 'created_at'),
        Index('idx_user_notifications_priority', 'priority'),
    )

# Main function for schema creation
def create_all_tables(engine):
    """Create all database tables."""
    try:
        Base.metadata.create_all(engine)
        logger.info("All database tables created successfully")
    except Exception as e:
        logger.error(f"Failed to create database tables: {e}")
        raise

def drop_all_tables(engine):
    """Drop all database tables (use with caution!)."""
    try:
        Base.metadata.drop_all(engine)
        logger.info("All database tables dropped")
    except Exception as e:
        logger.error(f"Failed to drop database tables: {e}")
        raise

# Main function for standalone testing
if __name__ == "__main__":
    from sqlalchemy import create_engine
    from sqlalchemy.orm import sessionmaker
    
    def test_schema():
        """Test database schema creation."""
        # Create in-memory SQLite database for testing
        engine = create_engine("sqlite:///:memory:", echo=True)
        
        # Create all tables
        create_all_tables(engine)
        
        # Create session
        Session = sessionmaker(bind=engine)
        session = Session()
        
        try:
            # Test creating a user
            user = User(
                username="testuser",
                email="test@example.com",
                password_hash="hashed_password",
                full_name="Test User",
                role=UserRole.USER.value
            )
            session.add(user)
            session.commit()
            
            # Test creating a document
            document = Document(
                filename="test.pdf",
                original_filename="test_document.pdf",
                file_path="/uploads/test.pdf",
                file_size=1024,
                file_hash="abc123",
                mime_type="application/pdf",
                document_type=DocumentType.PDF.value,
                owner_id=user.id
            )
            session.add(document)
            session.commit()
            
            # Test creating extracted data
            extracted_data = ExtractedData(
                document_id=document.id,
                field_name="title",
                field_type="text",
                field_value="Test Document Title",
                confidence_score=0.95
            )
            session.add(extracted_data)
            session.commit()
            
            print("Schema test completed successfully!")
            print(f"Created user: {user.username}")
            print(f"Created document: {document.filename}")
            print(f"Created extracted data: {extracted_data.field_name}")
            
        except Exception as e:
            print(f"Schema test failed: {e}")
            session.rollback()
        finally:
            session.close()
    
    test_schema()