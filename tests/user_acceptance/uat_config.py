#!/usr/bin/env python3
"""
User Acceptance Testing - Configuration Management

Centralized configuration management for UAT components including test settings,
thresholds, and environment configurations for the Malaysian document parser.
"""

import os
import json
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, field, asdict
from datetime import datetime, timedelta


@dataclass
class DocumentTypeConfig:
    """Configuration for specific document types."""
    name: str
    min_accuracy_threshold: float = 0.85
    confidence_threshold: float = 0.7
    required_fields: List[str] = field(default_factory=list)
    optional_fields: List[str] = field(default_factory=list)
    validation_rules: Dict[str, Any] = field(default_factory=dict)
    edge_case_scenarios: List[str] = field(default_factory=list)
    performance_requirements: Dict[str, float] = field(default_factory=dict)


@dataclass
class TestEnvironmentConfig:
    """Test environment configuration."""
    # Environment settings
    environment_name: str = "uat"
    api_base_url: str = "http://localhost:8000"
    api_timeout_seconds: int = 30
    api_retry_attempts: int = 3
    api_retry_delay_seconds: float = 1.0
    
    # Database settings
    database_url: str = "sqlite:///uat_test.db"
    database_pool_size: int = 10
    database_timeout_seconds: int = 30
    
    # File system settings
    temp_directory: str = "/tmp/uat_temp"
    upload_directory: str = "uploads/uat"
    max_file_size_mb: int = 50
    allowed_file_extensions: List[str] = field(default_factory=lambda: ['.jpg', '.jpeg', '.png', '.pdf'])
    
    # Security settings
    enable_ssl_verification: bool = False
    api_key: Optional[str] = None
    auth_token: Optional[str] = None
    
    # Logging settings
    log_level: str = "INFO"
    log_format: str = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    log_file_max_size_mb: int = 100
    log_file_backup_count: int = 5


@dataclass
class PerformanceConfig:
    """Performance testing configuration."""
    # Processing time thresholds
    max_processing_time_seconds: float = 10.0
    average_processing_time_threshold: float = 5.0
    
    # Memory usage thresholds
    max_memory_usage_mb: float = 512.0
    average_memory_usage_threshold_mb: float = 256.0
    
    # Throughput requirements
    min_throughput_docs_per_minute: int = 60
    target_throughput_docs_per_minute: int = 120
    
    # Concurrent processing
    max_concurrent_requests: int = 10
    concurrent_processing_timeout_seconds: int = 60
    
    # Load testing parameters
    load_test_duration_minutes: int = 10
    load_test_ramp_up_seconds: int = 30
    load_test_users: int = 5
    
    # Stress testing parameters
    stress_test_duration_minutes: int = 5
    stress_test_max_users: int = 20
    stress_test_failure_threshold_percent: float = 5.0
    
    # Resource monitoring
    cpu_usage_threshold_percent: float = 80.0
    memory_leak_detection_enabled: bool = True
    memory_leak_threshold_mb: float = 50.0


@dataclass
class AccuracyConfig:
    """Accuracy testing configuration."""
    # Overall accuracy requirements
    minimum_overall_accuracy: float = 0.85
    target_overall_accuracy: float = 0.95
    
    # Field-specific accuracy requirements
    critical_field_accuracy: float = 0.98  # Name, IC number, etc.
    important_field_accuracy: float = 0.95  # Address, date of birth, etc.
    standard_field_accuracy: float = 0.90   # Other fields
    
    # Confidence score requirements
    minimum_confidence_score: float = 0.7
    high_confidence_threshold: float = 0.9
    
    # Validation requirements
    require_manual_validation: bool = True
    min_validators_per_document: int = 2
    validator_agreement_threshold: float = 0.8
    
    # Error tolerance
    max_false_positive_rate: float = 0.05
    max_false_negative_rate: float = 0.03
    
    # Quality metrics
    image_quality_threshold: float = 0.7
    text_clarity_threshold: float = 0.8
    document_completeness_threshold: float = 0.95


@dataclass
class FeedbackConfig:
    """User feedback configuration."""
    # Feedback collection settings
    collection_enabled: bool = True
    web_interface_port: int = 5001
    session_duration_hours: int = 24
    auto_save_interval_minutes: int = 5
    
    # Feedback categories
    satisfaction_categories: List[str] = field(default_factory=lambda: [
        'accuracy', 'speed', 'usability', 'reliability', 'overall'
    ])
    
    # Rating scales
    rating_scale_min: int = 1
    rating_scale_max: int = 5
    satisfaction_threshold: float = 3.5
    
    # Feedback analysis
    sentiment_analysis_enabled: bool = True
    keyword_extraction_enabled: bool = True
    trend_analysis_enabled: bool = True
    
    # Notification settings
    email_notifications_enabled: bool = False
    notification_email: Optional[str] = None
    critical_feedback_threshold: float = 2.0
    
    # Data retention
    feedback_retention_days: int = 365
    anonymous_feedback_allowed: bool = True


@dataclass
class EdgeCaseConfig:
    """Edge case testing configuration."""
    # Image quality scenarios
    poor_quality_scenarios: List[str] = field(default_factory=lambda: [
        'low_resolution', 'high_noise', 'poor_lighting', 'motion_blur',
        'out_of_focus', 'overexposed', 'underexposed'
    ])
    
    # Document condition scenarios
    damaged_document_scenarios: List[str] = field(default_factory=lambda: [
        'torn_edges', 'water_damage', 'faded_text', 'stains',
        'creases', 'holes', 'partial_document'
    ])
    
    # Orientation scenarios
    orientation_scenarios: List[str] = field(default_factory=lambda: [
        'rotated_90', 'rotated_180', 'rotated_270', 'skewed',
        'upside_down', 'tilted'
    ])
    
    # Format scenarios
    format_scenarios: List[str] = field(default_factory=lambda: [
        'different_sizes', 'non_standard_format', 'multiple_pages',
        'embedded_in_larger_image', 'cropped'
    ])
    
    # Success rate thresholds for edge cases
    edge_case_success_thresholds: Dict[str, float] = field(default_factory=lambda: {
        'poor_quality': 0.6,
        'damaged_document': 0.5,
        'orientation': 0.8,
        'format_variations': 0.7
    })
    
    # Edge case generation settings
    generate_synthetic_edge_cases: bool = True
    synthetic_case_count_per_type: int = 10
    edge_case_severity_levels: List[str] = field(default_factory=lambda: [
        'mild', 'moderate', 'severe'
    ])


@dataclass
class ReportingConfig:
    """Reporting configuration."""
    # Report generation settings
    generate_html_report: bool = True
    generate_pdf_report: bool = False
    generate_json_report: bool = True
    generate_csv_export: bool = True
    
    # Report content settings
    include_detailed_metrics: bool = True
    include_performance_charts: bool = True
    include_accuracy_breakdown: bool = True
    include_edge_case_analysis: bool = True
    include_feedback_summary: bool = True
    include_recommendations: bool = True
    
    # Report styling
    report_template: str = "default"
    company_logo_path: Optional[str] = None
    report_title: str = "User Acceptance Test Report"
    report_subtitle: str = "Malaysian Document Parser"
    
    # Report distribution
    email_reports: bool = False
    email_recipients: List[str] = field(default_factory=list)
    upload_to_cloud: bool = False
    cloud_storage_path: Optional[str] = None
    
    # Report retention
    report_retention_days: int = 90
    archive_old_reports: bool = True
    archive_compression: bool = True


@dataclass
class UATMasterConfig:
    """Master UAT configuration containing all sub-configurations."""
    # Meta information
    config_version: str = "1.0.0"
    created_at: str = field(default_factory=lambda: datetime.now().isoformat())
    updated_at: str = field(default_factory=lambda: datetime.now().isoformat())
    
    # Sub-configurations
    environment: TestEnvironmentConfig = field(default_factory=TestEnvironmentConfig)
    performance: PerformanceConfig = field(default_factory=PerformanceConfig)
    accuracy: AccuracyConfig = field(default_factory=AccuracyConfig)
    feedback: FeedbackConfig = field(default_factory=FeedbackConfig)
    edge_cases: EdgeCaseConfig = field(default_factory=EdgeCaseConfig)
    reporting: ReportingConfig = field(default_factory=ReportingConfig)
    
    # Document type configurations
    document_types: Dict[str, DocumentTypeConfig] = field(default_factory=dict)
    
    # Test execution settings
    parallel_workers: int = 4
    test_timeout_minutes: int = 60
    retry_failed_tests: bool = True
    max_retry_attempts: int = 3
    
    # Data management
    test_data_directory: str = "test_data/real_documents"
    output_directory: str = "uat_results"
    backup_results: bool = True
    cleanup_temp_files: bool = True
    
    # Notification settings
    notifications_enabled: bool = True
    notification_channels: List[str] = field(default_factory=lambda: ['email', 'slack'])
    
    def __post_init__(self):
        """Initialize default document type configurations."""
        if not self.document_types:
            self.document_types = self._create_default_document_configs()
    
    def _create_default_document_configs(self) -> Dict[str, DocumentTypeConfig]:
        """Create default document type configurations."""
        configs = {}
        
        # MyKad configuration
        configs['mykad'] = DocumentTypeConfig(
            name='mykad',
            min_accuracy_threshold=0.90,
            confidence_threshold=0.75,
            required_fields=[
                'ic_number', 'name', 'date_of_birth', 'place_of_birth',
                'address', 'religion', 'race', 'gender'
            ],
            optional_fields=['occupation', 'sector', 'citizenship'],
            validation_rules={
                'ic_number': {'pattern': r'^\d{6}-\d{2}-\d{4}$', 'length': 14},
                'name': {'min_length': 2, 'max_length': 100},
                'date_of_birth': {'format': 'DD/MM/YYYY'},
                'gender': {'values': ['LELAKI', 'PEREMPUAN']}
            },
            edge_case_scenarios=[
                'worn_card', 'reflective_surface', 'poor_lighting',
                'partial_occlusion', 'damaged_corners'
            ],
            performance_requirements={
                'max_processing_time': 8.0,
                'min_confidence': 0.75,
                'max_memory_usage': 256.0
            }
        )
        
        # SPK configuration
        configs['spk'] = DocumentTypeConfig(
            name='spk',
            min_accuracy_threshold=0.88,
            confidence_threshold=0.70,
            required_fields=[
                'certificate_number', 'student_name', 'ic_number',
                'school_name', 'year', 'subjects', 'grades'
            ],
            optional_fields=['remarks', 'principal_signature'],
            validation_rules={
                'certificate_number': {'pattern': r'^[A-Z0-9]{8,12}$'},
                'student_name': {'min_length': 2, 'max_length': 100},
                'ic_number': {'pattern': r'^\d{6}-\d{2}-\d{4}$'},
                'year': {'min_value': 1990, 'max_value': 2030},
                'grades': {'values': ['A+', 'A', 'A-', 'B+', 'B', 'C+', 'C', 'D', 'E', 'G']}
            },
            edge_case_scenarios=[
                'multiple_pages', 'table_format', 'handwritten_sections',
                'seal_overlays', 'watermarks'
            ],
            performance_requirements={
                'max_processing_time': 12.0,
                'min_confidence': 0.70,
                'max_memory_usage': 384.0
            }
        )
        
        return configs
    
    def update_timestamp(self):
        """Update the configuration timestamp."""
        self.updated_at = datetime.now().isoformat()
    
    def validate_config(self) -> List[str]:
        """Validate configuration and return list of issues."""
        issues = []
        
        # Validate thresholds
        if self.accuracy.minimum_overall_accuracy > self.accuracy.target_overall_accuracy:
            issues.append("Minimum accuracy cannot be higher than target accuracy")
        
        if self.performance.max_processing_time_seconds <= 0:
            issues.append("Max processing time must be positive")
        
        if self.performance.max_memory_usage_mb <= 0:
            issues.append("Max memory usage must be positive")
        
        # Validate document type configs
        for doc_type, config in self.document_types.items():
            if config.min_accuracy_threshold > 1.0 or config.min_accuracy_threshold < 0.0:
                issues.append(f"Invalid accuracy threshold for {doc_type}")
            
            if config.confidence_threshold > 1.0 or config.confidence_threshold < 0.0:
                issues.append(f"Invalid confidence threshold for {doc_type}")
        
        # Validate directories
        if not self.test_data_directory:
            issues.append("Test data directory must be specified")
        
        if not self.output_directory:
            issues.append("Output directory must be specified")
        
        # Validate worker count
        if self.parallel_workers <= 0:
            issues.append("Parallel workers must be positive")
        
        # Validate timeout
        if self.test_timeout_minutes <= 0:
            issues.append("Test timeout must be positive")
        
        return issues
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert configuration to dictionary."""
        return asdict(self)
    
    def to_json(self, indent: int = 2) -> str:
        """Convert configuration to JSON string."""
        return json.dumps(self.to_dict(), indent=indent, ensure_ascii=False, default=str)
    
    def save_to_file(self, file_path: str):
        """Save configuration to file."""
        self.update_timestamp()
        
        with open(file_path, 'w', encoding='utf-8') as f:
            f.write(self.to_json())
    
    @classmethod
    def load_from_file(cls, file_path: str) -> 'UATMasterConfig':
        """Load configuration from file."""
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        return cls.from_dict(data)
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'UATMasterConfig':
        """Create configuration from dictionary."""
        # Extract sub-configurations
        environment_data = data.get('environment', {})
        performance_data = data.get('performance', {})
        accuracy_data = data.get('accuracy', {})
        feedback_data = data.get('feedback', {})
        edge_cases_data = data.get('edge_cases', {})
        reporting_data = data.get('reporting', {})
        document_types_data = data.get('document_types', {})
        
        # Create sub-configuration objects
        environment = TestEnvironmentConfig(**environment_data)
        performance = PerformanceConfig(**performance_data)
        accuracy = AccuracyConfig(**accuracy_data)
        feedback = FeedbackConfig(**feedback_data)
        edge_cases = EdgeCaseConfig(**edge_cases_data)
        reporting = ReportingConfig(**reporting_data)
        
        # Create document type configurations
        document_types = {}
        for doc_type, doc_data in document_types_data.items():
            document_types[doc_type] = DocumentTypeConfig(**doc_data)
        
        # Create main configuration
        config = cls(
            config_version=data.get('config_version', '1.0.0'),
            created_at=data.get('created_at', datetime.now().isoformat()),
            updated_at=data.get('updated_at', datetime.now().isoformat()),
            environment=environment,
            performance=performance,
            accuracy=accuracy,
            feedback=feedback,
            edge_cases=edge_cases,
            reporting=reporting,
            document_types=document_types,
            parallel_workers=data.get('parallel_workers', 4),
            test_timeout_minutes=data.get('test_timeout_minutes', 60),
            retry_failed_tests=data.get('retry_failed_tests', True),
            max_retry_attempts=data.get('max_retry_attempts', 3),
            test_data_directory=data.get('test_data_directory', 'test_data/real_documents'),
            output_directory=data.get('output_directory', 'uat_results'),
            backup_results=data.get('backup_results', True),
            cleanup_temp_files=data.get('cleanup_temp_files', True),
            notifications_enabled=data.get('notifications_enabled', True),
            notification_channels=data.get('notification_channels', ['email', 'slack'])
        )
        
        return config


class UATConfigManager:
    """Configuration manager for UAT."""
    
    def __init__(self, config_file: Optional[str] = None):
        self.config_file = config_file or 'uat_config.json'
        self.config = self._load_or_create_config()
    
    def _load_or_create_config(self) -> UATMasterConfig:
        """Load existing configuration or create default."""
        config_path = Path(self.config_file)
        
        if config_path.exists():
            try:
                return UATMasterConfig.load_from_file(str(config_path))
            except Exception as e:
                print(f"Error loading config file {config_path}: {e}")
                print("Creating default configuration")
        
        # Create default configuration
        config = UATMasterConfig()
        self.save_config(config)
        return config
    
    def save_config(self, config: Optional[UATMasterConfig] = None):
        """Save configuration to file."""
        config = config or self.config
        config.save_to_file(self.config_file)
    
    def get_config(self) -> UATMasterConfig:
        """Get current configuration."""
        return self.config
    
    def update_config(self, updates: Dict[str, Any]):
        """Update configuration with new values."""
        # Apply updates to configuration
        for key, value in updates.items():
            if hasattr(self.config, key):
                setattr(self.config, key, value)
        
        self.config.update_timestamp()
        self.save_config()
    
    def validate_config(self) -> List[str]:
        """Validate current configuration."""
        return self.config.validate_config()
    
    def reset_to_defaults(self):
        """Reset configuration to defaults."""
        self.config = UATMasterConfig()
        self.save_config()
    
    def export_config(self, export_path: str):
        """Export configuration to specified path."""
        self.config.save_to_file(export_path)
    
    def import_config(self, import_path: str):
        """Import configuration from specified path."""
        self.config = UATMasterConfig.load_from_file(import_path)
        self.save_config()


# Global configuration manager instance
_config_manager = None


def get_config_manager(config_file: Optional[str] = None) -> UATConfigManager:
    """Get global configuration manager instance."""
    global _config_manager
    
    if _config_manager is None:
        _config_manager = UATConfigManager(config_file)
    
    return _config_manager


def get_uat_config() -> UATMasterConfig:
    """Get current UAT configuration."""
    return get_config_manager().get_config()


def create_sample_config_file(file_path: str = 'uat_config_sample.json'):
    """Create a sample configuration file for reference."""
    config = UATMasterConfig()
    config.save_to_file(file_path)
    print(f"Sample configuration file created: {file_path}")


if __name__ == "__main__":
    # Create sample configuration file when run directly
    create_sample_config_file()
    
    # Validate default configuration
    config = UATMasterConfig()
    issues = config.validate_config()
    
    if issues:
        print("Configuration validation issues:")
        for issue in issues:
            print(f"  - {issue}")
    else:
        print("Configuration validation passed")