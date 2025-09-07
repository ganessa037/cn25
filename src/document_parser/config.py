#!/usr/bin/env python3
"""
Configuration Module

Centralized configuration management for the document parser system.
Handles settings for OCR engines, model parameters, API configuration,
and environment-specific settings.
"""

import os
import json
import logging
from pathlib import Path
from typing import Dict, List, Any, Optional, Union
from dataclasses import dataclass, field
from enum import Enum

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class Environment(Enum):
    """Environment types."""
    DEVELOPMENT = "development"
    TESTING = "testing"
    STAGING = "staging"
    PRODUCTION = "production"

class LogLevel(Enum):
    """Logging levels."""
    DEBUG = "DEBUG"
    INFO = "INFO"
    WARNING = "WARNING"
    ERROR = "ERROR"
    CRITICAL = "CRITICAL"

@dataclass
class DatabaseConfig:
    """Database configuration."""
    host: str = "localhost"
    port: int = 5432
    database: str = "document_parser"
    username: str = "postgres"
    password: str = ""
    pool_size: int = 10
    max_overflow: int = 20
    pool_timeout: int = 30
    pool_recycle: int = 3600
    echo: bool = False
    
    @property
    def url(self) -> str:
        """Get database URL."""
        if self.password:
            return f"postgresql://{self.username}:{self.password}@{self.host}:{self.port}/{self.database}"
        else:
            return f"postgresql://{self.username}@{self.host}:{self.port}/{self.database}"

@dataclass
class RedisConfig:
    """Redis configuration."""
    host: str = "localhost"
    port: int = 6379
    database: int = 0
    password: Optional[str] = None
    socket_timeout: int = 30
    socket_connect_timeout: int = 30
    socket_keepalive: bool = True
    socket_keepalive_options: Dict = field(default_factory=dict)
    connection_pool_max_connections: int = 50
    
    @property
    def url(self) -> str:
        """Get Redis URL."""
        if self.password:
            return f"redis://:{self.password}@{self.host}:{self.port}/{self.database}"
        else:
            return f"redis://{self.host}:{self.port}/{self.database}"

@dataclass
class OCRConfig:
    """OCR engine configuration."""
    default_engine: str = "tesseract"
    tesseract_config: str = "--psm 6"
    tesseract_languages: List[str] = field(default_factory=lambda: ["eng", "msa"])
    easyocr_languages: List[str] = field(default_factory=lambda: ["en", "ms"])
    easyocr_gpu: bool = False
    confidence_threshold: float = 0.6
    preprocessing_enabled: bool = True
    max_image_size: int = 2048  # pixels
    supported_formats: List[str] = field(default_factory=lambda: ["jpg", "jpeg", "png", "tiff", "bmp"])
    
    def get_engine_config(self, engine: str) -> Dict[str, Any]:
        """Get configuration for specific OCR engine."""
        if engine == "tesseract":
            return {
                "config": self.tesseract_config,
                "languages": self.tesseract_languages
            }
        elif engine == "easyocr":
            return {
                "languages": self.easyocr_languages,
                "gpu": self.easyocr_gpu
            }
        else:
            raise ValueError(f"Unsupported OCR engine: {engine}")

@dataclass
class ModelConfig:
    """Model configuration."""
    models_directory: str = "./models"
    classification_model_name: str = "document_classifier"
    field_extraction_model_name: str = "field_extractor"
    ocr_model_name: str = "ocr_engine"
    auto_load_models: bool = True
    model_cache_enabled: bool = True
    model_cache_ttl: int = 3600  # seconds
    confidence_thresholds: Dict[str, float] = field(default_factory=lambda: {
        "classification": 0.8,
        "field_extraction": 0.7,
        "ocr": 0.6
    })
    
    @property
    def models_path(self) -> Path:
        """Get models directory path."""
        return Path(self.models_directory).resolve()

@dataclass
class APIConfig:
    """API configuration."""
    host: str = "0.0.0.0"
    port: int = 8000
    debug: bool = False
    reload: bool = False
    workers: int = 1
    max_request_size: int = 16 * 1024 * 1024  # 16MB
    request_timeout: int = 300  # seconds
    cors_enabled: bool = True
    cors_origins: List[str] = field(default_factory=lambda: ["*"])
    cors_methods: List[str] = field(default_factory=lambda: ["GET", "POST", "PUT", "DELETE"])
    cors_headers: List[str] = field(default_factory=lambda: ["*"])
    rate_limiting_enabled: bool = True
    rate_limit_requests: int = 100
    rate_limit_window: int = 60  # seconds
    
    @property
    def server_url(self) -> str:
        """Get server URL."""
        return f"http://{self.host}:{self.port}"

@dataclass
class SecurityConfig:
    """Security configuration."""
    secret_key: str = "your-secret-key-change-in-production"
    algorithm: str = "HS256"
    access_token_expire_minutes: int = 30
    refresh_token_expire_days: int = 7
    password_min_length: int = 8
    max_login_attempts: int = 5
    lockout_duration_minutes: int = 15
    require_https: bool = False
    api_key_header: str = "X-API-Key"
    
    def generate_secret_key(self) -> str:
        """Generate a new secret key."""
        import secrets
        return secrets.token_urlsafe(32)

@dataclass
class LoggingConfig:
    """Logging configuration."""
    level: LogLevel = LogLevel.INFO
    format: str = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    date_format: str = "%Y-%m-%d %H:%M:%S"
    file_enabled: bool = True
    file_path: str = "./logs/document_parser.log"
    file_max_size: int = 10 * 1024 * 1024  # 10MB
    file_backup_count: int = 5
    console_enabled: bool = True
    json_format: bool = False
    
    @property
    def log_file_path(self) -> Path:
        """Get log file path."""
        return Path(self.file_path).resolve()

@dataclass
class ProcessingConfig:
    """Document processing configuration."""
    max_concurrent_jobs: int = 10
    job_timeout_seconds: int = 300
    retry_attempts: int = 3
    retry_delay_seconds: int = 5
    temp_directory: str = "./temp"
    cleanup_temp_files: bool = True
    max_file_size_mb: int = 50
    supported_mime_types: List[str] = field(default_factory=lambda: [
        "image/jpeg", "image/png", "image/tiff", "image/bmp",
        "application/pdf", "text/plain"
    ])
    
    @property
    def temp_path(self) -> Path:
        """Get temporary directory path."""
        return Path(self.temp_directory).resolve()

@dataclass
class MonitoringConfig:
    """Monitoring and metrics configuration."""
    enabled: bool = True
    metrics_endpoint: str = "/metrics"
    health_endpoint: str = "/health"
    prometheus_enabled: bool = False
    prometheus_port: int = 9090
    collect_detailed_metrics: bool = True
    performance_tracking: bool = True
    error_tracking: bool = True
    
class Config:
    """Main configuration class."""
    
    def __init__(self, config_file: Optional[Union[str, Path]] = None, environment: Optional[str] = None):
        self.environment = Environment(environment or os.getenv("ENVIRONMENT", "development"))
        self.config_file = Path(config_file) if config_file else None
        
        # Initialize configuration sections
        self.database = DatabaseConfig()
        self.redis = RedisConfig()
        self.ocr = OCRConfig()
        self.models = ModelConfig()
        self.api = APIConfig()
        self.security = SecurityConfig()
        self.logging = LoggingConfig()
        self.processing = ProcessingConfig()
        self.monitoring = MonitoringConfig()
        
        # Load configuration
        self._load_from_environment()
        if self.config_file and self.config_file.exists():
            self._load_from_file()
        
        # Apply environment-specific overrides
        self._apply_environment_overrides()
        
        # Validate configuration
        self._validate_config()
        
        logger.info(f"Configuration loaded for {self.environment.value} environment")
    
    def _load_from_environment(self):
        """Load configuration from environment variables."""
        # Database
        self.database.host = os.getenv("DB_HOST", self.database.host)
        self.database.port = int(os.getenv("DB_PORT", self.database.port))
        self.database.database = os.getenv("DB_NAME", self.database.database)
        self.database.username = os.getenv("DB_USER", self.database.username)
        self.database.password = os.getenv("DB_PASSWORD", self.database.password)
        
        # Redis
        self.redis.host = os.getenv("REDIS_HOST", self.redis.host)
        self.redis.port = int(os.getenv("REDIS_PORT", self.redis.port))
        self.redis.password = os.getenv("REDIS_PASSWORD", self.redis.password)
        
        # API
        self.api.host = os.getenv("API_HOST", self.api.host)
        self.api.port = int(os.getenv("API_PORT", self.api.port))
        self.api.debug = os.getenv("API_DEBUG", "false").lower() == "true"
        
        # Security
        self.security.secret_key = os.getenv("SECRET_KEY", self.security.secret_key)
        
        # Models
        self.models.models_directory = os.getenv("MODELS_DIR", self.models.models_directory)
        
        # Logging
        log_level = os.getenv("LOG_LEVEL", self.logging.level.value)
        try:
            self.logging.level = LogLevel(log_level.upper())
        except ValueError:
            logger.warning(f"Invalid log level '{log_level}', using default")
    
    def _load_from_file(self):
        """Load configuration from JSON file."""
        try:
            with open(self.config_file, 'r') as f:
                config_data = json.load(f)
            
            # Update configuration sections
            for section_name, section_data in config_data.items():
                if hasattr(self, section_name):
                    section = getattr(self, section_name)
                    for key, value in section_data.items():
                        if hasattr(section, key):
                            setattr(section, key, value)
            
            logger.info(f"Configuration loaded from {self.config_file}")
            
        except Exception as e:
            logger.error(f"Failed to load configuration from {self.config_file}: {e}")
    
    def _apply_environment_overrides(self):
        """Apply environment-specific configuration overrides."""
        if self.environment == Environment.DEVELOPMENT:
            self.api.debug = True
            self.api.reload = True
            self.logging.level = LogLevel.DEBUG
            self.logging.console_enabled = True
            
        elif self.environment == Environment.TESTING:
            self.database.database = f"{self.database.database}_test"
            self.redis.database = 1
            self.logging.level = LogLevel.WARNING
            self.logging.file_enabled = False
            
        elif self.environment == Environment.PRODUCTION:
            self.api.debug = False
            self.api.reload = False
            self.security.require_https = True
            self.logging.level = LogLevel.INFO
            self.monitoring.enabled = True
            self.monitoring.prometheus_enabled = True
    
    def _validate_config(self):
        """Validate configuration settings."""
        errors = []
        
        # Validate required settings
        if not self.security.secret_key or self.security.secret_key == "your-secret-key-change-in-production":
            if self.environment == Environment.PRODUCTION:
                errors.append("Secret key must be set for production environment")
        
        # Validate paths
        try:
            self.models.models_path.mkdir(parents=True, exist_ok=True)
        except Exception as e:
            errors.append(f"Cannot create models directory: {e}")
        
        try:
            self.processing.temp_path.mkdir(parents=True, exist_ok=True)
        except Exception as e:
            errors.append(f"Cannot create temp directory: {e}")
        
        try:
            self.logging.log_file_path.parent.mkdir(parents=True, exist_ok=True)
        except Exception as e:
            errors.append(f"Cannot create logs directory: {e}")
        
        # Validate numeric ranges
        if self.api.port < 1 or self.api.port > 65535:
            errors.append(f"Invalid API port: {self.api.port}")
        
        if self.database.port < 1 or self.database.port > 65535:
            errors.append(f"Invalid database port: {self.database.port}")
        
        if self.redis.port < 1 or self.redis.port > 65535:
            errors.append(f"Invalid Redis port: {self.redis.port}")
        
        if errors:
            error_msg = "Configuration validation failed:\n" + "\n".join(f"  - {error}" for error in errors)
            raise ValueError(error_msg)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert configuration to dictionary."""
        return {
            "environment": self.environment.value,
            "database": {
                "host": self.database.host,
                "port": self.database.port,
                "database": self.database.database,
                "username": self.database.username,
                "pool_size": self.database.pool_size,
                "max_overflow": self.database.max_overflow,
                "pool_timeout": self.database.pool_timeout,
                "pool_recycle": self.database.pool_recycle,
                "echo": self.database.echo
            },
            "redis": {
                "host": self.redis.host,
                "port": self.redis.port,
                "database": self.redis.database,
                "socket_timeout": self.redis.socket_timeout,
                "connection_pool_max_connections": self.redis.connection_pool_max_connections
            },
            "ocr": {
                "default_engine": self.ocr.default_engine,
                "tesseract_config": self.ocr.tesseract_config,
                "tesseract_languages": self.ocr.tesseract_languages,
                "easyocr_languages": self.ocr.easyocr_languages,
                "easyocr_gpu": self.ocr.easyocr_gpu,
                "confidence_threshold": self.ocr.confidence_threshold,
                "preprocessing_enabled": self.ocr.preprocessing_enabled,
                "max_image_size": self.ocr.max_image_size,
                "supported_formats": self.ocr.supported_formats
            },
            "models": {
                "models_directory": self.models.models_directory,
                "classification_model_name": self.models.classification_model_name,
                "field_extraction_model_name": self.models.field_extraction_model_name,
                "ocr_model_name": self.models.ocr_model_name,
                "auto_load_models": self.models.auto_load_models,
                "model_cache_enabled": self.models.model_cache_enabled,
                "model_cache_ttl": self.models.model_cache_ttl,
                "confidence_thresholds": self.models.confidence_thresholds
            },
            "api": {
                "host": self.api.host,
                "port": self.api.port,
                "debug": self.api.debug,
                "reload": self.api.reload,
                "workers": self.api.workers,
                "max_request_size": self.api.max_request_size,
                "request_timeout": self.api.request_timeout,
                "cors_enabled": self.api.cors_enabled,
                "rate_limiting_enabled": self.api.rate_limiting_enabled,
                "rate_limit_requests": self.api.rate_limit_requests,
                "rate_limit_window": self.api.rate_limit_window
            },
            "security": {
                "algorithm": self.security.algorithm,
                "access_token_expire_minutes": self.security.access_token_expire_minutes,
                "refresh_token_expire_days": self.security.refresh_token_expire_days,
                "password_min_length": self.security.password_min_length,
                "max_login_attempts": self.security.max_login_attempts,
                "lockout_duration_minutes": self.security.lockout_duration_minutes,
                "require_https": self.security.require_https,
                "api_key_header": self.security.api_key_header
            },
            "logging": {
                "level": self.logging.level.value,
                "format": self.logging.format,
                "date_format": self.logging.date_format,
                "file_enabled": self.logging.file_enabled,
                "file_path": self.logging.file_path,
                "file_max_size": self.logging.file_max_size,
                "file_backup_count": self.logging.file_backup_count,
                "console_enabled": self.logging.console_enabled,
                "json_format": self.logging.json_format
            },
            "processing": {
                "max_concurrent_jobs": self.processing.max_concurrent_jobs,
                "job_timeout_seconds": self.processing.job_timeout_seconds,
                "retry_attempts": self.processing.retry_attempts,
                "retry_delay_seconds": self.processing.retry_delay_seconds,
                "temp_directory": self.processing.temp_directory,
                "cleanup_temp_files": self.processing.cleanup_temp_files,
                "max_file_size_mb": self.processing.max_file_size_mb,
                "supported_mime_types": self.processing.supported_mime_types
            },
            "monitoring": {
                "enabled": self.monitoring.enabled,
                "metrics_endpoint": self.monitoring.metrics_endpoint,
                "health_endpoint": self.monitoring.health_endpoint,
                "prometheus_enabled": self.monitoring.prometheus_enabled,
                "prometheus_port": self.monitoring.prometheus_port,
                "collect_detailed_metrics": self.monitoring.collect_detailed_metrics,
                "performance_tracking": self.monitoring.performance_tracking,
                "error_tracking": self.monitoring.error_tracking
            }
        }
    
    def save_to_file(self, file_path: Union[str, Path]):
        """Save configuration to JSON file."""
        file_path = Path(file_path)
        file_path.parent.mkdir(parents=True, exist_ok=True)
        
        config_dict = self.to_dict()
        
        with open(file_path, 'w') as f:
            json.dump(config_dict, f, indent=2)
        
        logger.info(f"Configuration saved to {file_path}")
    
    def get_database_url(self) -> str:
        """Get database connection URL."""
        return self.database.url
    
    def get_redis_url(self) -> str:
        """Get Redis connection URL."""
        return self.redis.url
    
    def is_development(self) -> bool:
        """Check if running in development environment."""
        return self.environment == Environment.DEVELOPMENT
    
    def is_production(self) -> bool:
        """Check if running in production environment."""
        return self.environment == Environment.PRODUCTION
    
    def is_testing(self) -> bool:
        """Check if running in testing environment."""
        return self.environment == Environment.TESTING

# Global configuration instance
_config: Optional[Config] = None

def get_config() -> Config:
    """Get global configuration instance."""
    global _config
    if _config is None:
        _config = Config()
    return _config

def init_config(config_file: Optional[Union[str, Path]] = None, environment: Optional[str] = None) -> Config:
    """Initialize global configuration."""
    global _config
    _config = Config(config_file=config_file, environment=environment)
    return _config

def reload_config() -> Config:
    """Reload global configuration."""
    global _config
    if _config:
        config_file = _config.config_file
        environment = _config.environment.value
        _config = Config(config_file=config_file, environment=environment)
    else:
        _config = Config()
    return _config

# Configuration utilities
def create_default_config_file(file_path: Union[str, Path]):
    """Create a default configuration file."""
    config = Config()
    config.save_to_file(file_path)
    logger.info(f"Default configuration file created at {file_path}")

def validate_config_file(file_path: Union[str, Path]) -> bool:
    """Validate a configuration file."""
    try:
        Config(config_file=file_path)
        logger.info(f"Configuration file {file_path} is valid")
        return True
    except Exception as e:
        logger.error(f"Configuration file {file_path} is invalid: {e}")
        return False