#!/usr/bin/env python3
"""
Health Check Routes

FastAPI routes for system health monitoring, readiness checks, and metrics.
Provides comprehensive health status for all system components.
"""

import logging
import asyncio
import psutil
from datetime import datetime, timedelta
from typing import Dict, Any, List, Optional
import platform
import sys
from pathlib import Path

from fastapi import APIRouter, Depends, HTTPException, status, Query
from fastapi.responses import JSONResponse, PlainTextResponse
from pydantic import BaseModel, Field
import redis.asyncio as redis
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import text

from ..dependencies import get_redis_client, get_database_session, get_config
from ...models.document_models import ProcessingStatus

# Configure logging
logger = logging.getLogger(__name__)

# Create router
router = APIRouter()

# Response models

class ComponentHealth(BaseModel):
    """Health status of a system component."""
    name: str = Field(..., description="Component name")
    status: str = Field(..., description="Health status (healthy, degraded, unhealthy)")
    message: str = Field(..., description="Status message")
    response_time_ms: Optional[float] = Field(None, description="Response time in milliseconds")
    last_check: datetime = Field(..., description="Last health check time")
    details: Optional[Dict[str, Any]] = Field(None, description="Additional details")

class SystemHealth(BaseModel):
    """Overall system health status."""
    status: str = Field(..., description="Overall system status")
    timestamp: datetime = Field(..., description="Health check timestamp")
    version: str = Field(..., description="Application version")
    uptime_seconds: float = Field(..., description="System uptime in seconds")
    components: List[ComponentHealth] = Field(..., description="Component health statuses")
    system_info: Dict[str, Any] = Field(..., description="System information")

class ReadinessCheck(BaseModel):
    """Readiness check response."""
    ready: bool = Field(..., description="Whether system is ready to serve requests")
    timestamp: datetime = Field(..., description="Check timestamp")
    checks: List[ComponentHealth] = Field(..., description="Readiness check results")
    message: str = Field(..., description="Overall readiness message")

class LivenessCheck(BaseModel):
    """Liveness check response."""
    alive: bool = Field(..., description="Whether system is alive")
    timestamp: datetime = Field(..., description="Check timestamp")
    uptime_seconds: float = Field(..., description="System uptime")
    message: str = Field(..., description="Liveness message")

class MetricsResponse(BaseModel):
    """System metrics response."""
    timestamp: datetime = Field(..., description="Metrics timestamp")
    system_metrics: Dict[str, Any] = Field(..., description="System resource metrics")
    application_metrics: Dict[str, Any] = Field(..., description="Application-specific metrics")
    performance_metrics: Dict[str, Any] = Field(..., description="Performance metrics")

# Global variables for tracking
start_time = datetime.utcnow()
last_health_check = None
cached_health_status = None
health_cache_duration = timedelta(seconds=30)  # Cache health status for 30 seconds

# Helper functions

async def check_database_health() -> ComponentHealth:
    """Check database connectivity and health."""
    start_time = datetime.utcnow()
    
    try:
        # Get database session
        async with get_database_session() as session:
            # Simple query to test connectivity
            result = await session.execute(text("SELECT 1"))
            await result.fetchone()
            
            # Check database version
            version_result = await session.execute(text("SELECT version()"))
            db_version = await version_result.fetchone()
            
            response_time = (datetime.utcnow() - start_time).total_seconds() * 1000
            
            return ComponentHealth(
                name="database",
                status="healthy",
                message="Database connection successful",
                response_time_ms=response_time,
                last_check=datetime.utcnow(),
                details={
                    "version": str(db_version[0]) if db_version else "unknown",
                    "connection_pool": "active"
                }
            )
            
    except Exception as e:
        response_time = (datetime.utcnow() - start_time).total_seconds() * 1000
        logger.error(f"Database health check failed: {e}")
        
        return ComponentHealth(
            name="database",
            status="unhealthy",
            message=f"Database connection failed: {str(e)}",
            response_time_ms=response_time,
            last_check=datetime.utcnow(),
            details={"error": str(e)}
        )

async def check_redis_health() -> ComponentHealth:
    """Check Redis connectivity and health."""
    start_time = datetime.utcnow()
    
    try:
        # Get Redis client
        redis_client = await get_redis_client()
        
        # Test basic operations
        await redis_client.ping()
        
        # Get Redis info
        redis_info = await redis_client.info()
        
        # Test set/get operation
        test_key = "health_check_test"
        await redis_client.set(test_key, "test_value", ex=60)
        test_value = await redis_client.get(test_key)
        await redis_client.delete(test_key)
        
        if test_value != b"test_value":
            raise Exception("Redis set/get test failed")
        
        response_time = (datetime.utcnow() - start_time).total_seconds() * 1000
        
        return ComponentHealth(
            name="redis",
            status="healthy",
            message="Redis connection and operations successful",
            response_time_ms=response_time,
            last_check=datetime.utcnow(),
            details={
                "version": redis_info.get("redis_version", "unknown"),
                "connected_clients": redis_info.get("connected_clients", 0),
                "used_memory_human": redis_info.get("used_memory_human", "unknown"),
                "uptime_in_seconds": redis_info.get("uptime_in_seconds", 0)
            }
        )
        
    except Exception as e:
        response_time = (datetime.utcnow() - start_time).total_seconds() * 1000
        logger.error(f"Redis health check failed: {e}")
        
        return ComponentHealth(
            name="redis",
            status="unhealthy",
            message=f"Redis connection failed: {str(e)}",
            response_time_ms=response_time,
            last_check=datetime.utcnow(),
            details={"error": str(e)}
        )

async def check_ocr_service_health() -> ComponentHealth:
    """Check OCR service health."""
    start_time = datetime.utcnow()
    
    try:
        # Import OCR service
        from ...ocr_service import OCRService
        
        # Initialize OCR service
        ocr_service = OCRService()
        
        # Check if OCR engines are available
        available_engines = []
        
        # Check Tesseract
        try:
            import pytesseract
            tesseract_version = pytesseract.get_tesseract_version()
            available_engines.append(f"tesseract-{tesseract_version}")
        except Exception:
            pass
        
        # Check EasyOCR
        try:
            import easyocr
            available_engines.append("easyocr")
        except Exception:
            pass
        
        response_time = (datetime.utcnow() - start_time).total_seconds() * 1000
        
        if available_engines:
            return ComponentHealth(
                name="ocr_service",
                status="healthy",
                message="OCR service initialized successfully",
                response_time_ms=response_time,
                last_check=datetime.utcnow(),
                details={
                    "available_engines": available_engines,
                    "supported_languages": ["eng", "msa", "chi_sim", "chi_tra"]
                }
            )
        else:
            return ComponentHealth(
                name="ocr_service",
                status="degraded",
                message="No OCR engines available",
                response_time_ms=response_time,
                last_check=datetime.utcnow(),
                details={"available_engines": []}
            )
            
    except Exception as e:
        response_time = (datetime.utcnow() - start_time).total_seconds() * 1000
        logger.error(f"OCR service health check failed: {e}")
        
        return ComponentHealth(
            name="ocr_service",
            status="unhealthy",
            message=f"OCR service check failed: {str(e)}",
            response_time_ms=response_time,
            last_check=datetime.utcnow(),
            details={"error": str(e)}
        )

async def check_ml_models_health() -> ComponentHealth:
    """Check ML models health."""
    start_time = datetime.utcnow()
    
    try:
        # Check if required ML libraries are available
        ml_libraries = {}
        
        try:
            import spacy
            ml_libraries["spacy"] = spacy.__version__
        except ImportError:
            ml_libraries["spacy"] = "not_available"
        
        try:
            import transformers
            ml_libraries["transformers"] = transformers.__version__
        except ImportError:
            ml_libraries["transformers"] = "not_available"
        
        try:
            import torch
            ml_libraries["torch"] = torch.__version__
        except ImportError:
            ml_libraries["torch"] = "not_available"
        
        # Check model files existence
        model_files_exist = {
            "classification_model": False,
            "field_extraction_model": False,
            "spacy_model": False
        }
        
        # Check if spaCy model is available
        try:
            import spacy
            nlp = spacy.load("en_core_web_sm")
            model_files_exist["spacy_model"] = True
        except Exception:
            pass
        
        response_time = (datetime.utcnow() - start_time).total_seconds() * 1000
        
        # Determine status
        if ml_libraries.get("spacy") != "not_available":
            status = "healthy"
            message = "ML models and libraries available"
        else:
            status = "degraded"
            message = "Some ML libraries not available"
        
        return ComponentHealth(
            name="ml_models",
            status=status,
            message=message,
            response_time_ms=response_time,
            last_check=datetime.utcnow(),
            details={
                "libraries": ml_libraries,
                "model_files": model_files_exist
            }
        )
        
    except Exception as e:
        response_time = (datetime.utcnow() - start_time).total_seconds() * 1000
        logger.error(f"ML models health check failed: {e}")
        
        return ComponentHealth(
            name="ml_models",
            status="unhealthy",
            message=f"ML models check failed: {str(e)}",
            response_time_ms=response_time,
            last_check=datetime.utcnow(),
            details={"error": str(e)}
        )

async def check_file_system_health() -> ComponentHealth:
    """Check file system health."""
    start_time = datetime.utcnow()
    
    try:
        # Check disk usage
        disk_usage = psutil.disk_usage('/')
        disk_free_percent = (disk_usage.free / disk_usage.total) * 100
        
        # Check temp directory
        temp_dir = Path("/tmp")
        temp_writable = temp_dir.is_dir() and os.access(temp_dir, os.W_OK)
        
        # Test file operations
        test_file = temp_dir / f"health_check_{datetime.utcnow().timestamp()}.tmp"
        try:
            test_file.write_text("health check test")
            test_content = test_file.read_text()
            test_file.unlink()
            file_ops_working = test_content == "health check test"
        except Exception:
            file_ops_working = False
        
        response_time = (datetime.utcnow() - start_time).total_seconds() * 1000
        
        # Determine status
        if disk_free_percent < 10:
            status = "unhealthy"
            message = "Low disk space"
        elif disk_free_percent < 20 or not file_ops_working:
            status = "degraded"
            message = "File system issues detected"
        else:
            status = "healthy"
            message = "File system healthy"
        
        return ComponentHealth(
            name="file_system",
            status=status,
            message=message,
            response_time_ms=response_time,
            last_check=datetime.utcnow(),
            details={
                "disk_free_percent": round(disk_free_percent, 2),
                "disk_free_gb": round(disk_usage.free / (1024**3), 2),
                "disk_total_gb": round(disk_usage.total / (1024**3), 2),
                "temp_writable": temp_writable,
                "file_operations": file_ops_working
            }
        )
        
    except Exception as e:
        response_time = (datetime.utcnow() - start_time).total_seconds() * 1000
        logger.error(f"File system health check failed: {e}")
        
        return ComponentHealth(
            name="file_system",
            status="unhealthy",
            message=f"File system check failed: {str(e)}",
            response_time_ms=response_time,
            last_check=datetime.utcnow(),
            details={"error": str(e)}
        )

async def get_system_metrics() -> Dict[str, Any]:
    """Get system resource metrics."""
    try:
        # CPU metrics
        cpu_percent = psutil.cpu_percent(interval=1)
        cpu_count = psutil.cpu_count()
        cpu_freq = psutil.cpu_freq()
        
        # Memory metrics
        memory = psutil.virtual_memory()
        swap = psutil.swap_memory()
        
        # Disk metrics
        disk_usage = psutil.disk_usage('/')
        
        # Network metrics
        network_io = psutil.net_io_counters()
        
        # Process metrics
        process = psutil.Process()
        process_memory = process.memory_info()
        process_cpu = process.cpu_percent()
        
        return {
            "cpu": {
                "percent": cpu_percent,
                "count": cpu_count,
                "frequency_mhz": cpu_freq.current if cpu_freq else None
            },
            "memory": {
                "total_gb": round(memory.total / (1024**3), 2),
                "available_gb": round(memory.available / (1024**3), 2),
                "used_percent": memory.percent,
                "swap_total_gb": round(swap.total / (1024**3), 2),
                "swap_used_percent": swap.percent
            },
            "disk": {
                "total_gb": round(disk_usage.total / (1024**3), 2),
                "free_gb": round(disk_usage.free / (1024**3), 2),
                "used_percent": round((disk_usage.used / disk_usage.total) * 100, 2)
            },
            "network": {
                "bytes_sent": network_io.bytes_sent,
                "bytes_recv": network_io.bytes_recv,
                "packets_sent": network_io.packets_sent,
                "packets_recv": network_io.packets_recv
            },
            "process": {
                "memory_mb": round(process_memory.rss / (1024**2), 2),
                "cpu_percent": process_cpu,
                "threads": process.num_threads(),
                "open_files": len(process.open_files())
            }
        }
        
    except Exception as e:
        logger.error(f"Failed to get system metrics: {e}")
        return {"error": str(e)}

async def get_application_metrics(redis_client: redis.Redis) -> Dict[str, Any]:
    """Get application-specific metrics."""
    try:
        # Get document processing stats from Redis
        pattern = "document:*"
        keys = await redis_client.keys(pattern)
        
        status_counts = {
            ProcessingStatus.QUEUED.value: 0,
            ProcessingStatus.PROCESSING.value: 0,
            ProcessingStatus.COMPLETED.value: 0,
            ProcessingStatus.FAILED.value: 0
        }
        
        total_documents = len(keys)
        
        # Count documents by status
        for key in keys[:100]:  # Limit to avoid performance issues
            try:
                doc_data = await redis_client.hgetall(key)
                status = doc_data.get("status")
                if status in status_counts:
                    status_counts[status] += 1
            except Exception:
                continue
        
        # Get batch processing stats
        batch_pattern = "batch:*"
        batch_keys = await redis_client.keys(batch_pattern)
        total_batches = len(batch_keys)
        
        return {
            "documents": {
                "total": total_documents,
                "queued": status_counts[ProcessingStatus.QUEUED.value],
                "processing": status_counts[ProcessingStatus.PROCESSING.value],
                "completed": status_counts[ProcessingStatus.COMPLETED.value],
                "failed": status_counts[ProcessingStatus.FAILED.value]
            },
            "batches": {
                "total": total_batches
            },
            "cache": {
                "redis_keys": total_documents + total_batches
            }
        }
        
    except Exception as e:
        logger.error(f"Failed to get application metrics: {e}")
        return {"error": str(e)}

# Routes

@router.get("/health", response_model=SystemHealth)
async def health_check():
    """Comprehensive health check for all system components."""
    global last_health_check, cached_health_status
    
    # Check if we can use cached result
    if (last_health_check and cached_health_status and 
        datetime.utcnow() - last_health_check < health_cache_duration):
        return cached_health_status
    
    try:
        # Run all health checks concurrently
        health_checks = await asyncio.gather(
            check_database_health(),
            check_redis_health(),
            check_ocr_service_health(),
            check_ml_models_health(),
            check_file_system_health(),
            return_exceptions=True
        )
        
        # Filter out exceptions and create component list
        components = []
        for check in health_checks:
            if isinstance(check, ComponentHealth):
                components.append(check)
            else:
                logger.error(f"Health check failed: {check}")
                components.append(ComponentHealth(
                    name="unknown",
                    status="unhealthy",
                    message=f"Health check error: {str(check)}",
                    last_check=datetime.utcnow()
                ))
        
        # Determine overall status
        unhealthy_count = sum(1 for c in components if c.status == "unhealthy")
        degraded_count = sum(1 for c in components if c.status == "degraded")
        
        if unhealthy_count > 0:
            overall_status = "unhealthy"
        elif degraded_count > 0:
            overall_status = "degraded"
        else:
            overall_status = "healthy"
        
        # Get system information
        uptime = (datetime.utcnow() - start_time).total_seconds()
        
        system_info = {
            "platform": platform.platform(),
            "python_version": sys.version,
            "architecture": platform.architecture()[0],
            "processor": platform.processor(),
            "hostname": platform.node()
        }
        
        # Create response
        health_response = SystemHealth(
            status=overall_status,
            timestamp=datetime.utcnow(),
            version="1.0.0",  # Should come from config
            uptime_seconds=uptime,
            components=components,
            system_info=system_info
        )
        
        # Cache the result
        last_health_check = datetime.utcnow()
        cached_health_status = health_response
        
        return health_response
        
    except Exception as e:
        logger.error(f"Health check failed: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Health check failed"
        )

@router.get("/ready", response_model=ReadinessCheck)
async def readiness_check():
    """Check if the system is ready to serve requests."""
    try:
        # Run critical readiness checks
        readiness_checks = await asyncio.gather(
            check_database_health(),
            check_redis_health(),
            return_exceptions=True
        )
        
        # Process results
        checks = []
        ready = True
        
        for check in readiness_checks:
            if isinstance(check, ComponentHealth):
                checks.append(check)
                if check.status == "unhealthy":
                    ready = False
            else:
                logger.error(f"Readiness check failed: {check}")
                checks.append(ComponentHealth(
                    name="unknown",
                    status="unhealthy",
                    message=f"Readiness check error: {str(check)}",
                    last_check=datetime.utcnow()
                ))
                ready = False
        
        message = "System ready" if ready else "System not ready"
        
        response = ReadinessCheck(
            ready=ready,
            timestamp=datetime.utcnow(),
            checks=checks,
            message=message
        )
        
        # Return appropriate HTTP status
        if ready:
            return response
        else:
            return JSONResponse(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                content=response.dict()
            )
            
    except Exception as e:
        logger.error(f"Readiness check failed: {e}", exc_info=True)
        return JSONResponse(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            content={
                "ready": False,
                "timestamp": datetime.utcnow().isoformat(),
                "checks": [],
                "message": f"Readiness check failed: {str(e)}"
            }
        )

@router.get("/live", response_model=LivenessCheck)
async def liveness_check():
    """Check if the system is alive."""
    try:
        uptime = (datetime.utcnow() - start_time).total_seconds()
        
        return LivenessCheck(
            alive=True,
            timestamp=datetime.utcnow(),
            uptime_seconds=uptime,
            message="System is alive"
        )
        
    except Exception as e:
        logger.error(f"Liveness check failed: {e}", exc_info=True)
        return JSONResponse(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            content={
                "alive": False,
                "timestamp": datetime.utcnow().isoformat(),
                "uptime_seconds": 0,
                "message": f"Liveness check failed: {str(e)}"
            }
        )

@router.get("/metrics", response_model=MetricsResponse)
async def get_metrics(
    redis_client: redis.Redis = Depends(get_redis_client)
):
    """Get system and application metrics."""
    try:
        # Get metrics concurrently
        system_metrics, app_metrics = await asyncio.gather(
            get_system_metrics(),
            get_application_metrics(redis_client),
            return_exceptions=True
        )
        
        # Handle exceptions
        if isinstance(system_metrics, Exception):
            logger.error(f"System metrics failed: {system_metrics}")
            system_metrics = {"error": str(system_metrics)}
        
        if isinstance(app_metrics, Exception):
            logger.error(f"Application metrics failed: {app_metrics}")
            app_metrics = {"error": str(app_metrics)}
        
        # Performance metrics
        uptime = (datetime.utcnow() - start_time).total_seconds()
        performance_metrics = {
            "uptime_seconds": uptime,
            "uptime_hours": round(uptime / 3600, 2),
            "requests_per_second": 0,  # Would need request counter
            "average_response_time_ms": 0,  # Would need response time tracking
            "error_rate_percent": 0  # Would need error tracking
        }
        
        return MetricsResponse(
            timestamp=datetime.utcnow(),
            system_metrics=system_metrics,
            application_metrics=app_metrics,
            performance_metrics=performance_metrics
        )
        
    except Exception as e:
        logger.error(f"Metrics collection failed: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Metrics collection failed"
        )

@router.get("/metrics/prometheus", response_class=PlainTextResponse)
async def get_prometheus_metrics(
    redis_client: redis.Redis = Depends(get_redis_client)
):
    """Get metrics in Prometheus format."""
    try:
        # Get system metrics
        system_metrics = await get_system_metrics()
        app_metrics = await get_application_metrics(redis_client)
        
        # Format as Prometheus metrics
        prometheus_metrics = []
        
        # System metrics
        if "cpu" in system_metrics:
            prometheus_metrics.append(f"system_cpu_percent {system_metrics['cpu']['percent']}")
            prometheus_metrics.append(f"system_cpu_count {system_metrics['cpu']['count']}")
        
        if "memory" in system_metrics:
            prometheus_metrics.append(f"system_memory_used_percent {system_metrics['memory']['used_percent']}")
            prometheus_metrics.append(f"system_memory_total_gb {system_metrics['memory']['total_gb']}")
        
        if "disk" in system_metrics:
            prometheus_metrics.append(f"system_disk_used_percent {system_metrics['disk']['used_percent']}")
            prometheus_metrics.append(f"system_disk_free_gb {system_metrics['disk']['free_gb']}")
        
        # Application metrics
        if "documents" in app_metrics:
            docs = app_metrics["documents"]
            prometheus_metrics.append(f"documents_total {docs['total']}")
            prometheus_metrics.append(f"documents_queued {docs['queued']}")
            prometheus_metrics.append(f"documents_processing {docs['processing']}")
            prometheus_metrics.append(f"documents_completed {docs['completed']}")
            prometheus_metrics.append(f"documents_failed {docs['failed']}")
        
        # Uptime
        uptime = (datetime.utcnow() - start_time).total_seconds()
        prometheus_metrics.append(f"system_uptime_seconds {uptime}")
        
        return "\n".join(prometheus_metrics) + "\n"
        
    except Exception as e:
        logger.error(f"Prometheus metrics failed: {e}", exc_info=True)
        return f"# Error generating metrics: {str(e)}\n"

@router.get("/version")
async def get_version():
    """Get application version information."""
    return {
        "version": "1.0.0",
        "build_date": "2024-01-01",
        "commit_hash": "unknown",
        "python_version": sys.version,
        "platform": platform.platform()
    }

@router.get("/ping")
async def ping():
    """Simple ping endpoint."""
    return {"message": "pong", "timestamp": datetime.utcnow()}