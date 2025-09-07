#!/usr/bin/env python3
"""
Admin Routes

FastAPI routes for system administration, configuration management,
and monitoring. Requires admin authentication.
"""

import logging
import asyncio
import json
from datetime import datetime, timedelta
from typing import Dict, Any, List, Optional, Union
from pathlib import Path
import tempfile
import shutil

from fastapi import APIRouter, Depends, HTTPException, status, UploadFile, File, Form, Query, BackgroundTasks
from fastapi.responses import JSONResponse, FileResponse
from pydantic import BaseModel, Field, validator
from sqlalchemy.ext.asyncio import AsyncSession
import redis.asyncio as redis

from ...models.document_models import DocumentType, ProcessingStatus
from ..dependencies import (
    RequestContext, get_request_context, require_admin_authentication,
    get_config, get_redis_client, get_database_session,
    log_api_usage
)
from ...utils import DocumentUtils

# Configure logging
logger = logging.getLogger(__name__)

# Create router
router = APIRouter()

# Request/Response models

class SystemStats(BaseModel):
    """System statistics response."""
    total_documents: int = Field(..., description="Total documents processed")
    documents_by_status: Dict[str, int] = Field(..., description="Documents grouped by status")
    documents_by_type: Dict[str, int] = Field(..., description="Documents grouped by type")
    total_batches: int = Field(..., description="Total batches processed")
    processing_times: Dict[str, float] = Field(..., description="Average processing times")
    error_rates: Dict[str, float] = Field(..., description="Error rates by component")
    system_usage: Dict[str, Any] = Field(..., description="System resource usage")
    timestamp: datetime = Field(..., description="Statistics timestamp")

class ConfigurationUpdate(BaseModel):
    """Configuration update request."""
    section: str = Field(..., description="Configuration section")
    key: str = Field(..., description="Configuration key")
    value: Any = Field(..., description="New configuration value")
    description: Optional[str] = Field(None, description="Change description")

class TemplateUpload(BaseModel):
    """Template upload response."""
    template_name: str = Field(..., description="Template name")
    document_type: DocumentType = Field(..., description="Document type")
    version: str = Field(..., description="Template version")
    uploaded_at: datetime = Field(..., description="Upload timestamp")
    status: str = Field(..., description="Upload status")

class ModelUpload(BaseModel):
    """Model upload response."""
    model_name: str = Field(..., description="Model name")
    model_type: str = Field(..., description="Model type")
    version: str = Field(..., description="Model version")
    file_size_mb: float = Field(..., description="File size in MB")
    uploaded_at: datetime = Field(..., description="Upload timestamp")
    status: str = Field(..., description="Upload status")

class CacheOperation(BaseModel):
    """Cache operation request."""
    operation: str = Field(..., description="Operation type (clear, flush, info)")
    pattern: Optional[str] = Field(None, description="Key pattern for selective operations")
    ttl: Optional[int] = Field(None, description="TTL for cache entries")

class LogQuery(BaseModel):
    """Log query request."""
    level: Optional[str] = Field(None, description="Log level filter")
    component: Optional[str] = Field(None, description="Component filter")
    start_time: Optional[datetime] = Field(None, description="Start time filter")
    end_time: Optional[datetime] = Field(None, description="End time filter")
    limit: int = Field(100, ge=1, le=1000, description="Maximum number of log entries")

class MaintenanceMode(BaseModel):
    """Maintenance mode configuration."""
    enabled: bool = Field(..., description="Enable/disable maintenance mode")
    message: Optional[str] = Field(None, description="Maintenance message")
    estimated_duration: Optional[int] = Field(None, description="Estimated duration in minutes")
    allowed_ips: Optional[List[str]] = Field(None, description="IPs allowed during maintenance")

# Helper functions

async def get_system_statistics(redis_client: redis.Redis) -> SystemStats:
    """Get comprehensive system statistics."""
    try:
        # Get all document keys
        doc_pattern = "document:*"
        doc_keys = await redis_client.keys(doc_pattern)
        
        # Get all batch keys
        batch_pattern = "batch:*"
        batch_keys = await redis_client.keys(batch_pattern)
        
        # Initialize counters
        status_counts = {status.value: 0 for status in ProcessingStatus}
        type_counts = {doc_type.value: 0 for doc_type in DocumentType}
        processing_times = []
        error_count = 0
        
        # Process document statistics
        for key in doc_keys:
            try:
                doc_data = await redis_client.hgetall(key)
                
                # Count by status
                doc_status = doc_data.get("status")
                if doc_status in status_counts:
                    status_counts[doc_status] += 1
                
                # Count by type (if available in result)
                if "result" in doc_data:
                    try:
                        result = eval(doc_data["result"])  # Use proper JSON parsing in production
                        doc_type = result.get("document_type")
                        if doc_type in type_counts:
                            type_counts[doc_type] += 1
                    except Exception:
                        pass
                
                # Collect processing times
                if "processing_metadata" in doc_data:
                    try:
                        result = eval(doc_data["result"])
                        metadata = result.get("processing_metadata", {})
                        proc_time = metadata.get("processing_time")
                        if proc_time:
                            processing_times.append(float(proc_time))
                    except Exception:
                        pass
                
                # Count errors
                if doc_status == ProcessingStatus.FAILED.value:
                    error_count += 1
                    
            except Exception as e:
                logger.error(f"Error processing document stats for {key}: {e}")
                continue
        
        # Calculate averages
        avg_processing_time = sum(processing_times) / len(processing_times) if processing_times else 0
        error_rate = (error_count / len(doc_keys)) * 100 if doc_keys else 0
        
        # Get system resource usage
        import psutil
        system_usage = {
            "cpu_percent": psutil.cpu_percent(),
            "memory_percent": psutil.virtual_memory().percent,
            "disk_percent": psutil.disk_usage('/').percent if hasattr(psutil.disk_usage('/'), 'percent') else 0
        }
        
        return SystemStats(
            total_documents=len(doc_keys),
            documents_by_status=status_counts,
            documents_by_type=type_counts,
            total_batches=len(batch_keys),
            processing_times={
                "average_seconds": round(avg_processing_time, 2),
                "min_seconds": round(min(processing_times), 2) if processing_times else 0,
                "max_seconds": round(max(processing_times), 2) if processing_times else 0
            },
            error_rates={
                "document_processing": round(error_rate, 2),
                "ocr_errors": 0,  # Would need specific tracking
                "validation_errors": 0  # Would need specific tracking
            },
            system_usage=system_usage,
            timestamp=datetime.utcnow()
        )
        
    except Exception as e:
        logger.error(f"Failed to get system statistics: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to retrieve system statistics"
        )

async def cleanup_expired_data(redis_client: redis.Redis) -> Dict[str, int]:
    """Clean up expired data from Redis."""
    try:
        cleanup_stats = {
            "documents_cleaned": 0,
            "batches_cleaned": 0,
            "cache_entries_cleaned": 0
        }
        
        # Clean up old documents (older than 7 days)
        cutoff_time = datetime.utcnow() - timedelta(days=7)
        
        # Get all document keys
        doc_keys = await redis_client.keys("document:*")
        
        for key in doc_keys:
            try:
                doc_data = await redis_client.hgetall(key)
                created_at_str = doc_data.get("created_at")
                
                if created_at_str:
                    created_at = datetime.fromisoformat(created_at_str)
                    if created_at < cutoff_time:
                        await redis_client.delete(key)
                        cleanup_stats["documents_cleaned"] += 1
                        
            except Exception as e:
                logger.error(f"Error cleaning document {key}: {e}")
                continue
        
        # Clean up old batches
        batch_keys = await redis_client.keys("batch:*")
        
        for key in batch_keys:
            try:
                batch_data = await redis_client.hgetall(key)
                created_at_str = batch_data.get("created_at")
                
                if created_at_str:
                    created_at = datetime.fromisoformat(created_at_str)
                    if created_at < cutoff_time:
                        await redis_client.delete(key)
                        cleanup_stats["batches_cleaned"] += 1
                        
            except Exception as e:
                logger.error(f"Error cleaning batch {key}: {e}")
                continue
        
        return cleanup_stats
        
    except Exception as e:
        logger.error(f"Data cleanup failed: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Data cleanup failed"
        )

# Routes

@router.get("/stats", response_model=SystemStats)
async def get_admin_stats(
    context: RequestContext = Depends(require_admin_authentication),
    redis_client: redis.Redis = Depends(get_redis_client)
):
    """Get comprehensive system statistics."""
    start_time = datetime.utcnow()
    
    try:
        stats = await get_system_statistics(redis_client)
        
        # Log API usage
        processing_time = (datetime.utcnow() - start_time).total_seconds()
        await log_api_usage(
            context,
            "/admin/stats",
            "GET",
            200,
            processing_time
        )
        
        return stats
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to get admin stats: {e}", exc_info=True)
        
        # Log API usage with error
        processing_time = (datetime.utcnow() - start_time).total_seconds()
        await log_api_usage(
            context,
            "/admin/stats",
            "GET",
            500,
            processing_time,
            error=str(e)
        )
        
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to retrieve admin statistics"
        )

@router.post("/config")
async def update_configuration(
    config_update: ConfigurationUpdate,
    context: RequestContext = Depends(require_admin_authentication)
):
    """Update system configuration."""
    start_time = datetime.utcnow()
    
    try:
        # Get current config
        config = await get_config()
        
        # Validate section and key
        if not hasattr(config, config_update.section):
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Invalid configuration section: {config_update.section}"
            )
        
        section_config = getattr(config, config_update.section)
        if not hasattr(section_config, config_update.key):
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Invalid configuration key: {config_update.key}"
            )
        
        # Store old value for rollback
        old_value = getattr(section_config, config_update.key)
        
        # Update configuration
        setattr(section_config, config_update.key, config_update.value)
        
        # Save configuration
        await config.save_config()
        
        # Log configuration change
        logger.info(
            f"Configuration updated by {context.user_id}: "
            f"{config_update.section}.{config_update.key} = {config_update.value} "
            f"(was: {old_value})"
        )
        
        # Log API usage
        processing_time = (datetime.utcnow() - start_time).total_seconds()
        await log_api_usage(
            context,
            "/admin/config",
            "POST",
            200,
            processing_time
        )
        
        return {
            "message": "Configuration updated successfully",
            "section": config_update.section,
            "key": config_update.key,
            "old_value": old_value,
            "new_value": config_update.value,
            "updated_by": context.user_id,
            "updated_at": datetime.utcnow()
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Configuration update failed: {e}", exc_info=True)
        
        # Log API usage with error
        processing_time = (datetime.utcnow() - start_time).total_seconds()
        await log_api_usage(
            context,
            "/admin/config",
            "POST",
            500,
            processing_time,
            error=str(e)
        )
        
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Configuration update failed"
        )

@router.get("/config")
async def get_configuration(
    context: RequestContext = Depends(require_admin_authentication),
    section: Optional[str] = Query(None, description="Specific section to retrieve")
):
    """Get current system configuration."""
    start_time = datetime.utcnow()
    
    try:
        config = await get_config()
        
        if section:
            if hasattr(config, section):
                section_config = getattr(config, section)
                config_dict = section_config.dict() if hasattr(section_config, 'dict') else vars(section_config)
            else:
                raise HTTPException(
                    status_code=status.HTTP_404_NOT_FOUND,
                    detail=f"Configuration section not found: {section}"
                )
        else:
            config_dict = config.dict() if hasattr(config, 'dict') else vars(config)
        
        # Log API usage
        processing_time = (datetime.utcnow() - start_time).total_seconds()
        await log_api_usage(
            context,
            "/admin/config",
            "GET",
            200,
            processing_time
        )
        
        return {
            "configuration": config_dict,
            "section": section,
            "retrieved_at": datetime.utcnow()
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to get configuration: {e}", exc_info=True)
        
        # Log API usage with error
        processing_time = (datetime.utcnow() - start_time).total_seconds()
        await log_api_usage(
            context,
            "/admin/config",
            "GET",
            500,
            processing_time,
            error=str(e)
        )
        
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to retrieve configuration"
        )

@router.post("/templates/upload", response_model=TemplateUpload)
async def upload_template(
    context: RequestContext = Depends(require_admin_authentication),
    file: UploadFile = File(..., description="Template JSON file"),
    document_type: DocumentType = Form(..., description="Document type for template"),
    version: str = Form("1.0", description="Template version")
):
    """Upload a new document template."""
    start_time = datetime.utcnow()
    
    try:
        # Validate file type
        if not file.filename.endswith('.json'):
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Template file must be a JSON file"
            )
        
        # Read and validate JSON content
        content = await file.read()
        try:
            template_data = json.loads(content)
        except json.JSONDecodeError as e:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Invalid JSON format: {str(e)}"
            )
        
        # Validate template structure
        required_fields = ["document_type", "fields", "validation_rules", "extraction_rules"]
        for field in required_fields:
            if field not in template_data:
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail=f"Missing required field in template: {field}"
                )
        
        # Ensure document type matches
        if template_data["document_type"] != document_type.value:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Document type in template doesn't match specified type"
            )
        
        # Save template file
        template_dir = Path("src/document_parser/templates")
        template_dir.mkdir(parents=True, exist_ok=True)
        
        template_filename = f"{document_type.value}_template_v{version}.json"
        template_path = template_dir / template_filename
        
        with open(template_path, 'w') as f:
            json.dump(template_data, f, indent=2)
        
        # Log template upload
        logger.info(
            f"Template uploaded by {context.user_id}: {template_filename} "
            f"for {document_type.value}"
        )
        
        # Log API usage
        processing_time = (datetime.utcnow() - start_time).total_seconds()
        await log_api_usage(
            context,
            "/admin/templates/upload",
            "POST",
            200,
            processing_time
        )
        
        return TemplateUpload(
            template_name=template_filename,
            document_type=document_type,
            version=version,
            uploaded_at=datetime.utcnow(),
            status="uploaded"
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Template upload failed: {e}", exc_info=True)
        
        # Log API usage with error
        processing_time = (datetime.utcnow() - start_time).total_seconds()
        await log_api_usage(
            context,
            "/admin/templates/upload",
            "POST",
            500,
            processing_time,
            error=str(e)
        )
        
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Template upload failed"
        )

@router.get("/templates")
async def list_templates(
    context: RequestContext = Depends(require_admin_authentication)
):
    """List all available templates."""
    start_time = datetime.utcnow()
    
    try:
        template_dir = Path("src/document_parser/templates")
        
        if not template_dir.exists():
            return {"templates": [], "count": 0}
        
        templates = []
        for template_file in template_dir.glob("*.json"):
            try:
                with open(template_file, 'r') as f:
                    template_data = json.load(f)
                
                file_stat = template_file.stat()
                templates.append({
                    "filename": template_file.name,
                    "document_type": template_data.get("document_type"),
                    "version": template_data.get("version", "unknown"),
                    "created_at": datetime.fromtimestamp(file_stat.st_ctime),
                    "modified_at": datetime.fromtimestamp(file_stat.st_mtime),
                    "size_bytes": file_stat.st_size
                })
                
            except Exception as e:
                logger.error(f"Error reading template {template_file}: {e}")
                continue
        
        # Log API usage
        processing_time = (datetime.utcnow() - start_time).total_seconds()
        await log_api_usage(
            context,
            "/admin/templates",
            "GET",
            200,
            processing_time
        )
        
        return {
            "templates": templates,
            "count": len(templates),
            "retrieved_at": datetime.utcnow()
        }
        
    except Exception as e:
        logger.error(f"Failed to list templates: {e}", exc_info=True)
        
        # Log API usage with error
        processing_time = (datetime.utcnow() - start_time).total_seconds()
        await log_api_usage(
            context,
            "/admin/templates",
            "GET",
            500,
            processing_time,
            error=str(e)
        )
        
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to list templates"
        )

@router.post("/cache")
async def manage_cache(
    cache_op: CacheOperation,
    context: RequestContext = Depends(require_admin_authentication),
    redis_client: redis.Redis = Depends(get_redis_client)
):
    """Manage Redis cache operations."""
    start_time = datetime.utcnow()
    
    try:
        result = {}
        
        if cache_op.operation == "clear":
            # Clear specific pattern or all keys
            pattern = cache_op.pattern or "*"
            keys = await redis_client.keys(pattern)
            
            if keys:
                deleted_count = await redis_client.delete(*keys)
                result = {"operation": "clear", "deleted_keys": deleted_count, "pattern": pattern}
            else:
                result = {"operation": "clear", "deleted_keys": 0, "pattern": pattern}
                
        elif cache_op.operation == "flush":
            # Flush entire Redis database
            await redis_client.flushdb()
            result = {"operation": "flush", "message": "All cache data flushed"}
            
        elif cache_op.operation == "info":
            # Get Redis info
            redis_info = await redis_client.info()
            result = {
                "operation": "info",
                "redis_info": {
                    "version": redis_info.get("redis_version"),
                    "connected_clients": redis_info.get("connected_clients"),
                    "used_memory_human": redis_info.get("used_memory_human"),
                    "total_commands_processed": redis_info.get("total_commands_processed"),
                    "uptime_in_seconds": redis_info.get("uptime_in_seconds")
                }
            }
            
        else:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Invalid cache operation: {cache_op.operation}"
            )
        
        # Log cache operation
        logger.info(
            f"Cache operation performed by {context.user_id}: {cache_op.operation}"
        )
        
        # Log API usage
        processing_time = (datetime.utcnow() - start_time).total_seconds()
        await log_api_usage(
            context,
            "/admin/cache",
            "POST",
            200,
            processing_time
        )
        
        result["performed_by"] = context.user_id
        result["performed_at"] = datetime.utcnow()
        
        return result
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Cache operation failed: {e}", exc_info=True)
        
        # Log API usage with error
        processing_time = (datetime.utcnow() - start_time).total_seconds()
        await log_api_usage(
            context,
            "/admin/cache",
            "POST",
            500,
            processing_time,
            error=str(e)
        )
        
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Cache operation failed"
        )

@router.post("/cleanup")
async def cleanup_system(
    background_tasks: BackgroundTasks,
    context: RequestContext = Depends(require_admin_authentication),
    redis_client: redis.Redis = Depends(get_redis_client)
):
    """Clean up expired data and temporary files."""
    start_time = datetime.utcnow()
    
    try:
        # Start cleanup in background
        background_tasks.add_task(cleanup_expired_data, redis_client)
        
        # Clean up temporary files
        temp_dir = Path(tempfile.gettempdir()) / "document_parser"
        temp_files_cleaned = 0
        
        if temp_dir.exists():
            for user_dir in temp_dir.iterdir():
                if user_dir.is_dir():
                    try:
                        # Remove files older than 1 day
                        cutoff_time = datetime.utcnow() - timedelta(days=1)
                        
                        for temp_file in user_dir.iterdir():
                            if temp_file.is_file():
                                file_time = datetime.fromtimestamp(temp_file.stat().st_mtime)
                                if file_time < cutoff_time:
                                    temp_file.unlink()
                                    temp_files_cleaned += 1
                                    
                    except Exception as e:
                        logger.error(f"Error cleaning temp directory {user_dir}: {e}")
                        continue
        
        # Log cleanup operation
        logger.info(f"System cleanup initiated by {context.user_id}")
        
        # Log API usage
        processing_time = (datetime.utcnow() - start_time).total_seconds()
        await log_api_usage(
            context,
            "/admin/cleanup",
            "POST",
            200,
            processing_time
        )
        
        return {
            "message": "System cleanup initiated",
            "temp_files_cleaned": temp_files_cleaned,
            "background_cleanup": "started",
            "initiated_by": context.user_id,
            "initiated_at": datetime.utcnow()
        }
        
    except Exception as e:
        logger.error(f"System cleanup failed: {e}", exc_info=True)
        
        # Log API usage with error
        processing_time = (datetime.utcnow() - start_time).total_seconds()
        await log_api_usage(
            context,
            "/admin/cleanup",
            "POST",
            500,
            processing_time,
            error=str(e)
        )
        
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="System cleanup failed"
        )

@router.get("/logs")
async def get_system_logs(
    context: RequestContext = Depends(require_admin_authentication),
    level: Optional[str] = Query(None, description="Log level filter"),
    component: Optional[str] = Query(None, description="Component filter"),
    limit: int = Query(100, ge=1, le=1000, description="Maximum number of log entries")
):
    """Get system logs with filtering."""
    start_time = datetime.utcnow()
    
    try:
        # This is a simplified implementation
        # In production, you'd integrate with your logging system
        
        logs = [
            {
                "timestamp": datetime.utcnow() - timedelta(minutes=5),
                "level": "INFO",
                "component": "document_processor",
                "message": "Document processing completed successfully",
                "user_id": "user123",
                "document_id": "doc456"
            },
            {
                "timestamp": datetime.utcnow() - timedelta(minutes=10),
                "level": "ERROR",
                "component": "ocr_service",
                "message": "OCR processing failed for document",
                "error": "Tesseract not found",
                "document_id": "doc789"
            }
        ]
        
        # Apply filters
        filtered_logs = logs
        
        if level:
            filtered_logs = [log for log in filtered_logs if log["level"].lower() == level.lower()]
        
        if component:
            filtered_logs = [log for log in filtered_logs if component.lower() in log["component"].lower()]
        
        # Apply limit
        filtered_logs = filtered_logs[:limit]
        
        # Log API usage
        processing_time = (datetime.utcnow() - start_time).total_seconds()
        await log_api_usage(
            context,
            "/admin/logs",
            "GET",
            200,
            processing_time
        )
        
        return {
            "logs": filtered_logs,
            "count": len(filtered_logs),
            "filters": {
                "level": level,
                "component": component,
                "limit": limit
            },
            "retrieved_at": datetime.utcnow()
        }
        
    except Exception as e:
        logger.error(f"Failed to get system logs: {e}", exc_info=True)
        
        # Log API usage with error
        processing_time = (datetime.utcnow() - start_time).total_seconds()
        await log_api_usage(
            context,
            "/admin/logs",
            "GET",
            500,
            processing_time,
            error=str(e)
        )
        
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to retrieve system logs"
        )

@router.post("/maintenance")
async def set_maintenance_mode(
    maintenance: MaintenanceMode,
    context: RequestContext = Depends(require_admin_authentication),
    redis_client: redis.Redis = Depends(get_redis_client)
):
    """Enable or disable maintenance mode."""
    start_time = datetime.utcnow()
    
    try:
        # Store maintenance mode configuration in Redis
        maintenance_key = "system:maintenance"
        
        if maintenance.enabled:
            maintenance_data = {
                "enabled": "true",
                "message": maintenance.message or "System is under maintenance",
                "estimated_duration": str(maintenance.estimated_duration or 60),
                "allowed_ips": ",".join(maintenance.allowed_ips or []),
                "enabled_by": context.user_id,
                "enabled_at": datetime.utcnow().isoformat()
            }
            
            await redis_client.hset(maintenance_key, mapping=maintenance_data)
            
            # Set expiration if duration is specified
            if maintenance.estimated_duration:
                await redis_client.expire(maintenance_key, maintenance.estimated_duration * 60)
                
            message = "Maintenance mode enabled"
            
        else:
            await redis_client.delete(maintenance_key)
            message = "Maintenance mode disabled"
        
        # Log maintenance mode change
        logger.info(
            f"Maintenance mode {'enabled' if maintenance.enabled else 'disabled'} "
            f"by {context.user_id}"
        )
        
        # Log API usage
        processing_time = (datetime.utcnow() - start_time).total_seconds()
        await log_api_usage(
            context,
            "/admin/maintenance",
            "POST",
            200,
            processing_time
        )
        
        return {
            "message": message,
            "maintenance_enabled": maintenance.enabled,
            "configured_by": context.user_id,
            "configured_at": datetime.utcnow()
        }
        
    except Exception as e:
        logger.error(f"Failed to set maintenance mode: {e}", exc_info=True)
        
        # Log API usage with error
        processing_time = (datetime.utcnow() - start_time).total_seconds()
        await log_api_usage(
            context,
            "/admin/maintenance",
            "POST",
            500,
            processing_time,
            error=str(e)
        )
        
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to set maintenance mode"
        )

@router.get("/users")
async def list_users(
    context: RequestContext = Depends(require_admin_authentication),
    limit: int = Query(50, ge=1, le=200, description="Maximum number of users to return"),
    offset: int = Query(0, ge=0, description="Number of users to skip")
):
    """List system users and their activity."""
    start_time = datetime.utcnow()
    
    try:
        # This is a simplified implementation
        # In production, you'd query your user database
        
        users = [
            {
                "user_id": "user123",
                "username": "john.doe",
                "email": "john.doe@example.com",
                "role": "user",
                "created_at": datetime.utcnow() - timedelta(days=30),
                "last_login": datetime.utcnow() - timedelta(hours=2),
                "documents_processed": 45,
                "status": "active"
            },
            {
                "user_id": "admin456",
                "username": "admin",
                "email": "admin@example.com",
                "role": "admin",
                "created_at": datetime.utcnow() - timedelta(days=90),
                "last_login": datetime.utcnow() - timedelta(minutes=10),
                "documents_processed": 0,
                "status": "active"
            }
        ]
        
        # Apply pagination
        total_users = len(users)
        paginated_users = users[offset:offset + limit]
        
        # Log API usage
        processing_time = (datetime.utcnow() - start_time).total_seconds()
        await log_api_usage(
            context,
            "/admin/users",
            "GET",
            200,
            processing_time
        )
        
        return {
            "users": paginated_users,
            "total_count": total_users,
            "limit": limit,
            "offset": offset,
            "has_more": offset + limit < total_users,
            "retrieved_at": datetime.utcnow()
        }
        
    except Exception as e:
        logger.error(f"Failed to list users: {e}", exc_info=True)
        
        # Log API usage with error
        processing_time = (datetime.utcnow() - start_time).total_seconds()
        await log_api_usage(
            context,
            "/admin/users",
            "GET",
            500,
            processing_time,
            error=str(e)
        )
        
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to list users"
        )