#!/usr/bin/env python3
"""
Asynchronous Document Processing System

Celery-based asynchronous processing system for scalable document processing.
Provides queue management, task distribution, and result tracking.
"""

import logging
import asyncio
import json
import traceback
from datetime import datetime, timedelta
from typing import Dict, Any, Optional, List, Union
from enum import Enum
from dataclasses import dataclass, asdict
from pathlib import Path

import redis
from celery import Celery, Task
from celery.result import AsyncResult
from celery.signals import task_prerun, task_postrun, task_failure
from kombu import Queue
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import create_engine, text

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Task Status Enum
class TaskStatus(str, Enum):
    """Task execution status."""
    PENDING = "pending"
    STARTED = "started"
    PROCESSING = "processing"
    SUCCESS = "success"
    FAILURE = "failure"
    RETRY = "retry"
    REVOKED = "revoked"

# Task Priority Enum
class TaskPriority(str, Enum):
    """Task priority levels."""
    LOW = "low"
    NORMAL = "normal"
    HIGH = "high"
    CRITICAL = "critical"

# Processing Type Enum
class ProcessingType(str, Enum):
    """Document processing types."""
    SINGLE_DOCUMENT = "single_document"
    BATCH_PROCESSING = "batch_processing"
    BULK_UPLOAD = "bulk_upload"
    REPROCESSING = "reprocessing"
    VALIDATION = "validation"

@dataclass
class TaskMetadata:
    """Task metadata for tracking and monitoring."""
    task_id: str
    user_id: str
    processing_type: ProcessingType
    priority: TaskPriority
    created_at: datetime
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    status: TaskStatus = TaskStatus.PENDING
    progress: float = 0.0
    error_message: Optional[str] = None
    retry_count: int = 0
    max_retries: int = 3
    estimated_duration: Optional[int] = None  # seconds
    actual_duration: Optional[int] = None  # seconds
    file_count: int = 1
    processed_files: int = 0
    queue_name: str = "default"
    worker_id: Optional[str] = None
    memory_usage: Optional[float] = None  # MB
    cpu_usage: Optional[float] = None  # percentage

@dataclass
class ProcessingResult:
    """Document processing result."""
    success: bool
    task_id: str
    document_id: Optional[str] = None
    extracted_data: Optional[Dict[str, Any]] = None
    confidence_scores: Optional[Dict[str, float]] = None
    processing_time: Optional[float] = None
    error_message: Optional[str] = None
    warnings: Optional[List[str]] = None
    metadata: Optional[Dict[str, Any]] = None

# Celery Configuration
class CeleryConfig:
    """Celery configuration settings."""
    
    # Broker settings
    broker_url = "redis://localhost:6379/0"
    result_backend = "redis://localhost:6379/0"
    
    # Task settings
    task_serializer = "json"
    accept_content = ["json"]
    result_serializer = "json"
    timezone = "UTC"
    enable_utc = True
    
    # Worker settings
    worker_prefetch_multiplier = 1
    worker_max_tasks_per_child = 1000
    worker_disable_rate_limits = False
    
    # Task routing
    task_routes = {
        "document_parser.processing.process_single_document": {"queue": "documents"},
        "document_parser.processing.process_batch": {"queue": "batch"},
        "document_parser.processing.process_bulk_upload": {"queue": "bulk"},
        "document_parser.processing.validate_document": {"queue": "validation"},
    }
    
    # Queue definitions
    task_default_queue = "default"
    task_queues = (
        Queue("default", routing_key="default"),
        Queue("documents", routing_key="documents"),
        Queue("batch", routing_key="batch"),
        Queue("bulk", routing_key="bulk"),
        Queue("validation", routing_key="validation"),
        Queue("priority", routing_key="priority"),
    )
    
    # Retry settings
    task_acks_late = True
    task_reject_on_worker_lost = True
    task_default_retry_delay = 60  # seconds
    task_max_retries = 3
    
    # Result settings
    result_expires = 3600  # 1 hour
    result_persistent = True
    
    # Monitoring
    worker_send_task_events = True
    task_send_sent_event = True
    
    # Security
    worker_hijack_root_logger = False
    worker_log_color = False

# Initialize Celery app
celery_app = Celery("document_parser")
celery_app.config_from_object(CeleryConfig)

# Custom Task Base Class
class DocumentProcessingTask(Task):
    """Custom task base class with enhanced tracking."""
    
    def __init__(self):
        self.redis_client = None
        self.db_session = None
    
    def on_success(self, retval, task_id, args, kwargs):
        """Called when task succeeds."""
        logger.info(f"Task {task_id} completed successfully")
        self._update_task_status(task_id, TaskStatus.SUCCESS, progress=100.0)
    
    def on_failure(self, exc, task_id, args, kwargs, einfo):
        """Called when task fails."""
        logger.error(f"Task {task_id} failed: {exc}")
        self._update_task_status(
            task_id, 
            TaskStatus.FAILURE, 
            error_message=str(exc)
        )
    
    def on_retry(self, exc, task_id, args, kwargs, einfo):
        """Called when task is retried."""
        logger.warning(f"Task {task_id} retrying: {exc}")
        self._update_task_status(task_id, TaskStatus.RETRY)
    
    def _update_task_status(self, task_id: str, status: TaskStatus, 
                           progress: Optional[float] = None,
                           error_message: Optional[str] = None):
        """Update task status in Redis."""
        try:
            if not self.redis_client:
                self.redis_client = redis.Redis.from_url(CeleryConfig.broker_url)
            
            # Get existing metadata
            metadata_key = f"task_metadata:{task_id}"
            existing_data = self.redis_client.get(metadata_key)
            
            if existing_data:
                metadata = json.loads(existing_data)
                metadata["status"] = status.value
                
                if progress is not None:
                    metadata["progress"] = progress
                
                if error_message:
                    metadata["error_message"] = error_message
                
                if status == TaskStatus.STARTED:
                    metadata["started_at"] = datetime.utcnow().isoformat()
                elif status in [TaskStatus.SUCCESS, TaskStatus.FAILURE]:
                    metadata["completed_at"] = datetime.utcnow().isoformat()
                    
                    # Calculate duration
                    if metadata.get("started_at"):
                        started = datetime.fromisoformat(metadata["started_at"])
                        duration = (datetime.utcnow() - started).total_seconds()
                        metadata["actual_duration"] = duration
                
                # Update Redis
                self.redis_client.setex(
                    metadata_key, 
                    3600,  # 1 hour TTL
                    json.dumps(metadata)
                )
                
        except Exception as e:
            logger.error(f"Failed to update task status: {e}")

# Task Definitions

@celery_app.task(base=DocumentProcessingTask, bind=True)
def process_single_document(self, document_path: str, user_id: str, 
                          processing_options: Dict[str, Any]) -> Dict[str, Any]:
    """Process a single document asynchronously."""
    task_id = self.request.id
    logger.info(f"Starting document processing: {task_id}")
    
    try:
        # Update status to started
        self._update_task_status(task_id, TaskStatus.STARTED)
        
        # Simulate document processing
        import time
        
        # Progress updates
        for i in range(1, 6):
            time.sleep(1)  # Simulate processing time
            progress = i * 20
            self._update_task_status(task_id, TaskStatus.PROCESSING, progress=progress)
            
            # Update current state
            self.update_state(
                state="PROGRESS",
                meta={"current": i, "total": 5, "status": f"Processing step {i}/5"}
            )
        
        # TODO: Implement actual document processing logic
        # This would involve:
        # 1. Load document from storage
        # 2. Apply ML models for extraction
        # 3. Validate extracted data
        # 4. Store results in database
        # 5. Update document status
        
        result = ProcessingResult(
            success=True,
            task_id=task_id,
            document_id=f"doc_{task_id[:8]}",
            extracted_data={
                "title": "Sample Document",
                "content": "Extracted content...",
                "metadata": {"pages": 1, "format": "pdf"}
            },
            confidence_scores={"title": 0.95, "content": 0.88},
            processing_time=5.0
        )
        
        logger.info(f"Document processing completed: {task_id}")
        return asdict(result)
        
    except Exception as e:
        logger.error(f"Document processing failed: {task_id}, error: {e}")
        result = ProcessingResult(
            success=False,
            task_id=task_id,
            error_message=str(e)
        )
        return asdict(result)

@celery_app.task(base=DocumentProcessingTask, bind=True)
def process_batch(self, document_paths: List[str], user_id: str,
                 processing_options: Dict[str, Any]) -> Dict[str, Any]:
    """Process multiple documents in batch."""
    task_id = self.request.id
    total_docs = len(document_paths)
    logger.info(f"Starting batch processing: {task_id}, {total_docs} documents")
    
    try:
        self._update_task_status(task_id, TaskStatus.STARTED)
        
        results = []
        processed = 0
        
        for i, doc_path in enumerate(document_paths):
            try:
                # Process individual document
                # TODO: Call actual processing function
                import time
                time.sleep(0.5)  # Simulate processing
                
                doc_result = {
                    "document_path": doc_path,
                    "success": True,
                    "extracted_data": {"title": f"Document {i+1}"},
                    "processing_time": 0.5
                }
                
                results.append(doc_result)
                processed += 1
                
                # Update progress
                progress = (processed / total_docs) * 100
                self._update_task_status(task_id, TaskStatus.PROCESSING, progress=progress)
                
                self.update_state(
                    state="PROGRESS",
                    meta={
                        "current": processed,
                        "total": total_docs,
                        "status": f"Processed {processed}/{total_docs} documents"
                    }
                )
                
            except Exception as e:
                logger.error(f"Failed to process document {doc_path}: {e}")
                results.append({
                    "document_path": doc_path,
                    "success": False,
                    "error_message": str(e)
                })
        
        batch_result = {
            "success": True,
            "task_id": task_id,
            "total_documents": total_docs,
            "processed_documents": processed,
            "failed_documents": total_docs - processed,
            "results": results,
            "processing_time": total_docs * 0.5
        }
        
        logger.info(f"Batch processing completed: {task_id}")
        return batch_result
        
    except Exception as e:
        logger.error(f"Batch processing failed: {task_id}, error: {e}")
        return {
            "success": False,
            "task_id": task_id,
            "error_message": str(e)
        }

@celery_app.task(base=DocumentProcessingTask, bind=True)
def validate_document(self, document_path: str, validation_rules: Dict[str, Any]) -> Dict[str, Any]:
    """Validate document against specified rules."""
    task_id = self.request.id
    logger.info(f"Starting document validation: {task_id}")
    
    try:
        self._update_task_status(task_id, TaskStatus.STARTED)
        
        # TODO: Implement document validation logic
        import time
        time.sleep(2)  # Simulate validation
        
        validation_result = {
            "success": True,
            "task_id": task_id,
            "document_path": document_path,
            "validation_passed": True,
            "validation_score": 0.92,
            "issues": [],
            "recommendations": ["Document quality is good"]
        }
        
        logger.info(f"Document validation completed: {task_id}")
        return validation_result
        
    except Exception as e:
        logger.error(f"Document validation failed: {task_id}, error: {e}")
        return {
            "success": False,
            "task_id": task_id,
            "error_message": str(e)
        }

# Task Management Class
class AsyncTaskManager:
    """Manages asynchronous task execution and monitoring."""
    
    def __init__(self, redis_url: str = "redis://localhost:6379/0"):
        self.redis_client = redis.Redis.from_url(redis_url)
        self.celery_app = celery_app
    
    async def submit_document_processing(self, document_path: str, user_id: str,
                                       processing_options: Dict[str, Any],
                                       priority: TaskPriority = TaskPriority.NORMAL) -> str:
        """Submit document for asynchronous processing."""
        try:
            # Create task metadata
            task_metadata = TaskMetadata(
                task_id="",  # Will be set by Celery
                user_id=user_id,
                processing_type=ProcessingType.SINGLE_DOCUMENT,
                priority=priority,
                created_at=datetime.utcnow(),
                queue_name="documents"
            )
            
            # Submit task
            result = process_single_document.apply_async(
                args=[document_path, user_id, processing_options],
                queue="documents",
                priority=self._get_priority_value(priority)
            )
            
            # Update metadata with task ID
            task_metadata.task_id = result.id
            
            # Store metadata in Redis
            metadata_key = f"task_metadata:{result.id}"
            self.redis_client.setex(
                metadata_key,
                3600,  # 1 hour TTL
                json.dumps(asdict(task_metadata), default=str)
            )
            
            logger.info(f"Document processing task submitted: {result.id}")
            return result.id
            
        except Exception as e:
            logger.error(f"Failed to submit document processing task: {e}")
            raise
    
    async def submit_batch_processing(self, document_paths: List[str], user_id: str,
                                    processing_options: Dict[str, Any],
                                    priority: TaskPriority = TaskPriority.NORMAL) -> str:
        """Submit batch of documents for processing."""
        try:
            task_metadata = TaskMetadata(
                task_id="",
                user_id=user_id,
                processing_type=ProcessingType.BATCH_PROCESSING,
                priority=priority,
                created_at=datetime.utcnow(),
                file_count=len(document_paths),
                queue_name="batch"
            )
            
            result = process_batch.apply_async(
                args=[document_paths, user_id, processing_options],
                queue="batch",
                priority=self._get_priority_value(priority)
            )
            
            task_metadata.task_id = result.id
            
            metadata_key = f"task_metadata:{result.id}"
            self.redis_client.setex(
                metadata_key,
                3600,
                json.dumps(asdict(task_metadata), default=str)
            )
            
            logger.info(f"Batch processing task submitted: {result.id}")
            return result.id
            
        except Exception as e:
            logger.error(f"Failed to submit batch processing task: {e}")
            raise
    
    async def get_task_status(self, task_id: str) -> Optional[Dict[str, Any]]:
        """Get task status and metadata."""
        try:
            # Get Celery task result
            result = AsyncResult(task_id, app=self.celery_app)
            
            # Get metadata from Redis
            metadata_key = f"task_metadata:{task_id}"
            metadata_data = self.redis_client.get(metadata_key)
            
            status_info = {
                "task_id": task_id,
                "status": result.status,
                "result": result.result if result.ready() else None,
                "traceback": result.traceback if result.failed() else None,
                "metadata": json.loads(metadata_data) if metadata_data else None
            }
            
            return status_info
            
        except Exception as e:
            logger.error(f"Failed to get task status: {e}")
            return None
    
    async def cancel_task(self, task_id: str) -> bool:
        """Cancel a running task."""
        try:
            self.celery_app.control.revoke(task_id, terminate=True)
            
            # Update metadata
            metadata_key = f"task_metadata:{task_id}"
            metadata_data = self.redis_client.get(metadata_key)
            
            if metadata_data:
                metadata = json.loads(metadata_data)
                metadata["status"] = TaskStatus.REVOKED.value
                metadata["completed_at"] = datetime.utcnow().isoformat()
                
                self.redis_client.setex(metadata_key, 3600, json.dumps(metadata))
            
            logger.info(f"Task cancelled: {task_id}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to cancel task: {e}")
            return False
    
    async def get_user_tasks(self, user_id: str, limit: int = 50) -> List[Dict[str, Any]]:
        """Get tasks for a specific user."""
        try:
            # Scan for user's task metadata
            pattern = "task_metadata:*"
            tasks = []
            
            for key in self.redis_client.scan_iter(match=pattern):
                metadata_data = self.redis_client.get(key)
                if metadata_data:
                    metadata = json.loads(metadata_data)
                    if metadata.get("user_id") == user_id:
                        task_id = metadata.get("task_id")
                        if task_id:
                            task_status = await self.get_task_status(task_id)
                            if task_status:
                                tasks.append(task_status)
            
            # Sort by creation time (newest first)
            tasks.sort(key=lambda x: x.get("metadata", {}).get("created_at", ""), reverse=True)
            
            return tasks[:limit]
            
        except Exception as e:
            logger.error(f"Failed to get user tasks: {e}")
            return []
    
    def _get_priority_value(self, priority: TaskPriority) -> int:
        """Convert priority enum to numeric value."""
        priority_map = {
            TaskPriority.LOW: 1,
            TaskPriority.NORMAL: 5,
            TaskPriority.HIGH: 8,
            TaskPriority.CRITICAL: 10
        }
        return priority_map.get(priority, 5)

# Signal Handlers
@task_prerun.connect
def task_prerun_handler(sender=None, task_id=None, task=None, args=None, kwargs=None, **kwds):
    """Handle task pre-run signal."""
    logger.info(f"Task starting: {task_id}")

@task_postrun.connect
def task_postrun_handler(sender=None, task_id=None, task=None, args=None, kwargs=None, retval=None, state=None, **kwds):
    """Handle task post-run signal."""
    logger.info(f"Task finished: {task_id}, state: {state}")

@task_failure.connect
def task_failure_handler(sender=None, task_id=None, exception=None, traceback=None, einfo=None, **kwds):
    """Handle task failure signal."""
    logger.error(f"Task failed: {task_id}, exception: {exception}")

# Main function for standalone testing
if __name__ == "__main__":
    import asyncio
    
    async def test_async_processing():
        """Test asynchronous processing system."""
        manager = AsyncTaskManager()
        
        # Test single document processing
        task_id = await manager.submit_document_processing(
            document_path="/path/to/test.pdf",
            user_id="test_user",
            processing_options={"extract_text": True, "extract_images": False}
        )
        
        print(f"Submitted task: {task_id}")
        
        # Monitor task progress
        for i in range(10):
            await asyncio.sleep(1)
            status = await manager.get_task_status(task_id)
            if status:
                print(f"Task status: {status['status']}")
                if status['status'] in ['SUCCESS', 'FAILURE']:
                    break
        
        # Test batch processing
        batch_task_id = await manager.submit_batch_processing(
            document_paths=["/path/to/doc1.pdf", "/path/to/doc2.pdf"],
            user_id="test_user",
            processing_options={"extract_text": True}
        )
        
        print(f"Submitted batch task: {batch_task_id}")
        
        print("Async processing test completed!")
    
    # Run test
    asyncio.run(test_async_processing())