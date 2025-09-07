#!/usr/bin/env python3
"""
Document Processing Routes

FastAPI routes for document upload, processing, and result retrieval.
Handles single and batch document processing with comprehensive validation.
"""

import logging
import asyncio
import uuid
from datetime import datetime, timedelta
from typing import List, Optional, Dict, Any, Union
from pathlib import Path
import tempfile
import os

from fastapi import APIRouter, Depends, HTTPException, status, UploadFile, File, Form, BackgroundTasks, Query
from fastapi.responses import JSONResponse, FileResponse
from pydantic import BaseModel, Field, validator
from sqlalchemy.ext.asyncio import AsyncSession
import redis.asyncio as redis

from ...models.document_models import DocumentType, ProcessingStatus, ExtractionResult
from ..dependencies import (
    RequestContext, get_request_context, require_authentication,
    get_document_classifier, get_ocr_service, get_field_extractor, get_document_validator,
    log_api_usage
)
from ...utils import DocumentUtils, validate_file_type, get_file_size_mb

# Configure logging
logger = logging.getLogger(__name__)

# Create router
router = APIRouter()

# Request/Response models

class DocumentUploadResponse(BaseModel):
    """Response model for document upload."""
    document_id: str = Field(..., description="Unique document identifier")
    status: ProcessingStatus = Field(..., description="Processing status")
    message: str = Field(..., description="Status message")
    estimated_completion: Optional[datetime] = Field(None, description="Estimated completion time")
    processing_url: str = Field(..., description="URL to check processing status")

class DocumentProcessingStatus(BaseModel):
    """Response model for processing status."""
    document_id: str = Field(..., description="Document identifier")
    status: ProcessingStatus = Field(..., description="Current processing status")
    progress: float = Field(..., ge=0, le=100, description="Processing progress percentage")
    message: str = Field(..., description="Status message")
    created_at: datetime = Field(..., description="Document upload time")
    updated_at: datetime = Field(..., description="Last update time")
    estimated_completion: Optional[datetime] = Field(None, description="Estimated completion time")
    error_details: Optional[Dict[str, Any]] = Field(None, description="Error details if failed")

class DocumentProcessingResult(BaseModel):
    """Response model for processing results."""
    document_id: str = Field(..., description="Document identifier")
    document_type: DocumentType = Field(..., description="Detected document type")
    confidence_score: float = Field(..., ge=0, le=1, description="Overall confidence score")
    extracted_fields: Dict[str, Any] = Field(..., description="Extracted field data")
    validation_results: Dict[str, Any] = Field(..., description="Validation results")
    processing_metadata: Dict[str, Any] = Field(..., description="Processing metadata")
    created_at: datetime = Field(..., description="Processing completion time")

class BatchUploadRequest(BaseModel):
    """Request model for batch upload."""
    batch_name: Optional[str] = Field(None, description="Optional batch name")
    priority: int = Field(1, ge=1, le=5, description="Processing priority (1=highest, 5=lowest)")
    callback_url: Optional[str] = Field(None, description="Webhook URL for completion notification")
    options: Optional[Dict[str, Any]] = Field(None, description="Processing options")

class BatchProcessingResponse(BaseModel):
    """Response model for batch processing."""
    batch_id: str = Field(..., description="Unique batch identifier")
    document_ids: List[str] = Field(..., description="List of document IDs in batch")
    total_documents: int = Field(..., description="Total number of documents")
    status: ProcessingStatus = Field(..., description="Batch processing status")
    estimated_completion: Optional[datetime] = Field(None, description="Estimated completion time")
    batch_url: str = Field(..., description="URL to check batch status")

class ProcessingOptions(BaseModel):
    """Processing options for document upload."""
    enable_preprocessing: bool = Field(True, description="Enable image preprocessing")
    ocr_language: List[str] = Field(["eng", "msa"], description="OCR languages")
    confidence_threshold: float = Field(0.7, ge=0, le=1, description="Minimum confidence threshold")
    enable_validation: bool = Field(True, description="Enable field validation")
    return_raw_text: bool = Field(False, description="Include raw OCR text in response")
    return_bounding_boxes: bool = Field(False, description="Include bounding box coordinates")

# Helper functions

async def save_uploaded_file(upload_file: UploadFile, user_id: str) -> Path:
    """Save uploaded file to temporary storage."""
    # Create user-specific temp directory
    temp_dir = Path(tempfile.gettempdir()) / "document_parser" / user_id
    temp_dir.mkdir(parents=True, exist_ok=True)
    
    # Generate unique filename
    file_extension = Path(upload_file.filename).suffix if upload_file.filename else ".tmp"
    unique_filename = f"{uuid.uuid4()}{file_extension}"
    file_path = temp_dir / unique_filename
    
    # Save file
    try:
        with open(file_path, "wb") as buffer:
            content = await upload_file.read()
            buffer.write(content)
        
        logger.info(f"Saved uploaded file: {file_path}")
        return file_path
        
    except Exception as e:
        logger.error(f"Failed to save uploaded file: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to save uploaded file"
        )

async def validate_upload_file(upload_file: UploadFile) -> None:
    """Validate uploaded file."""
    # Check file size
    if upload_file.size and upload_file.size > 10 * 1024 * 1024:  # 10MB limit
        raise HTTPException(
            status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
            detail="File size exceeds 10MB limit"
        )
    
    # Check file type
    if upload_file.filename:
        if not validate_file_type(upload_file.filename):
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Unsupported file type. Supported: jpg, jpeg, png, tiff, bmp, pdf"
            )
    
    # Check content type
    allowed_content_types = [
        "image/jpeg", "image/png", "image/tiff", "image/bmp", "application/pdf"
    ]
    
    if upload_file.content_type not in allowed_content_types:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Unsupported content type: {upload_file.content_type}"
        )

async def process_document_async(
    document_id: str,
    file_path: Path,
    user_id: str,
    options: ProcessingOptions,
    redis_client: redis.Redis
):
    """Process document asynchronously."""
    try:
        # Update status to processing
        await redis_client.hset(
            f"document:{document_id}",
            mapping={
                "status": ProcessingStatus.PROCESSING.value,
                "progress": "10",
                "message": "Starting document processing",
                "updated_at": datetime.utcnow().isoformat()
            }
        )
        
        # Initialize services
        classifier = await get_document_classifier()
        ocr_service = await get_ocr_service()
        field_extractor = await get_field_extractor()
        validator = await get_document_validator()
        
        # Step 1: Document Classification
        await redis_client.hset(
            f"document:{document_id}",
            mapping={
                "progress": "25",
                "message": "Classifying document type"
            }
        )
        
        classification_result = await classifier.classify_document(file_path)
        document_type = classification_result.document_type
        classification_confidence = classification_result.confidence
        
        # Step 2: OCR Processing
        await redis_client.hset(
            f"document:{document_id}",
            mapping={
                "progress": "50",
                "message": "Extracting text with OCR"
            }
        )
        
        ocr_result = await ocr_service.extract_text(
            file_path,
            languages=options.ocr_language,
            preprocess=options.enable_preprocessing
        )
        
        # Step 3: Field Extraction
        await redis_client.hset(
            f"document:{document_id}",
            mapping={
                "progress": "75",
                "message": "Extracting document fields"
            }
        )
        
        extraction_result = await field_extractor.extract_fields(
            document_type,
            ocr_result.text,
            ocr_result.word_boxes if options.return_bounding_boxes else None,
            confidence_threshold=options.confidence_threshold
        )
        
        # Step 4: Validation
        validation_result = None
        if options.enable_validation:
            await redis_client.hset(
                f"document:{document_id}",
                mapping={
                    "progress": "90",
                    "message": "Validating extracted data"
                }
            )
            
            validation_result = await validator.validate_document(
                document_type,
                extraction_result.fields
            )
        
        # Prepare final result
        processing_result = {
            "document_id": document_id,
            "document_type": document_type.value,
            "confidence_score": min(classification_confidence, extraction_result.confidence),
            "extracted_fields": extraction_result.fields,
            "validation_results": validation_result.to_dict() if validation_result else None,
            "processing_metadata": {
                "classification_confidence": classification_confidence,
                "extraction_confidence": extraction_result.confidence,
                "ocr_confidence": ocr_result.confidence,
                "processing_time": (datetime.utcnow() - datetime.fromisoformat(
                    await redis_client.hget(f"document:{document_id}", "created_at")
                )).total_seconds(),
                "file_size_mb": get_file_size_mb(file_path),
                "ocr_language": options.ocr_language,
                "preprocessing_enabled": options.enable_preprocessing
            },
            "created_at": datetime.utcnow().isoformat()
        }
        
        # Include optional data
        if options.return_raw_text:
            processing_result["raw_text"] = ocr_result.text
        
        if options.return_bounding_boxes:
            processing_result["bounding_boxes"] = ocr_result.word_boxes
        
        # Store final result
        await redis_client.hset(
            f"document:{document_id}",
            mapping={
                "status": ProcessingStatus.COMPLETED.value,
                "progress": "100",
                "message": "Document processing completed successfully",
                "result": str(processing_result),
                "updated_at": datetime.utcnow().isoformat()
            }
        )
        
        # Set expiration (24 hours)
        await redis_client.expire(f"document:{document_id}", 86400)
        
        logger.info(f"Document {document_id} processed successfully")
        
    except Exception as e:
        logger.error(f"Document processing failed for {document_id}: {e}", exc_info=True)
        
        # Update status to failed
        await redis_client.hset(
            f"document:{document_id}",
            mapping={
                "status": ProcessingStatus.FAILED.value,
                "progress": "0",
                "message": f"Processing failed: {str(e)}",
                "error_details": str({"error": str(e), "type": type(e).__name__}),
                "updated_at": datetime.utcnow().isoformat()
            }
        )
    
    finally:
        # Clean up temporary file
        try:
            if file_path.exists():
                file_path.unlink()
                logger.info(f"Cleaned up temporary file: {file_path}")
        except Exception as e:
            logger.error(f"Failed to clean up temporary file {file_path}: {e}")

# Routes

@router.post("/upload", response_model=DocumentUploadResponse)
async def upload_document(
    background_tasks: BackgroundTasks,
    context: RequestContext = Depends(get_request_context),
    file: UploadFile = File(..., description="Document file to process"),
    options: str = Form("{}", description="Processing options as JSON string")
):
    """Upload and process a single document."""
    start_time = datetime.utcnow()
    
    try:
        # Parse processing options
        import json
        try:
            options_dict = json.loads(options)
            processing_options = ProcessingOptions(**options_dict)
        except (json.JSONDecodeError, ValueError) as e:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Invalid processing options: {e}"
            )
        
        # Validate file
        await validate_upload_file(file)
        
        # Generate document ID
        document_id = DocumentUtils.generate_document_id()
        
        # Save uploaded file
        user_id = context.user_id or "anonymous"
        file_path = await save_uploaded_file(file, user_id)
        
        # Store initial document info in Redis
        await context.redis_client.hset(
            f"document:{document_id}",
            mapping={
                "user_id": user_id,
                "filename": file.filename or "unknown",
                "content_type": file.content_type or "unknown",
                "file_size": file.size or 0,
                "status": ProcessingStatus.QUEUED.value,
                "progress": "0",
                "message": "Document queued for processing",
                "created_at": datetime.utcnow().isoformat(),
                "updated_at": datetime.utcnow().isoformat()
            }
        )
        
        # Start background processing
        background_tasks.add_task(
            process_document_async,
            document_id,
            file_path,
            user_id,
            processing_options,
            context.redis_client
        )
        
        # Log API usage
        processing_time = (datetime.utcnow() - start_time).total_seconds()
        await log_api_usage(
            context,
            "/documents/upload",
            "POST",
            200,
            processing_time
        )
        
        return DocumentUploadResponse(
            document_id=document_id,
            status=ProcessingStatus.QUEUED,
            message="Document uploaded successfully and queued for processing",
            estimated_completion=datetime.utcnow() + timedelta(minutes=2),
            processing_url=f"/api/v1/documents/{document_id}"
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Document upload failed: {e}", exc_info=True)
        
        # Log API usage with error
        processing_time = (datetime.utcnow() - start_time).total_seconds()
        await log_api_usage(
            context,
            "/documents/upload",
            "POST",
            500,
            processing_time,
            error=str(e)
        )
        
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Document upload failed"
        )

@router.get("/{document_id}", response_model=Union[DocumentProcessingStatus, DocumentProcessingResult])
async def get_document_status(
    document_id: str,
    context: RequestContext = Depends(get_request_context)
):
    """Get document processing status or results."""
    start_time = datetime.utcnow()
    
    try:
        # Get document info from Redis
        doc_data = await context.redis_client.hgetall(f"document:{document_id}")
        
        if not doc_data:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Document not found"
            )
        
        # Check if user has access to this document
        doc_user_id = doc_data.get("user_id")
        if context.user_id and doc_user_id != context.user_id and not context.has_role("admin"):
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="Access denied"
            )
        
        status_value = doc_data.get("status")
        processing_status = ProcessingStatus(status_value)
        
        # Log API usage
        processing_time = (datetime.utcnow() - start_time).total_seconds()
        await log_api_usage(
            context,
            f"/documents/{document_id}",
            "GET",
            200,
            processing_time
        )
        
        if processing_status == ProcessingStatus.COMPLETED:
            # Return full results
            result_data = eval(doc_data.get("result", "{}"))  # Note: Use proper JSON parsing in production
            return DocumentProcessingResult(**result_data)
        else:
            # Return status
            return DocumentProcessingStatus(
                document_id=document_id,
                status=processing_status,
                progress=float(doc_data.get("progress", 0)),
                message=doc_data.get("message", ""),
                created_at=datetime.fromisoformat(doc_data.get("created_at")),
                updated_at=datetime.fromisoformat(doc_data.get("updated_at")),
                estimated_completion=datetime.fromisoformat(doc_data.get("estimated_completion")) if doc_data.get("estimated_completion") else None,
                error_details=eval(doc_data.get("error_details", "None")) if doc_data.get("error_details") else None
            )
            
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to get document status: {e}", exc_info=True)
        
        # Log API usage with error
        processing_time = (datetime.utcnow() - start_time).total_seconds()
        await log_api_usage(
            context,
            f"/documents/{document_id}",
            "GET",
            500,
            processing_time,
            error=str(e)
        )
        
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to retrieve document status"
        )

@router.post("/batch", response_model=BatchProcessingResponse)
async def upload_batch_documents(
    background_tasks: BackgroundTasks,
    context: RequestContext = Depends(get_request_context),
    files: List[UploadFile] = File(..., description="List of document files to process"),
    batch_request: str = Form("{}", description="Batch processing options as JSON string")
):
    """Upload and process multiple documents in batch."""
    start_time = datetime.utcnow()
    
    try:
        # Parse batch request
        import json
        try:
            batch_data = json.loads(batch_request)
            batch_options = BatchUploadRequest(**batch_data)
        except (json.JSONDecodeError, ValueError) as e:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Invalid batch options: {e}"
            )
        
        # Validate batch size
        if len(files) > 50:  # Limit batch size
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Batch size exceeds maximum limit of 50 documents"
            )
        
        # Validate all files
        for file in files:
            await validate_upload_file(file)
        
        # Generate batch ID
        batch_id = f"batch_{uuid.uuid4()}"
        document_ids = []
        
        # Process each file
        user_id = context.user_id or "anonymous"
        processing_options = ProcessingOptions(**(batch_options.options or {}))
        
        for file in files:
            # Generate document ID
            document_id = DocumentUtils.generate_document_id()
            document_ids.append(document_id)
            
            # Save uploaded file
            file_path = await save_uploaded_file(file, user_id)
            
            # Store initial document info
            await context.redis_client.hset(
                f"document:{document_id}",
                mapping={
                    "batch_id": batch_id,
                    "user_id": user_id,
                    "filename": file.filename or "unknown",
                    "content_type": file.content_type or "unknown",
                    "file_size": file.size or 0,
                    "status": ProcessingStatus.QUEUED.value,
                    "progress": "0",
                    "message": "Document queued for batch processing",
                    "created_at": datetime.utcnow().isoformat(),
                    "updated_at": datetime.utcnow().isoformat()
                }
            )
            
            # Start background processing
            background_tasks.add_task(
                process_document_async,
                document_id,
                file_path,
                user_id,
                processing_options,
                context.redis_client
            )
        
        # Store batch info
        await context.redis_client.hset(
            f"batch:{batch_id}",
            mapping={
                "user_id": user_id,
                "batch_name": batch_options.batch_name or f"Batch {batch_id}",
                "total_documents": len(document_ids),
                "document_ids": ",".join(document_ids),
                "status": ProcessingStatus.QUEUED.value,
                "priority": batch_options.priority,
                "callback_url": batch_options.callback_url or "",
                "created_at": datetime.utcnow().isoformat(),
                "updated_at": datetime.utcnow().isoformat()
            }
        )
        
        # Set expiration (24 hours)
        await context.redis_client.expire(f"batch:{batch_id}", 86400)
        
        # Log API usage
        processing_time = (datetime.utcnow() - start_time).total_seconds()
        await log_api_usage(
            context,
            "/documents/batch",
            "POST",
            200,
            processing_time
        )
        
        return BatchProcessingResponse(
            batch_id=batch_id,
            document_ids=document_ids,
            total_documents=len(document_ids),
            status=ProcessingStatus.QUEUED,
            estimated_completion=datetime.utcnow() + timedelta(minutes=len(document_ids) * 2),
            batch_url=f"/api/v1/documents/batch/{batch_id}"
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Batch upload failed: {e}", exc_info=True)
        
        # Log API usage with error
        processing_time = (datetime.utcnow() - start_time).total_seconds()
        await log_api_usage(
            context,
            "/documents/batch",
            "POST",
            500,
            processing_time,
            error=str(e)
        )
        
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Batch upload failed"
        )

@router.get("/batch/{batch_id}")
async def get_batch_status(
    batch_id: str,
    context: RequestContext = Depends(get_request_context)
):
    """Get batch processing status."""
    start_time = datetime.utcnow()
    
    try:
        # Get batch info
        batch_data = await context.redis_client.hgetall(f"batch:{batch_id}")
        
        if not batch_data:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Batch not found"
            )
        
        # Check access
        batch_user_id = batch_data.get("user_id")
        if context.user_id and batch_user_id != context.user_id and not context.has_role("admin"):
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="Access denied"
            )
        
        # Get document statuses
        document_ids = batch_data.get("document_ids", "").split(",")
        document_statuses = []
        
        completed_count = 0
        failed_count = 0
        processing_count = 0
        
        for doc_id in document_ids:
            if doc_id:
                doc_data = await context.redis_client.hgetall(f"document:{doc_id}")
                if doc_data:
                    doc_status = doc_data.get("status")
                    document_statuses.append({
                        "document_id": doc_id,
                        "status": doc_status,
                        "progress": float(doc_data.get("progress", 0)),
                        "message": doc_data.get("message", "")
                    })
                    
                    if doc_status == ProcessingStatus.COMPLETED.value:
                        completed_count += 1
                    elif doc_status == ProcessingStatus.FAILED.value:
                        failed_count += 1
                    elif doc_status == ProcessingStatus.PROCESSING.value:
                        processing_count += 1
        
        # Determine batch status
        total_docs = len(document_ids)
        if completed_count == total_docs:
            batch_status = ProcessingStatus.COMPLETED
        elif failed_count == total_docs:
            batch_status = ProcessingStatus.FAILED
        elif completed_count + failed_count == total_docs:
            batch_status = ProcessingStatus.COMPLETED  # Partial completion
        else:
            batch_status = ProcessingStatus.PROCESSING
        
        # Calculate overall progress
        overall_progress = (completed_count + failed_count) / total_docs * 100 if total_docs > 0 else 0
        
        # Log API usage
        processing_time = (datetime.utcnow() - start_time).total_seconds()
        await log_api_usage(
            context,
            f"/documents/batch/{batch_id}",
            "GET",
            200,
            processing_time
        )
        
        return {
            "batch_id": batch_id,
            "batch_name": batch_data.get("batch_name"),
            "status": batch_status.value,
            "total_documents": total_docs,
            "completed_documents": completed_count,
            "failed_documents": failed_count,
            "processing_documents": processing_count,
            "overall_progress": overall_progress,
            "created_at": batch_data.get("created_at"),
            "updated_at": datetime.utcnow().isoformat(),
            "documents": document_statuses
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to get batch status: {e}", exc_info=True)
        
        # Log API usage with error
        processing_time = (datetime.utcnow() - start_time).total_seconds()
        await log_api_usage(
            context,
            f"/documents/batch/{batch_id}",
            "GET",
            500,
            processing_time,
            error=str(e)
        )
        
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to retrieve batch status"
        )

@router.delete("/{document_id}")
async def delete_document(
    document_id: str,
    context: RequestContext = Depends(get_request_context)
):
    """Delete document and its processing results."""
    start_time = datetime.utcnow()
    
    try:
        # Get document info
        doc_data = await context.redis_client.hgetall(f"document:{document_id}")
        
        if not doc_data:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Document not found"
            )
        
        # Check access
        doc_user_id = doc_data.get("user_id")
        if context.user_id and doc_user_id != context.user_id and not context.has_role("admin"):
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="Access denied"
            )
        
        # Delete from Redis
        await context.redis_client.delete(f"document:{document_id}")
        
        # Log API usage
        processing_time = (datetime.utcnow() - start_time).total_seconds()
        await log_api_usage(
            context,
            f"/documents/{document_id}",
            "DELETE",
            200,
            processing_time
        )
        
        return {"message": "Document deleted successfully"}
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to delete document: {e}", exc_info=True)
        
        # Log API usage with error
        processing_time = (datetime.utcnow() - start_time).total_seconds()
        await log_api_usage(
            context,
            f"/documents/{document_id}",
            "DELETE",
            500,
            processing_time,
            error=str(e)
        )
        
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to delete document"
        )

@router.get("/user/documents")
async def get_user_documents(
    context: RequestContext = Depends(require_authentication),
    limit: int = Query(10, ge=1, le=100, description="Number of documents to return"),
    offset: int = Query(0, ge=0, description="Number of documents to skip")
):
    """Get user's document processing history."""
    start_time = datetime.utcnow()
    
    try:
        # Get user's documents from Redis
        # Note: In production, use proper database with indexing
        pattern = f"document:*"
        keys = await context.redis_client.keys(pattern)
        
        user_documents = []
        for key in keys:
            doc_data = await context.redis_client.hgetall(key)
            if doc_data.get("user_id") == context.user_id:
                document_id = key.split(":")[1]
                user_documents.append({
                    "document_id": document_id,
                    "filename": doc_data.get("filename"),
                    "status": doc_data.get("status"),
                    "created_at": doc_data.get("created_at"),
                    "updated_at": doc_data.get("updated_at")
                })
        
        # Sort by creation time (newest first)
        user_documents.sort(key=lambda x: x["created_at"], reverse=True)
        
        # Apply pagination
        total_count = len(user_documents)
        paginated_documents = user_documents[offset:offset + limit]
        
        # Log API usage
        processing_time = (datetime.utcnow() - start_time).total_seconds()
        await log_api_usage(
            context,
            "/documents/user/documents",
            "GET",
            200,
            processing_time
        )
        
        return {
            "documents": paginated_documents,
            "total_count": total_count,
            "limit": limit,
            "offset": offset,
            "has_more": offset + limit < total_count
        }
        
    except Exception as e:
        logger.error(f"Failed to get user documents: {e}", exc_info=True)
        
        # Log API usage with error
        processing_time = (datetime.utcnow() - start_time).total_seconds()
        await log_api_usage(
            context,
            "/documents/user/documents",
            "GET",
            500,
            processing_time,
            error=str(e)
        )
        
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to retrieve user documents"
        )