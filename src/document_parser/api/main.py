#!/usr/bin/env python3
"""
Main FastAPI Application

Core FastAPI application with middleware, exception handlers,
and route registration for the document parser API.
"""

import logging
import time
import uuid
from contextlib import asynccontextmanager
from typing import Dict, Any, Optional

from fastapi import FastAPI, Request, HTTPException, status, Depends
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.trustedhost import TrustedHostMiddleware
from fastapi.responses import JSONResponse
from fastapi.openapi.docs import get_swagger_ui_html
from fastapi.openapi.utils import get_openapi
from fastapi.security import HTTPBearer, OAuth2PasswordBearer

from prometheus_fastapi_instrumentator import Instrumentator

from ..config import get_config
from ..database import init_database
from ..models.document_models import DocumentType
from .routes import documents, health, admin
from .dependencies import get_database, get_redis, get_request_context, RequestContext
from .auth import AuthenticationService, JWTManager, APIKeyManager, UserRole
from .rate_limiting import RateLimiter, RateLimitMiddleware, UserTier
from .models import APIResponse, APIError, ErrorDetail, ErrorType

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Global variables for services
config = get_config()
redis_client = None
ml_models = {}
processing_stats = {
    "total_requests": 0,
    "successful_requests": 0,
    "failed_requests": 0,
    "average_processing_time": 0.0
}

# Authentication and rate limiting services
jwt_manager = None
api_key_manager = None
authentication_service = None
rate_limiter = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan manager."""
    # Startup
    logger.info("Starting Document Parser API...")
    
    try:
        # Initialize database
        await init_database()
        logger.info("Database initialized")
        
        # Initialize Redis
        global redis_client
        redis_client = await get_redis()
        await redis_client.ping()
        logger.info("Redis connection established")
        
        # Initialize authentication services
        global jwt_manager, api_key_manager, authentication_service, rate_limiter
        jwt_manager = JWTManager(config.security.secret_key)
        api_key_manager = APIKeyManager(redis_client)
        
        # Initialize rate limiter
        rate_limiter = RateLimiter(redis_client)
        logger.info("Authentication and rate limiting services initialized")
        
        # Load ML models
        logger.info("Loading ML models...")
        # Add model loading logic here
        logger.info("ML models loaded successfully")
        
        logger.info("Document Parser API started successfully")
        
    except Exception as e:
        logger.error(f"Failed to start application: {e}")
        raise
    
    yield
    
    # Shutdown
    logger.info("Shutting down Document Parser API...")
    
    try:
        # Close Redis connection
        if redis_client:
            await redis_client.close()
            logger.info("Redis connection closed")
        
        # Cleanup ML models
        global ml_models
        ml_models.clear()
        logger.info("ML models cleaned up")
        
        logger.info("Document Parser API shut down successfully")
        
    except Exception as e:
        logger.error(f"Error during shutdown: {e}")

# Create FastAPI application
app = FastAPI(
    title="Document Parser API",
    description="Malaysian Document Parsing and Field Extraction API",
    version="1.0.0",
    docs_url="/docs" if config.api.enable_docs else None,
    redoc_url="/redoc" if config.api.enable_docs else None,
    openapi_url="/openapi.json" if config.api.enable_docs else None,
    lifespan=lifespan
)

# Add middleware

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=config.api.cors_origins,
    allow_credentials=True,
    allow_methods=["GET", "POST", "PUT", "DELETE"],
    allow_headers=["*"],
)

# Trusted host middleware
if config.api.trusted_hosts:
    app.add_middleware(
        TrustedHostMiddleware,
        allowed_hosts=config.api.trusted_hosts
    )

# Rate limiting middleware
@app.middleware("http")
async def rate_limit_middleware(request: Request, call_next):
    """Rate limiting middleware."""
    if rate_limiter:
        middleware = RateLimitMiddleware(rate_limiter)
        return await middleware(request, call_next)
    return await call_next(request)

# Request timing middleware
@app.middleware("http")
async def add_process_time_header(request: Request, call_next):
    """Add processing time to response headers."""
    start_time = time.time()
    response = await call_next(request)
    process_time = time.time() - start_time
    response.headers["X-Process-Time"] = str(process_time)
    return response

# Request logging middleware
@app.middleware("http")
async def log_requests(request: Request, call_next):
    """Log all incoming requests."""
    start_time = time.time()
    
    # Log request
    logger.info(
        f"Request: {request.method} {request.url.path} "
        f"from {request.client.host if request.client else 'unknown'}"
    )
    
    response = await call_next(request)
    
    # Log response
    process_time = time.time() - start_time
    logger.info(
        f"Response: {response.status_code} "
        f"({process_time:.3f}s)"
    )
    
    return response

# Exception handlers

@app.exception_handler(HTTPException)
async def http_exception_handler(request: Request, exc: HTTPException):
    """Handle HTTP exceptions."""
    return JSONResponse(
        status_code=exc.status_code,
        content={
            "error": {
                "code": exc.status_code,
                "message": exc.detail,
                "type": "http_error",
                "timestamp": time.time()
            }
        }
    )

@app.exception_handler(ValueError)
async def value_error_handler(request: Request, exc: ValueError):
    """Handle value errors."""
    logger.error(f"ValueError in {request.url.path}: {exc}")
    return JSONResponse(
        status_code=status.HTTP_400_BAD_REQUEST,
        content={
            "error": {
                "code": 400,
                "message": str(exc),
                "type": "validation_error",
                "timestamp": time.time()
            }
        }
    )

@app.exception_handler(Exception)
async def general_exception_handler(request: Request, exc: Exception):
    """Handle general exceptions."""
    logger.error(f"Unhandled exception in {request.url.path}: {exc}", exc_info=True)
    return JSONResponse(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        content={
            "error": {
                "code": 500,
                "message": "Internal server error",
                "type": "internal_error",
                "timestamp": time.time()
            }
        }
    )

# Custom OpenAPI schema
def custom_openapi():
    """Generate custom OpenAPI schema."""
    if app.openapi_schema:
        return app.openapi_schema
    
    openapi_schema = get_openapi(
        title="Document Parser API",
        version="1.0.0",
        description="""
        ## Malaysian Document Parsing API
        
        This API provides comprehensive document parsing capabilities for Malaysian documents including:
        
        - **MyKad (Malaysian Identity Card)**
        - **SPK (Sijil Pelajaran Malaysia)**
        - **Passport**
        - **Driving License**
        
        ### Features
        
        - **Document Classification**: Automatic document type detection
        - **OCR Processing**: Multi-language text extraction with preprocessing
        - **Field Extraction**: Intelligent field identification and extraction
        - **Data Validation**: Business logic validation for Malaysian standards
        - **Confidence Scoring**: Quality assessment for extracted data
        - **Batch Processing**: Multiple document processing support
        
        ### Authentication
        
        API uses JWT token-based authentication. Include the token in the Authorization header:
        ```
        Authorization: Bearer <your-jwt-token>
        ```
        
        ### Rate Limiting
        
        - **Standard users**: 100 requests per minute
        - **Premium users**: 1000 requests per minute
        
        ### Error Handling
        
        All errors follow a consistent format:
        ```json
        {
            "error": {
                "code": 400,
                "message": "Error description",
                "type": "error_type",
                "timestamp": 1642678800.123
            }
        }
        ```
        """,
        routes=app.routes,
    )
    
    # Add custom info
    openapi_schema["info"]["x-logo"] = {
        "url": "/static/logo.png"
    }
    
    # Add security schemes
    openapi_schema["components"]["securitySchemes"] = {
        "BearerAuth": {
            "type": "http",
            "scheme": "bearer",
            "bearerFormat": "JWT"
        },
        "ApiKeyAuth": {
            "type": "apiKey",
            "in": "header",
            "name": "X-API-Key"
        }
    }
    
    app.openapi_schema = openapi_schema
    return app.openapi_schema

app.openapi = custom_openapi

# Include routers
app.include_router(
    health.router,
    prefix="/health",
    tags=["Health"]
)

app.include_router(
    documents.router,
    prefix="/api/v1/documents",
    tags=["Documents"]
)

app.include_router(
    admin.router,
    prefix="/api/v1/admin",
    tags=["Admin"]
)

from .routes import auth
app.include_router(
    auth.router,
    prefix="/api/v1",
    tags=["Authentication"]
)

# Root endpoint
@app.get("/", tags=["Root"])
async def root() -> Dict[str, Any]:
    """API root endpoint with basic information."""
    return {
        "service": "Document Parser API",
        "version": "1.0.0",
        "status": "operational",
        "docs_url": "/docs",
        "health_check": "/health",
        "supported_documents": [
            "MyKad (Malaysian Identity Card)",
            "SPK (Sijil Pelajaran Malaysia)",
            "Passport",
            "Driving License"
        ],
        "features": [
            "Document Classification",
            "OCR Text Extraction",
            "Field Extraction",
            "Data Validation",
            "Confidence Scoring",
            "Batch Processing"
        ]
    }

# Metrics endpoint (Prometheus)
if config.monitoring.enable_metrics:
    instrumentator = Instrumentator(
        should_group_status_codes=False,
        should_ignore_untemplated=True,
        should_respect_env_var=True,
        should_instrument_requests_inprogress=True,
        excluded_handlers=["/metrics"],
        env_var_name="ENABLE_METRICS",
        inprogress_name="inprogress",
        inprogress_labels=True,
    )
    
    instrumentator.instrument(app).expose(app)

# Custom metrics
from prometheus_client import Counter, Histogram, Gauge

# Document processing metrics
document_processed_total = Counter(
    "document_processed_total",
    "Total number of documents processed",
    ["document_type", "status"]
)

processing_duration = Histogram(
    "document_processing_duration_seconds",
    "Time spent processing documents",
    ["document_type", "operation"]
)

active_processing_jobs = Gauge(
    "active_processing_jobs",
    "Number of documents currently being processed"
)

ocr_confidence_score = Histogram(
    "ocr_confidence_score",
    "OCR confidence scores",
    ["document_type"]
)

# Make metrics available to routes
app.state.metrics = {
    "document_processed_total": document_processed_total,
    "processing_duration": processing_duration,
    "active_processing_jobs": active_processing_jobs,
    "ocr_confidence_score": ocr_confidence_score
}

if __name__ == "__main__":
    import uvicorn
    
    uvicorn.run(
        "main:app",
        host=config.api.host,
        port=config.api.port,
        reload=config.api.debug,
        log_level="info" if not config.api.debug else "debug"
    )