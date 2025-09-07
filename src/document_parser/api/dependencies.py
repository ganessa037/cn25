#!/usr/bin/env python3
"""
API Dependencies

Dependency injection functions for FastAPI routes.
Provides database connections, authentication, rate limiting, and other shared resources.
"""

import logging
from typing import Optional, Dict, Any, Annotated
from functools import lru_cache

from fastapi import Depends, HTTPException, status, Request, Header
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from sqlalchemy.ext.asyncio import AsyncSession
import redis.asyncio as redis
import jwt
from datetime import datetime, timedelta

from ..config import Config, get_config
from ..database import get_async_session
from ..document_classifier import DocumentClassifier
from ..ocr_service import OCRService
from ..field_extractor import FieldExtractor
from ..validator import DocumentValidator

# Configure logging
logger = logging.getLogger(__name__)

# Security
security = HTTPBearer(auto_error=False)

# Cache for expensive operations
@lru_cache()
def get_cached_config() -> Config:
    """Get cached configuration."""
    return get_config()

# Database dependency
async def get_database() -> AsyncSession:
    """Get database session."""
    async with get_async_session() as session:
        try:
            yield session
        except Exception as e:
            logger.error(f"Database error: {e}")
            await session.rollback()
            raise
        finally:
            await session.close()

# Redis dependency
_redis_client: Optional[redis.Redis] = None

async def get_redis() -> redis.Redis:
    """Get Redis client."""
    global _redis_client
    
    if _redis_client is None:
        config = get_cached_config()
        _redis_client = redis.Redis(
            host=config.redis.host,
            port=config.redis.port,
            password=config.redis.password,
            db=config.redis.db,
            decode_responses=True,
            socket_connect_timeout=5,
            socket_timeout=5,
            retry_on_timeout=True
        )
    
    try:
        # Test connection
        await _redis_client.ping()
        return _redis_client
    except Exception as e:
        logger.error(f"Redis connection error: {e}")
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Cache service unavailable"
        )

# Configuration dependency
def get_config_dependency() -> Config:
    """Get application configuration."""
    return get_cached_config()

# Authentication dependencies

class AuthenticationError(Exception):
    """Authentication error."""
    pass

class AuthorizationError(Exception):
    """Authorization error."""
    pass

async def verify_jwt_token(token: str) -> Dict[str, Any]:
    """Verify JWT token and return payload."""
    config = get_cached_config()
    
    try:
        payload = jwt.decode(
            token,
            config.security.jwt_secret,
            algorithms=[config.security.jwt_algorithm]
        )
        
        # Check expiration
        exp = payload.get('exp')
        if exp and datetime.utcnow().timestamp() > exp:
            raise AuthenticationError("Token expired")
        
        return payload
        
    except jwt.InvalidTokenError as e:
        raise AuthenticationError(f"Invalid token: {e}")
    except Exception as e:
        raise AuthenticationError(f"Token verification failed: {e}")

async def get_current_user(
    credentials: Annotated[Optional[HTTPAuthorizationCredentials], Depends(security)],
    redis_client: Annotated[redis.Redis, Depends(get_redis)]
) -> Optional[Dict[str, Any]]:
    """Get current authenticated user."""
    if not credentials:
        return None
    
    try:
        # Verify token
        payload = await verify_jwt_token(credentials.credentials)
        
        # Check if token is blacklisted
        token_id = payload.get('jti')
        if token_id:
            is_blacklisted = await redis_client.get(f"blacklist:{token_id}")
            if is_blacklisted:
                raise AuthenticationError("Token has been revoked")
        
        # Get user info from payload
        user_info = {
            'user_id': payload.get('sub'),
            'username': payload.get('username'),
            'email': payload.get('email'),
            'roles': payload.get('roles', []),
            'permissions': payload.get('permissions', []),
            'plan': payload.get('plan', 'free'),
            'token_id': token_id
        }
        
        return user_info
        
    except AuthenticationError:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid authentication credentials",
            headers={"WWW-Authenticate": "Bearer"},
        )
    except Exception as e:
        logger.error(f"Authentication error: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Authentication service error"
        )

async def require_authentication(
    current_user: Annotated[Optional[Dict[str, Any]], Depends(get_current_user)]
) -> Dict[str, Any]:
    """Require user to be authenticated."""
    if not current_user:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Authentication required",
            headers={"WWW-Authenticate": "Bearer"},
        )
    return current_user

async def require_admin(
    current_user: Annotated[Dict[str, Any], Depends(require_authentication)]
) -> Dict[str, Any]:
    """Require user to have admin role."""
    if 'admin' not in current_user.get('roles', []):
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Admin access required"
        )
    return current_user

# API Key authentication
async def verify_api_key(
    x_api_key: Annotated[Optional[str], Header(alias="X-API-Key")] = None,
    redis_client: Annotated[redis.Redis, Depends(get_redis)] = None
) -> Optional[Dict[str, Any]]:
    """Verify API key."""
    if not x_api_key:
        return None
    
    try:
        # Check API key in Redis
        api_key_data = await redis_client.hgetall(f"api_key:{x_api_key}")
        
        if not api_key_data:
            raise AuthenticationError("Invalid API key")
        
        # Check if API key is active
        if api_key_data.get('status') != 'active':
            raise AuthenticationError("API key is not active")
        
        # Check expiration
        expires_at = api_key_data.get('expires_at')
        if expires_at and datetime.utcnow().timestamp() > float(expires_at):
            raise AuthenticationError("API key expired")
        
        return {
            'api_key': x_api_key,
            'user_id': api_key_data.get('user_id'),
            'plan': api_key_data.get('plan', 'free'),
            'permissions': api_key_data.get('permissions', '').split(',') if api_key_data.get('permissions') else [],
            'rate_limit': int(api_key_data.get('rate_limit', 100))
        }
        
    except AuthenticationError:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid API key"
        )
    except Exception as e:
        logger.error(f"API key verification error: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="API key verification service error"
        )

# Rate limiting
class RateLimiter:
    """Rate limiter using Redis."""
    
    def __init__(self, redis_client: redis.Redis):
        self.redis = redis_client
    
    async def is_allowed(
        self,
        key: str,
        limit: int,
        window: int = 60
    ) -> tuple[bool, Dict[str, Any]]:
        """Check if request is allowed under rate limit."""
        try:
            # Use sliding window rate limiting
            now = datetime.utcnow().timestamp()
            pipeline = self.redis.pipeline()
            
            # Remove old entries
            pipeline.zremrangebyscore(key, 0, now - window)
            
            # Count current requests
            pipeline.zcard(key)
            
            # Add current request
            pipeline.zadd(key, {str(now): now})
            
            # Set expiration
            pipeline.expire(key, window)
            
            results = await pipeline.execute()
            current_count = results[1]
            
            # Calculate remaining and reset time
            remaining = max(0, limit - current_count - 1)
            reset_time = int(now + window)
            
            rate_limit_info = {
                'limit': limit,
                'remaining': remaining,
                'reset': reset_time,
                'current': current_count + 1
            }
            
            return current_count < limit, rate_limit_info
            
        except Exception as e:
            logger.error(f"Rate limiting error: {e}")
            # Allow request if rate limiting fails
            return True, {'limit': limit, 'remaining': limit - 1, 'reset': int(now + window), 'current': 1}

async def check_rate_limit(
    request: Request,
    redis_client: Annotated[redis.Redis, Depends(get_redis)],
    current_user: Annotated[Optional[Dict[str, Any]], Depends(get_current_user)] = None,
    api_key_user: Annotated[Optional[Dict[str, Any]], Depends(verify_api_key)] = None
) -> Dict[str, Any]:
    """Check rate limit for current request."""
    # Determine user and rate limit
    user = current_user or api_key_user
    
    if user:
        # Authenticated user
        user_id = user.get('user_id', 'unknown')
        plan = user.get('plan', 'free')
        rate_limit = user.get('rate_limit', 100 if plan == 'free' else 1000)
        rate_key = f"rate_limit:user:{user_id}"
    else:
        # Anonymous user (by IP)
        client_ip = request.client.host if request.client else 'unknown'
        rate_limit = 10  # Very low limit for anonymous users
        rate_key = f"rate_limit:ip:{client_ip}"
    
    # Check rate limit
    rate_limiter = RateLimiter(redis_client)
    is_allowed, rate_info = await rate_limiter.is_allowed(rate_key, rate_limit)
    
    if not is_allowed:
        raise HTTPException(
            status_code=status.HTTP_429_TOO_MANY_REQUESTS,
            detail="Rate limit exceeded",
            headers={
                "X-RateLimit-Limit": str(rate_info['limit']),
                "X-RateLimit-Remaining": str(rate_info['remaining']),
                "X-RateLimit-Reset": str(rate_info['reset']),
                "Retry-After": str(rate_info['reset'] - int(datetime.utcnow().timestamp()))
            }
    )
    
    return rate_info

# Service dependencies

_document_classifier: Optional[DocumentClassifier] = None
_ocr_service: Optional[OCRService] = None
_field_extractor: Optional[FieldExtractor] = None
_document_validator: Optional[DocumentValidator] = None

async def get_document_classifier() -> DocumentClassifier:
    """Get document classifier service."""
    global _document_classifier
    
    if _document_classifier is None:
        config = get_cached_config()
        _document_classifier = DocumentClassifier(config)
        await _document_classifier.load_models()
    
    return _document_classifier

async def get_ocr_service() -> OCRService:
    """Get OCR service."""
    global _ocr_service
    
    if _ocr_service is None:
        config = get_cached_config()
        _ocr_service = OCRService(config)
        await _ocr_service.initialize()
    
    return _ocr_service

async def get_field_extractor() -> FieldExtractor:
    """Get field extractor service."""
    global _field_extractor
    
    if _field_extractor is None:
        config = get_cached_config()
        _field_extractor = FieldExtractor(config)
        await _field_extractor.load_models()
    
    return _field_extractor

async def get_document_validator() -> DocumentValidator:
    """Get document validator service."""
    global _document_validator
    
    if _document_validator is None:
        config = get_cached_config()
        _document_validator = DocumentValidator(config)
    
    return _document_validator

# Request context
class RequestContext:
    """Request context with user info and services."""
    
    def __init__(
        self,
        request: Request,
        user: Optional[Dict[str, Any]] = None,
        rate_limit_info: Optional[Dict[str, Any]] = None,
        db_session: Optional[AsyncSession] = None,
        redis_client: Optional[redis.Redis] = None
    ):
        self.request = request
        self.user = user
        self.rate_limit_info = rate_limit_info
        self.db_session = db_session
        self.redis_client = redis_client
        self.request_id = request.headers.get('X-Request-ID', 'unknown')
        self.start_time = datetime.utcnow()
    
    @property
    def is_authenticated(self) -> bool:
        return self.user is not None
    
    @property
    def user_id(self) -> Optional[str]:
        return self.user.get('user_id') if self.user else None
    
    @property
    def user_plan(self) -> str:
        return self.user.get('plan', 'free') if self.user else 'anonymous'
    
    def has_permission(self, permission: str) -> bool:
        if not self.user:
            return False
        return permission in self.user.get('permissions', [])
    
    def has_role(self, role: str) -> bool:
        if not self.user:
            return False
        return role in self.user.get('roles', [])

async def get_request_context(
    request: Request,
    db_session: Annotated[AsyncSession, Depends(get_database)],
    redis_client: Annotated[redis.Redis, Depends(get_redis)],
    rate_limit_info: Annotated[Dict[str, Any], Depends(check_rate_limit)],
    current_user: Annotated[Optional[Dict[str, Any]], Depends(get_current_user)] = None,
    api_key_user: Annotated[Optional[Dict[str, Any]], Depends(verify_api_key)] = None
) -> RequestContext:
    """Get request context with all dependencies."""
    user = current_user or api_key_user
    
    return RequestContext(
        request=request,
        user=user,
        rate_limit_info=rate_limit_info,
        db_session=db_session,
        redis_client=redis_client
    )

# Utility functions

def get_client_ip(request: Request) -> str:
    """Get client IP address from request."""
    # Check for forwarded headers
    forwarded_for = request.headers.get('X-Forwarded-For')
    if forwarded_for:
        return forwarded_for.split(',')[0].strip()
    
    real_ip = request.headers.get('X-Real-IP')
    if real_ip:
        return real_ip
    
    # Fallback to direct client IP
    return request.client.host if request.client else 'unknown'

def get_user_agent(request: Request) -> str:
    """Get user agent from request."""
    return request.headers.get('User-Agent', 'unknown')

async def log_api_usage(
    context: RequestContext,
    endpoint: str,
    method: str,
    status_code: int,
    processing_time: float,
    document_type: Optional[str] = None,
    error: Optional[str] = None
):
    """Log API usage for analytics."""
    try:
        usage_data = {
            'timestamp': context.start_time.isoformat(),
            'request_id': context.request_id,
            'user_id': context.user_id,
            'user_plan': context.user_plan,
            'endpoint': endpoint,
            'method': method,
            'status_code': status_code,
            'processing_time': processing_time,
            'document_type': document_type,
            'error': error,
            'client_ip': get_client_ip(context.request),
            'user_agent': get_user_agent(context.request)
        }
        
        # Store in Redis for analytics
        await context.redis_client.lpush(
            'api_usage_log',
            str(usage_data)
        )
        
        # Keep only last 10000 entries
        await context.redis_client.ltrim('api_usage_log', 0, 9999)
        
    except Exception as e:
        logger.error(f"Failed to log API usage: {e}")