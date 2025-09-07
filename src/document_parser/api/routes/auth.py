#!/usr/bin/env python3
"""
Authentication Routes

FastAPI routes for user authentication, token management, and API key operations.
Provides comprehensive authentication endpoints with JWT and API key support.
"""

import logging
from datetime import datetime, timedelta
from typing import Dict, Any, Optional, List

from fastapi import APIRouter, Depends, HTTPException, status, Request, Form
from fastapi.security import OAuth2PasswordRequestForm
from pydantic import BaseModel, Field, EmailStr
from sqlalchemy.ext.asyncio import AsyncSession

from ..dependencies import get_database, get_redis, get_request_context, RequestContext
from ..auth import (
    AuthenticationService, JWTManager, APIKeyManager, UserRole, TokenType,
    TokenPayload, AuthenticationResult, APIKeyInfo
)
from ..models import (
    APIResponse, LoginRequest, TokenResponse, RefreshTokenRequest, UserInfo,
    APIError, ErrorDetail, ErrorType
)

# Configure logging
logger = logging.getLogger(__name__)

# Create router
router = APIRouter(prefix="/auth", tags=["Authentication"])

# Request/Response Models

class RegisterRequest(BaseModel):
    """User registration request."""
    username: str = Field(..., min_length=3, max_length=50, description="Username")
    email: EmailStr = Field(..., description="Email address")
    password: str = Field(..., min_length=8, description="Password")
    full_name: Optional[str] = Field(None, max_length=100, description="Full name")
    organization: Optional[str] = Field(None, max_length=100, description="Organization")
    terms_accepted: bool = Field(..., description="Terms and conditions acceptance")

class PasswordChangeRequest(BaseModel):
    """Password change request."""
    current_password: str = Field(..., description="Current password")
    new_password: str = Field(..., min_length=8, description="New password")

class PasswordResetRequest(BaseModel):
    """Password reset request."""
    email: EmailStr = Field(..., description="Email address")

class PasswordResetConfirm(BaseModel):
    """Password reset confirmation."""
    token: str = Field(..., description="Reset token")
    new_password: str = Field(..., min_length=8, description="New password")

class APIKeyCreateRequest(BaseModel):
    """API key creation request."""
    name: str = Field(..., max_length=100, description="API key name")
    permissions: List[str] = Field(default_factory=list, description="API key permissions")
    rate_limit: int = Field(1000, ge=1, le=10000, description="Rate limit per hour")
    expires_in_days: Optional[int] = Field(None, ge=1, le=365, description="Expiration in days")
    allowed_ips: Optional[List[str]] = Field(None, description="Allowed IP addresses")

class APIKeyResponse(BaseModel):
    """API key creation response."""
    key_id: str = Field(..., description="API key ID")
    api_key: str = Field(..., description="API key (shown only once)")
    name: str = Field(..., description="API key name")
    permissions: List[str] = Field(..., description="API key permissions")
    rate_limit: int = Field(..., description="Rate limit per hour")
    created_at: datetime = Field(..., description="Creation timestamp")
    expires_at: Optional[datetime] = Field(None, description="Expiration timestamp")

class APIKeyListResponse(APIResponse):
    """API key list response."""
    api_keys: List[Dict[str, Any]] = Field(..., description="List of API keys")

# Helper Functions

def get_auth_service(context: RequestContext = Depends(get_request_context)) -> AuthenticationService:
    """Get authentication service instance."""
    # This would be injected from the main app
    from ..main import jwt_manager, api_key_manager
    
    if not jwt_manager or not api_key_manager:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Authentication service not available"
        )
    
    return AuthenticationService(
        jwt_manager=jwt_manager,
        api_key_manager=api_key_manager,
        redis_client=context.redis_client,
        db_session=context.db_session
    )

# Authentication Routes

@router.post("/register", response_model=APIResponse)
async def register_user(
    request: RegisterRequest,
    context: RequestContext = Depends(get_request_context)
):
    """Register a new user account."""
    try:
        # Check if terms are accepted
        if not request.terms_accepted:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Terms and conditions must be accepted"
            )
        
        # TODO: Implement user registration logic
        # This would involve:
        # 1. Check if username/email already exists
        # 2. Hash password
        # 3. Create user record
        # 4. Send verification email
        
        logger.info(f"User registration attempt: {request.username}")
        
        return APIResponse(
            success=True,
            message="Registration successful. Please check your email for verification."
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Registration error: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Registration failed"
        )

@router.post("/login", response_model=TokenResponse)
async def login_user(
    form_data: OAuth2PasswordRequestForm = Depends(),
    request: Request = None,
    auth_service: AuthenticationService = Depends(get_auth_service)
):
    """Authenticate user and return access token."""
    try:
        # Get client information
        ip_address = request.client.host if request and request.client else "unknown"
        user_agent = request.headers.get("user-agent", "unknown") if request else "unknown"
        
        # Authenticate user
        auth_result = await auth_service.authenticate_user(
            username=form_data.username,
            password=form_data.password,
            ip_address=ip_address,
            user_agent=user_agent
        )
        
        if not auth_result.success:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail=auth_result.error_message or "Authentication failed",
                headers={"WWW-Authenticate": "Bearer"}
            )
        
        # Create session
        session_id = await auth_service.create_session(
            user_id=auth_result.user_id,
            ip_address=ip_address,
            user_agent=user_agent
        )
        
        # Create token payload
        token_payload = TokenPayload(
            user_id=auth_result.user_id,
            username=auth_result.username,
            email=auth_result.email,
            role=auth_result.role,
            permissions=auth_result.permissions,
            token_type=TokenType.ACCESS,
            issued_at=datetime.utcnow(),
            expires_at=datetime.utcnow() + timedelta(hours=1),
            session_id=session_id,
            ip_address=ip_address
        )
        
        # Generate tokens
        access_token = auth_service.jwt_manager.create_access_token(token_payload)
        refresh_token = auth_service.jwt_manager.create_refresh_token(token_payload)
        
        logger.info(f"User login successful: {auth_result.username}")
        
        return TokenResponse(
            access_token=access_token,
            token_type="bearer",
            expires_in=3600,  # 1 hour
            refresh_token=refresh_token,
            user_id=auth_result.user_id,
            permissions=auth_result.permissions
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Login error: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Login failed"
        )

@router.post("/refresh", response_model=TokenResponse)
async def refresh_token(
    request: RefreshTokenRequest,
    auth_service: AuthenticationService = Depends(get_auth_service)
):
    """Refresh access token using refresh token."""
    try:
        # Verify refresh token
        token_payload = auth_service.jwt_manager.verify_token(request.refresh_token)
        
        if token_payload.token_type != TokenType.REFRESH:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Invalid token type"
            )
        
        # Create new access token
        new_payload = TokenPayload(
            user_id=token_payload.user_id,
            username=token_payload.username,
            email=token_payload.email,
            role=token_payload.role,
            permissions=token_payload.permissions,
            token_type=TokenType.ACCESS,
            issued_at=datetime.utcnow(),
            expires_at=datetime.utcnow() + timedelta(hours=1),
            session_id=token_payload.session_id
        )
        
        access_token = auth_service.jwt_manager.create_access_token(new_payload)
        
        return TokenResponse(
            access_token=access_token,
            token_type="bearer",
            expires_in=3600,
            user_id=token_payload.user_id,
            permissions=token_payload.permissions
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Token refresh error: {e}")
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Token refresh failed"
        )

@router.post("/logout", response_model=APIResponse)
async def logout_user(
    context: RequestContext = Depends(get_request_context),
    auth_service: AuthenticationService = Depends(get_auth_service)
):
    """Logout user and revoke session."""
    try:
        if not context.is_authenticated:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Not authenticated"
            )
        
        # Get session ID from token
        auth_header = context.request.headers.get("authorization")
        if auth_header and auth_header.startswith("Bearer "):
            token = auth_header[7:]
            token_payload = auth_service.jwt_manager.verify_token(token)
            
            if token_payload.session_id:
                await auth_service.revoke_session(token_payload.session_id)
        
        logger.info(f"User logout: {context.user_id}")
        
        return APIResponse(
            success=True,
            message="Logout successful"
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Logout error: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Logout failed"
        )

@router.get("/me", response_model=UserInfo)
async def get_current_user(
    context: RequestContext = Depends(get_request_context)
):
    """Get current user information."""
    if not context.is_authenticated:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Not authenticated"
        )
    
    # TODO: Get full user information from database
    return UserInfo(
        user_id=context.user_id,
        username=context.user.get("username", "unknown"),
        email=context.user.get("email", "unknown"),
        role=context.user.get("role", "user"),
        permissions=context.user.get("permissions", []),
        created_at=datetime.utcnow(),  # TODO: Get from database
        is_active=True
    )

# API Key Management Routes

@router.post("/api-keys", response_model=APIKeyResponse)
async def create_api_key(
    request: APIKeyCreateRequest,
    context: RequestContext = Depends(get_request_context),
    auth_service: AuthenticationService = Depends(get_auth_service)
):
    """Create a new API key."""
    if not context.is_authenticated:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Authentication required"
        )
    
    try:
        # Create API key
        api_key, key_info = await auth_service.api_key_manager.create_api_key(
            user_id=context.user_id,
            name=request.name,
            permissions=request.permissions,
            rate_limit=request.rate_limit,
            expires_in_days=request.expires_in_days,
            allowed_ips=request.allowed_ips
        )
        
        logger.info(f"API key created: {request.name} for user {context.user_id}")
        
        return APIKeyResponse(
            key_id=key_info.key_id,
            api_key=api_key,  # Only shown once
            name=key_info.name,
            permissions=key_info.permissions,
            rate_limit=key_info.rate_limit,
            created_at=key_info.created_at,
            expires_at=key_info.expires_at
        )
        
    except Exception as e:
        logger.error(f"API key creation error: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to create API key"
        )

@router.get("/api-keys", response_model=APIKeyListResponse)
async def list_api_keys(
    context: RequestContext = Depends(get_request_context)
):
    """List user's API keys."""
    if not context.is_authenticated:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Authentication required"
        )
    
    try:
        # TODO: Implement API key listing from database
        api_keys = []  # Placeholder
        
        return APIKeyListResponse(
            success=True,
            message="API keys retrieved successfully",
            api_keys=api_keys
        )
        
    except Exception as e:
        logger.error(f"API key listing error: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to retrieve API keys"
        )

@router.delete("/api-keys/{key_id}", response_model=APIResponse)
async def revoke_api_key(
    key_id: str,
    context: RequestContext = Depends(get_request_context)
):
    """Revoke an API key."""
    if not context.is_authenticated:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Authentication required"
        )
    
    try:
        # TODO: Implement API key revocation
        logger.info(f"API key revoked: {key_id} by user {context.user_id}")
        
        return APIResponse(
            success=True,
            message="API key revoked successfully"
        )
        
    except Exception as e:
        logger.error(f"API key revocation error: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to revoke API key"
        )

# Password Management Routes

@router.post("/change-password", response_model=APIResponse)
async def change_password(
    request: PasswordChangeRequest,
    context: RequestContext = Depends(get_request_context),
    auth_service: AuthenticationService = Depends(get_auth_service)
):
    """Change user password."""
    if not context.is_authenticated:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Authentication required"
        )
    
    try:
        # TODO: Implement password change logic
        # 1. Verify current password
        # 2. Hash new password
        # 3. Update database
        # 4. Invalidate all sessions
        
        logger.info(f"Password changed for user: {context.user_id}")
        
        return APIResponse(
            success=True,
            message="Password changed successfully"
        )
        
    except Exception as e:
        logger.error(f"Password change error: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to change password"
        )

@router.post("/reset-password", response_model=APIResponse)
async def request_password_reset(
    request: PasswordResetRequest,
    context: RequestContext = Depends(get_request_context)
):
    """Request password reset."""
    try:
        # TODO: Implement password reset request
        # 1. Check if email exists
        # 2. Generate reset token
        # 3. Send reset email
        
        logger.info(f"Password reset requested for: {request.email}")
        
        return APIResponse(
            success=True,
            message="Password reset instructions sent to your email"
        )
        
    except Exception as e:
        logger.error(f"Password reset request error: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to process password reset request"
        )

@router.post("/reset-password/confirm", response_model=APIResponse)
async def confirm_password_reset(
    request: PasswordResetConfirm,
    context: RequestContext = Depends(get_request_context)
):
    """Confirm password reset with token."""
    try:
        # TODO: Implement password reset confirmation
        # 1. Verify reset token
        # 2. Hash new password
        # 3. Update database
        # 4. Invalidate all sessions
        
        logger.info("Password reset confirmed")
        
        return APIResponse(
            success=True,
            message="Password reset successfully"
        )
        
    except Exception as e:
        logger.error(f"Password reset confirmation error: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to reset password"
        )

# Main function for standalone testing
if __name__ == "__main__":
    import asyncio
    from fastapi.testclient import TestClient
    from fastapi import FastAPI
    
    def test_auth_routes():
        """Test authentication routes."""
        app = FastAPI()
        app.include_router(router)
        
        client = TestClient(app)
        
        # Test registration endpoint
        response = client.post("/auth/register", json={
            "username": "testuser",
            "email": "test@example.com",
            "password": "testpassword123",
            "terms_accepted": True
        })
        
        print(f"Registration test: {response.status_code}")
        
        # Test login endpoint (will fail without proper setup)
        response = client.post("/auth/login", data={
            "username": "testuser",
            "password": "testpassword123"
        })
        
        print(f"Login test: {response.status_code}")
        
        print("Authentication routes test completed!")
    
    test_auth_routes()