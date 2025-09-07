#!/usr/bin/env python3
"""
Authentication System

Comprehensive authentication system with JWT tokens, API keys, OAuth integration,
and role-based access control for the document processing API.
"""

import logging
import secrets
import hashlib
from datetime import datetime, timedelta
from typing import Dict, Any, Optional, List, Union
from enum import Enum
from dataclasses import dataclass
import json
import uuid

import jwt
import bcrypt
from fastapi import HTTPException, status, Request
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
import redis.asyncio as redis
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, update, delete
from sqlalchemy.orm import declarative_base, sessionmaker
from sqlalchemy import Column, String, DateTime, Boolean, Integer, Text, JSON

# Configure logging
logger = logging.getLogger(__name__)

# Enums

class UserRole(str, Enum):
    """User role enumeration."""
    ADMIN = "admin"
    USER = "user"
    API_CLIENT = "api_client"
    READONLY = "readonly"
    PREMIUM = "premium"

class TokenType(str, Enum):
    """Token type enumeration."""
    ACCESS = "access"
    REFRESH = "refresh"
    API_KEY = "api_key"
    RESET = "reset"
    VERIFICATION = "verification"

class AuthProvider(str, Enum):
    """Authentication provider enumeration."""
    LOCAL = "local"
    GOOGLE = "google"
    MICROSOFT = "microsoft"
    GITHUB = "github"
    OAUTH2 = "oauth2"

class SessionStatus(str, Enum):
    """Session status enumeration."""
    ACTIVE = "active"
    EXPIRED = "expired"
    REVOKED = "revoked"
    SUSPENDED = "suspended"

# Data Classes

@dataclass
class TokenPayload:
    """JWT token payload structure."""
    user_id: str
    username: str
    email: str
    role: UserRole
    permissions: List[str]
    token_type: TokenType
    issued_at: datetime
    expires_at: datetime
    session_id: Optional[str] = None
    device_id: Optional[str] = None
    ip_address: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JWT encoding."""
        return {
            "sub": self.user_id,
            "username": self.username,
            "email": self.email,
            "role": self.role.value,
            "permissions": self.permissions,
            "token_type": self.token_type.value,
            "iat": int(self.issued_at.timestamp()),
            "exp": int(self.expires_at.timestamp()),
            "session_id": self.session_id,
            "device_id": self.device_id,
            "ip_address": self.ip_address
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'TokenPayload':
        """Create from dictionary."""
        return cls(
            user_id=data["sub"],
            username=data["username"],
            email=data["email"],
            role=UserRole(data["role"]),
            permissions=data["permissions"],
            token_type=TokenType(data["token_type"]),
            issued_at=datetime.fromtimestamp(data["iat"]),
            expires_at=datetime.fromtimestamp(data["exp"]),
            session_id=data.get("session_id"),
            device_id=data.get("device_id"),
            ip_address=data.get("ip_address")
        )

@dataclass
class AuthenticationResult:
    """Authentication result structure."""
    success: bool
    user_id: Optional[str] = None
    username: Optional[str] = None
    email: Optional[str] = None
    role: Optional[UserRole] = None
    permissions: Optional[List[str]] = None
    session_id: Optional[str] = None
    error_message: Optional[str] = None
    error_code: Optional[str] = None
    requires_2fa: bool = False
    account_locked: bool = False
    password_expired: bool = False

@dataclass
class APIKeyInfo:
    """API key information structure."""
    key_id: str
    user_id: str
    name: str
    permissions: List[str]
    rate_limit: int
    created_at: datetime
    last_used: Optional[datetime] = None
    expires_at: Optional[datetime] = None
    is_active: bool = True
    usage_count: int = 0
    allowed_ips: Optional[List[str]] = None

# Database Models (simplified for this example)

Base = declarative_base()

class User(Base):
    """User database model."""
    __tablename__ = "users"
    
    id = Column(String, primary_key=True)
    username = Column(String, unique=True, nullable=False)
    email = Column(String, unique=True, nullable=False)
    password_hash = Column(String, nullable=False)
    role = Column(String, nullable=False, default=UserRole.USER.value)
    permissions = Column(JSON, default=list)
    is_active = Column(Boolean, default=True)
    is_verified = Column(Boolean, default=False)
    created_at = Column(DateTime, default=datetime.utcnow)
    last_login = Column(DateTime)
    failed_login_attempts = Column(Integer, default=0)
    account_locked_until = Column(DateTime)
    password_changed_at = Column(DateTime, default=datetime.utcnow)
    two_factor_enabled = Column(Boolean, default=False)
    two_factor_secret = Column(String)
    auth_provider = Column(String, default=AuthProvider.LOCAL.value)
    provider_id = Column(String)
    metadata = Column(JSON, default=dict)

class APIKey(Base):
    """API key database model."""
    __tablename__ = "api_keys"
    
    id = Column(String, primary_key=True)
    user_id = Column(String, nullable=False)
    name = Column(String, nullable=False)
    key_hash = Column(String, nullable=False)
    permissions = Column(JSON, default=list)
    rate_limit = Column(Integer, default=1000)
    created_at = Column(DateTime, default=datetime.utcnow)
    last_used = Column(DateTime)
    expires_at = Column(DateTime)
    is_active = Column(Boolean, default=True)
    usage_count = Column(Integer, default=0)
    allowed_ips = Column(JSON, default=list)
    metadata = Column(JSON, default=dict)

class UserSession(Base):
    """User session database model."""
    __tablename__ = "user_sessions"
    
    id = Column(String, primary_key=True)
    user_id = Column(String, nullable=False)
    device_id = Column(String)
    ip_address = Column(String)
    user_agent = Column(String)
    status = Column(String, default=SessionStatus.ACTIVE.value)
    created_at = Column(DateTime, default=datetime.utcnow)
    last_activity = Column(DateTime, default=datetime.utcnow)
    expires_at = Column(DateTime)
    metadata = Column(JSON, default=dict)

# Authentication Classes

class PasswordManager:
    """Password hashing and verification manager."""
    
    @staticmethod
    def hash_password(password: str) -> str:
        """Hash password using bcrypt."""
        salt = bcrypt.gensalt()
        return bcrypt.hashpw(password.encode('utf-8'), salt).decode('utf-8')
    
    @staticmethod
    def verify_password(password: str, hashed: str) -> bool:
        """Verify password against hash."""
        return bcrypt.checkpw(password.encode('utf-8'), hashed.encode('utf-8'))
    
    @staticmethod
    def generate_secure_token(length: int = 32) -> str:
        """Generate secure random token."""
        return secrets.token_urlsafe(length)

class JWTManager:
    """JWT token management."""
    
    def __init__(self, secret_key: str, algorithm: str = "HS256"):
        self.secret_key = secret_key
        self.algorithm = algorithm
        self.access_token_expire = timedelta(hours=1)
        self.refresh_token_expire = timedelta(days=30)
    
    def create_access_token(self, payload: TokenPayload) -> str:
        """Create JWT access token."""
        payload.token_type = TokenType.ACCESS
        payload.expires_at = datetime.utcnow() + self.access_token_expire
        
        return jwt.encode(
            payload.to_dict(),
            self.secret_key,
            algorithm=self.algorithm
        )
    
    def create_refresh_token(self, payload: TokenPayload) -> str:
        """Create JWT refresh token."""
        payload.token_type = TokenType.REFRESH
        payload.expires_at = datetime.utcnow() + self.refresh_token_expire
        
        return jwt.encode(
            payload.to_dict(),
            self.secret_key,
            algorithm=self.algorithm
        )
    
    def verify_token(self, token: str) -> TokenPayload:
        """Verify and decode JWT token."""
        try:
            payload = jwt.decode(
                token,
                self.secret_key,
                algorithms=[self.algorithm]
            )
            return TokenPayload.from_dict(payload)
        except jwt.ExpiredSignatureError:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Token has expired"
            )
        except jwt.InvalidTokenError:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Invalid token"
            )

class APIKeyManager:
    """API key management."""
    
    def __init__(self, redis_client: redis.Redis):
        self.redis = redis_client
        self.key_prefix = "api_key:"
        self.usage_prefix = "api_usage:"
    
    def generate_api_key(self) -> tuple[str, str]:
        """Generate API key and its hash."""
        # Generate key with prefix for identification
        key = f"dp_{secrets.token_urlsafe(32)}"
        key_hash = hashlib.sha256(key.encode()).hexdigest()
        return key, key_hash
    
    async def create_api_key(
        self,
        user_id: str,
        name: str,
        permissions: List[str],
        rate_limit: int = 1000,
        expires_in_days: Optional[int] = None,
        allowed_ips: Optional[List[str]] = None
    ) -> tuple[str, APIKeyInfo]:
        """Create new API key."""
        key, key_hash = self.generate_api_key()
        key_id = str(uuid.uuid4())
        
        expires_at = None
        if expires_in_days:
            expires_at = datetime.utcnow() + timedelta(days=expires_in_days)
        
        api_key_info = APIKeyInfo(
            key_id=key_id,
            user_id=user_id,
            name=name,
            permissions=permissions,
            rate_limit=rate_limit,
            created_at=datetime.utcnow(),
            expires_at=expires_at,
            allowed_ips=allowed_ips or []
        )
        
        # Store in Redis for fast lookup
        await self.redis.hset(
            f"{self.key_prefix}{key_hash}",
            mapping={
                "key_id": key_id,
                "user_id": user_id,
                "name": name,
                "permissions": json.dumps(permissions),
                "rate_limit": rate_limit,
                "created_at": api_key_info.created_at.isoformat(),
                "expires_at": api_key_info.expires_at.isoformat() if expires_at else "",
                "is_active": "true",
                "usage_count": "0",
                "allowed_ips": json.dumps(allowed_ips or [])
            }
        )
        
        return key, api_key_info
    
    async def verify_api_key(self, key: str, ip_address: str) -> Optional[APIKeyInfo]:
        """Verify API key and return info."""
        key_hash = hashlib.sha256(key.encode()).hexdigest()
        
        # Get from Redis
        key_data = await self.redis.hgetall(f"{self.key_prefix}{key_hash}")
        
        if not key_data:
            return None
        
        # Check if active
        if key_data.get("is_active") != "true":
            return None
        
        # Check expiration
        expires_at_str = key_data.get("expires_at")
        if expires_at_str:
            expires_at = datetime.fromisoformat(expires_at_str)
            if datetime.utcnow() > expires_at:
                return None
        
        # Check IP restrictions
        allowed_ips = json.loads(key_data.get("allowed_ips", "[]"))
        if allowed_ips and ip_address not in allowed_ips:
            return None
        
        # Update usage
        await self.redis.hincrby(f"{self.key_prefix}{key_hash}", "usage_count", 1)
        await self.redis.hset(f"{self.key_prefix}{key_hash}", "last_used", datetime.utcnow().isoformat())
        
        return APIKeyInfo(
            key_id=key_data["key_id"],
            user_id=key_data["user_id"],
            name=key_data["name"],
            permissions=json.loads(key_data["permissions"]),
            rate_limit=int(key_data["rate_limit"]),
            created_at=datetime.fromisoformat(key_data["created_at"]),
            last_used=datetime.utcnow(),
            expires_at=datetime.fromisoformat(expires_at_str) if expires_at_str else None,
            is_active=True,
            usage_count=int(key_data["usage_count"]) + 1,
            allowed_ips=allowed_ips
        )

class AuthenticationService:
    """Main authentication service."""
    
    def __init__(
        self,
        jwt_manager: JWTManager,
        api_key_manager: APIKeyManager,
        redis_client: redis.Redis,
        db_session: AsyncSession
    ):
        self.jwt_manager = jwt_manager
        self.api_key_manager = api_key_manager
        self.redis = redis_client
        self.db = db_session
        self.password_manager = PasswordManager()
        self.max_failed_attempts = 5
        self.lockout_duration = timedelta(minutes=30)
    
    async def authenticate_user(
        self,
        username: str,
        password: str,
        ip_address: str,
        user_agent: str
    ) -> AuthenticationResult:
        """Authenticate user with username/password."""
        try:
            # Get user from database
            result = await self.db.execute(
                select(User).where(
                    (User.username == username) | (User.email == username)
                )
            )
            user = result.scalar_one_or_none()
            
            if not user:
                return AuthenticationResult(
                    success=False,
                    error_message="Invalid credentials",
                    error_code="INVALID_CREDENTIALS"
                )
            
            # Check if account is locked
            if user.account_locked_until and datetime.utcnow() < user.account_locked_until:
                return AuthenticationResult(
                    success=False,
                    error_message="Account is locked",
                    error_code="ACCOUNT_LOCKED",
                    account_locked=True
                )
            
            # Verify password
            if not self.password_manager.verify_password(password, user.password_hash):
                # Increment failed attempts
                await self._handle_failed_login(user)
                return AuthenticationResult(
                    success=False,
                    error_message="Invalid credentials",
                    error_code="INVALID_CREDENTIALS"
                )
            
            # Check if account is active
            if not user.is_active:
                return AuthenticationResult(
                    success=False,
                    error_message="Account is disabled",
                    error_code="ACCOUNT_DISABLED"
                )
            
            # Reset failed attempts on successful login
            await self.db.execute(
                update(User)
                .where(User.id == user.id)
                .values(
                    failed_login_attempts=0,
                    last_login=datetime.utcnow(),
                    account_locked_until=None
                )
            )
            await self.db.commit()
            
            return AuthenticationResult(
                success=True,
                user_id=user.id,
                username=user.username,
                email=user.email,
                role=UserRole(user.role),
                permissions=user.permissions or []
            )
            
        except Exception as e:
            logger.error(f"Authentication error: {e}")
            return AuthenticationResult(
                success=False,
                error_message="Authentication service error",
                error_code="SERVICE_ERROR"
            )
    
    async def _handle_failed_login(self, user: User):
        """Handle failed login attempt."""
        failed_attempts = user.failed_login_attempts + 1
        
        update_values = {"failed_login_attempts": failed_attempts}
        
        if failed_attempts >= self.max_failed_attempts:
            update_values["account_locked_until"] = datetime.utcnow() + self.lockout_duration
        
        await self.db.execute(
            update(User)
            .where(User.id == user.id)
            .values(**update_values)
        )
        await self.db.commit()
    
    async def create_session(
        self,
        user_id: str,
        ip_address: str,
        user_agent: str,
        device_id: Optional[str] = None
    ) -> str:
        """Create user session."""
        session_id = str(uuid.uuid4())
        
        session = UserSession(
            id=session_id,
            user_id=user_id,
            device_id=device_id,
            ip_address=ip_address,
            user_agent=user_agent,
            expires_at=datetime.utcnow() + timedelta(days=30)
        )
        
        self.db.add(session)
        await self.db.commit()
        
        return session_id
    
    async def revoke_session(self, session_id: str):
        """Revoke user session."""
        await self.db.execute(
            update(UserSession)
            .where(UserSession.id == session_id)
            .values(status=SessionStatus.REVOKED.value)
        )
        await self.db.commit()

# Main function for standalone testing
if __name__ == "__main__":
    import asyncio
    
    async def main():
        """Test authentication system."""
        # Initialize components
        jwt_manager = JWTManager("test-secret-key")
        
        # Test password hashing
        password = "test_password_123"
        hashed = PasswordManager.hash_password(password)
        print(f"Password hashed: {hashed}")
        print(f"Password verified: {PasswordManager.verify_password(password, hashed)}")
        
        # Test token creation
        payload = TokenPayload(
            user_id="test-user-123",
            username="testuser",
            email="test@example.com",
            role=UserRole.USER,
            permissions=["read", "write"],
            token_type=TokenType.ACCESS,
            issued_at=datetime.utcnow(),
            expires_at=datetime.utcnow() + timedelta(hours=1)
        )
        
        token = jwt_manager.create_access_token(payload)
        print(f"Token created: {token[:50]}...")
        
        # Test token verification
        verified_payload = jwt_manager.verify_token(token)
        print(f"Token verified: {verified_payload.username}")
        
        print("Authentication system test completed successfully!")
    
    asyncio.run(main())