#!/usr/bin/env python3
"""
Rate Limiting System

Advanced rate limiting implementation with multiple strategies, user tiers,
and comprehensive protection against API abuse.
"""

import logging
import time
import json
from datetime import datetime, timedelta
from typing import Dict, Any, Optional, List, Tuple, Union
from enum import Enum
from dataclasses import dataclass, asdict
import asyncio
from collections import defaultdict

import redis.asyncio as redis
from fastapi import HTTPException, status, Request
from pydantic import BaseModel

# Configure logging
logger = logging.getLogger(__name__)

# Enums

class RateLimitStrategy(str, Enum):
    """Rate limiting strategy enumeration."""
    FIXED_WINDOW = "fixed_window"
    SLIDING_WINDOW = "sliding_window"
    TOKEN_BUCKET = "token_bucket"
    LEAKY_BUCKET = "leaky_bucket"
    ADAPTIVE = "adaptive"

class UserTier(str, Enum):
    """User tier enumeration for different rate limits."""
    FREE = "free"
    BASIC = "basic"
    PREMIUM = "premium"
    ENTERPRISE = "enterprise"
    ADMIN = "admin"

class RateLimitScope(str, Enum):
    """Rate limit scope enumeration."""
    GLOBAL = "global"
    USER = "user"
    IP = "ip"
    API_KEY = "api_key"
    ENDPOINT = "endpoint"
    COMBINED = "combined"

class ViolationType(str, Enum):
    """Rate limit violation type."""
    SOFT_LIMIT = "soft_limit"
    HARD_LIMIT = "hard_limit"
    BURST_LIMIT = "burst_limit"
    DAILY_QUOTA = "daily_quota"
    MONTHLY_QUOTA = "monthly_quota"

# Data Classes

@dataclass
class RateLimitConfig:
    """Rate limit configuration."""
    strategy: RateLimitStrategy
    requests_per_minute: int
    requests_per_hour: int
    requests_per_day: int
    burst_limit: int
    window_size: int = 60  # seconds
    bucket_capacity: Optional[int] = None
    refill_rate: Optional[float] = None
    adaptive_factor: float = 1.0
    grace_period: int = 5  # seconds
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return asdict(self)

@dataclass
class RateLimitStatus:
    """Current rate limit status."""
    allowed: bool
    remaining: int
    reset_time: datetime
    retry_after: Optional[int] = None
    current_usage: int = 0
    limit: int = 0
    window_start: Optional[datetime] = None
    violation_type: Optional[ViolationType] = None
    warning: bool = False
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "allowed": self.allowed,
            "remaining": self.remaining,
            "reset_time": self.reset_time.isoformat(),
            "retry_after": self.retry_after,
            "current_usage": self.current_usage,
            "limit": self.limit,
            "window_start": self.window_start.isoformat() if self.window_start else None,
            "violation_type": self.violation_type.value if self.violation_type else None,
            "warning": self.warning
        }

@dataclass
class RateLimitViolation:
    """Rate limit violation record."""
    identifier: str
    violation_type: ViolationType
    timestamp: datetime
    endpoint: str
    ip_address: str
    user_agent: str
    current_usage: int
    limit: int
    severity: str = "medium"
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return asdict(self)

# Rate Limiting Implementations

class FixedWindowRateLimiter:
    """Fixed window rate limiting implementation."""
    
    def __init__(self, redis_client: redis.Redis):
        self.redis = redis_client
        self.key_prefix = "rate_limit:fixed:"
    
    async def is_allowed(
        self,
        identifier: str,
        limit: int,
        window_size: int = 60
    ) -> RateLimitStatus:
        """Check if request is allowed under fixed window."""
        current_time = int(time.time())
        window_start = current_time - (current_time % window_size)
        key = f"{self.key_prefix}{identifier}:{window_start}"
        
        # Get current count
        current_count = await self.redis.get(key)
        current_count = int(current_count) if current_count else 0
        
        if current_count >= limit:
            reset_time = datetime.fromtimestamp(window_start + window_size)
            return RateLimitStatus(
                allowed=False,
                remaining=0,
                reset_time=reset_time,
                retry_after=window_start + window_size - current_time,
                current_usage=current_count,
                limit=limit,
                window_start=datetime.fromtimestamp(window_start),
                violation_type=ViolationType.HARD_LIMIT
            )
        
        # Increment counter
        pipe = self.redis.pipeline()
        pipe.incr(key)
        pipe.expire(key, window_size)
        await pipe.execute()
        
        reset_time = datetime.fromtimestamp(window_start + window_size)
        remaining = limit - current_count - 1
        
        return RateLimitStatus(
            allowed=True,
            remaining=remaining,
            reset_time=reset_time,
            current_usage=current_count + 1,
            limit=limit,
            window_start=datetime.fromtimestamp(window_start),
            warning=remaining < limit * 0.1  # Warning when 90% used
        )

class SlidingWindowRateLimiter:
    """Sliding window rate limiting implementation."""
    
    def __init__(self, redis_client: redis.Redis):
        self.redis = redis_client
        self.key_prefix = "rate_limit:sliding:"
    
    async def is_allowed(
        self,
        identifier: str,
        limit: int,
        window_size: int = 60
    ) -> RateLimitStatus:
        """Check if request is allowed under sliding window."""
        current_time = time.time()
        window_start = current_time - window_size
        key = f"{self.key_prefix}{identifier}"
        
        # Remove old entries and count current
        pipe = self.redis.pipeline()
        pipe.zremrangebyscore(key, 0, window_start)
        pipe.zcard(key)
        pipe.expire(key, window_size)
        results = await pipe.execute()
        
        current_count = results[1]
        
        if current_count >= limit:
            # Get oldest entry to determine reset time
            oldest_entries = await self.redis.zrange(key, 0, 0, withscores=True)
            if oldest_entries:
                oldest_time = oldest_entries[0][1]
                reset_time = datetime.fromtimestamp(oldest_time + window_size)
                retry_after = int(oldest_time + window_size - current_time)
            else:
                reset_time = datetime.fromtimestamp(current_time + window_size)
                retry_after = window_size
            
            return RateLimitStatus(
                allowed=False,
                remaining=0,
                reset_time=reset_time,
                retry_after=retry_after,
                current_usage=current_count,
                limit=limit,
                violation_type=ViolationType.HARD_LIMIT
            )
        
        # Add current request
        await self.redis.zadd(key, {str(current_time): current_time})
        await self.redis.expire(key, window_size)
        
        remaining = limit - current_count - 1
        reset_time = datetime.fromtimestamp(current_time + window_size)
        
        return RateLimitStatus(
            allowed=True,
            remaining=remaining,
            reset_time=reset_time,
            current_usage=current_count + 1,
            limit=limit,
            warning=remaining < limit * 0.1
        )

class TokenBucketRateLimiter:
    """Token bucket rate limiting implementation."""
    
    def __init__(self, redis_client: redis.Redis):
        self.redis = redis_client
        self.key_prefix = "rate_limit:bucket:"
    
    async def is_allowed(
        self,
        identifier: str,
        capacity: int,
        refill_rate: float,
        tokens_requested: int = 1
    ) -> RateLimitStatus:
        """Check if request is allowed under token bucket."""
        current_time = time.time()
        key = f"{self.key_prefix}{identifier}"
        
        # Get current bucket state
        bucket_data = await self.redis.hmget(key, "tokens", "last_refill")
        
        if bucket_data[0] is None:
            # Initialize bucket
            tokens = capacity
            last_refill = current_time
        else:
            tokens = float(bucket_data[0])
            last_refill = float(bucket_data[1])
        
        # Calculate tokens to add
        time_passed = current_time - last_refill
        tokens_to_add = time_passed * refill_rate
        tokens = min(capacity, tokens + tokens_to_add)
        
        if tokens < tokens_requested:
            # Calculate when enough tokens will be available
            tokens_needed = tokens_requested - tokens
            time_to_wait = tokens_needed / refill_rate
            reset_time = datetime.fromtimestamp(current_time + time_to_wait)
            
            return RateLimitStatus(
                allowed=False,
                remaining=int(tokens),
                reset_time=reset_time,
                retry_after=int(time_to_wait),
                current_usage=capacity - int(tokens),
                limit=capacity,
                violation_type=ViolationType.HARD_LIMIT
            )
        
        # Consume tokens
        tokens -= tokens_requested
        
        # Update bucket state
        await self.redis.hmset(key, {
            "tokens": tokens,
            "last_refill": current_time
        })
        await self.redis.expire(key, 3600)  # Expire after 1 hour of inactivity
        
        return RateLimitStatus(
            allowed=True,
            remaining=int(tokens),
            reset_time=datetime.fromtimestamp(current_time + (capacity - tokens) / refill_rate),
            current_usage=capacity - int(tokens),
            limit=capacity,
            warning=tokens < capacity * 0.1
        )

class AdaptiveRateLimiter:
    """Adaptive rate limiting that adjusts based on system load."""
    
    def __init__(self, redis_client: redis.Redis):
        self.redis = redis_client
        self.key_prefix = "rate_limit:adaptive:"
        self.load_key = "system:load"
    
    async def is_allowed(
        self,
        identifier: str,
        base_limit: int,
        window_size: int = 60
    ) -> RateLimitStatus:
        """Check if request is allowed with adaptive limiting."""
        # Get system load factor
        load_factor = await self.redis.get(self.load_key)
        load_factor = float(load_factor) if load_factor else 1.0
        
        # Adjust limit based on load
        adjusted_limit = int(base_limit * load_factor)
        
        # Use sliding window for actual limiting
        sliding_limiter = SlidingWindowRateLimiter(self.redis)
        status = await sliding_limiter.is_allowed(identifier, adjusted_limit, window_size)
        
        # Update status with adaptive information
        status.limit = adjusted_limit
        
        return status
    
    async def update_system_load(self, load_factor: float):
        """Update system load factor."""
        await self.redis.set(self.load_key, load_factor, ex=300)  # Expire after 5 minutes

# Main Rate Limiter

class RateLimiter:
    """Main rate limiter with multiple strategies and user tiers."""
    
    def __init__(self, redis_client: redis.Redis):
        self.redis = redis_client
        self.fixed_window = FixedWindowRateLimiter(redis_client)
        self.sliding_window = SlidingWindowRateLimiter(redis_client)
        self.token_bucket = TokenBucketRateLimiter(redis_client)
        self.adaptive = AdaptiveRateLimiter(redis_client)
        
        # Default configurations for different user tiers
        self.tier_configs = {
            UserTier.FREE: RateLimitConfig(
                strategy=RateLimitStrategy.FIXED_WINDOW,
                requests_per_minute=10,
                requests_per_hour=100,
                requests_per_day=1000,
                burst_limit=5
            ),
            UserTier.BASIC: RateLimitConfig(
                strategy=RateLimitStrategy.SLIDING_WINDOW,
                requests_per_minute=50,
                requests_per_hour=1000,
                requests_per_day=10000,
                burst_limit=20
            ),
            UserTier.PREMIUM: RateLimitConfig(
                strategy=RateLimitStrategy.TOKEN_BUCKET,
                requests_per_minute=200,
                requests_per_hour=5000,
                requests_per_day=50000,
                burst_limit=100,
                bucket_capacity=200,
                refill_rate=3.33  # 200 per minute
            ),
            UserTier.ENTERPRISE: RateLimitConfig(
                strategy=RateLimitStrategy.ADAPTIVE,
                requests_per_minute=1000,
                requests_per_hour=20000,
                requests_per_day=200000,
                burst_limit=500
            ),
            UserTier.ADMIN: RateLimitConfig(
                strategy=RateLimitStrategy.TOKEN_BUCKET,
                requests_per_minute=10000,
                requests_per_hour=100000,
                requests_per_day=1000000,
                burst_limit=1000,
                bucket_capacity=1000,
                refill_rate=166.67  # 10000 per minute
            )
        }
        
        self.violation_key = "rate_limit:violations"
    
    async def check_rate_limit(
        self,
        identifier: str,
        user_tier: UserTier,
        scope: RateLimitScope = RateLimitScope.USER,
        endpoint: Optional[str] = None,
        ip_address: Optional[str] = None,
        user_agent: Optional[str] = None
    ) -> RateLimitStatus:
        """Check rate limit for a request."""
        config = self.tier_configs[user_tier]
        
        # Create scope-specific identifier
        scoped_identifier = f"{scope.value}:{identifier}"
        if endpoint:
            scoped_identifier += f":{endpoint}"
        
        # Check different time windows
        minute_status = await self._check_window(
            scoped_identifier + ":minute",
            config.requests_per_minute,
            60,
            config
        )
        
        if not minute_status.allowed:
            await self._record_violation(
                identifier, ViolationType.HARD_LIMIT, endpoint or "unknown",
                ip_address or "unknown", user_agent or "unknown",
                minute_status.current_usage, minute_status.limit
            )
            return minute_status
        
        hour_status = await self._check_window(
            scoped_identifier + ":hour",
            config.requests_per_hour,
            3600,
            config
        )
        
        if not hour_status.allowed:
            await self._record_violation(
                identifier, ViolationType.DAILY_QUOTA, endpoint or "unknown",
                ip_address or "unknown", user_agent or "unknown",
                hour_status.current_usage, hour_status.limit
            )
            return hour_status
        
        day_status = await self._check_window(
            scoped_identifier + ":day",
            config.requests_per_day,
            86400,
            config
        )
        
        if not day_status.allowed:
            await self._record_violation(
                identifier, ViolationType.MONTHLY_QUOTA, endpoint or "unknown",
                ip_address or "unknown", user_agent or "unknown",
                day_status.current_usage, day_status.limit
            )
            return day_status
        
        # Return the most restrictive status (minute-level)
        return minute_status
    
    async def _check_window(
        self,
        identifier: str,
        limit: int,
        window_size: int,
        config: RateLimitConfig
    ) -> RateLimitStatus:
        """Check rate limit for a specific window."""
        if config.strategy == RateLimitStrategy.FIXED_WINDOW:
            return await self.fixed_window.is_allowed(identifier, limit, window_size)
        elif config.strategy == RateLimitStrategy.SLIDING_WINDOW:
            return await self.sliding_window.is_allowed(identifier, limit, window_size)
        elif config.strategy == RateLimitStrategy.TOKEN_BUCKET:
            return await self.token_bucket.is_allowed(
                identifier, config.bucket_capacity or limit, config.refill_rate or (limit / 60)
            )
        elif config.strategy == RateLimitStrategy.ADAPTIVE:
            return await self.adaptive.is_allowed(identifier, limit, window_size)
        else:
            # Default to fixed window
            return await self.fixed_window.is_allowed(identifier, limit, window_size)
    
    async def _record_violation(
        self,
        identifier: str,
        violation_type: ViolationType,
        endpoint: str,
        ip_address: str,
        user_agent: str,
        current_usage: int,
        limit: int
    ):
        """Record rate limit violation."""
        violation = RateLimitViolation(
            identifier=identifier,
            violation_type=violation_type,
            timestamp=datetime.utcnow(),
            endpoint=endpoint,
            ip_address=ip_address,
            user_agent=user_agent,
            current_usage=current_usage,
            limit=limit
        )
        
        # Store violation in Redis
        await self.redis.lpush(
            f"{self.violation_key}:{identifier}",
            json.dumps(violation.to_dict(), default=str)
        )
        await self.redis.expire(f"{self.violation_key}:{identifier}", 86400)  # Keep for 24 hours
        
        # Log violation
        logger.warning(
            f"Rate limit violation: {identifier} exceeded {violation_type.value} "
            f"({current_usage}/{limit}) on {endpoint}"
        )
    
    async def get_violations(
        self,
        identifier: str,
        limit: int = 100
    ) -> List[RateLimitViolation]:
        """Get recent violations for an identifier."""
        violations_data = await self.redis.lrange(
            f"{self.violation_key}:{identifier}", 0, limit - 1
        )
        
        violations = []
        for violation_json in violations_data:
            try:
                violation_dict = json.loads(violation_json)
                violation_dict['timestamp'] = datetime.fromisoformat(violation_dict['timestamp'])
                violations.append(RateLimitViolation(**violation_dict))
            except Exception as e:
                logger.error(f"Error parsing violation data: {e}")
        
        return violations
    
    async def reset_rate_limit(self, identifier: str, scope: RateLimitScope = RateLimitScope.USER):
        """Reset rate limit for an identifier."""
        scoped_identifier = f"{scope.value}:{identifier}"
        
        # Delete all rate limit keys for this identifier
        keys_to_delete = []
        for window in ["minute", "hour", "day"]:
            # Fixed window keys
            pattern = f"rate_limit:fixed:{scoped_identifier}:{window}:*"
            keys = await self.redis.keys(pattern)
            keys_to_delete.extend(keys)
            
            # Sliding window keys
            keys_to_delete.append(f"rate_limit:sliding:{scoped_identifier}:{window}")
            
            # Token bucket keys
            keys_to_delete.append(f"rate_limit:bucket:{scoped_identifier}:{window}")
            
            # Adaptive keys
            keys_to_delete.append(f"rate_limit:adaptive:{scoped_identifier}:{window}")
        
        if keys_to_delete:
            await self.redis.delete(*keys_to_delete)
        
        logger.info(f"Rate limit reset for {identifier}")

# Middleware Integration

class RateLimitMiddleware:
    """FastAPI middleware for rate limiting."""
    
    def __init__(self, rate_limiter: RateLimiter):
        self.rate_limiter = rate_limiter
    
    async def __call__(self, request: Request, call_next):
        """Process request with rate limiting."""
        # Extract request information
        ip_address = self._get_client_ip(request)
        user_agent = request.headers.get("user-agent", "unknown")
        endpoint = f"{request.method} {request.url.path}"
        
        # Determine user tier (this would come from authentication)
        user_tier = UserTier.FREE  # Default
        identifier = ip_address  # Default to IP-based limiting
        
        # Check if user is authenticated
        if hasattr(request.state, "user") and request.state.user:
            user_tier = UserTier(request.state.user.get("tier", UserTier.FREE.value))
            identifier = request.state.user["user_id"]
        
        # Check rate limit
        status = await self.rate_limiter.check_rate_limit(
            identifier=identifier,
            user_tier=user_tier,
            endpoint=endpoint,
            ip_address=ip_address,
            user_agent=user_agent
        )
        
        if not status.allowed:
            raise HTTPException(
                status_code=status.HTTP_429_TOO_MANY_REQUESTS,
                detail={
                    "error": "Rate limit exceeded",
                    "retry_after": status.retry_after,
                    "reset_time": status.reset_time.isoformat(),
                    "violation_type": status.violation_type.value if status.violation_type else None
                },
                headers={
                    "X-RateLimit-Limit": str(status.limit),
                    "X-RateLimit-Remaining": str(status.remaining),
                    "X-RateLimit-Reset": str(int(status.reset_time.timestamp())),
                    "Retry-After": str(status.retry_after) if status.retry_after else "60"
                }
            )
        
        # Add rate limit headers to response
        response = await call_next(request)
        response.headers["X-RateLimit-Limit"] = str(status.limit)
        response.headers["X-RateLimit-Remaining"] = str(status.remaining)
        response.headers["X-RateLimit-Reset"] = str(int(status.reset_time.timestamp()))
        
        if status.warning:
            response.headers["X-RateLimit-Warning"] = "Approaching rate limit"
        
        return response
    
    def _get_client_ip(self, request: Request) -> str:
        """Extract client IP address from request."""
        # Check for forwarded headers
        forwarded_for = request.headers.get("x-forwarded-for")
        if forwarded_for:
            return forwarded_for.split(",")[0].strip()
        
        real_ip = request.headers.get("x-real-ip")
        if real_ip:
            return real_ip
        
        return request.client.host if request.client else "unknown"

# Main function for standalone testing
if __name__ == "__main__":
    import asyncio
    
    async def main():
        """Test rate limiting system."""
        # Mock Redis client for testing
        class MockRedis:
            def __init__(self):
                self.data = {}
                self.expiry = {}
            
            async def get(self, key):
                if key in self.expiry and time.time() > self.expiry[key]:
                    del self.data[key]
                    del self.expiry[key]
                return self.data.get(key)
            
            async def set(self, key, value, ex=None):
                self.data[key] = value
                if ex:
                    self.expiry[key] = time.time() + ex
            
            async def incr(self, key):
                current = int(self.data.get(key, 0))
                self.data[key] = str(current + 1)
                return current + 1
            
            async def expire(self, key, seconds):
                self.expiry[key] = time.time() + seconds
            
            def pipeline(self):
                return self
            
            async def execute(self):
                return [None, None]
        
        # Test rate limiter
        mock_redis = MockRedis()
        rate_limiter = RateLimiter(mock_redis)
        
        # Test different user tiers
        for tier in UserTier:
            print(f"\nTesting {tier.value} tier:")
            
            status = await rate_limiter.check_rate_limit(
                identifier=f"test-user-{tier.value}",
                user_tier=tier,
                endpoint="POST /documents/upload",
                ip_address="192.168.1.1"
            )
            
            print(f"  Allowed: {status.allowed}")
            print(f"  Remaining: {status.remaining}")
            print(f"  Limit: {status.limit}")
            print(f"  Warning: {status.warning}")
        
        print("\nRate limiting system test completed successfully!")
    
    asyncio.run(main())