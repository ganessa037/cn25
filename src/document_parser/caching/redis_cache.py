#!/usr/bin/env python3
"""
Redis Caching Layer

Comprehensive Redis-based caching system for:
- Frequently accessed data caching
- Session management
- Rate limiting data
- Processing results
- User preferences and settings
"""

import logging
import json
import pickle
import hashlib
import asyncio
from datetime import datetime, timedelta
from typing import Dict, Any, Optional, List, Union, Callable
from enum import Enum
from dataclasses import dataclass, asdict
from functools import wraps

import redis.asyncio as redis
from redis.asyncio import Redis
from redis.exceptions import RedisError, ConnectionError

# Configure logging
logger = logging.getLogger(__name__)

# Cache Configuration
class CacheNamespace(str, Enum):
    """Cache namespace prefixes."""
    USER_SESSION = "session"
    USER_PREFERENCES = "user_prefs"
    DOCUMENT_METADATA = "doc_meta"
    EXTRACTED_DATA = "extracted"
    PROCESSING_RESULTS = "proc_result"
    API_RATE_LIMIT = "rate_limit"
    SYSTEM_CONFIG = "sys_config"
    TEMPORARY_DATA = "temp"
    ANALYTICS = "analytics"
    NOTIFICATIONS = "notifications"

class CacheStrategy(str, Enum):
    """Cache invalidation strategies."""
    TTL = "ttl"  # Time-to-live
    LRU = "lru"  # Least Recently Used
    LFU = "lfu"  # Least Frequently Used
    MANUAL = "manual"  # Manual invalidation

@dataclass
class CacheConfig:
    """Cache configuration settings."""
    default_ttl: int = 3600  # 1 hour
    max_memory: str = "256mb"
    eviction_policy: str = "allkeys-lru"
    compression_enabled: bool = True
    serialization_format: str = "json"  # json, pickle, msgpack
    key_prefix: str = "docparser"
    cluster_enabled: bool = False
    sentinel_enabled: bool = False

@dataclass
class SessionData:
    """User session data structure."""
    user_id: str
    username: str
    email: str
    role: str
    permissions: List[str]
    created_at: datetime
    last_accessed: datetime
    ip_address: str
    user_agent: str
    is_active: bool = True
    metadata: Optional[Dict[str, Any]] = None

class RedisCache:
    """Redis-based caching system with advanced features."""
    
    def __init__(self, 
                 redis_url: str = "redis://localhost:6379/0",
                 config: Optional[CacheConfig] = None):
        self.redis_url = redis_url
        self.config = config or CacheConfig()
        self.redis_client: Optional[Redis] = None
        self.is_connected = False
        
        # Cache statistics
        self.stats = {
            "hits": 0,
            "misses": 0,
            "sets": 0,
            "deletes": 0,
            "errors": 0
        }
    
    async def connect(self) -> bool:
        """Establish Redis connection."""
        try:
            self.redis_client = redis.from_url(
                self.redis_url,
                encoding="utf-8",
                decode_responses=True,
                socket_connect_timeout=5,
                socket_timeout=5,
                retry_on_timeout=True,
                health_check_interval=30
            )
            
            # Test connection
            await self.redis_client.ping()
            self.is_connected = True
            
            # Configure Redis settings
            await self._configure_redis()
            
            logger.info("Redis cache connected successfully")
            return True
            
        except Exception as e:
            logger.error(f"Failed to connect to Redis: {e}")
            self.is_connected = False
            return False
    
    async def disconnect(self):
        """Close Redis connection."""
        if self.redis_client:
            await self.redis_client.close()
            self.is_connected = False
            logger.info("Redis cache disconnected")
    
    async def _configure_redis(self):
        """Configure Redis settings."""
        try:
            # Set memory policy
            await self.redis_client.config_set("maxmemory-policy", self.config.eviction_policy)
            
            # Set max memory if specified
            if self.config.max_memory:
                await self.redis_client.config_set("maxmemory", self.config.max_memory)
            
            logger.info("Redis configuration applied")
            
        except Exception as e:
            logger.warning(f"Failed to configure Redis: {e}")
    
    def _generate_key(self, namespace: CacheNamespace, key: str) -> str:
        """Generate cache key with namespace and prefix."""
        return f"{self.config.key_prefix}:{namespace.value}:{key}"
    
    def _serialize_data(self, data: Any) -> str:
        """Serialize data for storage."""
        try:
            if self.config.serialization_format == "json":
                return json.dumps(data, default=str)
            elif self.config.serialization_format == "pickle":
                return pickle.dumps(data).hex()
            else:
                return str(data)
        except Exception as e:
            logger.error(f"Serialization error: {e}")
            raise
    
    def _deserialize_data(self, data: str) -> Any:
        """Deserialize data from storage."""
        try:
            if self.config.serialization_format == "json":
                return json.loads(data)
            elif self.config.serialization_format == "pickle":
                return pickle.loads(bytes.fromhex(data))
            else:
                return data
        except Exception as e:
            logger.error(f"Deserialization error: {e}")
            raise
    
    async def get(self, namespace: CacheNamespace, key: str) -> Optional[Any]:
        """Get value from cache."""
        if not self.is_connected:
            return None
        
        try:
            cache_key = self._generate_key(namespace, key)
            data = await self.redis_client.get(cache_key)
            
            if data is not None:
                self.stats["hits"] += 1
                return self._deserialize_data(data)
            else:
                self.stats["misses"] += 1
                return None
                
        except Exception as e:
            logger.error(f"Cache get error: {e}")
            self.stats["errors"] += 1
            return None
    
    async def set(self, namespace: CacheNamespace, key: str, value: Any, 
                  ttl: Optional[int] = None) -> bool:
        """Set value in cache with optional TTL."""
        if not self.is_connected:
            return False
        
        try:
            cache_key = self._generate_key(namespace, key)
            serialized_data = self._serialize_data(value)
            
            ttl = ttl or self.config.default_ttl
            
            if ttl > 0:
                await self.redis_client.setex(cache_key, ttl, serialized_data)
            else:
                await self.redis_client.set(cache_key, serialized_data)
            
            self.stats["sets"] += 1
            return True
            
        except Exception as e:
            logger.error(f"Cache set error: {e}")
            self.stats["errors"] += 1
            return False
    
    async def delete(self, namespace: CacheNamespace, key: str) -> bool:
        """Delete value from cache."""
        if not self.is_connected:
            return False
        
        try:
            cache_key = self._generate_key(namespace, key)
            result = await self.redis_client.delete(cache_key)
            
            if result > 0:
                self.stats["deletes"] += 1
                return True
            return False
            
        except Exception as e:
            logger.error(f"Cache delete error: {e}")
            self.stats["errors"] += 1
            return False
    
    async def exists(self, namespace: CacheNamespace, key: str) -> bool:
        """Check if key exists in cache."""
        if not self.is_connected:
            return False
        
        try:
            cache_key = self._generate_key(namespace, key)
            return await self.redis_client.exists(cache_key) > 0
        except Exception as e:
            logger.error(f"Cache exists error: {e}")
            return False
    
    async def expire(self, namespace: CacheNamespace, key: str, ttl: int) -> bool:
        """Set expiration time for a key."""
        if not self.is_connected:
            return False
        
        try:
            cache_key = self._generate_key(namespace, key)
            return await self.redis_client.expire(cache_key, ttl)
        except Exception as e:
            logger.error(f"Cache expire error: {e}")
            return False
    
    async def get_ttl(self, namespace: CacheNamespace, key: str) -> int:
        """Get remaining TTL for a key."""
        if not self.is_connected:
            return -1
        
        try:
            cache_key = self._generate_key(namespace, key)
            return await self.redis_client.ttl(cache_key)
        except Exception as e:
            logger.error(f"Cache TTL error: {e}")
            return -1
    
    async def increment(self, namespace: CacheNamespace, key: str, 
                       amount: int = 1, ttl: Optional[int] = None) -> Optional[int]:
        """Increment a numeric value in cache."""
        if not self.is_connected:
            return None
        
        try:
            cache_key = self._generate_key(namespace, key)
            
            # Use pipeline for atomic operations
            async with self.redis_client.pipeline() as pipe:
                await pipe.incr(cache_key, amount)
                if ttl:
                    await pipe.expire(cache_key, ttl)
                results = await pipe.execute()
                
            return results[0]
            
        except Exception as e:
            logger.error(f"Cache increment error: {e}")
            return None
    
    async def get_multiple(self, namespace: CacheNamespace, 
                          keys: List[str]) -> Dict[str, Any]:
        """Get multiple values from cache."""
        if not self.is_connected or not keys:
            return {}
        
        try:
            cache_keys = [self._generate_key(namespace, key) for key in keys]
            values = await self.redis_client.mget(cache_keys)
            
            result = {}
            for i, (original_key, value) in enumerate(zip(keys, values)):
                if value is not None:
                    result[original_key] = self._deserialize_data(value)
                    self.stats["hits"] += 1
                else:
                    self.stats["misses"] += 1
            
            return result
            
        except Exception as e:
            logger.error(f"Cache get_multiple error: {e}")
            self.stats["errors"] += 1
            return {}
    
    async def set_multiple(self, namespace: CacheNamespace, 
                          data: Dict[str, Any], ttl: Optional[int] = None) -> bool:
        """Set multiple values in cache."""
        if not self.is_connected or not data:
            return False
        
        try:
            # Prepare data for mset
            cache_data = {}
            for key, value in data.items():
                cache_key = self._generate_key(namespace, key)
                cache_data[cache_key] = self._serialize_data(value)
            
            # Use pipeline for atomic operations
            async with self.redis_client.pipeline() as pipe:
                await pipe.mset(cache_data)
                
                # Set TTL for all keys if specified
                if ttl:
                    for cache_key in cache_data.keys():
                        await pipe.expire(cache_key, ttl)
                
                await pipe.execute()
            
            self.stats["sets"] += len(data)
            return True
            
        except Exception as e:
            logger.error(f"Cache set_multiple error: {e}")
            self.stats["errors"] += 1
            return False
    
    async def clear_namespace(self, namespace: CacheNamespace) -> int:
        """Clear all keys in a namespace."""
        if not self.is_connected:
            return 0
        
        try:
            pattern = f"{self.config.key_prefix}:{namespace.value}:*"
            keys = []
            
            # Scan for keys to avoid blocking
            async for key in self.redis_client.scan_iter(match=pattern):
                keys.append(key)
            
            if keys:
                deleted = await self.redis_client.delete(*keys)
                self.stats["deletes"] += deleted
                return deleted
            
            return 0
            
        except Exception as e:
            logger.error(f"Cache clear_namespace error: {e}")
            self.stats["errors"] += 1
            return 0
    
    async def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics."""
        stats = self.stats.copy()
        
        if self.is_connected:
            try:
                info = await self.redis_client.info()
                stats.update({
                    "redis_memory_used": info.get("used_memory_human", "unknown"),
                    "redis_connected_clients": info.get("connected_clients", 0),
                    "redis_total_commands": info.get("total_commands_processed", 0),
                    "redis_keyspace_hits": info.get("keyspace_hits", 0),
                    "redis_keyspace_misses": info.get("keyspace_misses", 0)
                })
            except Exception as e:
                logger.error(f"Failed to get Redis info: {e}")
        
        # Calculate hit ratio
        total_requests = stats["hits"] + stats["misses"]
        stats["hit_ratio"] = stats["hits"] / total_requests if total_requests > 0 else 0
        
        return stats

# Session Management
class SessionManager:
    """Redis-based session management."""
    
    def __init__(self, cache: RedisCache, session_ttl: int = 3600):
        self.cache = cache
        self.session_ttl = session_ttl
    
    async def create_session(self, session_data: SessionData) -> str:
        """Create a new user session."""
        try:
            session_id = self._generate_session_id(session_data.user_id)
            
            # Store session data
            await self.cache.set(
                CacheNamespace.USER_SESSION,
                session_id,
                asdict(session_data),
                ttl=self.session_ttl
            )
            
            logger.info(f"Session created: {session_id} for user {session_data.user_id}")
            return session_id
            
        except Exception as e:
            logger.error(f"Failed to create session: {e}")
            raise
    
    async def get_session(self, session_id: str) -> Optional[SessionData]:
        """Get session data by session ID."""
        try:
            data = await self.cache.get(CacheNamespace.USER_SESSION, session_id)
            
            if data:
                # Update last accessed time
                data["last_accessed"] = datetime.utcnow().isoformat()
                await self.cache.set(
                    CacheNamespace.USER_SESSION,
                    session_id,
                    data,
                    ttl=self.session_ttl
                )
                
                # Convert back to SessionData
                data["created_at"] = datetime.fromisoformat(data["created_at"])
                data["last_accessed"] = datetime.fromisoformat(data["last_accessed"])
                
                return SessionData(**data)
            
            return None
            
        except Exception as e:
            logger.error(f"Failed to get session: {e}")
            return None
    
    async def update_session(self, session_id: str, updates: Dict[str, Any]) -> bool:
        """Update session data."""
        try:
            session_data = await self.get_session(session_id)
            if not session_data:
                return False
            
            # Apply updates
            session_dict = asdict(session_data)
            session_dict.update(updates)
            session_dict["last_accessed"] = datetime.utcnow().isoformat()
            
            # Save updated session
            return await self.cache.set(
                CacheNamespace.USER_SESSION,
                session_id,
                session_dict,
                ttl=self.session_ttl
            )
            
        except Exception as e:
            logger.error(f"Failed to update session: {e}")
            return False
    
    async def delete_session(self, session_id: str) -> bool:
        """Delete a session."""
        try:
            result = await self.cache.delete(CacheNamespace.USER_SESSION, session_id)
            if result:
                logger.info(f"Session deleted: {session_id}")
            return result
            
        except Exception as e:
            logger.error(f"Failed to delete session: {e}")
            return False
    
    async def extend_session(self, session_id: str, additional_ttl: int = None) -> bool:
        """Extend session expiration time."""
        try:
            ttl = additional_ttl or self.session_ttl
            return await self.cache.expire(CacheNamespace.USER_SESSION, session_id, ttl)
            
        except Exception as e:
            logger.error(f"Failed to extend session: {e}")
            return False
    
    def _generate_session_id(self, user_id: str) -> str:
        """Generate unique session ID."""
        import secrets
        timestamp = str(datetime.utcnow().timestamp())
        random_part = secrets.token_hex(16)
        return hashlib.sha256(f"{user_id}:{timestamp}:{random_part}".encode()).hexdigest()

# Cache Decorators
def cached(namespace: CacheNamespace, ttl: int = 3600, key_func: Optional[Callable] = None):
    """Decorator for caching function results."""
    def decorator(func):
        @wraps(func)
        async def wrapper(*args, **kwargs):
            # Generate cache key
            if key_func:
                cache_key = key_func(*args, **kwargs)
            else:
                # Default key generation
                key_parts = [func.__name__]
                key_parts.extend(str(arg) for arg in args)
                key_parts.extend(f"{k}={v}" for k, v in sorted(kwargs.items()))
                cache_key = hashlib.md5(":".join(key_parts).encode()).hexdigest()
            
            # Try to get from cache
            cache = getattr(func, '_cache', None)
            if cache:
                result = await cache.get(namespace, cache_key)
                if result is not None:
                    return result
            
            # Execute function and cache result
            result = await func(*args, **kwargs)
            
            if cache and result is not None:
                await cache.set(namespace, cache_key, result, ttl)
            
            return result
        
        return wrapper
    return decorator

# Cache Warming
class CacheWarmer:
    """Preload frequently accessed data into cache."""
    
    def __init__(self, cache: RedisCache):
        self.cache = cache
    
    async def warm_user_preferences(self, user_ids: List[str]):
        """Preload user preferences into cache."""
        try:
            # TODO: Implement user preferences loading from database
            logger.info(f"Warming cache for {len(user_ids)} users")
            
            for user_id in user_ids:
                # Simulate loading user preferences
                preferences = {
                    "theme": "light",
                    "language": "en",
                    "notifications": True,
                    "auto_save": True
                }
                
                await self.cache.set(
                    CacheNamespace.USER_PREFERENCES,
                    user_id,
                    preferences,
                    ttl=7200  # 2 hours
                )
            
            logger.info("User preferences cache warming completed")
            
        except Exception as e:
            logger.error(f"Cache warming failed: {e}")
    
    async def warm_system_config(self):
        """Preload system configuration into cache."""
        try:
            # TODO: Implement system config loading from database
            config = {
                "max_file_size": 10485760,  # 10MB
                "allowed_file_types": ["pdf", "docx", "txt"],
                "processing_timeout": 300,  # 5 minutes
                "rate_limit_per_hour": 1000
            }
            
            await self.cache.set(
                CacheNamespace.SYSTEM_CONFIG,
                "global",
                config,
                ttl=3600  # 1 hour
            )
            
            logger.info("System configuration cache warming completed")
            
        except Exception as e:
            logger.error(f"System config cache warming failed: {e}")

# Main function for standalone testing
if __name__ == "__main__":
    async def test_redis_cache():
        """Test Redis cache functionality."""
        # Initialize cache
        cache = RedisCache()
        
        # Connect to Redis
        connected = await cache.connect()
        if not connected:
            print("Failed to connect to Redis")
            return
        
        try:
            # Test basic operations
            await cache.set(CacheNamespace.TEMPORARY_DATA, "test_key", {"message": "Hello, Redis!"})
            
            result = await cache.get(CacheNamespace.TEMPORARY_DATA, "test_key")
            print(f"Cache get result: {result}")
            
            # Test session management
            session_manager = SessionManager(cache)
            
            session_data = SessionData(
                user_id="user123",
                username="testuser",
                email="test@example.com",
                role="user",
                permissions=["read", "write"],
                created_at=datetime.utcnow(),
                last_accessed=datetime.utcnow(),
                ip_address="127.0.0.1",
                user_agent="Test Agent"
            )
            
            session_id = await session_manager.create_session(session_data)
            print(f"Created session: {session_id}")
            
            retrieved_session = await session_manager.get_session(session_id)
            print(f"Retrieved session: {retrieved_session.username if retrieved_session else 'None'}")
            
            # Test cache statistics
            stats = await cache.get_stats()
            print(f"Cache statistics: {stats}")
            
            print("Redis cache test completed successfully!")
            
        except Exception as e:
            print(f"Redis cache test failed: {e}")
        finally:
            await cache.disconnect()
    
    # Run test
    asyncio.run(test_redis_cache())