#!/usr/bin/env python3
"""
Load Balancing and Health Monitoring System

Comprehensive system for:
- Load balancing strategies
- Health monitoring and checks
- Service discovery
- Auto-scaling capabilities
- Circuit breaker pattern
- Performance monitoring
"""

import logging
import asyncio
import time
import statistics
from datetime import datetime, timedelta
from typing import Dict, Any, Optional, List, Union, Callable
from enum import Enum
from dataclasses import dataclass, field
import json
import aiohttp
from urllib.parse import urljoin
import random
import hashlib

# Configure logging
logger = logging.getLogger(__name__)

# Load Balancing Configuration
class LoadBalancingStrategy(str, Enum):
    """Load balancing strategies."""
    ROUND_ROBIN = "round_robin"
    WEIGHTED_ROUND_ROBIN = "weighted_round_robin"
    LEAST_CONNECTIONS = "least_connections"
    LEAST_RESPONSE_TIME = "least_response_time"
    IP_HASH = "ip_hash"
    RANDOM = "random"
    WEIGHTED_RANDOM = "weighted_random"

class HealthStatus(str, Enum):
    """Service health status."""
    HEALTHY = "healthy"
    UNHEALTHY = "unhealthy"
    DEGRADED = "degraded"
    UNKNOWN = "unknown"
    MAINTENANCE = "maintenance"

class CircuitState(str, Enum):
    """Circuit breaker states."""
    CLOSED = "closed"
    OPEN = "open"
    HALF_OPEN = "half_open"

@dataclass
class ServiceInstance:
    """Service instance configuration."""
    id: str
    host: str
    port: int
    weight: int = 1
    health_status: HealthStatus = HealthStatus.UNKNOWN
    last_health_check: Optional[datetime] = None
    response_times: List[float] = field(default_factory=list)
    active_connections: int = 0
    total_requests: int = 0
    failed_requests: int = 0
    cpu_usage: float = 0.0
    memory_usage: float = 0.0
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    @property
    def url(self) -> str:
        return f"http://{self.host}:{self.port}"
    
    @property
    def average_response_time(self) -> float:
        return statistics.mean(self.response_times) if self.response_times else 0.0
    
    @property
    def success_rate(self) -> float:
        if self.total_requests == 0:
            return 1.0
        return (self.total_requests - self.failed_requests) / self.total_requests

@dataclass
class HealthCheckConfig:
    """Health check configuration."""
    endpoint: str = "/health"
    interval: int = 30  # seconds
    timeout: int = 5    # seconds
    retries: int = 3
    failure_threshold: int = 3
    success_threshold: int = 2
    expected_status_codes: List[int] = field(default_factory=lambda: [200])
    expected_response_time: float = 1.0  # seconds

@dataclass
class CircuitBreakerConfig:
    """Circuit breaker configuration."""
    failure_threshold: int = 5
    recovery_timeout: int = 60  # seconds
    half_open_max_calls: int = 3
    monitoring_window: int = 300  # seconds

@dataclass
class LoadBalancerConfig:
    """Load balancer configuration."""
    strategy: LoadBalancingStrategy = LoadBalancingStrategy.ROUND_ROBIN
    health_check: HealthCheckConfig = field(default_factory=HealthCheckConfig)
    circuit_breaker: CircuitBreakerConfig = field(default_factory=CircuitBreakerConfig)
    sticky_sessions: bool = False
    session_timeout: int = 3600  # seconds
    max_retries: int = 3
    retry_delay: float = 1.0  # seconds
    enable_metrics: bool = True
    metrics_retention: int = 86400  # seconds (24 hours)

class CircuitBreaker:
    """Circuit breaker implementation."""
    
    def __init__(self, config: CircuitBreakerConfig):
        self.config = config
        self.state = CircuitState.CLOSED
        self.failure_count = 0
        self.last_failure_time: Optional[datetime] = None
        self.half_open_calls = 0
        self.failure_times: List[datetime] = []
    
    def can_execute(self) -> bool:
        """Check if request can be executed."""
        now = datetime.utcnow()
        
        if self.state == CircuitState.CLOSED:
            return True
        elif self.state == CircuitState.OPEN:
            if (self.last_failure_time and 
                now - self.last_failure_time > timedelta(seconds=self.config.recovery_timeout)):
                self.state = CircuitState.HALF_OPEN
                self.half_open_calls = 0
                return True
            return False
        elif self.state == CircuitState.HALF_OPEN:
            return self.half_open_calls < self.config.half_open_max_calls
        
        return False
    
    def record_success(self):
        """Record successful request."""
        if self.state == CircuitState.HALF_OPEN:
            self.half_open_calls += 1
            if self.half_open_calls >= self.config.half_open_max_calls:
                self.state = CircuitState.CLOSED
                self.failure_count = 0
                self.failure_times.clear()
    
    def record_failure(self):
        """Record failed request."""
        now = datetime.utcnow()
        self.failure_count += 1
        self.last_failure_time = now
        self.failure_times.append(now)
        
        # Clean old failures outside monitoring window
        cutoff_time = now - timedelta(seconds=self.config.monitoring_window)
        self.failure_times = [t for t in self.failure_times if t > cutoff_time]
        
        if self.state == CircuitState.CLOSED:
            if len(self.failure_times) >= self.config.failure_threshold:
                self.state = CircuitState.OPEN
        elif self.state == CircuitState.HALF_OPEN:
            self.state = CircuitState.OPEN

class HealthMonitor:
    """Health monitoring system."""
    
    def __init__(self, config: HealthCheckConfig):
        self.config = config
        self.session: Optional[aiohttp.ClientSession] = None
        self.monitoring_tasks: Dict[str, asyncio.Task] = {}
    
    async def start(self):
        """Start health monitoring."""
        self.session = aiohttp.ClientSession(
            timeout=aiohttp.ClientTimeout(total=self.config.timeout)
        )
        logger.info("Health monitor started")
    
    async def stop(self):
        """Stop health monitoring."""
        # Cancel all monitoring tasks
        for task in self.monitoring_tasks.values():
            task.cancel()
        
        # Wait for tasks to complete
        if self.monitoring_tasks:
            await asyncio.gather(*self.monitoring_tasks.values(), return_exceptions=True)
        
        if self.session:
            await self.session.close()
        
        logger.info("Health monitor stopped")
    
    def start_monitoring(self, instance: ServiceInstance, 
                        callback: Callable[[ServiceInstance, HealthStatus], None]):
        """Start monitoring a service instance."""
        if instance.id in self.monitoring_tasks:
            return
        
        task = asyncio.create_task(
            self._monitor_instance(instance, callback)
        )
        self.monitoring_tasks[instance.id] = task
        
        logger.info(f"Started monitoring instance: {instance.id}")
    
    def stop_monitoring(self, instance_id: str):
        """Stop monitoring a service instance."""
        if instance_id in self.monitoring_tasks:
            self.monitoring_tasks[instance_id].cancel()
            del self.monitoring_tasks[instance_id]
            logger.info(f"Stopped monitoring instance: {instance_id}")
    
    async def _monitor_instance(self, instance: ServiceInstance,
                               callback: Callable[[ServiceInstance, HealthStatus], None]):
        """Monitor a single service instance."""
        consecutive_failures = 0
        consecutive_successes = 0
        
        while True:
            try:
                # Perform health check
                start_time = time.time()
                status = await self._check_health(instance)
                response_time = time.time() - start_time
                
                # Update response times
                instance.response_times.append(response_time)
                if len(instance.response_times) > 100:  # Keep last 100 measurements
                    instance.response_times.pop(0)
                
                instance.last_health_check = datetime.utcnow()
                
                # Determine health status
                if status and response_time <= self.config.expected_response_time:
                    consecutive_successes += 1
                    consecutive_failures = 0
                    
                    if (instance.health_status != HealthStatus.HEALTHY and 
                        consecutive_successes >= self.config.success_threshold):
                        instance.health_status = HealthStatus.HEALTHY
                        callback(instance, HealthStatus.HEALTHY)
                else:
                    consecutive_failures += 1
                    consecutive_successes = 0
                    
                    if consecutive_failures >= self.config.failure_threshold:
                        if response_time > self.config.expected_response_time * 2:
                            instance.health_status = HealthStatus.DEGRADED
                            callback(instance, HealthStatus.DEGRADED)
                        else:
                            instance.health_status = HealthStatus.UNHEALTHY
                            callback(instance, HealthStatus.UNHEALTHY)
                
                await asyncio.sleep(self.config.interval)
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Health check error for {instance.id}: {e}")
                consecutive_failures += 1
                consecutive_successes = 0
                
                if consecutive_failures >= self.config.failure_threshold:
                    instance.health_status = HealthStatus.UNHEALTHY
                    callback(instance, HealthStatus.UNHEALTHY)
                
                await asyncio.sleep(self.config.interval)
    
    async def _check_health(self, instance: ServiceInstance) -> bool:
        """Perform health check on service instance."""
        if not self.session:
            return False
        
        url = urljoin(instance.url, self.config.endpoint)
        
        for attempt in range(self.config.retries):
            try:
                async with self.session.get(url) as response:
                    return response.status in self.config.expected_status_codes
            except Exception as e:
                if attempt == self.config.retries - 1:
                    logger.warning(f"Health check failed for {instance.id}: {e}")
                await asyncio.sleep(0.5 * (attempt + 1))  # Exponential backoff
        
        return False

class LoadBalancer:
    """Main load balancer implementation."""
    
    def __init__(self, config: Optional[LoadBalancerConfig] = None):
        self.config = config or LoadBalancerConfig()
        self.instances: Dict[str, ServiceInstance] = {}
        self.circuit_breakers: Dict[str, CircuitBreaker] = {}
        self.health_monitor = HealthMonitor(self.config.health_check)
        
        # Load balancing state
        self.round_robin_index = 0
        self.sticky_sessions: Dict[str, str] = {}  # session_id -> instance_id
        
        # Metrics
        self.metrics = {
            "total_requests": 0,
            "successful_requests": 0,
            "failed_requests": 0,
            "average_response_time": 0.0,
            "requests_per_second": 0.0
        }
        self.request_times: List[float] = []
        
        logger.info(f"LoadBalancer initialized with strategy: {self.config.strategy}")
    
    async def start(self):
        """Start the load balancer."""
        await self.health_monitor.start()
        logger.info("Load balancer started")
    
    async def stop(self):
        """Stop the load balancer."""
        await self.health_monitor.stop()
        logger.info("Load balancer stopped")
    
    def add_instance(self, instance: ServiceInstance):
        """Add a service instance."""
        self.instances[instance.id] = instance
        self.circuit_breakers[instance.id] = CircuitBreaker(self.config.circuit_breaker)
        
        # Start health monitoring
        self.health_monitor.start_monitoring(instance, self._on_health_status_change)
        
        logger.info(f"Added instance: {instance.id} ({instance.url})")
    
    def remove_instance(self, instance_id: str):
        """Remove a service instance."""
        if instance_id in self.instances:
            self.health_monitor.stop_monitoring(instance_id)
            del self.instances[instance_id]
            del self.circuit_breakers[instance_id]
            
            # Remove from sticky sessions
            sessions_to_remove = [sid for sid, iid in self.sticky_sessions.items() if iid == instance_id]
            for session_id in sessions_to_remove:
                del self.sticky_sessions[session_id]
            
            logger.info(f"Removed instance: {instance_id}")
    
    def _on_health_status_change(self, instance: ServiceInstance, status: HealthStatus):
        """Handle health status changes."""
        logger.info(f"Instance {instance.id} health status changed to: {status}")
    
    def get_healthy_instances(self) -> List[ServiceInstance]:
        """Get list of healthy instances."""
        return [
            instance for instance in self.instances.values()
            if instance.health_status in [HealthStatus.HEALTHY, HealthStatus.DEGRADED]
        ]
    
    async def select_instance(self, client_ip: Optional[str] = None, 
                             session_id: Optional[str] = None) -> Optional[ServiceInstance]:
        """Select an instance based on load balancing strategy."""
        healthy_instances = self.get_healthy_instances()
        
        if not healthy_instances:
            logger.warning("No healthy instances available")
            return None
        
        # Check sticky sessions
        if self.config.sticky_sessions and session_id:
            if session_id in self.sticky_sessions:
                instance_id = self.sticky_sessions[session_id]
                if instance_id in self.instances:
                    instance = self.instances[instance_id]
                    if instance.health_status in [HealthStatus.HEALTHY, HealthStatus.DEGRADED]:
                        return instance
                # Remove invalid session
                del self.sticky_sessions[session_id]
        
        # Filter instances by circuit breaker
        available_instances = [
            instance for instance in healthy_instances
            if self.circuit_breakers[instance.id].can_execute()
        ]
        
        if not available_instances:
            logger.warning("No available instances (circuit breakers open)")
            return None
        
        # Select instance based on strategy
        selected_instance = None
        
        if self.config.strategy == LoadBalancingStrategy.ROUND_ROBIN:
            selected_instance = self._round_robin_select(available_instances)
        elif self.config.strategy == LoadBalancingStrategy.WEIGHTED_ROUND_ROBIN:
            selected_instance = self._weighted_round_robin_select(available_instances)
        elif self.config.strategy == LoadBalancingStrategy.LEAST_CONNECTIONS:
            selected_instance = self._least_connections_select(available_instances)
        elif self.config.strategy == LoadBalancingStrategy.LEAST_RESPONSE_TIME:
            selected_instance = self._least_response_time_select(available_instances)
        elif self.config.strategy == LoadBalancingStrategy.IP_HASH:
            selected_instance = self._ip_hash_select(available_instances, client_ip)
        elif self.config.strategy == LoadBalancingStrategy.RANDOM:
            selected_instance = self._random_select(available_instances)
        elif self.config.strategy == LoadBalancingStrategy.WEIGHTED_RANDOM:
            selected_instance = self._weighted_random_select(available_instances)
        
        # Create sticky session if enabled
        if (self.config.sticky_sessions and session_id and selected_instance):
            self.sticky_sessions[session_id] = selected_instance.id
        
        return selected_instance
    
    def _round_robin_select(self, instances: List[ServiceInstance]) -> ServiceInstance:
        """Round robin selection."""
        instance = instances[self.round_robin_index % len(instances)]
        self.round_robin_index += 1
        return instance
    
    def _weighted_round_robin_select(self, instances: List[ServiceInstance]) -> ServiceInstance:
        """Weighted round robin selection."""
        total_weight = sum(instance.weight for instance in instances)
        if total_weight == 0:
            return self._round_robin_select(instances)
        
        # Create weighted list
        weighted_instances = []
        for instance in instances:
            weighted_instances.extend([instance] * instance.weight)
        
        return self._round_robin_select(weighted_instances)
    
    def _least_connections_select(self, instances: List[ServiceInstance]) -> ServiceInstance:
        """Least connections selection."""
        return min(instances, key=lambda x: x.active_connections)
    
    def _least_response_time_select(self, instances: List[ServiceInstance]) -> ServiceInstance:
        """Least response time selection."""
        return min(instances, key=lambda x: x.average_response_time)
    
    def _ip_hash_select(self, instances: List[ServiceInstance], 
                       client_ip: Optional[str]) -> ServiceInstance:
        """IP hash selection."""
        if not client_ip:
            return self._round_robin_select(instances)
        
        hash_value = int(hashlib.md5(client_ip.encode()).hexdigest(), 16)
        index = hash_value % len(instances)
        return instances[index]
    
    def _random_select(self, instances: List[ServiceInstance]) -> ServiceInstance:
        """Random selection."""
        return random.choice(instances)
    
    def _weighted_random_select(self, instances: List[ServiceInstance]) -> ServiceInstance:
        """Weighted random selection."""
        total_weight = sum(instance.weight for instance in instances)
        if total_weight == 0:
            return self._random_select(instances)
        
        random_weight = random.uniform(0, total_weight)
        current_weight = 0
        
        for instance in instances:
            current_weight += instance.weight
            if random_weight <= current_weight:
                return instance
        
        return instances[-1]  # Fallback
    
    async def execute_request(self, method: str, path: str, 
                             client_ip: Optional[str] = None,
                             session_id: Optional[str] = None,
                             **kwargs) -> Dict[str, Any]:
        """Execute request with load balancing and circuit breaking."""
        start_time = time.time()
        
        for attempt in range(self.config.max_retries + 1):
            instance = await self.select_instance(client_ip, session_id)
            
            if not instance:
                return {
                    "success": False,
                    "error": "No available instances",
                    "status_code": 503
                }
            
            circuit_breaker = self.circuit_breakers[instance.id]
            
            if not circuit_breaker.can_execute():
                if attempt < self.config.max_retries:
                    await asyncio.sleep(self.config.retry_delay)
                    continue
                return {
                    "success": False,
                    "error": "Circuit breaker open",
                    "status_code": 503
                }
            
            try:
                # Increment active connections
                instance.active_connections += 1
                instance.total_requests += 1
                
                # Execute request
                result = await self._execute_http_request(instance, method, path, **kwargs)
                
                # Record success
                circuit_breaker.record_success()
                
                # Update metrics
                response_time = time.time() - start_time
                self._update_metrics(True, response_time)
                
                return result
                
            except Exception as e:
                logger.error(f"Request failed on instance {instance.id}: {e}")
                
                # Record failure
                circuit_breaker.record_failure()
                instance.failed_requests += 1
                
                if attempt < self.config.max_retries:
                    await asyncio.sleep(self.config.retry_delay * (attempt + 1))
                    continue
                
                # Update metrics
                response_time = time.time() - start_time
                self._update_metrics(False, response_time)
                
                return {
                    "success": False,
                    "error": str(e),
                    "status_code": 500
                }
            
            finally:
                # Decrement active connections
                instance.active_connections = max(0, instance.active_connections - 1)
        
        return {
            "success": False,
            "error": "Max retries exceeded",
            "status_code": 503
        }
    
    async def _execute_http_request(self, instance: ServiceInstance, 
                                   method: str, path: str, **kwargs) -> Dict[str, Any]:
        """Execute HTTP request to service instance."""
        url = urljoin(instance.url, path)
        
        timeout = aiohttp.ClientTimeout(total=30)
        async with aiohttp.ClientSession(timeout=timeout) as session:
            async with session.request(method, url, **kwargs) as response:
                content = await response.text()
                
                return {
                    "success": True,
                    "status_code": response.status,
                    "content": content,
                    "headers": dict(response.headers)
                }
    
    def _update_metrics(self, success: bool, response_time: float):
        """Update load balancer metrics."""
        if not self.config.enable_metrics:
            return
        
        self.metrics["total_requests"] += 1
        
        if success:
            self.metrics["successful_requests"] += 1
        else:
            self.metrics["failed_requests"] += 1
        
        # Update response times
        self.request_times.append(response_time)
        if len(self.request_times) > 1000:  # Keep last 1000 measurements
            self.request_times.pop(0)
        
        # Calculate average response time
        if self.request_times:
            self.metrics["average_response_time"] = statistics.mean(self.request_times)
    
    def get_metrics(self) -> Dict[str, Any]:
        """Get load balancer metrics."""
        metrics = self.metrics.copy()
        
        # Add instance metrics
        metrics["instances"] = {
            instance.id: {
                "health_status": instance.health_status,
                "active_connections": instance.active_connections,
                "total_requests": instance.total_requests,
                "failed_requests": instance.failed_requests,
                "success_rate": instance.success_rate,
                "average_response_time": instance.average_response_time,
                "circuit_breaker_state": self.circuit_breakers[instance.id].state
            }
            for instance in self.instances.values()
        }
        
        # Add circuit breaker metrics
        metrics["circuit_breakers"] = {
            instance_id: {
                "state": cb.state,
                "failure_count": cb.failure_count,
                "last_failure_time": cb.last_failure_time.isoformat() if cb.last_failure_time else None
            }
            for instance_id, cb in self.circuit_breakers.items()
        }
        
        return metrics
    
    def get_health_status(self) -> Dict[str, Any]:
        """Get overall health status."""
        healthy_instances = len(self.get_healthy_instances())
        total_instances = len(self.instances)
        
        return {
            "status": "healthy" if healthy_instances > 0 else "unhealthy",
            "healthy_instances": healthy_instances,
            "total_instances": total_instances,
            "health_ratio": healthy_instances / total_instances if total_instances > 0 else 0,
            "instances": {
                instance.id: {
                    "status": instance.health_status,
                    "url": instance.url,
                    "last_check": instance.last_health_check.isoformat() if instance.last_health_check else None
                }
                for instance in self.instances.values()
            }
        }

# Auto-scaling manager
class AutoScaler:
    """Auto-scaling manager for dynamic instance management."""
    
    def __init__(self, load_balancer: LoadBalancer):
        self.load_balancer = load_balancer
        self.scaling_rules = {
            "cpu_threshold": 80.0,
            "memory_threshold": 85.0,
            "response_time_threshold": 2.0,
            "min_instances": 2,
            "max_instances": 10,
            "scale_up_cooldown": 300,  # 5 minutes
            "scale_down_cooldown": 600  # 10 minutes
        }
        self.last_scale_action: Optional[datetime] = None
    
    async def evaluate_scaling(self) -> Dict[str, Any]:
        """Evaluate if scaling action is needed."""
        try:
            instances = self.load_balancer.get_healthy_instances()
            
            if not instances:
                return {"action": "none", "reason": "No healthy instances"}
            
            # Calculate metrics
            avg_cpu = statistics.mean(instance.cpu_usage for instance in instances)
            avg_memory = statistics.mean(instance.memory_usage for instance in instances)
            avg_response_time = statistics.mean(instance.average_response_time for instance in instances)
            
            current_time = datetime.utcnow()
            
            # Check if we're in cooldown period
            if self.last_scale_action:
                time_since_last_action = (current_time - self.last_scale_action).total_seconds()
                if time_since_last_action < self.scaling_rules["scale_up_cooldown"]:
                    return {"action": "none", "reason": "In cooldown period"}
            
            # Determine scaling action
            scale_up_needed = (
                avg_cpu > self.scaling_rules["cpu_threshold"] or
                avg_memory > self.scaling_rules["memory_threshold"] or
                avg_response_time > self.scaling_rules["response_time_threshold"]
            )
            
            scale_down_needed = (
                avg_cpu < self.scaling_rules["cpu_threshold"] * 0.5 and
                avg_memory < self.scaling_rules["memory_threshold"] * 0.5 and
                avg_response_time < self.scaling_rules["response_time_threshold"] * 0.5
            )
            
            if scale_up_needed and len(instances) < self.scaling_rules["max_instances"]:
                return {
                    "action": "scale_up",
                    "reason": f"High resource usage: CPU={avg_cpu:.1f}%, Memory={avg_memory:.1f}%, RT={avg_response_time:.2f}s",
                    "metrics": {"cpu": avg_cpu, "memory": avg_memory, "response_time": avg_response_time}
                }
            
            elif scale_down_needed and len(instances) > self.scaling_rules["min_instances"]:
                return {
                    "action": "scale_down",
                    "reason": f"Low resource usage: CPU={avg_cpu:.1f}%, Memory={avg_memory:.1f}%, RT={avg_response_time:.2f}s",
                    "metrics": {"cpu": avg_cpu, "memory": avg_memory, "response_time": avg_response_time}
                }
            
            return {"action": "none", "reason": "Metrics within acceptable range"}
            
        except Exception as e:
            logger.error(f"Auto-scaling evaluation failed: {e}")
            return {"action": "none", "reason": f"Evaluation error: {e}"}

# Main function for standalone testing
if __name__ == "__main__":
    async def test_load_balancer():
        """Test load balancer functionality."""
        # Initialize load balancer
        config = LoadBalancerConfig(
            strategy=LoadBalancingStrategy.ROUND_ROBIN,
            health_check=HealthCheckConfig(interval=10)
        )
        lb = LoadBalancer(config)
        
        # Add test instances
        instances = [
            ServiceInstance("instance1", "localhost", 8001, weight=1),
            ServiceInstance("instance2", "localhost", 8002, weight=2),
            ServiceInstance("instance3", "localhost", 8003, weight=1)
        ]
        
        try:
            await lb.start()
            
            for instance in instances:
                lb.add_instance(instance)
            
            # Test instance selection
            for i in range(10):
                selected = await lb.select_instance()
                print(f"Request {i+1}: {selected.id if selected else 'None'}")
            
            # Get metrics
            metrics = lb.get_metrics()
            print(f"Metrics: {json.dumps(metrics, indent=2)}")
            
            # Get health status
            health = lb.get_health_status()
            print(f"Health: {json.dumps(health, indent=2)}")
            
            print("Load balancer test completed successfully!")
            
        except Exception as e:
            print(f"Load balancer test failed: {e}")
        finally:
            await lb.stop()
    
    # Run test
    asyncio.run(test_load_balancer())