#!/usr/bin/env python3
"""
Performance Testing Framework

Comprehensive performance testing for the Malaysian document parser.
Includes load testing, stress testing, memory profiling, and performance monitoring.
"""

import os
import sys
import time
import psutil
import pytest
import threading
import multiprocessing
from pathlib import Path
from typing import Dict, List, Any, Optional, Callable
from dataclasses import dataclass, asdict
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
from unittest.mock import Mock, patch
import numpy as np
from PIL import Image
import io
import json

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))


@dataclass
class PerformanceMetrics:
    """Performance metrics for a single test run."""
    execution_time: float
    memory_usage_mb: float
    cpu_usage_percent: float
    throughput_docs_per_second: float
    peak_memory_mb: float
    error_rate: float
    success_count: int
    error_count: int
    total_requests: int


@dataclass
class LoadTestConfig:
    """Configuration for load testing."""
    concurrent_users: int = 10
    duration_seconds: int = 60
    ramp_up_seconds: int = 10
    target_throughput: float = 5.0  # docs per second
    max_response_time: float = 2.0  # seconds
    acceptable_error_rate: float = 0.05  # 5%


@dataclass
class StressTestConfig:
    """Configuration for stress testing."""
    max_concurrent_users: int = 100
    step_size: int = 10
    step_duration: int = 30
    breaking_point_threshold: float = 0.1  # 10% error rate
    memory_limit_mb: int = 2048


class PerformanceMonitor:
    """Monitor system performance during tests."""
    
    def __init__(self):
        self.process = psutil.Process()
        self.monitoring = False
        self.metrics = []
        self.monitor_thread = None
    
    def start_monitoring(self, interval: float = 0.1):
        """Start performance monitoring."""
        self.monitoring = True
        self.metrics = []
        self.monitor_thread = threading.Thread(
            target=self._monitor_loop,
            args=(interval,)
        )
        self.monitor_thread.start()
    
    def stop_monitoring(self) -> Dict[str, float]:
        """Stop monitoring and return aggregated metrics."""
        self.monitoring = False
        if self.monitor_thread:
            self.monitor_thread.join()
        
        if not self.metrics:
            return {
                'avg_memory_mb': 0.0,
                'peak_memory_mb': 0.0,
                'avg_cpu_percent': 0.0,
                'peak_cpu_percent': 0.0
            }
        
        memory_values = [m['memory_mb'] for m in self.metrics]
        cpu_values = [m['cpu_percent'] for m in self.metrics]
        
        return {
            'avg_memory_mb': np.mean(memory_values),
            'peak_memory_mb': np.max(memory_values),
            'avg_cpu_percent': np.mean(cpu_values),
            'peak_cpu_percent': np.max(cpu_values)
        }
    
    def _monitor_loop(self, interval: float):
        """Monitoring loop running in separate thread."""
        while self.monitoring:
            try:
                memory_info = self.process.memory_info()
                cpu_percent = self.process.cpu_percent()
                
                self.metrics.append({
                    'timestamp': time.time(),
                    'memory_mb': memory_info.rss / 1024 / 1024,
                    'cpu_percent': cpu_percent
                })
                
                time.sleep(interval)
            except Exception:
                # Process might have ended
                break


class DocumentProcessorMock:
    """Mock document processor for performance testing."""
    
    def __init__(self, processing_time: float = 0.1):
        self.processing_time = processing_time
        self.call_count = 0
        self.error_rate = 0.0
    
    def process_document(self, image_data: bytes) -> Dict[str, Any]:
        """Mock document processing with configurable delay."""
        self.call_count += 1
        
        # Simulate processing time
        time.sleep(self.processing_time)
        
        # Simulate errors based on error rate
        if np.random.random() < self.error_rate:
            raise Exception("Simulated processing error")
        
        # Return mock result
        return {
            'document_type': 'mykad',
            'fields': {
                'name': 'AHMAD BIN ALI',
                'ic_number': '123456-78-9012',
                'address': 'NO 123, JALAN ABC, 12345 KUALA LUMPUR'
            },
            'confidence': 0.95,
            'processing_time': self.processing_time
        }


class PerformanceTester:
    """Main performance testing class."""
    
    def __init__(self):
        self.monitor = PerformanceMonitor()
        self.processor = DocumentProcessorMock()
        self.test_images = self._generate_test_images()
    
    def _generate_test_images(self, count: int = 10) -> List[bytes]:
        """Generate test images for performance testing."""
        images = []
        
        for i in range(count):
            # Create a simple test image
            img = Image.new('RGB', (800, 600), color=(255, 255, 255))
            
            # Convert to bytes
            img_bytes = io.BytesIO()
            img.save(img_bytes, format='JPEG')
            images.append(img_bytes.getvalue())
        
        return images
    
    def run_single_request(self) -> Dict[str, Any]:
        """Run a single document processing request."""
        start_time = time.time()
        
        try:
            # Select random test image
            image_data = np.random.choice(self.test_images)
            
            # Process document
            result = self.processor.process_document(image_data)
            
            execution_time = time.time() - start_time
            
            return {
                'success': True,
                'execution_time': execution_time,
                'result': result,
                'error': None
            }
            
        except Exception as e:
            execution_time = time.time() - start_time
            
            return {
                'success': False,
                'execution_time': execution_time,
                'result': None,
                'error': str(e)
            }
    
    def run_load_test(self, config: LoadTestConfig) -> PerformanceMetrics:
        """Run load testing with specified configuration."""
        print(f"Starting load test: {config.concurrent_users} users, {config.duration_seconds}s")
        
        # Start monitoring
        self.monitor.start_monitoring()
        
        start_time = time.time()
        results = []
        
        def worker():
            """Worker function for load testing."""
            worker_results = []
            end_time = start_time + config.duration_seconds
            
            while time.time() < end_time:
                result = self.run_single_request()
                worker_results.append(result)
                
                # Add small delay to control throughput
                time.sleep(0.1)
            
            return worker_results
        
        # Run concurrent workers
        with ThreadPoolExecutor(max_workers=config.concurrent_users) as executor:
            futures = [executor.submit(worker) for _ in range(config.concurrent_users)]
            
            for future in futures:
                results.extend(future.result())
        
        # Stop monitoring
        monitor_metrics = self.monitor.stop_monitoring()
        
        # Calculate metrics
        total_time = time.time() - start_time
        success_results = [r for r in results if r['success']]
        error_results = [r for r in results if not r['success']]
        
        success_count = len(success_results)
        error_count = len(error_results)
        total_requests = len(results)
        
        avg_execution_time = np.mean([r['execution_time'] for r in success_results]) if success_results else 0
        throughput = success_count / total_time if total_time > 0 else 0
        error_rate = error_count / total_requests if total_requests > 0 else 0
        
        return PerformanceMetrics(
            execution_time=avg_execution_time,
            memory_usage_mb=monitor_metrics['avg_memory_mb'],
            cpu_usage_percent=monitor_metrics['avg_cpu_percent'],
            throughput_docs_per_second=throughput,
            peak_memory_mb=monitor_metrics['peak_memory_mb'],
            error_rate=error_rate,
            success_count=success_count,
            error_count=error_count,
            total_requests=total_requests
        )
    
    def run_stress_test(self, config: StressTestConfig) -> Dict[str, Any]:
        """Run stress testing to find breaking point."""
        print(f"Starting stress test: up to {config.max_concurrent_users} users")
        
        results = []
        breaking_point = None
        
        for users in range(config.step_size, config.max_concurrent_users + 1, config.step_size):
            print(f"Testing with {users} concurrent users...")
            
            load_config = LoadTestConfig(
                concurrent_users=users,
                duration_seconds=config.step_duration,
                ramp_up_seconds=5
            )
            
            metrics = self.run_load_test(load_config)
            results.append({
                'concurrent_users': users,
                'metrics': metrics
            })
            
            # Check if we've hit the breaking point
            if (metrics.error_rate > config.breaking_point_threshold or 
                metrics.peak_memory_mb > config.memory_limit_mb):
                breaking_point = users
                print(f"Breaking point reached at {users} users")
                break
            
            # Small delay between steps
            time.sleep(2)
        
        return {
            'results': results,
            'breaking_point': breaking_point,
            'max_stable_users': breaking_point - config.step_size if breaking_point else config.max_concurrent_users
        }
    
    def run_memory_profiling(self, duration: int = 60) -> Dict[str, Any]:
        """Run memory profiling test."""
        print(f"Starting memory profiling for {duration} seconds...")
        
        self.monitor.start_monitoring(interval=0.05)  # Higher frequency for memory profiling
        
        start_time = time.time()
        request_count = 0
        
        while time.time() - start_time < duration:
            self.run_single_request()
            request_count += 1
            time.sleep(0.1)
        
        monitor_metrics = self.monitor.stop_monitoring()
        
        return {
            'duration': duration,
            'requests_processed': request_count,
            'memory_metrics': monitor_metrics,
            'memory_growth': monitor_metrics['peak_memory_mb'] - monitor_metrics['avg_memory_mb']
        }


# Pytest fixtures
@pytest.fixture
def performance_tester():
    """Create performance tester instance."""
    return PerformanceTester()


@pytest.fixture
def load_test_config():
    """Default load test configuration."""
    return LoadTestConfig(
        concurrent_users=5,
        duration_seconds=30,
        target_throughput=2.0,
        max_response_time=1.0
    )


@pytest.fixture
def stress_test_config():
    """Default stress test configuration."""
    return StressTestConfig(
        max_concurrent_users=50,
        step_size=5,
        step_duration=15
    )


# Performance tests
@pytest.mark.performance
class TestPerformance:
    """Performance test suite."""
    
    def test_single_request_performance(self, performance_tester):
        """Test single request performance."""
        result = performance_tester.run_single_request()
        
        assert result['success'] is True
        assert result['execution_time'] < 2.0  # Should complete within 2 seconds
        assert result['result'] is not None
    
    def test_load_performance(self, performance_tester, load_test_config):
        """Test system performance under normal load."""
        metrics = performance_tester.run_load_test(load_test_config)
        
        # Performance assertions
        assert metrics.error_rate < 0.05  # Less than 5% error rate
        assert metrics.throughput_docs_per_second >= 1.0  # At least 1 doc/sec
        assert metrics.execution_time < 2.0  # Average response time under 2s
        assert metrics.peak_memory_mb < 1024  # Memory usage under 1GB
    
    @pytest.mark.slow
    def test_stress_performance(self, performance_tester, stress_test_config):
        """Test system breaking point under stress."""
        results = performance_tester.run_stress_test(stress_test_config)
        
        assert results['breaking_point'] is not None or results['max_stable_users'] > 0
        assert len(results['results']) > 0
        
        # Verify that performance degrades gracefully
        for i in range(1, len(results['results'])):
            current = results['results'][i]['metrics']
            previous = results['results'][i-1]['metrics']
            
            # Error rate should not increase too dramatically
            error_rate_increase = current.error_rate - previous.error_rate
            assert error_rate_increase < 0.5  # No more than 50% jump in error rate
    
    def test_memory_profiling(self, performance_tester):
        """Test memory usage and potential leaks."""
        results = performance_tester.run_memory_profiling(duration=30)
        
        # Memory growth should be reasonable
        assert results['memory_growth'] < 100  # Less than 100MB growth
        assert results['memory_metrics']['peak_memory_mb'] < 512  # Peak under 512MB
        assert results['requests_processed'] > 0
    
    def test_concurrent_processing(self, performance_tester):
        """Test concurrent document processing."""
        def process_batch():
            results = []
            for _ in range(10):
                result = performance_tester.run_single_request()
                results.append(result)
            return results
        
        # Run multiple batches concurrently
        with ThreadPoolExecutor(max_workers=3) as executor:
            futures = [executor.submit(process_batch) for _ in range(3)]
            all_results = []
            
            for future in futures:
                batch_results = future.result()
                all_results.extend(batch_results)
        
        # Verify all requests completed successfully
        success_count = sum(1 for r in all_results if r['success'])
        total_count = len(all_results)
        
        assert success_count / total_count >= 0.95  # 95% success rate
    
    def test_throughput_scaling(self, performance_tester):
        """Test throughput scaling with different user loads."""
        user_counts = [1, 2, 5, 10]
        throughputs = []
        
        for users in user_counts:
            config = LoadTestConfig(
                concurrent_users=users,
                duration_seconds=20
            )
            
            metrics = performance_tester.run_load_test(config)
            throughputs.append(metrics.throughput_docs_per_second)
        
        # Throughput should generally increase with more users (up to a point)
        assert throughputs[1] >= throughputs[0]  # 2 users >= 1 user
        assert throughputs[2] >= throughputs[1]  # 5 users >= 2 users
        
        # But shouldn't decrease dramatically at higher loads
        if len(throughputs) > 3:
            assert throughputs[3] >= throughputs[2] * 0.8  # 10 users >= 80% of 5 users


@pytest.mark.benchmark
class TestBenchmarks:
    """Benchmark tests for performance comparison."""
    
    def test_document_processing_benchmark(self, performance_tester, benchmark):
        """Benchmark document processing performance."""
        def process_document():
            return performance_tester.run_single_request()
        
        result = benchmark(process_document)
        assert result['success'] is True
    
    def test_batch_processing_benchmark(self, performance_tester, benchmark):
        """Benchmark batch processing performance."""
        def process_batch():
            results = []
            for _ in range(5):
                result = performance_tester.run_single_request()
                results.append(result)
            return results
        
        results = benchmark(process_batch)
        success_count = sum(1 for r in results if r['success'])
        assert success_count >= 4  # At least 80% success rate


if __name__ == '__main__':
    # Run performance tests directly
    tester = PerformanceTester()
    
    print("Running basic performance test...")
    result = tester.run_single_request()
    print(f"Single request result: {result}")
    
    print("\nRunning load test...")
    load_config = LoadTestConfig(concurrent_users=3, duration_seconds=10)
    metrics = tester.run_load_test(load_config)
    print(f"Load test metrics: {asdict(metrics)}")
    
    print("\nPerformance testing completed.")