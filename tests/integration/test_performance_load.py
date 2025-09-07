#!/usr/bin/env python3
"""
Performance and Load Testing for Document Parser

Comprehensive performance and load tests that validate system behavior
under stress, measure throughput, and identify performance bottlenecks.
"""

import os
import sys
import time
import json
import threading
import multiprocessing
import statistics
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
from dataclasses import dataclass
from datetime import datetime, timedelta

import pytest
import numpy as np
from PIL import Image
import psutil
import io

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

try:
    from src.document_parser import (
        DocumentClassifier,
        OCRService,
        FieldExtractor,
        DocumentValidator,
        ImagePreprocessor
    )
except ImportError as e:
    pytest.skip(f"Document parser modules not available: {e}", allow_module_level=True)


@dataclass
class PerformanceMetrics:
    """Performance metrics data structure."""
    operation: str
    execution_time: float
    memory_usage: int
    cpu_usage: float
    success: bool
    error_message: Optional[str] = None
    timestamp: datetime = None
    
    def __post_init__(self):
        if self.timestamp is None:
            self.timestamp = datetime.now()


@dataclass
class LoadTestResults:
    """Load test results data structure."""
    test_name: str
    total_requests: int
    successful_requests: int
    failed_requests: int
    average_response_time: float
    min_response_time: float
    max_response_time: float
    percentile_95: float
    percentile_99: float
    throughput: float  # requests per second
    error_rate: float
    memory_peak: int
    cpu_peak: float
    duration: float


class PerformanceMonitor:
    """System performance monitoring utility."""
    
    def __init__(self):
        self.process = psutil.Process(os.getpid())
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
    
    def stop_monitoring(self) -> Dict[str, Any]:
        """Stop monitoring and return metrics."""
        self.monitoring = False
        if self.monitor_thread:
            self.monitor_thread.join()
        
        if not self.metrics:
            return {}
        
        cpu_values = [m['cpu_percent'] for m in self.metrics]
        memory_values = [m['memory_rss'] for m in self.metrics]
        
        return {
            'cpu_avg': statistics.mean(cpu_values),
            'cpu_max': max(cpu_values),
            'cpu_min': min(cpu_values),
            'memory_avg': statistics.mean(memory_values),
            'memory_max': max(memory_values),
            'memory_min': min(memory_values),
            'samples': len(self.metrics)
        }
    
    def _monitor_loop(self, interval: float):
        """Monitoring loop."""
        while self.monitoring:
            try:
                cpu_percent = self.process.cpu_percent()
                memory_info = self.process.memory_info()
                
                self.metrics.append({
                    'timestamp': time.time(),
                    'cpu_percent': cpu_percent,
                    'memory_rss': memory_info.rss,
                    'memory_vms': memory_info.vms
                })
                
                time.sleep(interval)
            except Exception:
                break


class TestPerformanceLoad:
    """Performance and load testing for document parser."""
    
    @pytest.fixture(autouse=True)
    def setup_performance_testing(self):
        """Setup performance testing environment."""
        # Initialize components
        self.classifier = DocumentClassifier()
        self.ocr_service = OCRService()
        self.field_extractor = FieldExtractor()
        self.validator = DocumentValidator()
        self.preprocessor = ImagePreprocessor()
        
        # Performance monitor
        self.monitor = PerformanceMonitor()
        
        # Test configuration
        self.performance_config = {
            'max_response_time': 5.0,  # seconds
            'max_memory_usage': 500 * 1024 * 1024,  # 500MB
            'max_cpu_usage': 80.0,  # percent
            'min_throughput': 10.0,  # requests per second
            'error_threshold': 0.05  # 5% error rate
        }
        
        # Load test scenarios
        self.load_scenarios = {
            'light': {'concurrent_users': 5, 'requests_per_user': 10, 'duration': 30},
            'medium': {'concurrent_users': 20, 'requests_per_user': 25, 'duration': 60},
            'heavy': {'concurrent_users': 50, 'requests_per_user': 20, 'duration': 120},
            'stress': {'concurrent_users': 100, 'requests_per_user': 10, 'duration': 180}
        }
        
        # Results storage
        self.performance_results = []
        self.load_test_results = []
    
    def create_test_documents(self, count: int, doc_types: List[str] = None) -> List[np.ndarray]:
        """Create multiple test documents for load testing."""
        if doc_types is None:
            doc_types = ['mykad', 'spk']
        
        documents = []
        for i in range(count):
            doc_type = doc_types[i % len(doc_types)]
            
            if doc_type == 'mykad':
                img = np.ones((400, 600, 3), dtype=np.uint8) * 255
                img[50:70, 50:200] = [0, 0, 0]
                img[100:120, 50:150] = [0, 0, 0]
            else:  # spk
                img = np.ones((800, 600, 3), dtype=np.uint8) * 255
                img[100:120, 100:300] = [0, 0, 0]
                img[200:220, 100:250] = [0, 0, 0]
            
            # Add some variation
            noise = np.random.randint(0, 20, img.shape, dtype=np.uint8)
            img = np.clip(img.astype(int) + noise, 0, 255).astype(np.uint8)
            
            documents.append(img)
        
        return documents
    
    def measure_operation_performance(self, operation_func, *args, **kwargs) -> PerformanceMetrics:
        """Measure performance of a single operation."""
        # Start monitoring
        process = psutil.Process(os.getpid())
        start_memory = process.memory_info().rss
        start_cpu = process.cpu_percent()
        start_time = time.time()
        
        success = True
        error_message = None
        
        try:
            result = operation_func(*args, **kwargs)
        except Exception as e:
            success = False
            error_message = str(e)
            result = None
        
        # End monitoring
        end_time = time.time()
        end_memory = process.memory_info().rss
        end_cpu = process.cpu_percent()
        
        execution_time = end_time - start_time
        memory_usage = end_memory - start_memory
        cpu_usage = max(start_cpu, end_cpu)
        
        return PerformanceMetrics(
            operation=operation_func.__name__,
            execution_time=execution_time,
            memory_usage=memory_usage,
            cpu_usage=cpu_usage,
            success=success,
            error_message=error_message
        )
    
    @pytest.mark.performance
    def test_classification_performance(self):
        """Test document classification performance."""
        documents = self.create_test_documents(50)
        metrics = []
        
        for doc in documents:
            metric = self.measure_operation_performance(
                self.classifier.classify, doc
            )
            metrics.append(metric)
        
        # Analyze results
        successful_metrics = [m for m in metrics if m.success]
        assert len(successful_metrics) >= 45  # At least 90% success rate
        
        avg_time = statistics.mean([m.execution_time for m in successful_metrics])
        max_time = max([m.execution_time for m in successful_metrics])
        avg_memory = statistics.mean([m.memory_usage for m in successful_metrics])
        
        # Performance assertions
        assert avg_time < 2.0  # Average classification time under 2 seconds
        assert max_time < 5.0  # Maximum classification time under 5 seconds
        assert avg_memory < 50 * 1024 * 1024  # Average memory usage under 50MB
        
        self.performance_results.extend(metrics)
    
    @pytest.mark.performance
    def test_ocr_performance(self):
        """Test OCR service performance."""
        documents = self.create_test_documents(30)
        metrics = []
        
        for doc in documents:
            metric = self.measure_operation_performance(
                self.ocr_service.extract_text, doc
            )
            metrics.append(metric)
        
        # Analyze results
        successful_metrics = [m for m in metrics if m.success]
        assert len(successful_metrics) >= 25  # At least 83% success rate
        
        avg_time = statistics.mean([m.execution_time for m in successful_metrics])
        max_time = max([m.execution_time for m in successful_metrics])
        
        # OCR is typically the slowest operation
        assert avg_time < 4.0  # Average OCR time under 4 seconds
        assert max_time < 8.0  # Maximum OCR time under 8 seconds
        
        self.performance_results.extend(metrics)
    
    @pytest.mark.performance
    def test_field_extraction_performance(self):
        """Test field extraction performance."""
        # Create sample OCR text
        sample_texts = [
            "AHMAD BIN ALI\n123456-78-9012\nNO 123 JALAN ABC",
            "SITI FATIMAH BINTI HASSAN\n987654-32-1098\nNO 456 JALAN DEF",
            "CERTIFICATE OF EDUCATION\nJOHN DOE\nMATHEMATICS A"
        ] * 20
        
        doc_types = ['mykad', 'mykad', 'spk'] * 20
        metrics = []
        
        for text, doc_type in zip(sample_texts, doc_types):
            metric = self.measure_operation_performance(
                self.field_extractor.extract_fields, text, doc_type
            )
            metrics.append(metric)
        
        # Analyze results
        successful_metrics = [m for m in metrics if m.success]
        assert len(successful_metrics) >= 55  # At least 92% success rate
        
        avg_time = statistics.mean([m.execution_time for m in successful_metrics])
        max_time = max([m.execution_time for m in successful_metrics])
        
        # Field extraction should be fast
        assert avg_time < 0.5  # Average extraction time under 0.5 seconds
        assert max_time < 2.0  # Maximum extraction time under 2 seconds
        
        self.performance_results.extend(metrics)
    
    @pytest.mark.performance
    def test_validation_performance(self):
        """Test validation performance."""
        # Create sample extracted data
        from src.document_parser.schema import ExtractedData
        
        sample_data = [
            ExtractedData(
                document_type='mykad',
                fields={'name': 'AHMAD BIN ALI', 'ic_number': '123456-78-9012'},
                confidence=0.9
            ),
            ExtractedData(
                document_type='spk',
                fields={'candidate_name': 'JOHN DOE', 'subject': 'MATHEMATICS'},
                confidence=0.85
            )
        ] * 30
        
        metrics = []
        
        for data in sample_data:
            metric = self.measure_operation_performance(
                self.validator.validate, data
            )
            metrics.append(metric)
        
        # Analyze results
        successful_metrics = [m for m in metrics if m.success]
        assert len(successful_metrics) >= 58  # At least 97% success rate
        
        avg_time = statistics.mean([m.execution_time for m in successful_metrics])
        max_time = max([m.execution_time for m in successful_metrics])
        
        # Validation should be very fast
        assert avg_time < 0.1  # Average validation time under 0.1 seconds
        assert max_time < 0.5  # Maximum validation time under 0.5 seconds
        
        self.performance_results.extend(metrics)
    
    def run_load_test_scenario(self, scenario_name: str, scenario_config: Dict) -> LoadTestResults:
        """Run a specific load test scenario."""
        print(f"\nRunning {scenario_name} load test scenario...")
        
        concurrent_users = scenario_config['concurrent_users']
        requests_per_user = scenario_config['requests_per_user']
        duration = scenario_config['duration']
        
        # Create test documents
        total_requests = concurrent_users * requests_per_user
        documents = self.create_test_documents(total_requests)
        
        # Results tracking
        results = []
        start_time = time.time()
        
        # Start system monitoring
        self.monitor.start_monitoring()
        
        def user_simulation(user_id: int, user_documents: List[np.ndarray]):
            """Simulate a user making multiple requests."""
            user_results = []
            
            for i, doc in enumerate(user_documents):
                request_start = time.time()
                success = True
                error_msg = None
                
                try:
                    # Simulate complete pipeline
                    classification = self.classifier.classify(doc)
                    ocr_result = self.ocr_service.extract_text(doc)
                    extracted_data = self.field_extractor.extract_fields(
                        ocr_result['text'], 
                        classification['document_type']
                    )
                    validation = self.validator.validate(extracted_data)
                    
                except Exception as e:
                    success = False
                    error_msg = str(e)
                
                request_time = time.time() - request_start
                
                user_results.append({
                    'user_id': user_id,
                    'request_id': i,
                    'response_time': request_time,
                    'success': success,
                    'error': error_msg,
                    'timestamp': time.time()
                })
                
                # Check if duration exceeded
                if time.time() - start_time > duration:
                    break
            
            return user_results
        
        # Execute load test with thread pool
        with ThreadPoolExecutor(max_workers=concurrent_users) as executor:
            # Distribute documents among users
            docs_per_user = len(documents) // concurrent_users
            futures = []
            
            for user_id in range(concurrent_users):
                start_idx = user_id * docs_per_user
                end_idx = start_idx + docs_per_user
                user_docs = documents[start_idx:end_idx]
                
                future = executor.submit(user_simulation, user_id, user_docs)
                futures.append(future)
            
            # Collect results
            for future in futures:
                try:
                    user_results = future.result(timeout=duration + 30)
                    results.extend(user_results)
                except Exception as e:
                    print(f"User simulation failed: {e}")
        
        # Stop monitoring
        system_metrics = self.monitor.stop_monitoring()
        
        # Calculate metrics
        total_time = time.time() - start_time
        successful_requests = len([r for r in results if r['success']])
        failed_requests = len([r for r in results if not r['success']])
        total_requests_actual = len(results)
        
        if total_requests_actual == 0:
            return LoadTestResults(
                test_name=scenario_name,
                total_requests=0,
                successful_requests=0,
                failed_requests=0,
                average_response_time=0,
                min_response_time=0,
                max_response_time=0,
                percentile_95=0,
                percentile_99=0,
                throughput=0,
                error_rate=1.0,
                memory_peak=0,
                cpu_peak=0,
                duration=total_time
            )
        
        response_times = [r['response_time'] for r in results if r['success']]
        
        if response_times:
            avg_response_time = statistics.mean(response_times)
            min_response_time = min(response_times)
            max_response_time = max(response_times)
            percentile_95 = np.percentile(response_times, 95)
            percentile_99 = np.percentile(response_times, 99)
        else:
            avg_response_time = min_response_time = max_response_time = 0
            percentile_95 = percentile_99 = 0
        
        throughput = total_requests_actual / total_time
        error_rate = failed_requests / total_requests_actual
        
        return LoadTestResults(
            test_name=scenario_name,
            total_requests=total_requests_actual,
            successful_requests=successful_requests,
            failed_requests=failed_requests,
            average_response_time=avg_response_time,
            min_response_time=min_response_time,
            max_response_time=max_response_time,
            percentile_95=percentile_95,
            percentile_99=percentile_99,
            throughput=throughput,
            error_rate=error_rate,
            memory_peak=system_metrics.get('memory_max', 0),
            cpu_peak=system_metrics.get('cpu_max', 0),
            duration=total_time
        )
    
    @pytest.mark.performance
    @pytest.mark.slow
    def test_light_load_scenario(self):
        """Test system under light load."""
        results = self.run_load_test_scenario('light', self.load_scenarios['light'])
        
        # Assertions for light load
        assert results.error_rate < 0.02  # Less than 2% error rate
        assert results.average_response_time < 3.0  # Average response under 3 seconds
        assert results.percentile_95 < 5.0  # 95th percentile under 5 seconds
        assert results.throughput > 2.0  # At least 2 requests per second
        
        self.load_test_results.append(results)
    
    @pytest.mark.performance
    @pytest.mark.slow
    def test_medium_load_scenario(self):
        """Test system under medium load."""
        results = self.run_load_test_scenario('medium', self.load_scenarios['medium'])
        
        # Assertions for medium load
        assert results.error_rate < 0.05  # Less than 5% error rate
        assert results.average_response_time < 5.0  # Average response under 5 seconds
        assert results.percentile_95 < 8.0  # 95th percentile under 8 seconds
        assert results.throughput > 1.0  # At least 1 request per second
        
        self.load_test_results.append(results)
    
    @pytest.mark.performance
    @pytest.mark.slow
    def test_heavy_load_scenario(self):
        """Test system under heavy load."""
        results = self.run_load_test_scenario('heavy', self.load_scenarios['heavy'])
        
        # Assertions for heavy load (more lenient)
        assert results.error_rate < 0.10  # Less than 10% error rate
        assert results.average_response_time < 8.0  # Average response under 8 seconds
        assert results.percentile_95 < 15.0  # 95th percentile under 15 seconds
        assert results.throughput > 0.5  # At least 0.5 requests per second
        
        self.load_test_results.append(results)
    
    @pytest.mark.performance
    @pytest.mark.slow
    def test_stress_scenario(self):
        """Test system under stress conditions."""
        results = self.run_load_test_scenario('stress', self.load_scenarios['stress'])
        
        # Stress test - system should not crash but may have higher error rates
        assert results.error_rate < 0.20  # Less than 20% error rate
        assert results.average_response_time < 15.0  # Average response under 15 seconds
        
        # System should still process some requests
        assert results.successful_requests > 0
        
        self.load_test_results.append(results)
    
    @pytest.mark.performance
    def test_memory_leak_detection(self):
        """Test for memory leaks during extended operation."""
        import gc
        
        process = psutil.Process(os.getpid())
        initial_memory = process.memory_info().rss
        
        documents = self.create_test_documents(100)
        
        # Process documents in batches
        for i in range(0, len(documents), 10):
            batch = documents[i:i+10]
            
            for doc in batch:
                try:
                    # Process document
                    classification = self.classifier.classify(doc)
                    ocr_result = self.ocr_service.extract_text(doc)
                    extracted_data = self.field_extractor.extract_fields(
                        ocr_result['text'], 
                        classification['document_type']
                    )
                    validation = self.validator.validate(extracted_data)
                    
                    # Clean up references
                    del classification, ocr_result, extracted_data, validation
                    
                except Exception:
                    pass
            
            # Force garbage collection
            gc.collect()
            
            # Check memory usage
            current_memory = process.memory_info().rss
            memory_increase = current_memory - initial_memory
            
            # Memory increase should be reasonable
            assert memory_increase < 200 * 1024 * 1024  # Less than 200MB increase
    
    @pytest.mark.performance
    def test_cpu_usage_under_load(self):
        """Test CPU usage under sustained load."""
        documents = self.create_test_documents(50)
        
        # Start monitoring
        self.monitor.start_monitoring(interval=0.1)
        
        # Process documents continuously
        start_time = time.time()
        processed = 0
        
        while time.time() - start_time < 30:  # Run for 30 seconds
            doc = documents[processed % len(documents)]
            
            try:
                classification = self.classifier.classify(doc)
                processed += 1
            except Exception:
                pass
        
        # Stop monitoring and get results
        system_metrics = self.monitor.stop_monitoring()
        
        # CPU usage should be reasonable
        assert system_metrics['cpu_avg'] < 90.0  # Average CPU under 90%
        assert system_metrics['cpu_max'] < 100.0  # Max CPU under 100%
        
        # Should have processed some documents
        assert processed > 10
    
    def generate_performance_report(self) -> str:
        """Generate comprehensive performance report."""
        report_file = Path(__file__).parent.parent / "reports" / "performance_report.json"
        report_file.parent.mkdir(parents=True, exist_ok=True)
        
        report_data = {
            'timestamp': datetime.now().isoformat(),
            'performance_metrics': [
                {
                    'operation': m.operation,
                    'execution_time': m.execution_time,
                    'memory_usage': m.memory_usage,
                    'cpu_usage': m.cpu_usage,
                    'success': m.success,
                    'error_message': m.error_message
                }
                for m in self.performance_results
            ],
            'load_test_results': [
                {
                    'test_name': r.test_name,
                    'total_requests': r.total_requests,
                    'successful_requests': r.successful_requests,
                    'failed_requests': r.failed_requests,
                    'average_response_time': r.average_response_time,
                    'min_response_time': r.min_response_time,
                    'max_response_time': r.max_response_time,
                    'percentile_95': r.percentile_95,
                    'percentile_99': r.percentile_99,
                    'throughput': r.throughput,
                    'error_rate': r.error_rate,
                    'memory_peak': r.memory_peak,
                    'cpu_peak': r.cpu_peak,
                    'duration': r.duration
                }
                for r in self.load_test_results
            ]
        }
        
        with open(report_file, 'w') as f:
            json.dump(report_data, f, indent=2)
        
        return str(report_file)
    
    def teardown_method(self):
        """Cleanup after each test method."""
        # Stop monitoring if still running
        if self.monitor.monitoring:
            self.monitor.stop_monitoring()
        
        # Generate performance report
        if self.performance_results or self.load_test_results:
            report_file = self.generate_performance_report()
            print(f"\nPerformance report saved to: {report_file}")


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-m", "performance"])