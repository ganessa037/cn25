#!/usr/bin/env python3
"""
Comprehensive Testing Framework for Hybrid Autocorrect Model

This module provides extensive testing capabilities including:
- Functionality tests
- Performance benchmarks
- Edge case validation
- Comprehensive logging system

Author: AI Assistant
Date: 2024
"""

import os
import sys
import json
import time
import logging
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Tuple, Any, Optional
import pickle
import traceback
from dataclasses import dataclass, asdict
from contextlib import contextmanager

# Add parent directory to path for imports
sys.path.append(str(Path(__file__).parent.parent))

# Import required libraries
try:
    from fuzzywuzzy import fuzz, process
    from jellyfish import levenshtein_distance, jaro_winkler_similarity
    import matplotlib.pyplot as plt
    import seaborn as sns
    from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
    from sklearn.model_selection import cross_val_score
except ImportError as e:
    print(f"Warning: Some dependencies not available: {e}")

@dataclass
class TestResult:
    """Data class for storing test results"""
    test_name: str
    test_type: str  # 'functionality', 'performance', 'edge_case'
    status: str  # 'PASS', 'FAIL', 'ERROR'
    execution_time: float
    timestamp: str
    input_data: Any
    expected_output: Any
    actual_output: Any
    metrics: Dict[str, float]
    error_message: Optional[str] = None
    additional_info: Optional[Dict] = None

class TestLogger:
    """Comprehensive logging system for model testing"""
    
    def __init__(self, log_dir: str = "test_logs"):
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(exist_ok=True)
        
        # Initialize session ID first
        self.session_id = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Setup logging configuration
        self.setup_logging()
        
        # Test results storage
        self.test_results: List[TestResult] = []
        
    def setup_logging(self):
        """Configure logging with multiple handlers"""
        # Create logger
        self.logger = logging.getLogger('ModelTesting')
        self.logger.setLevel(logging.DEBUG)
        
        # Clear existing handlers
        self.logger.handlers.clear()
        
        # File handler for detailed logs
        log_file = self.log_dir / f"test_session_{self.session_id}.log"
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(logging.DEBUG)
        
        # Console handler for important messages
        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.INFO)
        
        # Formatter
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        file_handler.setFormatter(formatter)
        console_handler.setFormatter(formatter)
        
        # Add handlers
        self.logger.addHandler(file_handler)
        self.logger.addHandler(console_handler)
        
    def log_test_start(self, test_name: str, test_type: str, input_data: Any):
        """Log the start of a test"""
        self.logger.info(f"Starting {test_type} test: {test_name}")
        self.logger.debug(f"Input data: {input_data}")
        
    def log_test_result(self, result: TestResult):
        """Log test result and store for analysis"""
        self.test_results.append(result)
        
        if result.status == 'PASS':
            self.logger.info(f"✅ {result.test_name} - PASSED ({result.execution_time:.3f}s)")
        elif result.status == 'FAIL':
            self.logger.warning(f"❌ {result.test_name} - FAILED ({result.execution_time:.3f}s)")
            if result.error_message:
                self.logger.warning(f"   Error: {result.error_message}")
        else:  # ERROR
            self.logger.error(f"💥 {result.test_name} - ERROR ({result.execution_time:.3f}s)")
            if result.error_message:
                self.logger.error(f"   Error: {result.error_message}")
                
        # Log metrics if available
        if result.metrics:
            metrics_str = ", ".join([f"{k}: {v:.3f}" for k, v in result.metrics.items()])
            self.logger.debug(f"   Metrics: {metrics_str}")
    
    def save_results(self):
        """Save all test results to files"""
        # Save as JSON
        results_json = self.log_dir / f"test_results_{self.session_id}.json"
        with open(results_json, 'w') as f:
            json.dump([asdict(result) for result in self.test_results], f, indent=2, default=str)
        
        # Save as CSV for analysis
        results_csv = self.log_dir / f"test_results_{self.session_id}.csv"
        df = pd.DataFrame([asdict(result) for result in self.test_results])
        df.to_csv(results_csv, index=False)
        
        self.logger.info(f"Test results saved to {results_json} and {results_csv}")
        
    def generate_summary_report(self):
        """Generate comprehensive test summary report"""
        if not self.test_results:
            self.logger.warning("No test results to summarize")
            return
            
        # Calculate summary statistics
        total_tests = len(self.test_results)
        passed_tests = sum(1 for r in self.test_results if r.status == 'PASS')
        failed_tests = sum(1 for r in self.test_results if r.status == 'FAIL')
        error_tests = sum(1 for r in self.test_results if r.status == 'ERROR')
        
        avg_execution_time = np.mean([r.execution_time for r in self.test_results])
        total_execution_time = sum(r.execution_time for r in self.test_results)
        
        # Generate report
        report = f"""
🧪 COMPREHENSIVE MODEL TESTING REPORT
{'='*60}
Session ID: {self.session_id}
Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

📊 SUMMARY STATISTICS:
   Total Tests: {total_tests}
   ✅ Passed: {passed_tests} ({passed_tests/total_tests*100:.1f}%)
   ❌ Failed: {failed_tests} ({failed_tests/total_tests*100:.1f}%)
   💥 Errors: {error_tests} ({error_tests/total_tests*100:.1f}%)
   
⏱️ PERFORMANCE METRICS:
   Total Execution Time: {total_execution_time:.3f}s
   Average Test Time: {avg_execution_time:.3f}s
   Fastest Test: {min(r.execution_time for r in self.test_results):.3f}s
   Slowest Test: {max(r.execution_time for r in self.test_results):.3f}s

📋 TEST BREAKDOWN BY TYPE:
"""
        
        # Breakdown by test type
        test_types = {}
        for result in self.test_results:
            if result.test_type not in test_types:
                test_types[result.test_type] = {'total': 0, 'passed': 0, 'failed': 0, 'errors': 0}
            
            test_types[result.test_type]['total'] += 1
            if result.status == 'PASS':
                test_types[result.test_type]['passed'] += 1
            elif result.status == 'FAIL':
                test_types[result.test_type]['failed'] += 1
            else:
                test_types[result.test_type]['errors'] += 1
        
        for test_type, stats in test_types.items():
            report += f"   {test_type.upper()}:\n"
            report += f"     Total: {stats['total']}, Passed: {stats['passed']}, Failed: {stats['failed']}, Errors: {stats['errors']}\n"
        
        # Failed tests details
        failed_results = [r for r in self.test_results if r.status in ['FAIL', 'ERROR']]
        if failed_results:
            report += f"\n❌ FAILED/ERROR TESTS:\n"
            for result in failed_results:
                report += f"   • {result.test_name} ({result.test_type}) - {result.status}\n"
                if result.error_message:
                    report += f"     Error: {result.error_message}\n"
        
        # Save report
        report_file = self.log_dir / f"test_report_{self.session_id}.txt"
        with open(report_file, 'w') as f:
            f.write(report)
        
        print(report)
        self.logger.info(f"Summary report saved to {report_file}")
        
        return report

class ModelTester:
    """Comprehensive testing framework for the autocorrect model"""
    
    def __init__(self, model_path: str = "."):
        self.model_path = Path(model_path)
        self.logger = TestLogger()
        self.test_data = {}
        self.models = {}
        self.current_execution_time = 0.0
        
        # Load test datasets
        self.load_test_data()
        
    def load_test_data(self):
        """Load various test datasets"""
        try:
            # Load vehicle master data
            vehicle_file = self.model_path.parent / "data" / "vehicle_master.csv"
            if vehicle_file.exists():
                self.test_data['vehicle_master'] = pd.read_csv(vehicle_file)
                self.logger.logger.info(f"Loaded vehicle master data: {len(self.test_data['vehicle_master'])} records")
            
            # Create synthetic test cases
            self.create_test_cases()
            
        except Exception as e:
            self.logger.logger.error(f"Error loading test data: {e}")
    
    def create_test_cases(self):
        """Create comprehensive test cases for different scenarios"""
        # Functionality test cases
        self.test_data['functionality'] = [
            {'input': 'toyota', 'expected': 'toyota', 'category': 'exact_match'},
            {'input': 'toyot', 'expected': 'toyota', 'category': 'missing_char'},
            {'input': 'toyotaa', 'expected': 'toyota', 'category': 'extra_char'},
            {'input': 'tyoota', 'expected': 'toyota', 'category': 'transposition'},
            {'input': 'honda', 'expected': 'honda', 'category': 'exact_match'},
            {'input': 'hond', 'expected': 'honda', 'category': 'missing_char'},
            {'input': 'hondaa', 'expected': 'honda', 'category': 'extra_char'},
            {'input': 'civic', 'expected': 'civic', 'category': 'exact_match'},
            {'input': 'civicy', 'expected': 'civic', 'category': 'extra_char'},
            {'input': 'camry', 'expected': 'camry', 'category': 'exact_match'},
        ]
        
        # Performance test cases (larger dataset)
        self.test_data['performance'] = []
        if 'vehicle_master' in self.test_data:
            brands = self.test_data['vehicle_master']['brand'].unique()[:50]
            models = self.test_data['vehicle_master']['model'].unique()[:50]
            
            for brand in brands:
                # Add variations
                self.test_data['performance'].extend([
                    {'input': brand, 'expected': brand, 'category': 'exact'},
                    {'input': brand[:-1], 'expected': brand, 'category': 'truncated'},
                    {'input': brand + 'x', 'expected': brand, 'category': 'extra_char'},
                ])
            
            for model in models:
                if len(model) > 3:  # Only test longer model names
                    self.test_data['performance'].extend([
                        {'input': model, 'expected': model, 'category': 'exact'},
                        {'input': model[:-1], 'expected': model, 'category': 'truncated'},
                    ])
        
        # Edge cases
        self.test_data['edge_cases'] = [
            {'input': '', 'expected': '', 'category': 'empty_string'},
            {'input': ' ', 'expected': ' ', 'category': 'whitespace'},
            {'input': 'xyz123', 'expected': 'xyz123', 'category': 'unknown_brand'},
            {'input': 'a', 'expected': 'a', 'category': 'single_char'},
            {'input': 'TOYOTA', 'expected': 'toyota', 'category': 'uppercase'},
            {'input': 'ToYoTa', 'expected': 'toyota', 'category': 'mixed_case'},
            {'input': '12345', 'expected': '12345', 'category': 'numeric'},
            {'input': 'toyota camry', 'expected': 'toyota camry', 'category': 'multi_word'},
            {'input': 'very_long_unknown_brand_name_that_does_not_exist', 'expected': 'very_long_unknown_brand_name_that_does_not_exist', 'category': 'very_long'},
        ]
        
        self.logger.logger.info(f"Created test cases: {len(self.test_data['functionality'])} functionality, {len(self.test_data['performance'])} performance, {len(self.test_data['edge_cases'])} edge cases")
    
    @contextmanager
    def time_test(self, test_name: str):
        """Context manager for timing tests"""
        start_time = time.time()
        try:
            yield
        finally:
            execution_time = time.time() - start_time
            self.current_execution_time = execution_time
    
    def run_functionality_tests(self, corrector_func):
        """Run comprehensive functionality tests"""
        self.logger.logger.info("🔧 Starting functionality tests...")
        
        for i, test_case in enumerate(self.test_data['functionality']):
            test_name = f"functionality_test_{i+1}_{test_case['category']}"
            
            with self.time_test(test_name):
                try:
                    self.logger.log_test_start(test_name, "functionality", test_case['input'])
                    
                    # Run correction
                    if callable(corrector_func):
                        result = corrector_func(test_case['input'])
                        if isinstance(result, tuple):
                            actual_output, confidence, method = result
                        else:
                            actual_output = result
                            confidence = 1.0
                            method = 'unknown'
                    else:
                        actual_output = test_case['input']  # Fallback
                        confidence = 0.0
                        method = 'fallback'
                    
                    # Determine test status
                    if actual_output.lower() == test_case['expected'].lower():
                        status = 'PASS'
                        error_message = None
                    else:
                        status = 'FAIL'
                        error_message = f"Expected '{test_case['expected']}', got '{actual_output}'"
                    
                    # Create test result
                    test_result = TestResult(
                        test_name=test_name,
                        test_type="functionality",
                        status=status,
                        execution_time=self.current_execution_time,
                        timestamp=datetime.now().isoformat(),
                        input_data=test_case['input'],
                        expected_output=test_case['expected'],
                        actual_output=actual_output,
                        metrics={'confidence': confidence},
                        error_message=error_message,
                        additional_info={'category': test_case['category'], 'method': method}
                    )
                    
                    self.logger.log_test_result(test_result)
                    
                except Exception as e:
                    test_result = TestResult(
                        test_name=test_name,
                        test_type="functionality",
                        status="ERROR",
                        execution_time=self.current_execution_time,
                        timestamp=datetime.now().isoformat(),
                        input_data=test_case['input'],
                        expected_output=test_case['expected'],
                        actual_output=None,
                        metrics={},
                        error_message=str(e),
                        additional_info={'traceback': traceback.format_exc()}
                    )
                    
                    self.logger.log_test_result(test_result)
    
    def run_performance_tests(self, corrector_func):
        """Run performance benchmarking tests"""
        self.logger.logger.info("⚡ Starting performance tests...")
        
        # Batch performance test
        test_name = "batch_performance_test"
        batch_inputs = [tc['input'] for tc in self.test_data['performance'][:100]]  # Limit to 100 for performance
        
        with self.time_test(test_name):
            try:
                self.logger.log_test_start(test_name, "performance", f"{len(batch_inputs)} inputs")
                
                start_time = time.time()
                results = []
                
                for input_text in batch_inputs:
                    if callable(corrector_func):
                        result = corrector_func(input_text)
                        results.append(result)
                
                total_time = time.time() - start_time
                avg_time_per_correction = total_time / len(batch_inputs) if batch_inputs else 0
                throughput = len(batch_inputs) / total_time if total_time > 0 else 0
                
                # Performance metrics
                metrics = {
                    'total_time': total_time,
                    'avg_time_per_correction': avg_time_per_correction,
                    'throughput_per_second': throughput,
                    'batch_size': len(batch_inputs)
                }
                
                # Determine if performance is acceptable (< 10ms per correction)
                status = 'PASS' if avg_time_per_correction < 0.01 else 'FAIL'
                error_message = None if status == 'PASS' else f"Average time {avg_time_per_correction:.4f}s exceeds 10ms threshold"
                
                test_result = TestResult(
                    test_name=test_name,
                    test_type="performance",
                    status=status,
                    execution_time=self.current_execution_time,
                    timestamp=datetime.now().isoformat(),
                    input_data=f"{len(batch_inputs)} test inputs",
                    expected_output="< 10ms per correction",
                    actual_output=f"{avg_time_per_correction:.4f}s per correction",
                    metrics=metrics,
                    error_message=error_message
                )
                
                self.logger.log_test_result(test_result)
                
            except Exception as e:
                test_result = TestResult(
                    test_name=test_name,
                    test_type="performance",
                    status="ERROR",
                    execution_time=self.current_execution_time,
                    timestamp=datetime.now().isoformat(),
                    input_data=f"{len(batch_inputs)} test inputs",
                    expected_output="Performance metrics",
                    actual_output=None,
                    metrics={},
                    error_message=str(e),
                    additional_info={'traceback': traceback.format_exc()}
                )
                
                self.logger.log_test_result(test_result)
    
    def run_edge_case_tests(self, corrector_func):
        """Run edge case tests"""
        self.logger.logger.info("🔍 Starting edge case tests...")
        
        for i, test_case in enumerate(self.test_data['edge_cases']):
            test_name = f"edge_case_test_{i+1}_{test_case['category']}"
            
            with self.time_test(test_name):
                try:
                    self.logger.log_test_start(test_name, "edge_case", test_case['input'])
                    
                    # Run correction
                    if callable(corrector_func):
                        result = corrector_func(test_case['input'])
                        if isinstance(result, tuple):
                            actual_output, confidence, method = result
                        else:
                            actual_output = result
                            confidence = 1.0
                            method = 'unknown'
                    else:
                        actual_output = test_case['input']  # Fallback
                        confidence = 0.0
                        method = 'fallback'
                    
                    # For edge cases, we mainly check that the function doesn't crash
                    # and returns reasonable output
                    status = 'PASS'  # If we get here without exception, it's a pass
                    error_message = None
                    
                    # Additional validation for specific edge cases
                    if test_case['category'] == 'empty_string' and actual_output != '':
                        status = 'FAIL'
                        error_message = f"Empty string should return empty string, got '{actual_output}'"
                    elif test_case['category'] == 'uppercase' and actual_output != test_case['expected']:
                        status = 'FAIL'
                        error_message = f"Case normalization failed: expected '{test_case['expected']}', got '{actual_output}'"
                    
                    test_result = TestResult(
                        test_name=test_name,
                        test_type="edge_case",
                        status=status,
                        execution_time=self.current_execution_time,
                        timestamp=datetime.now().isoformat(),
                        input_data=test_case['input'],
                        expected_output=test_case['expected'],
                        actual_output=actual_output,
                        metrics={'confidence': confidence},
                        error_message=error_message,
                        additional_info={'category': test_case['category'], 'method': method}
                    )
                    
                    self.logger.log_test_result(test_result)
                    
                except Exception as e:
                    test_result = TestResult(
                        test_name=test_name,
                        test_type="edge_case",
                        status="ERROR",
                        execution_time=self.current_execution_time,
                        timestamp=datetime.now().isoformat(),
                        input_data=test_case['input'],
                        expected_output=test_case['expected'],
                        actual_output=None,
                        metrics={},
                        error_message=str(e),
                        additional_info={'traceback': traceback.format_exc()}
                    )
                    
                    self.logger.log_test_result(test_result)
    
    def run_all_tests(self, corrector_func=None):
        """Run all test suites"""
        self.logger.logger.info("🚀 Starting comprehensive model testing...")
        
        if corrector_func is None:
            # Create a dummy corrector for testing the framework
            def dummy_corrector(text):
                return text.lower(), 0.5, 'dummy'
            corrector_func = dummy_corrector
            self.logger.logger.warning("No corrector function provided, using dummy corrector")
        
        # Run all test suites
        self.run_functionality_tests(corrector_func)
        self.run_performance_tests(corrector_func)
        self.run_edge_case_tests(corrector_func)
        
        # Save results and generate report
        self.logger.save_results()
        self.logger.generate_summary_report()
        
        return self.logger.test_results

def main():
    """Main function to run comprehensive testing"""
    print("🧪 Comprehensive Model Testing Framework")
    print("=" * 50)
    
    # Initialize tester
    tester = ModelTester()
    
    # Run tests with dummy corrector (replace with actual model when available)
    results = tester.run_all_tests()
    
    print(f"\n✅ Testing completed! {len(results)} tests executed.")
    print(f"📁 Results saved in: {tester.logger.log_dir}")

if __name__ == "__main__":
    main()