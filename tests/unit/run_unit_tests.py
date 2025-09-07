#!/usr/bin/env python3
"""
Unit Test Runner for Document Parser

Comprehensive test runner that executes all unit tests with coverage reporting,
performance metrics, and detailed reporting following the autocorrect model's patterns.
"""

import os
import sys
import time
import json
import logging
import argparse
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Any, Optional
import subprocess

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

try:
    import pytest
    import coverage
    import psutil
except ImportError as e:
    print(f"Required testing dependencies not installed: {e}")
    print("Please install: pip install pytest coverage psutil")
    sys.exit(1)


class UnitTestRunner:
    """Comprehensive unit test runner with reporting."""
    
    def __init__(self, test_dir: str = None, output_dir: str = None):
        self.test_dir = Path(test_dir) if test_dir else Path(__file__).parent
        self.output_dir = Path(output_dir) if output_dir else self.test_dir.parent / "reports"
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Setup logging
        self.setup_logging()
        
        # Test configuration
        self.coverage_threshold = 80.0
        self.performance_threshold = 10.0  # seconds per test
        self.memory_threshold = 100 * 1024 * 1024  # 100MB
        
        # Results storage
        self.test_results = {
            'summary': {},
            'coverage': {},
            'performance': {},
            'failures': [],
            'errors': []
        }
    
    def setup_logging(self):
        """Setup logging configuration."""
        log_file = self.output_dir / f"unit_tests_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
        
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(log_file),
                logging.StreamHandler(sys.stdout)
            ]
        )
        
        self.logger = logging.getLogger(__name__)
    
    def discover_test_files(self) -> List[Path]:
        """Discover all test files in the test directory."""
        test_files = []
        
        # Find all test_*.py files
        for test_file in self.test_dir.glob("test_*.py"):
            if test_file.name != "test_config.py":  # Skip config file
                test_files.append(test_file)
        
        self.logger.info(f"Discovered {len(test_files)} test files: {[f.name for f in test_files]}")
        return test_files
    
    def run_tests_with_coverage(self, test_files: List[Path] = None, 
                              verbose: bool = True) -> Dict[str, Any]:
        """Run tests with coverage measurement."""
        self.logger.info("Starting unit tests with coverage measurement...")
        
        # Initialize coverage
        cov = coverage.Coverage(
            source=[str(project_root / "src" / "document_parser")],
            omit=[
                "*/tests/*",
                "*/test_*",
                "*/__pycache__/*",
                "*/venv/*",
                "*/env/*"
            ]
        )
        
        cov.start()
        
        try:
            # Run pytest
            pytest_args = [
                "-v" if verbose else "-q",
                "--tb=short",
                "--strict-markers",
                "--disable-warnings",
                f"--junitxml={self.output_dir}/junit_results.xml",
                f"--html={self.output_dir}/test_report.html",
                "--self-contained-html"
            ]
            
            if test_files:
                pytest_args.extend([str(f) for f in test_files])
            else:
                pytest_args.append(str(self.test_dir))
            
            # Capture start time and memory
            start_time = time.time()
            process = psutil.Process(os.getpid())
            start_memory = process.memory_info().rss
            
            # Run tests
            exit_code = pytest.main(pytest_args)
            
            # Capture end time and memory
            end_time = time.time()
            end_memory = process.memory_info().rss
            
            # Stop coverage and generate report
            cov.stop()
            cov.save()
            
            # Generate coverage reports
            coverage_report = self.generate_coverage_report(cov)
            
            # Calculate performance metrics
            execution_time = end_time - start_time
            memory_usage = end_memory - start_memory
            
            # Store results
            self.test_results['summary'] = {
                'exit_code': exit_code,
                'execution_time': execution_time,
                'memory_usage': memory_usage,
                'timestamp': datetime.now().isoformat()
            }
            
            self.test_results['coverage'] = coverage_report
            self.test_results['performance'] = {
                'execution_time': execution_time,
                'memory_usage': memory_usage,
                'performance_threshold_met': execution_time <= self.performance_threshold,
                'memory_threshold_met': memory_usage <= self.memory_threshold
            }
            
            self.logger.info(f"Tests completed in {execution_time:.2f} seconds")
            self.logger.info(f"Memory usage: {memory_usage / 1024 / 1024:.2f} MB")
            
            return self.test_results
            
        except Exception as e:
            self.logger.error(f"Error running tests: {e}")
            cov.stop()
            raise
    
    def generate_coverage_report(self, cov: coverage.Coverage) -> Dict[str, Any]:
        """Generate detailed coverage report."""
        self.logger.info("Generating coverage report...")
        
        # Generate text report
        coverage_file = self.output_dir / "coverage.txt"
        with open(coverage_file, 'w') as f:
            cov.report(file=f, show_missing=True)
        
        # Generate HTML report
        html_dir = self.output_dir / "coverage_html"
        cov.html_report(directory=str(html_dir))
        
        # Generate XML report
        xml_file = self.output_dir / "coverage.xml"
        cov.xml_report(outfile=str(xml_file))
        
        # Get coverage statistics
        coverage_data = cov.get_data()
        total_coverage = cov.report(file=None)
        
        # Analyze coverage by module
        module_coverage = {}
        for filename in coverage_data.measured_files():
            if "src/document_parser" in filename:
                module_name = Path(filename).stem
                analysis = cov._analyze(filename)
                if analysis:
                    executed = len(analysis.executed)
                    missing = len(analysis.missing)
                    total_lines = executed + missing
                    coverage_percent = (executed / total_lines * 100) if total_lines > 0 else 0
                    
                    module_coverage[module_name] = {
                        'executed': executed,
                        'missing': missing,
                        'total': total_lines,
                        'coverage': coverage_percent
                    }
        
        coverage_report = {
            'total_coverage': total_coverage,
            'threshold_met': total_coverage >= self.coverage_threshold,
            'threshold': self.coverage_threshold,
            'module_coverage': module_coverage,
            'reports': {
                'text': str(coverage_file),
                'html': str(html_dir / "index.html"),
                'xml': str(xml_file)
            }
        }
        
        self.logger.info(f"Total coverage: {total_coverage:.2f}%")
        if total_coverage >= self.coverage_threshold:
            self.logger.info(f"✓ Coverage threshold ({self.coverage_threshold}%) met")
        else:
            self.logger.warning(f"✗ Coverage threshold ({self.coverage_threshold}%) not met")
        
        return coverage_report
    
    def run_component_tests(self, component: str) -> Dict[str, Any]:
        """Run tests for a specific component."""
        self.logger.info(f"Running tests for component: {component}")
        
        test_file = self.test_dir / f"test_{component}.py"
        if not test_file.exists():
            raise FileNotFoundError(f"Test file not found: {test_file}")
        
        return self.run_tests_with_coverage([test_file])
    
    def run_performance_tests(self) -> Dict[str, Any]:
        """Run performance-focused tests."""
        self.logger.info("Running performance tests...")
        
        performance_results = {}
        test_files = self.discover_test_files()
        
        for test_file in test_files:
            self.logger.info(f"Performance testing: {test_file.name}")
            
            start_time = time.time()
            process = psutil.Process(os.getpid())
            start_memory = process.memory_info().rss
            
            # Run individual test file
            exit_code = pytest.main([
                str(test_file),
                "-v",
                "--tb=no",
                "-q"
            ])
            
            end_time = time.time()
            end_memory = process.memory_info().rss
            
            execution_time = end_time - start_time
            memory_usage = end_memory - start_memory
            
            performance_results[test_file.stem] = {
                'execution_time': execution_time,
                'memory_usage': memory_usage,
                'exit_code': exit_code,
                'performance_ok': execution_time <= self.performance_threshold,
                'memory_ok': memory_usage <= self.memory_threshold
            }
            
            self.logger.info(f"  Time: {execution_time:.2f}s, Memory: {memory_usage/1024/1024:.2f}MB")
        
        return performance_results
    
    def generate_summary_report(self) -> str:
        """Generate a comprehensive summary report."""
        self.logger.info("Generating summary report...")
        
        report_file = self.output_dir / "test_summary.json"
        
        # Add system information
        self.test_results['system_info'] = {
            'python_version': sys.version,
            'platform': sys.platform,
            'cpu_count': psutil.cpu_count(),
            'memory_total': psutil.virtual_memory().total,
            'timestamp': datetime.now().isoformat()
        }
        
        # Save detailed results
        with open(report_file, 'w') as f:
            json.dump(self.test_results, f, indent=2, default=str)
        
        # Generate human-readable summary
        summary_file = self.output_dir / "test_summary.txt"
        with open(summary_file, 'w') as f:
            f.write("DOCUMENT PARSER UNIT TEST SUMMARY\n")
            f.write("=" * 50 + "\n\n")
            
            # Test execution summary
            summary = self.test_results.get('summary', {})
            f.write(f"Execution Time: {summary.get('execution_time', 0):.2f} seconds\n")
            f.write(f"Memory Usage: {summary.get('memory_usage', 0) / 1024 / 1024:.2f} MB\n")
            f.write(f"Exit Code: {summary.get('exit_code', 'Unknown')}\n\n")
            
            # Coverage summary
            coverage = self.test_results.get('coverage', {})
            f.write(f"Coverage: {coverage.get('total_coverage', 0):.2f}%\n")
            f.write(f"Coverage Threshold: {coverage.get('threshold', 0):.2f}%\n")
            f.write(f"Threshold Met: {'✓' if coverage.get('threshold_met', False) else '✗'}\n\n")
            
            # Module coverage breakdown
            f.write("Module Coverage Breakdown:\n")
            f.write("-" * 30 + "\n")
            module_coverage = coverage.get('module_coverage', {})
            for module, data in module_coverage.items():
                f.write(f"{module}: {data.get('coverage', 0):.2f}% ({data.get('executed', 0)}/{data.get('total', 0)} lines)\n")
            
            f.write("\n")
            
            # Performance summary
            performance = self.test_results.get('performance', {})
            f.write(f"Performance Threshold Met: {'✓' if performance.get('performance_threshold_met', False) else '✗'}\n")
            f.write(f"Memory Threshold Met: {'✓' if performance.get('memory_threshold_met', False) else '✗'}\n")
        
        self.logger.info(f"Summary report saved to: {summary_file}")
        return str(summary_file)
    
    def run_all_tests(self, verbose: bool = True, 
                     include_performance: bool = True) -> Dict[str, Any]:
        """Run all unit tests with comprehensive reporting."""
        self.logger.info("Starting comprehensive unit test suite...")
        
        try:
            # Run main test suite
            results = self.run_tests_with_coverage(verbose=verbose)
            
            # Run performance tests if requested
            if include_performance:
                performance_results = self.run_performance_tests()
                results['detailed_performance'] = performance_results
            
            # Generate summary report
            summary_file = self.generate_summary_report()
            results['summary_report'] = summary_file
            
            # Log final results
            self.log_final_results(results)
            
            return results
            
        except Exception as e:
            self.logger.error(f"Test suite failed: {e}")
            raise
    
    def log_final_results(self, results: Dict[str, Any]):
        """Log final test results."""
        self.logger.info("\n" + "=" * 60)
        self.logger.info("UNIT TEST SUITE COMPLETED")
        self.logger.info("=" * 60)
        
        summary = results.get('summary', {})
        coverage = results.get('coverage', {})
        performance = results.get('performance', {})
        
        self.logger.info(f"Exit Code: {summary.get('exit_code', 'Unknown')}")
        self.logger.info(f"Execution Time: {summary.get('execution_time', 0):.2f} seconds")
        self.logger.info(f"Memory Usage: {summary.get('memory_usage', 0) / 1024 / 1024:.2f} MB")
        self.logger.info(f"Coverage: {coverage.get('total_coverage', 0):.2f}%")
        
        # Overall status
        exit_code = summary.get('exit_code', 1)
        coverage_ok = coverage.get('threshold_met', False)
        performance_ok = performance.get('performance_threshold_met', True)
        memory_ok = performance.get('memory_threshold_met', True)
        
        overall_success = (exit_code == 0 and coverage_ok and performance_ok and memory_ok)
        
        if overall_success:
            self.logger.info("\n✓ ALL TESTS PASSED SUCCESSFULLY!")
        else:
            self.logger.warning("\n✗ SOME TESTS FAILED OR THRESHOLDS NOT MET")
            if exit_code != 0:
                self.logger.warning("  - Test failures detected")
            if not coverage_ok:
                self.logger.warning("  - Coverage threshold not met")
            if not performance_ok:
                self.logger.warning("  - Performance threshold exceeded")
            if not memory_ok:
                self.logger.warning("  - Memory usage threshold exceeded")
        
        self.logger.info("=" * 60)


def main():
    """Main entry point for the test runner."""
    parser = argparse.ArgumentParser(description="Document Parser Unit Test Runner")
    parser.add_argument("--component", help="Run tests for specific component")
    parser.add_argument("--no-performance", action="store_true", help="Skip performance tests")
    parser.add_argument("--quiet", action="store_true", help="Run in quiet mode")
    parser.add_argument("--output-dir", help="Output directory for reports")
    parser.add_argument("--coverage-threshold", type=float, default=80.0, 
                       help="Coverage threshold percentage")
    
    args = parser.parse_args()
    
    # Create test runner
    runner = UnitTestRunner(output_dir=args.output_dir)
    runner.coverage_threshold = args.coverage_threshold
    
    try:
        if args.component:
            # Run tests for specific component
            results = runner.run_component_tests(args.component)
        else:
            # Run all tests
            results = runner.run_all_tests(
                verbose=not args.quiet,
                include_performance=not args.no_performance
            )
        
        # Exit with appropriate code
        exit_code = results.get('summary', {}).get('exit_code', 1)
        sys.exit(exit_code)
        
    except Exception as e:
        print(f"Test runner failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()