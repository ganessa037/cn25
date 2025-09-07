#!/usr/bin/env python3
"""
Integration Test Runner for Document Parser

Comprehensive integration test runner that executes end-to-end tests,
API tests, and performance tests with detailed reporting.
"""

import os
import sys
import time
import json
import logging
import argparse
import subprocess
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Any, Optional

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

try:
    import pytest
    import psutil
except ImportError as e:
    print(f"Required testing dependencies not installed: {e}")
    print("Please install: pip install pytest psutil")
    sys.exit(1)


class IntegrationTestRunner:
    """Comprehensive integration test runner with reporting."""
    
    def __init__(self, test_dir: str = None, output_dir: str = None):
        self.test_dir = Path(test_dir) if test_dir else Path(__file__).parent
        self.output_dir = Path(output_dir) if output_dir else self.test_dir.parent / "reports"
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Setup logging
        self.setup_logging()
        
        # Test configuration
        self.test_config = {
            'timeout': 300,  # 5 minutes per test
            'retry_count': 2,
            'parallel_workers': 4,
            'performance_threshold': {
                'response_time': 10.0,  # seconds
                'memory_usage': 1024 * 1024 * 1024,  # 1GB
                'error_rate': 0.05  # 5%
            }
        }
        
        # Test suites
        self.test_suites = {
            'end_to_end': {
                'file': 'test_end_to_end_pipeline.py',
                'markers': ['integration'],
                'timeout': 600,  # 10 minutes
                'description': 'End-to-end pipeline integration tests'
            },
            'api': {
                'file': 'test_api_endpoints.py',
                'markers': ['integration', 'api'],
                'timeout': 300,  # 5 minutes
                'description': 'API endpoint integration tests'
            },
            'performance': {
                'file': 'test_performance_load.py',
                'markers': ['performance'],
                'timeout': 1200,  # 20 minutes
                'description': 'Performance and load tests'
            }
        }
        
        # Results storage
        self.test_results = {
            'summary': {},
            'suites': {},
            'performance': {},
            'errors': [],
            'warnings': []
        }
    
    def setup_logging(self):
        """Setup logging configuration."""
        log_file = self.output_dir / f"integration_tests_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
        
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(log_file),
                logging.StreamHandler(sys.stdout)
            ]
        )
        
        self.logger = logging.getLogger(__name__)
    
    def check_prerequisites(self) -> bool:
        """Check if all prerequisites are met for integration testing."""
        self.logger.info("Checking integration test prerequisites...")
        
        prerequisites = {
            'python_version': sys.version_info >= (3, 8),
            'memory_available': psutil.virtual_memory().available > 2 * 1024 * 1024 * 1024,  # 2GB
            'disk_space': psutil.disk_usage('.').free > 1 * 1024 * 1024 * 1024,  # 1GB
            'test_files_exist': all(
                (self.test_dir / suite['file']).exists() 
                for suite in self.test_suites.values()
            )
        }
        
        all_good = True
        for check, result in prerequisites.items():
            if result:
                self.logger.info(f"✓ {check}: OK")
            else:
                self.logger.error(f"✗ {check}: FAILED")
                all_good = False
        
        if not all_good:
            self.logger.error("Prerequisites not met. Please resolve issues before running tests.")
        
        return all_good
    
    def run_test_suite(self, suite_name: str, suite_config: Dict[str, Any], 
                      verbose: bool = True, markers: List[str] = None) -> Dict[str, Any]:
        """Run a specific test suite."""
        self.logger.info(f"Running {suite_name} test suite...")
        
        test_file = self.test_dir / suite_config['file']
        if not test_file.exists():
            error_msg = f"Test file not found: {test_file}"
            self.logger.error(error_msg)
            return {
                'success': False,
                'error': error_msg,
                'execution_time': 0,
                'test_count': 0
            }
        
        # Prepare pytest arguments
        pytest_args = [
            str(test_file),
            "-v" if verbose else "-q",
            "--tb=short",
            "--strict-markers",
            "--disable-warnings",
            f"--timeout={suite_config.get('timeout', self.test_config['timeout'])}",
            f"--junitxml={self.output_dir}/{suite_name}_junit.xml",
            f"--html={self.output_dir}/{suite_name}_report.html",
            "--self-contained-html"
        ]
        
        # Add markers
        test_markers = markers or suite_config.get('markers', [])
        if test_markers:
            pytest_args.extend(["-m", " and ".join(test_markers)])
        
        # Add parallel execution for non-performance tests
        if suite_name != 'performance' and self.test_config['parallel_workers'] > 1:
            pytest_args.extend(["-n", str(self.test_config['parallel_workers'])])
        
        # Capture start time and system state
        start_time = time.time()
        process = psutil.Process(os.getpid())
        start_memory = process.memory_info().rss
        
        try:
            # Run pytest
            exit_code = pytest.main(pytest_args)
            
            # Capture end time and system state
            end_time = time.time()
            end_memory = process.memory_info().rss
            
            execution_time = end_time - start_time
            memory_usage = end_memory - start_memory
            
            # Parse JUnit XML for test count
            junit_file = self.output_dir / f"{suite_name}_junit.xml"
            test_count = self.parse_junit_results(junit_file)
            
            success = exit_code == 0
            
            result = {
                'success': success,
                'exit_code': exit_code,
                'execution_time': execution_time,
                'memory_usage': memory_usage,
                'test_count': test_count,
                'junit_file': str(junit_file),
                'html_report': str(self.output_dir / f"{suite_name}_report.html")
            }
            
            if success:
                self.logger.info(f"✓ {suite_name} suite completed successfully in {execution_time:.2f}s")
            else:
                self.logger.warning(f"✗ {suite_name} suite failed with exit code {exit_code}")
            
            return result
            
        except Exception as e:
            error_msg = f"Error running {suite_name} suite: {e}"
            self.logger.error(error_msg)
            return {
                'success': False,
                'error': error_msg,
                'execution_time': time.time() - start_time,
                'test_count': 0
            }
    
    def parse_junit_results(self, junit_file: Path) -> Dict[str, int]:
        """Parse JUnit XML results to extract test statistics."""
        if not junit_file.exists():
            return {'total': 0, 'passed': 0, 'failed': 0, 'skipped': 0}
        
        try:
            import xml.etree.ElementTree as ET
            tree = ET.parse(junit_file)
            root = tree.getroot()
            
            # Extract test statistics
            testsuite = root.find('testsuite')
            if testsuite is not None:
                return {
                    'total': int(testsuite.get('tests', 0)),
                    'passed': int(testsuite.get('tests', 0)) - int(testsuite.get('failures', 0)) - int(testsuite.get('errors', 0)) - int(testsuite.get('skipped', 0)),
                    'failed': int(testsuite.get('failures', 0)) + int(testsuite.get('errors', 0)),
                    'skipped': int(testsuite.get('skipped', 0))
                }
        except Exception as e:
            self.logger.warning(f"Could not parse JUnit results: {e}")
        
        return {'total': 0, 'passed': 0, 'failed': 0, 'skipped': 0}
    
    def run_smoke_tests(self) -> bool:
        """Run quick smoke tests to verify basic functionality."""
        self.logger.info("Running smoke tests...")
        
        smoke_args = [
            str(self.test_dir),
            "-v",
            "-m", "not slow and not performance",
            "--tb=short",
            "--maxfail=5",
            "--timeout=60",
            "-x"  # Stop on first failure
        ]
        
        try:
            exit_code = pytest.main(smoke_args)
            success = exit_code == 0
            
            if success:
                self.logger.info("✓ Smoke tests passed")
            else:
                self.logger.error("✗ Smoke tests failed")
            
            return success
            
        except Exception as e:
            self.logger.error(f"Smoke tests failed with error: {e}")
            return False
    
    def run_all_suites(self, include_performance: bool = True, 
                      verbose: bool = True, smoke_first: bool = True) -> Dict[str, Any]:
        """Run all integration test suites."""
        self.logger.info("Starting comprehensive integration test suite...")
        
        # Check prerequisites
        if not self.check_prerequisites():
            return {
                'success': False,
                'error': 'Prerequisites not met',
                'suites': {}
            }
        
        # Run smoke tests first if requested
        if smoke_first:
            if not self.run_smoke_tests():
                self.logger.warning("Smoke tests failed, but continuing with full suite...")
                self.test_results['warnings'].append("Smoke tests failed")
        
        # Run each test suite
        suite_results = {}
        total_start_time = time.time()
        
        for suite_name, suite_config in self.test_suites.items():
            # Skip performance tests if not requested
            if not include_performance and suite_name == 'performance':
                self.logger.info(f"Skipping {suite_name} suite (performance tests disabled)")
                continue
            
            self.logger.info(f"\n{'='*60}")
            self.logger.info(f"RUNNING {suite_name.upper()} TEST SUITE")
            self.logger.info(f"Description: {suite_config['description']}")
            self.logger.info(f"{'='*60}")
            
            suite_result = self.run_test_suite(suite_name, suite_config, verbose)
            suite_results[suite_name] = suite_result
            
            # Log suite summary
            if suite_result['success']:
                self.logger.info(f"✓ {suite_name} suite: PASSED ({suite_result['test_count']} tests)")
            else:
                self.logger.error(f"✗ {suite_name} suite: FAILED")
                if 'error' in suite_result:
                    self.test_results['errors'].append(f"{suite_name}: {suite_result['error']}")
        
        # Calculate overall results
        total_execution_time = time.time() - total_start_time
        successful_suites = [name for name, result in suite_results.items() if result['success']]
        failed_suites = [name for name, result in suite_results.items() if not result['success']]
        
        overall_success = len(failed_suites) == 0
        
        # Store results
        self.test_results['summary'] = {
            'overall_success': overall_success,
            'total_execution_time': total_execution_time,
            'successful_suites': successful_suites,
            'failed_suites': failed_suites,
            'total_suites': len(suite_results),
            'timestamp': datetime.now().isoformat()
        }
        
        self.test_results['suites'] = suite_results
        
        # Generate comprehensive report
        report_file = self.generate_comprehensive_report()
        self.test_results['report_file'] = report_file
        
        # Log final summary
        self.log_final_summary()
        
        return self.test_results
    
    def generate_comprehensive_report(self) -> str:
        """Generate comprehensive integration test report."""
        self.logger.info("Generating comprehensive test report...")
        
        # JSON report
        json_report_file = self.output_dir / "integration_test_report.json"
        with open(json_report_file, 'w') as f:
            json.dump(self.test_results, f, indent=2, default=str)
        
        # HTML summary report
        html_report_file = self.output_dir / "integration_test_summary.html"
        self.generate_html_summary(html_report_file)
        
        # Text summary report
        text_report_file = self.output_dir / "integration_test_summary.txt"
        self.generate_text_summary(text_report_file)
        
        self.logger.info(f"Reports generated:")
        self.logger.info(f"  JSON: {json_report_file}")
        self.logger.info(f"  HTML: {html_report_file}")
        self.logger.info(f"  Text: {text_report_file}")
        
        return str(json_report_file)
    
    def generate_html_summary(self, html_file: Path):
        """Generate HTML summary report."""
        summary = self.test_results['summary']
        suites = self.test_results['suites']
        
        html_content = f"""
<!DOCTYPE html>
<html>
<head>
    <title>Integration Test Report</title>
    <style>
        body {{ font-family: Arial, sans-serif; margin: 20px; }}
        .header {{ background-color: #f0f0f0; padding: 20px; border-radius: 5px; }}
        .success {{ color: green; }}
        .failure {{ color: red; }}
        .warning {{ color: orange; }}
        table {{ border-collapse: collapse; width: 100%; margin: 20px 0; }}
        th, td {{ border: 1px solid #ddd; padding: 8px; text-align: left; }}
        th {{ background-color: #f2f2f2; }}
        .suite-success {{ background-color: #d4edda; }}
        .suite-failure {{ background-color: #f8d7da; }}
    </style>
</head>
<body>
    <div class="header">
        <h1>Document Parser Integration Test Report</h1>
        <p><strong>Generated:</strong> {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
        <p><strong>Overall Status:</strong> 
            <span class="{'success' if summary['overall_success'] else 'failure'}">
                {'PASSED' if summary['overall_success'] else 'FAILED'}
            </span>
        </p>
        <p><strong>Total Execution Time:</strong> {summary['total_execution_time']:.2f} seconds</p>
    </div>
    
    <h2>Test Suite Results</h2>
    <table>
        <tr>
            <th>Suite</th>
            <th>Status</th>
            <th>Tests</th>
            <th>Execution Time</th>
            <th>Memory Usage</th>
            <th>Reports</th>
        </tr>
"""
        
        for suite_name, suite_result in suites.items():
            status_class = 'suite-success' if suite_result['success'] else 'suite-failure'
            status_text = 'PASSED' if suite_result['success'] else 'FAILED'
            
            test_count = suite_result.get('test_count', {})
            if isinstance(test_count, dict):
                test_summary = f"{test_count.get('total', 0)} total, {test_count.get('passed', 0)} passed, {test_count.get('failed', 0)} failed"
            else:
                test_summary = str(test_count)
            
            memory_mb = suite_result.get('memory_usage', 0) / (1024 * 1024)
            
            html_content += f"""
        <tr class="{status_class}">
            <td>{suite_name}</td>
            <td>{status_text}</td>
            <td>{test_summary}</td>
            <td>{suite_result.get('execution_time', 0):.2f}s</td>
            <td>{memory_mb:.1f} MB</td>
            <td>
                <a href="{suite_name}_report.html">HTML</a> | 
                <a href="{suite_name}_junit.xml">JUnit</a>
            </td>
        </tr>
"""
        
        html_content += """
    </table>
    
    <h2>System Information</h2>
    <ul>
        <li><strong>Python Version:</strong> {}</li>
        <li><strong>Platform:</strong> {}</li>
        <li><strong>CPU Count:</strong> {}</li>
        <li><strong>Total Memory:</strong> {:.1f} GB</li>
    </ul>
</body>
</html>
""".format(
            sys.version,
            sys.platform,
            psutil.cpu_count(),
            psutil.virtual_memory().total / (1024**3)
        )
        
        with open(html_file, 'w') as f:
            f.write(html_content)
    
    def generate_text_summary(self, text_file: Path):
        """Generate text summary report."""
        summary = self.test_results['summary']
        suites = self.test_results['suites']
        
        with open(text_file, 'w') as f:
            f.write("DOCUMENT PARSER INTEGRATION TEST SUMMARY\n")
            f.write("=" * 50 + "\n\n")
            
            f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"Overall Status: {'PASSED' if summary['overall_success'] else 'FAILED'}\n")
            f.write(f"Total Execution Time: {summary['total_execution_time']:.2f} seconds\n")
            f.write(f"Successful Suites: {len(summary['successful_suites'])}/{summary['total_suites']}\n\n")
            
            f.write("TEST SUITE RESULTS\n")
            f.write("-" * 30 + "\n")
            
            for suite_name, suite_result in suites.items():
                status = "PASSED" if suite_result['success'] else "FAILED"
                f.write(f"{suite_name}: {status} ({suite_result.get('execution_time', 0):.2f}s)\n")
                
                test_count = suite_result.get('test_count', {})
                if isinstance(test_count, dict):
                    f.write(f"  Tests: {test_count.get('total', 0)} total, {test_count.get('passed', 0)} passed, {test_count.get('failed', 0)} failed\n")
                
                if 'error' in suite_result:
                    f.write(f"  Error: {suite_result['error']}\n")
                
                f.write("\n")
            
            if self.test_results['errors']:
                f.write("ERRORS\n")
                f.write("-" * 10 + "\n")
                for error in self.test_results['errors']:
                    f.write(f"- {error}\n")
                f.write("\n")
            
            if self.test_results['warnings']:
                f.write("WARNINGS\n")
                f.write("-" * 10 + "\n")
                for warning in self.test_results['warnings']:
                    f.write(f"- {warning}\n")
    
    def log_final_summary(self):
        """Log final test summary."""
        summary = self.test_results['summary']
        
        self.logger.info("\n" + "=" * 60)
        self.logger.info("INTEGRATION TEST SUITE COMPLETED")
        self.logger.info("=" * 60)
        
        if summary['overall_success']:
            self.logger.info("\n✓ ALL INTEGRATION TESTS PASSED SUCCESSFULLY!")
        else:
            self.logger.error("\n✗ SOME INTEGRATION TESTS FAILED")
            for failed_suite in summary['failed_suites']:
                self.logger.error(f"  - {failed_suite} suite failed")
        
        self.logger.info(f"\nExecution Summary:")
        self.logger.info(f"  Total time: {summary['total_execution_time']:.2f} seconds")
        self.logger.info(f"  Successful suites: {len(summary['successful_suites'])}/{summary['total_suites']}")
        
        if self.test_results['errors']:
            self.logger.error(f"  Errors: {len(self.test_results['errors'])}")
        
        if self.test_results['warnings']:
            self.logger.warning(f"  Warnings: {len(self.test_results['warnings'])}")
        
        self.logger.info("=" * 60)


def main():
    """Main entry point for the integration test runner."""
    parser = argparse.ArgumentParser(description="Document Parser Integration Test Runner")
    parser.add_argument("--suite", help="Run specific test suite (end_to_end, api, performance)")
    parser.add_argument("--no-performance", action="store_true", help="Skip performance tests")
    parser.add_argument("--no-smoke", action="store_true", help="Skip smoke tests")
    parser.add_argument("--quiet", action="store_true", help="Run in quiet mode")
    parser.add_argument("--output-dir", help="Output directory for reports")
    parser.add_argument("--markers", help="Additional pytest markers to include")
    
    args = parser.parse_args()
    
    # Create test runner
    runner = IntegrationTestRunner(output_dir=args.output_dir)
    
    try:
        if args.suite:
            # Run specific suite
            if args.suite not in runner.test_suites:
                print(f"Unknown test suite: {args.suite}")
                print(f"Available suites: {', '.join(runner.test_suites.keys())}")
                sys.exit(1)
            
            suite_config = runner.test_suites[args.suite]
            markers = args.markers.split(',') if args.markers else None
            
            result = runner.run_test_suite(args.suite, suite_config, 
                                         verbose=not args.quiet, markers=markers)
            
            exit_code = 0 if result['success'] else 1
        else:
            # Run all suites
            results = runner.run_all_suites(
                include_performance=not args.no_performance,
                verbose=not args.quiet,
                smoke_first=not args.no_smoke
            )
            
            exit_code = 0 if results['summary']['overall_success'] else 1
        
        sys.exit(exit_code)
        
    except Exception as e:
        print(f"Integration test runner failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()