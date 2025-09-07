#!/usr/bin/env python3
"""
Comprehensive Test Automation Pipeline

Automated test execution pipeline for the Malaysian document parser project.
Supports unit tests, integration tests, user acceptance tests, and performance testing.
"""

import os
import sys
import json
import time
import argparse
import subprocess
from pathlib import Path
from typing import Dict, List, Any, Optional
from datetime import datetime
from dataclasses import dataclass, asdict

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))


@dataclass
class TestSuiteConfig:
    """Configuration for test suite execution."""
    name: str
    directory: str
    enabled: bool = True
    timeout: int = 300
    parallel_workers: int = 4
    coverage_threshold: float = 0.8
    markers: List[str] = None
    extra_args: List[str] = None


@dataclass
class TestResult:
    """Test execution result."""
    suite_name: str
    status: str  # 'passed', 'failed', 'skipped', 'error'
    duration: float
    tests_run: int
    tests_passed: int
    tests_failed: int
    tests_skipped: int
    coverage_percentage: float
    error_message: Optional[str] = None
    warnings: List[str] = None


@dataclass
class PipelineResult:
    """Overall pipeline execution result."""
    start_time: datetime
    end_time: datetime
    total_duration: float
    overall_status: str
    test_results: List[TestResult]
    summary: Dict[str, Any]
    artifacts: List[str]


class TestAutomationPipeline:
    """Comprehensive test automation pipeline."""
    
    def __init__(self, config_file: Optional[str] = None):
        self.project_root = project_root
        self.tests_dir = self.project_root / 'tests'
        self.reports_dir = self.tests_dir / 'reports'
        self.artifacts_dir = self.tests_dir / 'artifacts'
        
        # Create directories
        self.reports_dir.mkdir(parents=True, exist_ok=True)
        self.artifacts_dir.mkdir(parents=True, exist_ok=True)
        
        # Load configuration
        self.config = self._load_config(config_file)
        
        # Initialize results
        self.pipeline_result = None
        
    def _load_config(self, config_file: Optional[str] = None) -> Dict[str, TestSuiteConfig]:
        """Load test pipeline configuration."""
        default_config = {
            'unit': TestSuiteConfig(
                name='Unit Tests',
                directory='unit',
                enabled=True,
                timeout=180,
                parallel_workers=4,
                coverage_threshold=0.8,
                markers=['unit'],
                extra_args=['--cov=src', '--cov-report=html', '--cov-report=xml']
            ),
            'integration': TestSuiteConfig(
                name='Integration Tests',
                directory='integration',
                enabled=True,
                timeout=600,
                parallel_workers=2,
                coverage_threshold=0.7,
                markers=['integration'],
                extra_args=['--cov=src', '--cov-append']
            ),
            'user_acceptance': TestSuiteConfig(
                name='User Acceptance Tests',
                directory='user_acceptance',
                enabled=True,
                timeout=900,
                parallel_workers=1,
                coverage_threshold=0.6,
                markers=['acceptance'],
                extra_args=['--cov=src', '--cov-append']
            ),
            'performance': TestSuiteConfig(
                name='Performance Tests',
                directory='integration',
                enabled=False,  # Disabled by default
                timeout=1200,
                parallel_workers=1,
                coverage_threshold=0.0,
                markers=['performance'],
                extra_args=['--benchmark-only']
            )
        }
        
        if config_file and Path(config_file).exists():
            try:
                with open(config_file, 'r') as f:
                    custom_config = json.load(f)
                # Merge with default config
                for suite_name, suite_config in custom_config.items():
                    if suite_name in default_config:
                        # Update existing config
                        for key, value in suite_config.items():
                            setattr(default_config[suite_name], key, value)
                    else:
                        # Add new config
                        default_config[suite_name] = TestSuiteConfig(**suite_config)
            except Exception as e:
                print(f"Warning: Failed to load config file {config_file}: {e}")
                print("Using default configuration.")
        
        return default_config
    
    def _check_prerequisites(self) -> bool:
        """Check if all prerequisites are met."""
        print("Checking prerequisites...")
        
        # Check if pytest is installed
        try:
            result = subprocess.run(['pytest', '--version'], 
                                  capture_output=True, text=True, timeout=10)
            if result.returncode != 0:
                print("Error: pytest is not installed or not accessible")
                return False
        except (subprocess.TimeoutExpired, FileNotFoundError):
            print("Error: pytest is not installed or not accessible")
            return False
        
        # Check if test directories exist
        for suite_config in self.config.values():
            if suite_config.enabled:
                test_dir = self.tests_dir / suite_config.directory
                if not test_dir.exists():
                    print(f"Warning: Test directory {test_dir} does not exist")
        
        # Check if source code exists
        src_dir = self.project_root / 'src'
        if not src_dir.exists():
            print(f"Warning: Source directory {src_dir} does not exist")
        
        print("Prerequisites check completed.")
        return True
    
    def _run_test_suite(self, suite_name: str, suite_config: TestSuiteConfig) -> TestResult:
        """Run a specific test suite."""
        print(f"\nRunning {suite_config.name}...")
        
        if not suite_config.enabled:
            print(f"Skipping {suite_config.name} (disabled)")
            return TestResult(
                suite_name=suite_name,
                status='skipped',
                duration=0.0,
                tests_run=0,
                tests_passed=0,
                tests_failed=0,
                tests_skipped=0,
                coverage_percentage=0.0
            )
        
        test_dir = self.tests_dir / suite_config.directory
        if not test_dir.exists():
            print(f"Test directory {test_dir} does not exist")
            return TestResult(
                suite_name=suite_name,
                status='error',
                duration=0.0,
                tests_run=0,
                tests_passed=0,
                tests_failed=0,
                tests_skipped=0,
                coverage_percentage=0.0,
                error_message=f"Test directory {test_dir} does not exist"
            )
        
        # Prepare pytest command
        cmd = ['pytest', str(test_dir)]
        
        # Add markers
        if suite_config.markers:
            for marker in suite_config.markers:
                cmd.extend(['-m', marker])
        
        # Add parallel execution
        if suite_config.parallel_workers > 1:
            cmd.extend(['-n', str(suite_config.parallel_workers)])
        
        # Add output format
        report_file = self.reports_dir / f"{suite_name}_report.xml"
        cmd.extend(['--junitxml', str(report_file)])
        
        # Add verbose output
        cmd.extend(['-v', '--tb=short'])
        
        # Add extra arguments
        if suite_config.extra_args:
            cmd.extend(suite_config.extra_args)
        
        # Set environment variables
        env = os.environ.copy()
        env['TESTING'] = 'true'
        env['TEST_SUITE'] = suite_name
        
        start_time = time.time()
        
        try:
            # Run pytest
            result = subprocess.run(
                cmd,
                cwd=self.project_root,
                capture_output=True,
                text=True,
                timeout=suite_config.timeout,
                env=env
            )
            
            duration = time.time() - start_time
            
            # Parse results
            test_result = self._parse_pytest_output(result, suite_name, duration)
            
            # Save output
            output_file = self.artifacts_dir / f"{suite_name}_output.txt"
            with open(output_file, 'w') as f:
                f.write(f"Command: {' '.join(cmd)}\n\n")
                f.write(f"Return code: {result.returncode}\n\n")
                f.write("STDOUT:\n")
                f.write(result.stdout)
                f.write("\n\nSTDERR:\n")
                f.write(result.stderr)
            
            return test_result
            
        except subprocess.TimeoutExpired:
            duration = time.time() - start_time
            print(f"Test suite {suite_name} timed out after {duration:.2f} seconds")
            return TestResult(
                suite_name=suite_name,
                status='error',
                duration=duration,
                tests_run=0,
                tests_passed=0,
                tests_failed=0,
                tests_skipped=0,
                coverage_percentage=0.0,
                error_message=f"Test suite timed out after {suite_config.timeout} seconds"
            )
        
        except Exception as e:
            duration = time.time() - start_time
            print(f"Error running test suite {suite_name}: {e}")
            return TestResult(
                suite_name=suite_name,
                status='error',
                duration=duration,
                tests_run=0,
                tests_passed=0,
                tests_failed=0,
                tests_skipped=0,
                coverage_percentage=0.0,
                error_message=str(e)
            )
    
    def _parse_pytest_output(self, result: subprocess.CompletedProcess, 
                           suite_name: str, duration: float) -> TestResult:
        """Parse pytest output to extract test results."""
        stdout = result.stdout
        stderr = result.stderr
        
        # Initialize counters
        tests_run = 0
        tests_passed = 0
        tests_failed = 0
        tests_skipped = 0
        coverage_percentage = 0.0
        warnings = []
        
        # Parse test counts from output
        lines = stdout.split('\n')
        for line in lines:
            line = line.strip()
            
            # Look for test summary line
            if 'passed' in line or 'failed' in line or 'skipped' in line:
                if '::' not in line:  # Avoid individual test lines
                    # Extract numbers
                    import re
                    passed_match = re.search(r'(\d+) passed', line)
                    failed_match = re.search(r'(\d+) failed', line)
                    skipped_match = re.search(r'(\d+) skipped', line)
                    
                    if passed_match:
                        tests_passed = int(passed_match.group(1))
                    if failed_match:
                        tests_failed = int(failed_match.group(1))
                    if skipped_match:
                        tests_skipped = int(skipped_match.group(1))
            
            # Look for coverage percentage
            if 'TOTAL' in line and '%' in line:
                coverage_match = re.search(r'(\d+)%', line)
                if coverage_match:
                    coverage_percentage = float(coverage_match.group(1)) / 100.0
            
            # Look for warnings
            if 'warning' in line.lower():
                warnings.append(line)
        
        tests_run = tests_passed + tests_failed + tests_skipped
        
        # Determine status
        if result.returncode == 0:
            status = 'passed'
        elif tests_failed > 0:
            status = 'failed'
        else:
            status = 'error'
        
        return TestResult(
            suite_name=suite_name,
            status=status,
            duration=duration,
            tests_run=tests_run,
            tests_passed=tests_passed,
            tests_failed=tests_failed,
            tests_skipped=tests_skipped,
            coverage_percentage=coverage_percentage,
            error_message=stderr if stderr and result.returncode != 0 else None,
            warnings=warnings if warnings else None
        )
    
    def _generate_reports(self, pipeline_result: PipelineResult):
        """Generate comprehensive test reports."""
        print("\nGenerating reports...")
        
        # Generate JSON report
        json_report = self.reports_dir / 'pipeline_report.json'
        with open(json_report, 'w') as f:
            json.dump(asdict(pipeline_result), f, indent=2, default=str)
        
        # Generate HTML report
        html_report = self.reports_dir / 'pipeline_report.html'
        self._generate_html_report(pipeline_result, html_report)
        
        # Generate text summary
        text_report = self.reports_dir / 'pipeline_summary.txt'
        self._generate_text_report(pipeline_result, text_report)
        
        print(f"Reports generated in {self.reports_dir}")
    
    def _generate_html_report(self, pipeline_result: PipelineResult, output_file: Path):
        """Generate HTML test report."""
        html_content = f"""
<!DOCTYPE html>
<html>
<head>
    <title>Test Pipeline Report</title>
    <style>
        body {{ font-family: Arial, sans-serif; margin: 20px; }}
        .header {{ background-color: #f0f0f0; padding: 20px; border-radius: 5px; }}
        .summary {{ margin: 20px 0; }}
        .test-suite {{ margin: 20px 0; border: 1px solid #ddd; border-radius: 5px; }}
        .suite-header {{ background-color: #f8f8f8; padding: 10px; font-weight: bold; }}
        .suite-content {{ padding: 10px; }}
        .passed {{ color: green; }}
        .failed {{ color: red; }}
        .skipped {{ color: orange; }}
        .error {{ color: red; font-weight: bold; }}
        table {{ border-collapse: collapse; width: 100%; }}
        th, td {{ border: 1px solid #ddd; padding: 8px; text-align: left; }}
        th {{ background-color: #f2f2f2; }}
    </style>
</head>
<body>
    <div class="header">
        <h1>Test Pipeline Report</h1>
        <p><strong>Execution Time:</strong> {pipeline_result.start_time} - {pipeline_result.end_time}</p>
        <p><strong>Total Duration:</strong> {pipeline_result.total_duration:.2f} seconds</p>
        <p><strong>Overall Status:</strong> <span class="{pipeline_result.overall_status}">{pipeline_result.overall_status.upper()}</span></p>
    </div>
    
    <div class="summary">
        <h2>Summary</h2>
        <table>
            <tr><th>Metric</th><th>Value</th></tr>
            <tr><td>Total Tests Run</td><td>{pipeline_result.summary['total_tests_run']}</td></tr>
            <tr><td>Tests Passed</td><td class="passed">{pipeline_result.summary['total_tests_passed']}</td></tr>
            <tr><td>Tests Failed</td><td class="failed">{pipeline_result.summary['total_tests_failed']}</td></tr>
            <tr><td>Tests Skipped</td><td class="skipped">{pipeline_result.summary['total_tests_skipped']}</td></tr>
            <tr><td>Average Coverage</td><td>{pipeline_result.summary['average_coverage']:.1%}</td></tr>
        </table>
    </div>
    
    <div class="test-suites">
        <h2>Test Suite Results</h2>
"""
        
        for result in pipeline_result.test_results:
            html_content += f"""
        <div class="test-suite">
            <div class="suite-header">
                {result.suite_name} - <span class="{result.status}">{result.status.upper()}</span>
            </div>
            <div class="suite-content">
                <p><strong>Duration:</strong> {result.duration:.2f} seconds</p>
                <p><strong>Tests:</strong> {result.tests_run} total, 
                   <span class="passed">{result.tests_passed} passed</span>, 
                   <span class="failed">{result.tests_failed} failed</span>, 
                   <span class="skipped">{result.tests_skipped} skipped</span></p>
                <p><strong>Coverage:</strong> {result.coverage_percentage:.1%}</p>
                {f'<p><strong>Error:</strong> {result.error_message}</p>' if result.error_message else ''}
            </div>
        </div>
"""
        
        html_content += """
    </div>
</body>
</html>
"""
        
        with open(output_file, 'w') as f:
            f.write(html_content)
    
    def _generate_text_report(self, pipeline_result: PipelineResult, output_file: Path):
        """Generate text summary report."""
        content = f"""
TEST PIPELINE SUMMARY
{'=' * 50}

Execution Time: {pipeline_result.start_time} - {pipeline_result.end_time}
Total Duration: {pipeline_result.total_duration:.2f} seconds
Overall Status: {pipeline_result.overall_status.upper()}

SUMMARY
{'-' * 20}
Total Tests Run: {pipeline_result.summary['total_tests_run']}
Tests Passed: {pipeline_result.summary['total_tests_passed']}
Tests Failed: {pipeline_result.summary['total_tests_failed']}
Tests Skipped: {pipeline_result.summary['total_tests_skipped']}
Average Coverage: {pipeline_result.summary['average_coverage']:.1%}

TEST SUITE RESULTS
{'-' * 30}
"""
        
        for result in pipeline_result.test_results:
            content += f"""
{result.suite_name}:
  Status: {result.status.upper()}
  Duration: {result.duration:.2f} seconds
  Tests: {result.tests_run} total, {result.tests_passed} passed, {result.tests_failed} failed, {result.tests_skipped} skipped
  Coverage: {result.coverage_percentage:.1%}
"""
            if result.error_message:
                content += f"  Error: {result.error_message}\n"
        
        with open(output_file, 'w') as f:
            f.write(content)
    
    def run_pipeline(self, suites: Optional[List[str]] = None, 
                    skip_prerequisites: bool = False) -> PipelineResult:
        """Run the complete test pipeline."""
        start_time = datetime.now()
        print(f"Starting test pipeline at {start_time}")
        
        # Check prerequisites
        if not skip_prerequisites and not self._check_prerequisites():
            print("Prerequisites check failed. Aborting pipeline.")
            return PipelineResult(
                start_time=start_time,
                end_time=datetime.now(),
                total_duration=0.0,
                overall_status='error',
                test_results=[],
                summary={},
                artifacts=[]
            )
        
        # Determine which suites to run
        if suites:
            suites_to_run = {name: config for name, config in self.config.items() 
                           if name in suites}
        else:
            suites_to_run = {name: config for name, config in self.config.items() 
                           if config.enabled}
        
        # Run test suites
        test_results = []
        for suite_name, suite_config in suites_to_run.items():
            result = self._run_test_suite(suite_name, suite_config)
            test_results.append(result)
        
        end_time = datetime.now()
        total_duration = (end_time - start_time).total_seconds()
        
        # Calculate summary
        total_tests_run = sum(r.tests_run for r in test_results)
        total_tests_passed = sum(r.tests_passed for r in test_results)
        total_tests_failed = sum(r.tests_failed for r in test_results)
        total_tests_skipped = sum(r.tests_skipped for r in test_results)
        
        # Calculate average coverage (excluding zero coverage)
        coverage_values = [r.coverage_percentage for r in test_results if r.coverage_percentage > 0]
        average_coverage = sum(coverage_values) / len(coverage_values) if coverage_values else 0.0
        
        # Determine overall status
        if any(r.status == 'error' for r in test_results):
            overall_status = 'error'
        elif any(r.status == 'failed' for r in test_results):
            overall_status = 'failed'
        elif all(r.status in ['passed', 'skipped'] for r in test_results):
            overall_status = 'passed'
        else:
            overall_status = 'unknown'
        
        # Create pipeline result
        pipeline_result = PipelineResult(
            start_time=start_time,
            end_time=end_time,
            total_duration=total_duration,
            overall_status=overall_status,
            test_results=test_results,
            summary={
                'total_tests_run': total_tests_run,
                'total_tests_passed': total_tests_passed,
                'total_tests_failed': total_tests_failed,
                'total_tests_skipped': total_tests_skipped,
                'average_coverage': average_coverage,
                'suites_run': len(test_results),
                'suites_passed': len([r for r in test_results if r.status == 'passed']),
                'suites_failed': len([r for r in test_results if r.status == 'failed']),
                'suites_error': len([r for r in test_results if r.status == 'error'])
            },
            artifacts=list(self.artifacts_dir.glob('*'))
        )
        
        # Generate reports
        self._generate_reports(pipeline_result)
        
        # Print summary
        self._print_summary(pipeline_result)
        
        self.pipeline_result = pipeline_result
        return pipeline_result
    
    def _print_summary(self, pipeline_result: PipelineResult):
        """Print pipeline summary to console."""
        print(f"\n{'=' * 60}")
        print("TEST PIPELINE SUMMARY")
        print(f"{'=' * 60}")
        print(f"Overall Status: {pipeline_result.overall_status.upper()}")
        print(f"Total Duration: {pipeline_result.total_duration:.2f} seconds")
        print(f"Tests Run: {pipeline_result.summary['total_tests_run']}")
        print(f"Tests Passed: {pipeline_result.summary['total_tests_passed']}")
        print(f"Tests Failed: {pipeline_result.summary['total_tests_failed']}")
        print(f"Tests Skipped: {pipeline_result.summary['total_tests_skipped']}")
        print(f"Average Coverage: {pipeline_result.summary['average_coverage']:.1%}")
        print(f"\nReports available in: {self.reports_dir}")
        print(f"{'=' * 60}")


def main():
    """Main entry point for test pipeline."""
    parser = argparse.ArgumentParser(description='Run comprehensive test pipeline')
    parser.add_argument('--config', '-c', help='Configuration file path')
    parser.add_argument('--suites', '-s', nargs='+', 
                       choices=['unit', 'integration', 'user_acceptance', 'performance'],
                       help='Specific test suites to run')
    parser.add_argument('--skip-prerequisites', action='store_true',
                       help='Skip prerequisites check')
    parser.add_argument('--list-suites', action='store_true',
                       help='List available test suites')
    
    args = parser.parse_args()
    
    # Initialize pipeline
    pipeline = TestAutomationPipeline(args.config)
    
    if args.list_suites:
        print("Available test suites:")
        for name, config in pipeline.config.items():
            status = "enabled" if config.enabled else "disabled"
            print(f"  {name}: {config.name} ({status})")
        return
    
    # Run pipeline
    try:
        result = pipeline.run_pipeline(
            suites=args.suites,
            skip_prerequisites=args.skip_prerequisites
        )
        
        # Exit with appropriate code
        if result.overall_status == 'passed':
            sys.exit(0)
        elif result.overall_status == 'failed':
            sys.exit(1)
        else:
            sys.exit(2)
            
    except KeyboardInterrupt:
        print("\nPipeline interrupted by user")
        sys.exit(130)
    except Exception as e:
        print(f"\nPipeline failed with error: {e}")
        sys.exit(1)


if __name__ == '__main__':
    main()