#!/usr/bin/env python3
"""
User Acceptance Testing - Comprehensive Test Runner

Orchestrates all user acceptance testing components including real document testing,
user feedback collection, and accuracy validation for the Malaysian document parser.
"""

import os
import sys
import json
import time
import logging
import argparse
import subprocess
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime, timedelta
from dataclasses import dataclass, field
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading

import pytest
import requests
from flask import Flask

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

# Import UAT components
try:
    from tests.user_acceptance.test_real_documents import RealDocumentTester
    from tests.user_acceptance.test_user_feedback import UserFeedbackCollector
    from tests.user_acceptance.test_accuracy_validation import AccuracyValidationSystem
except ImportError as e:
    print(f"Warning: Could not import UAT components: {e}")
    RealDocumentTester = None
    UserFeedbackCollector = None
    AccuracyValidationSystem = None


@dataclass
class UATConfiguration:
    """User Acceptance Testing configuration."""
    # Test execution settings
    test_data_directory: str = "test_data/real_documents"
    output_directory: str = "uat_results"
    parallel_workers: int = 4
    test_timeout_minutes: int = 60
    
    # Document testing settings
    min_documents_per_type: int = 10
    accuracy_threshold: float = 0.85
    confidence_threshold: float = 0.7
    
    # Feedback collection settings
    feedback_collection_enabled: bool = True
    feedback_web_port: int = 5001
    feedback_session_duration_hours: int = 24
    
    # Validation settings
    validation_database: str = "uat_validation.db"
    require_manual_validation: bool = True
    min_validators_per_document: int = 2
    
    # Reporting settings
    generate_html_report: bool = True
    generate_pdf_report: bool = False
    include_detailed_metrics: bool = True
    
    # API testing settings
    api_base_url: str = "http://localhost:8000"
    api_timeout_seconds: int = 30
    
    # Performance requirements
    max_processing_time_seconds: float = 10.0
    max_memory_usage_mb: float = 512.0
    
    # Quality requirements
    min_image_resolution: Tuple[int, int] = (300, 300)
    supported_formats: List[str] = field(default_factory=lambda: ['jpg', 'jpeg', 'png', 'pdf'])


@dataclass
class UATResults:
    """Comprehensive UAT results."""
    test_session_id: str
    start_time: str
    end_time: Optional[str] = None
    duration_seconds: Optional[float] = None
    
    # Test execution results
    total_tests_run: int = 0
    tests_passed: int = 0
    tests_failed: int = 0
    tests_skipped: int = 0
    
    # Document testing results
    documents_tested: int = 0
    documents_passed: int = 0
    overall_accuracy: float = 0.0
    accuracy_by_document_type: Dict[str, float] = field(default_factory=dict)
    
    # Performance results
    average_processing_time: float = 0.0
    max_processing_time: float = 0.0
    average_memory_usage: float = 0.0
    max_memory_usage: float = 0.0
    
    # Feedback results
    feedback_sessions: int = 0
    user_satisfaction_score: float = 0.0
    feedback_summary: Dict[str, Any] = field(default_factory=dict)
    
    # Validation results
    validation_accuracy: float = 0.0
    field_accuracy_breakdown: Dict[str, float] = field(default_factory=dict)
    common_errors: List[Dict[str, Any]] = field(default_factory=list)
    
    # Quality metrics
    image_quality_scores: Dict[str, float] = field(default_factory=dict)
    edge_case_handling: Dict[str, float] = field(default_factory=dict)
    
    # Recommendations
    improvement_recommendations: List[str] = field(default_factory=list)
    critical_issues: List[str] = field(default_factory=list)
    
    # Test artifacts
    log_files: List[str] = field(default_factory=list)
    report_files: List[str] = field(default_factory=list)
    screenshot_files: List[str] = field(default_factory=list)


class UserAcceptanceTestRunner:
    """Comprehensive User Acceptance Test runner."""
    
    def __init__(self, config: UATConfiguration = None):
        self.config = config or UATConfiguration()
        self.results = UATResults(
            test_session_id=self._generate_session_id(),
            start_time=datetime.now().isoformat()
        )
        
        # Setup logging
        self.logger = self._setup_logging()
        
        # Initialize components
        self.real_document_tester = None
        self.feedback_collector = None
        self.validation_system = None
        self.feedback_app = None
        self.feedback_thread = None
        
        # Create output directory
        self.output_dir = Path(self.config.output_directory)
        self.output_dir.mkdir(exist_ok=True)
        
        # Test data directory
        self.test_data_dir = Path(self.config.test_data_directory)
        
        # Initialize components if available
        self._initialize_components()
    
    def _generate_session_id(self) -> str:
        """Generate unique session ID."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        return f"UAT_{timestamp}"
    
    def _setup_logging(self) -> logging.Logger:
        """Setup logging configuration."""
        log_file = self.output_dir / f"{self.results.test_session_id}.log"
        
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(log_file),
                logging.StreamHandler(sys.stdout)
            ]
        )
        
        logger = logging.getLogger(__name__)
        self.results.log_files.append(str(log_file))
        return logger
    
    def _initialize_components(self):
        """Initialize UAT components."""
        try:
            if RealDocumentTester:
                self.real_document_tester = RealDocumentTester(
                    api_base_url=self.config.api_base_url,
                    timeout_seconds=self.config.api_timeout_seconds
                )
                self.logger.info("Real document tester initialized")
            
            if UserFeedbackCollector:
                feedback_db = self.output_dir / "feedback.db"
                self.feedback_collector = UserFeedbackCollector(
                    database_path=str(feedback_db)
                )
                self.logger.info("User feedback collector initialized")
            
            if AccuracyValidationSystem:
                validation_db = self.output_dir / self.config.validation_database
                self.validation_system = AccuracyValidationSystem(
                    database_path=str(validation_db)
                )
                self.logger.info("Accuracy validation system initialized")
                
        except Exception as e:
            self.logger.error(f"Error initializing components: {e}")
    
    def run_comprehensive_uat(self) -> UATResults:
        """Run comprehensive user acceptance testing."""
        self.logger.info(f"Starting comprehensive UAT session: {self.results.test_session_id}")
        
        try:
            # Phase 1: Prerequisites check
            self.logger.info("Phase 1: Checking prerequisites")
            if not self._check_prerequisites():
                raise RuntimeError("Prerequisites check failed")
            
            # Phase 2: Real document testing
            self.logger.info("Phase 2: Real document testing")
            self._run_real_document_tests()
            
            # Phase 3: Start feedback collection (if enabled)
            if self.config.feedback_collection_enabled:
                self.logger.info("Phase 3: Starting feedback collection")
                self._start_feedback_collection()
            
            # Phase 4: Accuracy validation
            self.logger.info("Phase 4: Accuracy validation")
            self._run_accuracy_validation()
            
            # Phase 5: Performance testing
            self.logger.info("Phase 5: Performance testing")
            self._run_performance_tests()
            
            # Phase 6: Edge case testing
            self.logger.info("Phase 6: Edge case testing")
            self._run_edge_case_tests()
            
            # Phase 7: Generate reports
            self.logger.info("Phase 7: Generating reports")
            self._generate_reports()
            
            # Phase 8: Cleanup
            self.logger.info("Phase 8: Cleanup")
            self._cleanup()
            
            # Finalize results
            self.results.end_time = datetime.now().isoformat()
            start_dt = datetime.fromisoformat(self.results.start_time)
            end_dt = datetime.fromisoformat(self.results.end_time)
            self.results.duration_seconds = (end_dt - start_dt).total_seconds()
            
            self.logger.info(f"UAT session completed successfully in {self.results.duration_seconds:.2f} seconds")
            
        except Exception as e:
            self.logger.error(f"UAT session failed: {e}")
            self.results.critical_issues.append(f"UAT session failure: {str(e)}")
            raise
        
        return self.results
    
    def _check_prerequisites(self) -> bool:
        """Check prerequisites for UAT execution."""
        checks_passed = 0
        total_checks = 0
        
        # Check test data directory
        total_checks += 1
        if self.test_data_dir.exists():
            checks_passed += 1
            self.logger.info(f"✓ Test data directory exists: {self.test_data_dir}")
        else:
            self.logger.warning(f"✗ Test data directory not found: {self.test_data_dir}")
        
        # Check API availability
        total_checks += 1
        try:
            response = requests.get(
                f"{self.config.api_base_url}/health",
                timeout=5
            )
            if response.status_code == 200:
                checks_passed += 1
                self.logger.info("✓ API is accessible")
            else:
                self.logger.warning(f"✗ API returned status {response.status_code}")
        except Exception as e:
            self.logger.warning(f"✗ API not accessible: {e}")
        
        # Check required components
        total_checks += 1
        if all([self.real_document_tester, self.validation_system]):
            checks_passed += 1
            self.logger.info("✓ Required components initialized")
        else:
            self.logger.warning("✗ Some required components not available")
        
        # Check output directory permissions
        total_checks += 1
        try:
            test_file = self.output_dir / "test_write.tmp"
            test_file.write_text("test")
            test_file.unlink()
            checks_passed += 1
            self.logger.info("✓ Output directory is writable")
        except Exception as e:
            self.logger.warning(f"✗ Output directory not writable: {e}")
        
        success_rate = checks_passed / total_checks
        self.logger.info(f"Prerequisites check: {checks_passed}/{total_checks} passed ({success_rate:.1%})")
        
        return success_rate >= 0.75  # Require at least 75% of checks to pass
    
    def _run_real_document_tests(self):
        """Run real document testing."""
        if not self.real_document_tester:
            self.logger.warning("Real document tester not available, skipping")
            return
        
        # Find test documents
        test_documents = self._discover_test_documents()
        
        if not test_documents:
            self.logger.warning("No test documents found")
            return
        
        self.logger.info(f"Found {len(test_documents)} test documents")
        
        # Process documents
        results = []
        failed_documents = []
        
        with ThreadPoolExecutor(max_workers=self.config.parallel_workers) as executor:
            future_to_doc = {
                executor.submit(self._process_single_document, doc_path): doc_path
                for doc_path in test_documents
            }
            
            for future in as_completed(future_to_doc):
                doc_path = future_to_doc[future]
                try:
                    result = future.result(timeout=self.config.test_timeout_minutes * 60)
                    if result:
                        results.append(result)
                        self.results.documents_tested += 1
                        if result.overall_accuracy >= self.config.accuracy_threshold:
                            self.results.documents_passed += 1
                    else:
                        failed_documents.append(doc_path)
                        
                except Exception as e:
                    self.logger.error(f"Error processing {doc_path}: {e}")
                    failed_documents.append(doc_path)
        
        # Calculate overall metrics
        if results:
            self.results.overall_accuracy = sum(r.overall_accuracy for r in results) / len(results)
            
            # Group by document type
            doc_type_results = {}
            for result in results:
                doc_type = result.document_type
                if doc_type not in doc_type_results:
                    doc_type_results[doc_type] = []
                doc_type_results[doc_type].append(result.overall_accuracy)
            
            for doc_type, accuracies in doc_type_results.items():
                self.results.accuracy_by_document_type[doc_type] = sum(accuracies) / len(accuracies)
        
        self.logger.info(f"Real document testing completed: {self.results.documents_passed}/{self.results.documents_tested} passed")
        
        if failed_documents:
            self.results.critical_issues.extend([f"Failed to process: {doc}" for doc in failed_documents])
    
    def _discover_test_documents(self) -> List[Path]:
        """Discover test documents in the test data directory."""
        documents = []
        
        if not self.test_data_dir.exists():
            return documents
        
        for ext in self.config.supported_formats:
            pattern = f"*.{ext}"
            documents.extend(self.test_data_dir.rglob(pattern))
        
        return documents
    
    def _process_single_document(self, doc_path: Path):
        """Process a single document."""
        try:
            # Determine document type from path or filename
            doc_type = self._determine_document_type(doc_path)
            
            # Process document
            if doc_type == "mykad":
                result = self.real_document_tester.test_mykad_processing(
                    str(doc_path), f"test_{doc_path.stem}"
                )
            elif doc_type == "spk":
                result = self.real_document_tester.test_spk_processing(
                    str(doc_path), f"test_{doc_path.stem}"
                )
            else:
                self.logger.warning(f"Unknown document type for {doc_path}")
                return None
            
            return result
            
        except Exception as e:
            self.logger.error(f"Error processing document {doc_path}: {e}")
            return None
    
    def _determine_document_type(self, doc_path: Path) -> str:
        """Determine document type from path or filename."""
        path_str = str(doc_path).lower()
        
        if "mykad" in path_str or "ic" in path_str:
            return "mykad"
        elif "spk" in path_str or "certificate" in path_str:
            return "spk"
        else:
            # Default to mykad if unclear
            return "mykad"
    
    def _start_feedback_collection(self):
        """Start feedback collection web interface."""
        if not self.feedback_collector:
            self.logger.warning("Feedback collector not available")
            return
        
        try:
            # Create Flask app for feedback collection
            self.feedback_app = self.feedback_collector.create_web_interface()
            
            # Start in separate thread
            def run_feedback_app():
                self.feedback_app.run(
                    host='0.0.0.0',
                    port=self.config.feedback_web_port,
                    debug=False,
                    use_reloader=False
                )
            
            self.feedback_thread = threading.Thread(target=run_feedback_app, daemon=True)
            self.feedback_thread.start()
            
            # Wait a moment for server to start
            time.sleep(2)
            
            feedback_url = f"http://localhost:{self.config.feedback_web_port}"
            self.logger.info(f"Feedback collection started at: {feedback_url}")
            
            # Store feedback URL for later reference
            self.results.feedback_summary['feedback_url'] = feedback_url
            
        except Exception as e:
            self.logger.error(f"Error starting feedback collection: {e}")
    
    def _run_accuracy_validation(self):
        """Run accuracy validation tests."""
        if not self.validation_system:
            self.logger.warning("Validation system not available")
            return
        
        try:
            # Get validation metrics
            metrics = self.validation_system.get_validation_metrics()
            
            self.results.validation_accuracy = metrics.overall_accuracy_percentage / 100
            self.results.field_accuracy_breakdown = metrics.accuracy_by_field_type
            
            # Extract common errors
            self.results.common_errors = metrics.common_error_patterns[:10]  # Top 10
            
            # Generate improvement recommendations
            if metrics.improvement_areas:
                self.results.improvement_recommendations.extend(metrics.improvement_areas)
            
            self.logger.info(f"Validation accuracy: {self.results.validation_accuracy:.2%}")
            
        except Exception as e:
            self.logger.error(f"Error in accuracy validation: {e}")
    
    def _run_performance_tests(self):
        """Run performance tests."""
        if not self.real_document_tester:
            self.logger.warning("Real document tester not available for performance testing")
            return
        
        try:
            # Run performance benchmarks
            performance_results = self.real_document_tester.run_performance_benchmark(
                num_documents=min(10, self.results.documents_tested)
            )
            
            if performance_results:
                self.results.average_processing_time = performance_results.get('average_processing_time', 0)
                self.results.max_processing_time = performance_results.get('max_processing_time', 0)
                self.results.average_memory_usage = performance_results.get('average_memory_usage', 0)
                self.results.max_memory_usage = performance_results.get('max_memory_usage', 0)
                
                # Check against thresholds
                if self.results.max_processing_time > self.config.max_processing_time_seconds:
                    self.results.critical_issues.append(
                        f"Processing time exceeded threshold: {self.results.max_processing_time:.2f}s > {self.config.max_processing_time_seconds}s"
                    )
                
                if self.results.max_memory_usage > self.config.max_memory_usage_mb:
                    self.results.critical_issues.append(
                        f"Memory usage exceeded threshold: {self.results.max_memory_usage:.2f}MB > {self.config.max_memory_usage_mb}MB"
                    )
            
            self.logger.info(f"Performance testing completed - Avg time: {self.results.average_processing_time:.2f}s")
            
        except Exception as e:
            self.logger.error(f"Error in performance testing: {e}")
    
    def _run_edge_case_tests(self):
        """Run edge case testing."""
        if not self.real_document_tester:
            self.logger.warning("Real document tester not available for edge case testing")
            return
        
        try:
            # Define edge case scenarios
            edge_cases = [
                'poor_quality',
                'damaged_document',
                'partial_document',
                'rotated_document',
                'low_resolution',
                'high_contrast',
                'blurred_text'
            ]
            
            edge_case_results = {}
            
            for case in edge_cases:
                try:
                    # Run edge case test
                    result = self.real_document_tester.test_edge_case_handling(case)
                    if result:
                        edge_case_results[case] = result.success_rate
                    else:
                        edge_case_results[case] = 0.0
                        
                except Exception as e:
                    self.logger.warning(f"Edge case test '{case}' failed: {e}")
                    edge_case_results[case] = 0.0
            
            self.results.edge_case_handling = edge_case_results
            
            # Check for critical edge case failures
            failed_cases = [case for case, rate in edge_case_results.items() if rate < 0.5]
            if failed_cases:
                self.results.critical_issues.append(
                    f"Poor edge case handling: {', '.join(failed_cases)}"
                )
            
            self.logger.info(f"Edge case testing completed - {len(edge_case_results)} scenarios tested")
            
        except Exception as e:
            self.logger.error(f"Error in edge case testing: {e}")
    
    def _generate_reports(self):
        """Generate comprehensive test reports."""
        try:
            # Generate JSON report
            json_report = self.output_dir / f"{self.results.test_session_id}_report.json"
            with open(json_report, 'w', encoding='utf-8') as f:
                json.dump(self.results.__dict__, f, indent=2, ensure_ascii=False, default=str)
            
            self.results.report_files.append(str(json_report))
            self.logger.info(f"JSON report generated: {json_report}")
            
            # Generate HTML report if requested
            if self.config.generate_html_report:
                html_report = self._generate_html_report()
                if html_report:
                    self.results.report_files.append(html_report)
            
            # Generate summary report
            summary_report = self._generate_summary_report()
            if summary_report:
                self.results.report_files.append(summary_report)
            
        except Exception as e:
            self.logger.error(f"Error generating reports: {e}")
    
    def _generate_html_report(self) -> Optional[str]:
        """Generate HTML report."""
        try:
            html_file = self.output_dir / f"{self.results.test_session_id}_report.html"
            
            html_content = f"""
<!DOCTYPE html>
<html>
<head>
    <title>User Acceptance Test Report - {self.results.test_session_id}</title>
    <style>
        body {{ font-family: Arial, sans-serif; margin: 20px; }}
        .header {{ background-color: #f0f0f0; padding: 20px; border-radius: 5px; }}
        .section {{ margin: 20px 0; padding: 15px; border: 1px solid #ddd; border-radius: 5px; }}
        .metric {{ display: inline-block; margin: 10px; padding: 10px; background-color: #f9f9f9; border-radius: 3px; }}
        .success {{ color: green; }}
        .warning {{ color: orange; }}
        .error {{ color: red; }}
        table {{ border-collapse: collapse; width: 100%; }}
        th, td {{ border: 1px solid #ddd; padding: 8px; text-align: left; }}
        th {{ background-color: #f2f2f2; }}
    </style>
</head>
<body>
    <div class="header">
        <h1>User Acceptance Test Report</h1>
        <p><strong>Session ID:</strong> {self.results.test_session_id}</p>
        <p><strong>Duration:</strong> {self.results.duration_seconds:.2f} seconds</p>
        <p><strong>Generated:</strong> {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
    </div>
    
    <div class="section">
        <h2>Overall Results</h2>
        <div class="metric">Documents Tested: <strong>{self.results.documents_tested}</strong></div>
        <div class="metric">Documents Passed: <strong>{self.results.documents_passed}</strong></div>
        <div class="metric">Overall Accuracy: <strong>{self.results.overall_accuracy:.2%}</strong></div>
        <div class="metric">Validation Accuracy: <strong>{self.results.validation_accuracy:.2%}</strong></div>
    </div>
    
    <div class="section">
        <h2>Performance Metrics</h2>
        <div class="metric">Avg Processing Time: <strong>{self.results.average_processing_time:.2f}s</strong></div>
        <div class="metric">Max Processing Time: <strong>{self.results.max_processing_time:.2f}s</strong></div>
        <div class="metric">Avg Memory Usage: <strong>{self.results.average_memory_usage:.2f}MB</strong></div>
        <div class="metric">Max Memory Usage: <strong>{self.results.max_memory_usage:.2f}MB</strong></div>
    </div>
    
    <div class="section">
        <h2>Accuracy by Document Type</h2>
        <table>
            <tr><th>Document Type</th><th>Accuracy</th></tr>
"""
            
            for doc_type, accuracy in self.results.accuracy_by_document_type.items():
                html_content += f"<tr><td>{doc_type}</td><td>{accuracy:.2%}</td></tr>"
            
            html_content += """
        </table>
    </div>
    
    <div class="section">
        <h2>Field Accuracy Breakdown</h2>
        <table>
            <tr><th>Field Type</th><th>Accuracy</th></tr>
"""
            
            for field_type, accuracy in self.results.field_accuracy_breakdown.items():
                html_content += f"<tr><td>{field_type}</td><td>{accuracy:.2%}</td></tr>"
            
            html_content += """
        </table>
    </div>
"""
            
            if self.results.critical_issues:
                html_content += """
    <div class="section">
        <h2>Critical Issues</h2>
        <ul>
"""
                for issue in self.results.critical_issues:
                    html_content += f"<li class='error'>{issue}</li>"
                
                html_content += "</ul></div>"
            
            if self.results.improvement_recommendations:
                html_content += """
    <div class="section">
        <h2>Improvement Recommendations</h2>
        <ul>
"""
                for recommendation in self.results.improvement_recommendations:
                    html_content += f"<li>{recommendation}</li>"
                
                html_content += "</ul></div>"
            
            html_content += """
</body>
</html>
"""
            
            with open(html_file, 'w', encoding='utf-8') as f:
                f.write(html_content)
            
            self.logger.info(f"HTML report generated: {html_file}")
            return str(html_file)
            
        except Exception as e:
            self.logger.error(f"Error generating HTML report: {e}")
            return None
    
    def _generate_summary_report(self) -> Optional[str]:
        """Generate summary text report."""
        try:
            summary_file = self.output_dir / f"{self.results.test_session_id}_summary.txt"
            
            with open(summary_file, 'w', encoding='utf-8') as f:
                f.write(f"USER ACCEPTANCE TEST SUMMARY\n")
                f.write(f"{'=' * 50}\n\n")
                
                f.write(f"Session ID: {self.results.test_session_id}\n")
                f.write(f"Duration: {self.results.duration_seconds:.2f} seconds\n")
                f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
                
                f.write(f"OVERALL RESULTS\n")
                f.write(f"{'-' * 20}\n")
                f.write(f"Documents Tested: {self.results.documents_tested}\n")
                f.write(f"Documents Passed: {self.results.documents_passed}\n")
                f.write(f"Success Rate: {self.results.documents_passed/max(1, self.results.documents_tested):.2%}\n")
                f.write(f"Overall Accuracy: {self.results.overall_accuracy:.2%}\n")
                f.write(f"Validation Accuracy: {self.results.validation_accuracy:.2%}\n\n")
                
                f.write(f"PERFORMANCE METRICS\n")
                f.write(f"{'-' * 20}\n")
                f.write(f"Average Processing Time: {self.results.average_processing_time:.2f}s\n")
                f.write(f"Maximum Processing Time: {self.results.max_processing_time:.2f}s\n")
                f.write(f"Average Memory Usage: {self.results.average_memory_usage:.2f}MB\n")
                f.write(f"Maximum Memory Usage: {self.results.max_memory_usage:.2f}MB\n\n")
                
                if self.results.critical_issues:
                    f.write(f"CRITICAL ISSUES\n")
                    f.write(f"{'-' * 20}\n")
                    for issue in self.results.critical_issues:
                        f.write(f"• {issue}\n")
                    f.write("\n")
                
                if self.results.improvement_recommendations:
                    f.write(f"IMPROVEMENT RECOMMENDATIONS\n")
                    f.write(f"{'-' * 30}\n")
                    for recommendation in self.results.improvement_recommendations:
                        f.write(f"• {recommendation}\n")
                    f.write("\n")
                
                # Overall assessment
                f.write(f"OVERALL ASSESSMENT\n")
                f.write(f"{'-' * 20}\n")
                
                if self.results.overall_accuracy >= 0.9:
                    f.write("✓ EXCELLENT: System performance exceeds expectations\n")
                elif self.results.overall_accuracy >= 0.8:
                    f.write("✓ GOOD: System performance meets requirements\n")
                elif self.results.overall_accuracy >= 0.7:
                    f.write("⚠ FAIR: System performance needs improvement\n")
                else:
                    f.write("✗ POOR: System performance requires significant improvement\n")
                
                if not self.results.critical_issues:
                    f.write("✓ No critical issues identified\n")
                else:
                    f.write(f"✗ {len(self.results.critical_issues)} critical issues require attention\n")
            
            self.logger.info(f"Summary report generated: {summary_file}")
            return str(summary_file)
            
        except Exception as e:
            self.logger.error(f"Error generating summary report: {e}")
            return None
    
    def _cleanup(self):
        """Cleanup resources."""
        try:
            # Stop feedback collection if running
            if self.feedback_thread and self.feedback_thread.is_alive():
                self.logger.info("Stopping feedback collection")
                # Note: Flask server will stop when main thread exits
            
            # Collect final feedback metrics if available
            if self.feedback_collector:
                try:
                    analytics = self.feedback_collector.get_feedback_analytics()
                    if analytics:
                        self.results.feedback_sessions = analytics.total_feedback_count
                        self.results.user_satisfaction_score = analytics.average_satisfaction_score
                        self.results.feedback_summary.update({
                            'total_feedback': analytics.total_feedback_count,
                            'avg_satisfaction': analytics.average_satisfaction_score,
                            'feedback_categories': analytics.feedback_by_category
                        })
                except Exception as e:
                    self.logger.warning(f"Error collecting final feedback metrics: {e}")
            
            self.logger.info("Cleanup completed")
            
        except Exception as e:
            self.logger.error(f"Error during cleanup: {e}")


def main():
    """Main entry point for UAT runner."""
    parser = argparse.ArgumentParser(description="User Acceptance Test Runner")
    parser.add_argument('--config', type=str, help='Configuration file path')
    parser.add_argument('--test-data', type=str, help='Test data directory')
    parser.add_argument('--output', type=str, help='Output directory')
    parser.add_argument('--api-url', type=str, default='http://localhost:8000', help='API base URL')
    parser.add_argument('--workers', type=int, default=4, help='Number of parallel workers')
    parser.add_argument('--timeout', type=int, default=60, help='Test timeout in minutes')
    parser.add_argument('--no-feedback', action='store_true', help='Disable feedback collection')
    parser.add_argument('--feedback-port', type=int, default=5001, help='Feedback web interface port')
    parser.add_argument('--accuracy-threshold', type=float, default=0.85, help='Accuracy threshold')
    parser.add_argument('--verbose', '-v', action='store_true', help='Verbose logging')
    
    args = parser.parse_args()
    
    # Create configuration
    config = UATConfiguration()
    
    if args.test_data:
        config.test_data_directory = args.test_data
    if args.output:
        config.output_directory = args.output
    if args.api_url:
        config.api_base_url = args.api_url
    if args.workers:
        config.parallel_workers = args.workers
    if args.timeout:
        config.test_timeout_minutes = args.timeout
    if args.no_feedback:
        config.feedback_collection_enabled = False
    if args.feedback_port:
        config.feedback_web_port = args.feedback_port
    if args.accuracy_threshold:
        config.accuracy_threshold = args.accuracy_threshold
    
    # Load configuration file if provided
    if args.config and Path(args.config).exists():
        try:
            with open(args.config, 'r') as f:
                config_data = json.load(f)
                for key, value in config_data.items():
                    if hasattr(config, key):
                        setattr(config, key, value)
        except Exception as e:
            print(f"Error loading configuration file: {e}")
            return 1
    
    # Set logging level
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    try:
        # Create and run UAT
        runner = UserAcceptanceTestRunner(config)
        results = runner.run_comprehensive_uat()
        
        # Print summary
        print(f"\n{'=' * 60}")
        print(f"USER ACCEPTANCE TEST COMPLETED")
        print(f"{'=' * 60}")
        print(f"Session ID: {results.test_session_id}")
        print(f"Duration: {results.duration_seconds:.2f} seconds")
        print(f"Documents Tested: {results.documents_tested}")
        print(f"Documents Passed: {results.documents_passed}")
        print(f"Overall Accuracy: {results.overall_accuracy:.2%}")
        print(f"Validation Accuracy: {results.validation_accuracy:.2%}")
        
        if results.critical_issues:
            print(f"\nCRITICAL ISSUES ({len(results.critical_issues)}):")
            for issue in results.critical_issues:
                print(f"  • {issue}")
        
        if results.report_files:
            print(f"\nREPORTS GENERATED:")
            for report in results.report_files:
                print(f"  • {report}")
        
        # Return appropriate exit code
        if results.critical_issues:
            return 1
        elif results.overall_accuracy < config.accuracy_threshold:
            return 1
        else:
            return 0
            
    except Exception as e:
        print(f"UAT execution failed: {e}")
        return 1


if __name__ == "__main__":
    sys.exit(main())