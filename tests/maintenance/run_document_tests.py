#!/usr/bin/env python3
"""
Test Runner Script for Document Parser Testing

This script orchestrates comprehensive testing of the document parser models,
including both framework testing and trained model evaluation.

Usage:
    python run_document_tests.py [--mode framework|models|all] [--config test_config.json]

Follows the autocorrect model's testing pattern.

Author: AI Assistant
Date: 2024
"""

import os
import sys
import argparse
import json
import time
from pathlib import Path
from datetime import datetime

# Add current directory to path
sys.path.append(str(Path(__file__).parent))

try:
    from comprehensive_document_testing import DocumentParserTester, TestLogger
except ImportError as e:
    print(f"Warning: Could not import testing modules: {e}")
    print("Please ensure all required files are present.")

class DocumentTestRunner:
    """Orchestrates comprehensive testing of document parser models"""
    
    def __init__(self, config_file: str = None):
        self.config = self.load_config(config_file)
        self.results = {}
        self.start_time = datetime.now()
        
    def load_config(self, config_file: str = None) -> dict:
        """Load test configuration"""
        default_config = {
            "test_modes": {
                "framework": {
                    "enabled": True,
                    "description": "Test the testing framework itself"
                },
                "models": {
                    "enabled": True,
                    "description": "Test trained document parser models"
                }
            },
            "test_parameters": {
                "confidence_threshold": 0.8,
                "performance_threshold_ms": 100,
                "batch_size": 50,
                "timeout_seconds": 600
            },
            "logging": {
                "log_level": "INFO",
                "save_detailed_logs": True,
                "generate_html_report": True
            },
            "output": {
                "results_dir": "test_results",
                "save_plots": True,
                "export_csv": True
            },
            "paths": {
                "model_path": "model_artifacts/document_parser/models",
                "test_data_path": "data/test",
                "output_path": "test_results"
            }
        }
        
        if config_file and Path(config_file).exists():
            try:
                with open(config_file, 'r') as f:
                    user_config = json.load(f)
                # Merge with default config
                default_config.update(user_config)
                print(f"✅ Loaded configuration from {config_file}")
            except Exception as e:
                print(f"⚠️ Error loading config file: {e}. Using default configuration.")
        
        return default_config
    
    def setup_output_directory(self):
        """Setup output directory for test results"""
        results_dir = Path(self.config['output']['results_dir'])
        results_dir.mkdir(exist_ok=True)
        
        # Create session-specific subdirectory
        session_dir = results_dir / f"session_{self.start_time.strftime('%Y%m%d_%H%M%S')}"
        session_dir.mkdir(exist_ok=True)
        
        self.session_dir = session_dir
        print(f"📁 Test results will be saved to: {session_dir}")
        
        return session_dir
    
    def run_framework_tests(self):
        """Run framework validation tests"""
        print("\n🔧 FRAMEWORK TESTING")
        print("-" * 30)
        
        try:
            # Test framework initialization
            tester = DocumentParserTester()
            logger = TestLogger()
            
            print("✅ Framework components initialized successfully")
            
            # Test logging functionality
            from comprehensive_document_testing import TestResult
            test_result = TestResult(
                test_name="framework_test",
                test_type="framework",
                status="PASS",
                execution_time=0.001,
                timestamp=datetime.now().isoformat(),
                input_data="test_input",
                expected_output="test_output",
                actual_output="test_output",
                metrics={"test_metric": 1.0}
            )
            
            logger.log_test_result(test_result)
            print("✅ Logging functionality working")
            
            # Test summary generation
            summary = logger.generate_summary()
            print(f"✅ Summary generation working: {summary['total_tests']} test(s)")
            
            self.results['framework'] = {
                'status': 'PASS',
                'tests_run': 3,
                'execution_time': time.time()
            }
            
            return True
            
        except Exception as e:
            print(f"❌ Framework testing failed: {e}")
            self.results['framework'] = {
                'status': 'FAIL',
                'error': str(e),
                'execution_time': time.time()
            }
            return False
    
    def run_model_tests(self):
        """Run trained model tests"""
        print("\n🤖 MODEL TESTING")
        print("-" * 30)
        
        try:
            # Initialize tester with configured paths
            tester = DocumentParserTester(
                model_path=self.config['paths']['model_path'],
                test_data_path=self.config['paths']['test_data_path']
            )
            
            # Run comprehensive tests
            summary = tester.run_comprehensive_tests()
            
            self.results['models'] = {
                'status': 'PASS' if summary['success_rate'] > 0.8 else 'FAIL',
                'summary': summary,
                'execution_time': time.time()
            }
            
            print(f"✅ Model testing completed with {summary['success_rate']:.2%} success rate")
            
            return True
            
        except Exception as e:
            print(f"❌ Model testing failed: {e}")
            self.results['models'] = {
                'status': 'FAIL',
                'error': str(e),
                'execution_time': time.time()
            }
            return False
    
    def run_performance_monitoring(self):
        """Run performance monitoring tests"""
        print("\n📊 PERFORMANCE MONITORING")
        print("-" * 30)
        
        try:
            # Monitor system resources
            import psutil
            
            # Memory usage
            memory_info = psutil.virtual_memory()
            memory_usage_percent = memory_info.percent
            
            # CPU usage
            cpu_usage_percent = psutil.cpu_percent(interval=1)
            
            # Disk usage
            disk_info = psutil.disk_usage('/')
            disk_usage_percent = (disk_info.used / disk_info.total) * 100
            
            performance_metrics = {
                'memory_usage_percent': memory_usage_percent,
                'cpu_usage_percent': cpu_usage_percent,
                'disk_usage_percent': disk_usage_percent,
                'timestamp': datetime.now().isoformat()
            }
            
            # Check if performance is within acceptable limits
            performance_ok = (
                memory_usage_percent < 90 and
                cpu_usage_percent < 90 and
                disk_usage_percent < 90
            )
            
            self.results['performance'] = {
                'status': 'PASS' if performance_ok else 'WARN',
                'metrics': performance_metrics,
                'execution_time': time.time()
            }
            
            print(f"📈 Memory: {memory_usage_percent:.1f}%")
            print(f"🖥️ CPU: {cpu_usage_percent:.1f}%")
            print(f"💾 Disk: {disk_usage_percent:.1f}%")
            
            return True
            
        except ImportError:
            print("⚠️ psutil not available, skipping performance monitoring")
            self.results['performance'] = {
                'status': 'SKIP',
                'reason': 'psutil not available'
            }
            return True
        except Exception as e:
            print(f"❌ Performance monitoring failed: {e}")
            self.results['performance'] = {
                'status': 'FAIL',
                'error': str(e)
            }
            return False
    
    def generate_report(self):
        """Generate comprehensive test report"""
        print("\n📋 GENERATING REPORT")
        print("-" * 30)
        
        # Setup output directory
        self.setup_output_directory()
        
        # Create report data
        report_data = {
            'session_info': {
                'session_id': self.start_time.strftime('%Y%m%d_%H%M%S'),
                'start_time': self.start_time.isoformat(),
                'end_time': datetime.now().isoformat(),
                'duration_seconds': (datetime.now() - self.start_time).total_seconds()
            },
            'configuration': self.config,
            'results': self.results,
            'summary': self.generate_summary()
        }
        
        # Save JSON report
        report_file = self.session_dir / 'test_report.json'
        with open(report_file, 'w') as f:
            json.dump(report_data, f, indent=2, default=str)
        
        print(f"📄 Report saved to: {report_file}")
        
        # Generate HTML report if enabled
        if self.config['logging']['generate_html_report']:
            self.generate_html_report(report_data)
        
        return report_data
    
    def generate_html_report(self, report_data):
        """Generate HTML report"""
        html_content = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <title>Document Parser Test Report</title>
            <style>
                body {{ font-family: Arial, sans-serif; margin: 20px; }}
                .header {{ background-color: #f0f0f0; padding: 20px; border-radius: 5px; }}
                .section {{ margin: 20px 0; padding: 15px; border: 1px solid #ddd; border-radius: 5px; }}
                .pass {{ color: green; }}
                .fail {{ color: red; }}
                .warn {{ color: orange; }}
                .skip {{ color: gray; }}
                table {{ border-collapse: collapse; width: 100%; }}
                th, td {{ border: 1px solid #ddd; padding: 8px; text-align: left; }}
                th {{ background-color: #f2f2f2; }}
            </style>
        </head>
        <body>
            <div class="header">
                <h1>Document Parser Test Report</h1>
                <p>Session: {report_data['session_info']['session_id']}</p>
                <p>Generated: {report_data['session_info']['end_time']}</p>
                <p>Duration: {report_data['session_info']['duration_seconds']:.2f} seconds</p>
            </div>
            
            <div class="section">
                <h2>Summary</h2>
                <table>
                    <tr><th>Component</th><th>Status</th><th>Details</th></tr>
        """
        
        for component, result in self.results.items():
            status_class = result['status'].lower()
            html_content += f"""
                    <tr>
                        <td>{component.title()}</td>
                        <td class="{status_class}">{result['status']}</td>
                        <td>{result.get('summary', {}).get('success_rate', 'N/A')}</td>
                    </tr>
            """
        
        html_content += """
                </table>
            </div>
        </body>
        </html>
        """
        
        html_file = self.session_dir / 'test_report.html'
        with open(html_file, 'w') as f:
            f.write(html_content)
        
        print(f"🌐 HTML report saved to: {html_file}")
    
    def generate_summary(self):
        """Generate overall test summary"""
        total_components = len(self.results)
        passed_components = sum(1 for r in self.results.values() if r['status'] == 'PASS')
        
        return {
            'total_components': total_components,
            'passed_components': passed_components,
            'overall_success_rate': passed_components / total_components if total_components > 0 else 0,
            'components_status': {name: result['status'] for name, result in self.results.items()}
        }
    
    def run_all_tests(self, mode: str = 'all'):
        """Run all specified tests"""
        print(f"🚀 Starting Document Parser Testing - Mode: {mode}")
        print("=" * 60)
        
        success = True
        
        if mode in ['framework', 'all']:
            if not self.run_framework_tests():
                success = False
        
        if mode in ['models', 'all']:
            if not self.run_model_tests():
                success = False
        
        if mode in ['performance', 'all']:
            if not self.run_performance_monitoring():
                success = False
        
        # Generate report
        report = self.generate_report()
        
        # Print final summary
        print("\n" + "=" * 60)
        print("🏁 TESTING COMPLETE")
        print("=" * 60)
        
        summary = report['summary']
        print(f"📊 Overall Success Rate: {summary['overall_success_rate']:.2%}")
        print(f"✅ Passed Components: {summary['passed_components']}/{summary['total_components']}")
        
        for component, status in summary['components_status'].items():
            emoji = {'PASS': '✅', 'FAIL': '❌', 'WARN': '⚠️', 'SKIP': '⏭️'}.get(status, '❓')
            print(f"{emoji} {component.title()}: {status}")
        
        return success

def main():
    """Main function"""
    parser = argparse.ArgumentParser(description='Document Parser Test Runner')
    parser.add_argument('--mode', choices=['framework', 'models', 'performance', 'all'], 
                       default='all', help='Test mode to run')
    parser.add_argument('--config', help='Configuration file path')
    
    args = parser.parse_args()
    
    # Initialize test runner
    runner = DocumentTestRunner(config_file=args.config)
    
    # Run tests
    success = runner.run_all_tests(mode=args.mode)
    
    # Exit with appropriate code
    sys.exit(0 if success else 1)

if __name__ == "__main__":
    main()