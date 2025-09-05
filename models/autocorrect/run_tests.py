#!/usr/bin/env python3
"""
Test Runner Script for Autocorrect Model Testing

This script orchestrates comprehensive testing of the autocorrect models,
including both framework testing and trained model evaluation.

Usage:
    python run_tests.py [--mode framework|models|all] [--config test_config.json]

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
    from comprehensive_model_testing import ModelTester, TestLogger
    from test_trained_models import run_comprehensive_tests, TrainedModelLoader, ProductionAutocorrectTester
except ImportError as e:
    print(f"Warning: Could not import testing modules: {e}")
    print("Please ensure all required files are present.")

class TestRunner:
    """Orchestrates comprehensive testing of autocorrect models"""
    
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
                    "description": "Test trained autocorrect models"
                }
            },
            "test_parameters": {
                "confidence_threshold": 0.6,
                "performance_threshold_ms": 10,
                "batch_size": 100,
                "timeout_seconds": 300
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
        print("=" * 40)
        
        try:
            # Initialize tester with dummy corrector
            tester = ModelTester()
            
            def dummy_corrector(text):
                """Dummy corrector for framework testing"""
                if not text:
                    return text, 0.0, 'dummy'
                
                # Simple rule-based corrections for testing
                corrections = {
                    'toyot': 'toyota',
                    'hond': 'honda',
                    'camr': 'camry',
                    'civicy': 'civic'
                }
                
                text_lower = text.lower().strip()
                if text_lower in corrections:
                    return corrections[text_lower], 0.9, 'dummy_rule'
                else:
                    return text, 0.1, 'dummy_passthrough'
            
            # Run tests
            results = tester.run_all_tests(dummy_corrector)
            
            self.results['framework'] = {
                'status': 'completed',
                'total_tests': len(results),
                'passed': sum(1 for r in results if r.status == 'PASS'),
                'failed': sum(1 for r in results if r.status == 'FAIL'),
                'errors': sum(1 for r in results if r.status == 'ERROR'),
                'execution_time': sum(r.execution_time for r in results)
            }
            
            print(f"✅ Framework testing completed: {self.results['framework']['passed']}/{self.results['framework']['total_tests']} tests passed")
            
        except Exception as e:
            print(f"❌ Framework testing failed: {e}")
            self.results['framework'] = {
                'status': 'failed',
                'error': str(e)
            }
    
    def run_model_tests(self):
        """Run trained model tests"""
        print("\n🤖 TRAINED MODEL TESTING")
        print("=" * 40)
        
        try:
            # Check if models are available
            model_loader = TrainedModelLoader()
            
            if model_loader.load_models():
                print("✅ Models loaded successfully")
                
                # Run comprehensive model tests
                results = run_comprehensive_tests()
                
                self.results['models'] = {
                    'status': 'completed',
                    'models_loaded': len(model_loader.models),
                    'best_model': model_loader.get_best_model()[1] if model_loader.get_best_model()[1] else 'none',
                    'total_tests': len(results) if results else 0
                }
                
                print(f"✅ Model testing completed")
                
            else:
                print("⚠️ No trained models found. Skipping model tests.")
                print("   Please run the training notebook first to generate models.")
                
                self.results['models'] = {
                    'status': 'skipped',
                    'reason': 'no_models_found'
                }
                
        except Exception as e:
            print(f"❌ Model testing failed: {e}")
            self.results['models'] = {
                'status': 'failed',
                'error': str(e)
            }
    
    def run_integration_tests(self):
        """Run integration tests between components"""
        print("\n🔗 INTEGRATION TESTING")
        print("=" * 40)
        
        try:
            # Test data flow between components
            integration_tests = [
                self.test_data_loading,
                self.test_model_pipeline,
                self.test_correction_workflow,
                self.test_batch_processing
            ]
            
            passed = 0
            total = len(integration_tests)
            
            for test_func in integration_tests:
                try:
                    test_func()
                    passed += 1
                    print(f"✅ {test_func.__name__} - PASSED")
                except Exception as e:
                    print(f"❌ {test_func.__name__} - FAILED: {e}")
            
            self.results['integration'] = {
                'status': 'completed',
                'passed': passed,
                'total': total,
                'success_rate': passed / total if total > 0 else 0
            }
            
        except Exception as e:
            print(f"❌ Integration testing failed: {e}")
            self.results['integration'] = {
                'status': 'failed',
                'error': str(e)
            }
    
    def test_data_loading(self):
        """Test data loading functionality"""
        data_dir = Path(__file__).parent.parent / "data"
        
        # Check for required data files
        required_files = ['vehicle_master.csv']
        for file_name in required_files:
            file_path = data_dir / file_name
            if not file_path.exists():
                raise FileNotFoundError(f"Required data file not found: {file_path}")
        
        # Test loading
        import pandas as pd
        df = pd.read_csv(data_dir / 'vehicle_master.csv')
        if len(df) == 0:
            raise ValueError("Vehicle master data is empty")
    
    def test_model_pipeline(self):
        """Test model pipeline functionality"""
        # Test that we can create a basic correction pipeline
        model_loader = TrainedModelLoader()
        
        # This should not raise an exception
        model_loader.load_models()
    
    def test_correction_workflow(self):
        """Test end-to-end correction workflow"""
        model_loader = TrainedModelLoader()
        model_loader.load_models()
        
        corrector = ProductionAutocorrectTester(model_loader)
        
        # Test basic correction
        result = corrector.correct_text("toyota")
        if not isinstance(result, tuple) or len(result) != 3:
            raise ValueError("Correction function should return (text, confidence, method) tuple")
    
    def test_batch_processing(self):
        """Test batch processing functionality"""
        model_loader = TrainedModelLoader()
        model_loader.load_models()
        
        corrector = ProductionAutocorrectTester(model_loader)
        
        # Test batch correction
        test_inputs = ['toyota', 'honda', 'nissan']
        results = corrector.batch_correct(test_inputs)
        
        if len(results) != len(test_inputs):
            raise ValueError("Batch processing should return same number of results as inputs")
    
    def generate_final_report(self):
        """Generate comprehensive final report"""
        print("\n📊 GENERATING FINAL REPORT")
        print("=" * 40)
        
        end_time = datetime.now()
        total_duration = (end_time - self.start_time).total_seconds()
        
        report = f"""
🧪 COMPREHENSIVE AUTOCORRECT MODEL TESTING REPORT
{'='*70}
Session ID: {self.start_time.strftime('%Y%m%d_%H%M%S')}
Start Time: {self.start_time.strftime('%Y-%m-%d %H:%M:%S')}
End Time: {end_time.strftime('%Y-%m-%d %H:%M:%S')}
Total Duration: {total_duration:.2f} seconds

📋 TEST SUMMARY:
"""
        
        total_tests = 0
        total_passed = 0
        
        for test_type, results in self.results.items():
            report += f"\n{test_type.upper()} TESTS:\n"
            if results['status'] == 'completed':
                if 'total_tests' in results:
                    total_tests += results['total_tests']
                    total_passed += results.get('passed', 0)
                    report += f"   Status: ✅ Completed\n"
                    report += f"   Tests: {results.get('passed', 0)}/{results.get('total_tests', 0)} passed\n"
                elif 'total' in results:
                    total_tests += results['total']
                    total_passed += results.get('passed', 0)
                    report += f"   Status: ✅ Completed\n"
                    report += f"   Tests: {results.get('passed', 0)}/{results.get('total', 0)} passed\n"
                else:
                    report += f"   Status: ✅ Completed\n"
            elif results['status'] == 'skipped':
                report += f"   Status: ⏭️ Skipped - {results.get('reason', 'unknown')}\n"
            else:
                report += f"   Status: ❌ Failed - {results.get('error', 'unknown error')}\n"
        
        overall_success_rate = (total_passed / total_tests * 100) if total_tests > 0 else 0
        
        report += f"""
🎯 OVERALL RESULTS:
   Total Tests Executed: {total_tests}
   Total Tests Passed: {total_passed}
   Overall Success Rate: {overall_success_rate:.1f}%
   
📁 ARTIFACTS:
   Results Directory: {self.session_dir}
   Detailed Logs: Available in test_logs/
   
🔄 NEXT STEPS:
   • Review failed tests in detailed logs
   • Check model performance metrics
   • Consider retraining if accuracy is low
   • Deploy models if all tests pass
"""
        
        # Save report
        report_file = self.session_dir / "final_report.txt"
        with open(report_file, 'w') as f:
            f.write(report)
        
        # Save results as JSON
        results_file = self.session_dir / "test_results_summary.json"
        summary_data = {
            'session_id': self.start_time.strftime('%Y%m%d_%H%M%S'),
            'start_time': self.start_time.isoformat(),
            'end_time': end_time.isoformat(),
            'duration_seconds': total_duration,
            'results': self.results,
            'summary': {
                'total_tests': total_tests,
                'total_passed': total_passed,
                'success_rate': overall_success_rate
            }
        }
        
        with open(results_file, 'w') as f:
            json.dump(summary_data, f, indent=2)
        
        print(report)
        print(f"\n📄 Final report saved to: {report_file}")
        print(f"📊 Results summary saved to: {results_file}")
    
    def run_all_tests(self, mode: str = 'all'):
        """Run all specified tests"""
        print(f"🚀 STARTING COMPREHENSIVE TESTING - MODE: {mode.upper()}")
        print(f"⏰ Start Time: {self.start_time.strftime('%Y-%m-%d %H:%M:%S')}")
        print("=" * 70)
        
        # Setup output directory
        self.setup_output_directory()
        
        # Run tests based on mode
        if mode in ['framework', 'all']:
            self.run_framework_tests()
        
        if mode in ['models', 'all']:
            self.run_model_tests()
        
        if mode in ['integration', 'all']:
            self.run_integration_tests()
        
        # Generate final report
        self.generate_final_report()
        
        return self.results

def main():
    """Main function with command line interface"""
    parser = argparse.ArgumentParser(description='Comprehensive Autocorrect Model Testing')
    parser.add_argument('--mode', choices=['framework', 'models', 'integration', 'all'], 
                       default='all', help='Test mode to run')
    parser.add_argument('--config', type=str, help='Path to test configuration file')
    
    args = parser.parse_args()
    
    try:
        # Initialize test runner
        runner = TestRunner(args.config)
        
        # Run tests
        results = runner.run_all_tests(args.mode)
        
        print("\n🎉 Testing completed successfully!")
        return 0
        
    except KeyboardInterrupt:
        print("\n⚠️ Testing interrupted by user")
        return 1
    except Exception as e:
        print(f"\n❌ Testing failed with error: {e}")
        import traceback
        traceback.print_exc()
        return 1

if __name__ == "__main__":
    exit(main())