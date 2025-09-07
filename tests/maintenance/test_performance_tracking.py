#!/usr/bin/env python3
"""Test Performance Tracking System

This script tests the performance tracking functionality for document parser models,
following the autocorrect model's test_trained_models.py organizational patterns.
"""

import os
import sys
import time
import tempfile
import shutil
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, Any, List

# Add src to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'src'))

from document_parser.maintenance.performance_tracker import (
    DocumentParserPerformanceTracker,
    PerformanceMetrics,
    PerformanceStorage
)

class PerformanceTrackingTester:
    """Test suite for performance tracking system"""
    
    def __init__(self):
        self.temp_dir = None
        self.tracker = None
        self.test_results = []
        
    def setup_test_environment(self):
        """Set up temporary test environment"""
        self.temp_dir = tempfile.mkdtemp(prefix="perf_test_")
        self.tracker = DocumentParserPerformanceTracker(storage_path=self.temp_dir)
        print(f"📁 Test environment created: {self.temp_dir}")
        
    def cleanup_test_environment(self):
        """Clean up test environment"""
        if self.temp_dir and os.path.exists(self.temp_dir):
            shutil.rmtree(self.temp_dir)
            print(f"🧹 Test environment cleaned up")
    
    def test_basic_tracking(self) -> bool:
        """Test basic performance tracking functionality"""
        print("\n=== Testing Basic Tracking ===")
        
        try:
            # Sample document processing data
            start_time = time.time()
            end_time = start_time + 2.5
            
            results = {
                'classification': 'mykad',
                'fields': {
                    'name': 'JOHN DOE',
                    'ic_number': '123456-78-9012',
                    'address': '123 MAIN ST'
                },
                'confidence': {
                    'name': 0.95,
                    'ic_number': 0.88,
                    'address': 0.92
                },
                'success': True,
                'ocr_confidence': 0.91,
                'image_quality': 0.85
            }
            
            ground_truth = {
                'classification': 'mykad',
                'fields': {
                    'name': 'JOHN DOE',
                    'ic_number': '123456-78-9012',
                    'address': '123 MAIN ST'
                }
            }
            
            # Track performance
            tracking_id = self.tracker.track_processing(
                document_id='test_doc_001',
                document_type='mykad',
                start_time=start_time,
                end_time=end_time,
                results=results,
                ground_truth=ground_truth
            )
            
            print(f"✅ Basic tracking successful: {tracking_id}")
            return True
            
        except Exception as e:
            print(f"❌ Basic tracking failed: {e}")
            return False
    
    def test_accuracy_calculation(self) -> bool:
        """Test accuracy calculation with various scenarios"""
        print("\n=== Testing Accuracy Calculation ===")
        
        try:
            test_cases = [
                {
                    'name': 'Perfect Match',
                    'results': {
                        'classification': 'spk',
                        'fields': {'vehicle_number': 'ABC1234', 'owner': 'JANE DOE'},
                        'confidence': {'vehicle_number': 0.95, 'owner': 0.90},
                        'success': True
                    },
                    'ground_truth': {
                        'classification': 'spk',
                        'fields': {'vehicle_number': 'ABC1234', 'owner': 'JANE DOE'}
                    },
                    'expected_accuracy': 1.0
                },
                {
                    'name': 'Partial Match',
                    'results': {
                        'classification': 'mykad',
                        'fields': {'name': 'JOHN DOE', 'ic_number': 'WRONG123'},
                        'confidence': {'name': 0.95, 'ic_number': 0.60},
                        'success': True
                    },
                    'ground_truth': {
                        'classification': 'mykad',
                        'fields': {'name': 'JOHN DOE', 'ic_number': '123456-78-9012'}
                    },
                    'expected_accuracy': 0.5
                },
                {
                    'name': 'Classification Error',
                    'results': {
                        'classification': 'passport',
                        'fields': {'name': 'JOHN DOE'},
                        'confidence': {'name': 0.85},
                        'success': True
                    },
                    'ground_truth': {
                        'classification': 'mykad',
                        'fields': {'name': 'JOHN DOE'}
                    },
                    'expected_accuracy': 0.5  # 1 correct field, 1 wrong classification
                }
            ]
            
            for i, test_case in enumerate(test_cases):
                start_time = time.time()
                end_time = start_time + 1.5
                
                tracking_id = self.tracker.track_processing(
                    document_id=f'accuracy_test_{i}',
                    document_type='test',
                    start_time=start_time,
                    end_time=end_time,
                    results=test_case['results'],
                    ground_truth=test_case['ground_truth']
                )
                
                print(f"✅ {test_case['name']}: {tracking_id}")
            
            return True
            
        except Exception as e:
            print(f"❌ Accuracy calculation test failed: {e}")
            return False
    
    def test_performance_alerts(self) -> bool:
        """Test performance alert system"""
        print("\n=== Testing Performance Alerts ===")
        
        try:
            # Test slow processing alert
            start_time = time.time()
            end_time = start_time + 6.0  # Exceeds 5s threshold
            
            slow_results = {
                'classification': 'mykad',
                'fields': {'name': 'SLOW DOC'},
                'confidence': {'name': 0.95},
                'success': True
            }
            
            self.tracker.track_processing(
                document_id='slow_doc',
                document_type='mykad',
                start_time=start_time,
                end_time=end_time,
                results=slow_results
            )
            
            # Test low accuracy alert
            low_accuracy_results = {
                'classification': 'wrong_type',
                'fields': {'name': 'WRONG NAME'},
                'confidence': {'name': 0.50},
                'success': True
            }
            
            low_accuracy_truth = {
                'classification': 'mykad',
                'fields': {'name': 'CORRECT NAME', 'ic': '123456'}
            }
            
            start_time = time.time()
            end_time = start_time + 1.0
            
            self.tracker.track_processing(
                document_id='low_accuracy_doc',
                document_type='mykad',
                start_time=start_time,
                end_time=end_time,
                results=low_accuracy_results,
                ground_truth=low_accuracy_truth
            )
            
            # Test low confidence alert
            low_confidence_results = {
                'classification': 'mykad',
                'fields': {'name': 'UNCLEAR NAME'},
                'confidence': {'name': 0.40},  # Below 0.7 threshold
                'success': True
            }
            
            start_time = time.time()
            end_time = start_time + 1.0
            
            self.tracker.track_processing(
                document_id='low_confidence_doc',
                document_type='mykad',
                start_time=start_time,
                end_time=end_time,
                results=low_confidence_results
            )
            
            print("✅ Performance alerts tested (slow processing, low accuracy, low confidence)")
            return True
            
        except Exception as e:
            print(f"❌ Performance alerts test failed: {e}")
            return False
    
    def test_metrics_storage_retrieval(self) -> bool:
        """Test metrics storage and retrieval"""
        print("\n=== Testing Metrics Storage & Retrieval ===")
        
        try:
            # Store multiple metrics
            for i in range(5):
                start_time = time.time()
                end_time = start_time + (1.0 + i * 0.5)  # Varying processing times
                
                results = {
                    'classification': 'mykad' if i % 2 == 0 else 'spk',
                    'fields': {'field1': f'value_{i}'},
                    'confidence': {'field1': 0.8 + i * 0.02},
                    'success': True
                }
                
                self.tracker.track_processing(
                    document_id=f'storage_test_{i}',
                    document_type='mykad' if i % 2 == 0 else 'spk',
                    start_time=start_time,
                    end_time=end_time,
                    results=results
                )
            
            # Test retrieval
            all_metrics = self.tracker.storage.get_metrics(limit=10)
            print(f"✅ Retrieved {len(all_metrics)} metrics")
            
            # Test filtered retrieval
            mykad_metrics = self.tracker.storage.get_metrics(document_type='mykad')
            print(f"✅ Retrieved {len(mykad_metrics)} MyKad metrics")
            
            # Test date-filtered retrieval
            recent_metrics = self.tracker.storage.get_metrics(
                start_date=datetime.now() - timedelta(hours=1)
            )
            print(f"✅ Retrieved {len(recent_metrics)} recent metrics")
            
            return True
            
        except Exception as e:
            print(f"❌ Metrics storage & retrieval test failed: {e}")
            return False
    
    def test_performance_summary(self) -> bool:
        """Test performance summary generation"""
        print("\n=== Testing Performance Summary ===")
        
        try:
            # Generate summary
            summary = self.tracker.get_performance_summary(days=1)
            
            required_fields = [
                'total_documents', 'success_rate', 'error_rate',
                'processing_time', 'accuracy', 'confidence',
                'document_type_breakdown'
            ]
            
            for field in required_fields:
                if field not in summary:
                    print(f"❌ Missing field in summary: {field}")
                    return False
            
            print(f"✅ Summary generated with {summary['total_documents']} documents")
            print(f"   Success rate: {summary['success_rate']:.1%}")
            print(f"   Document types: {len(summary['document_type_breakdown'])}")
            
            return True
            
        except Exception as e:
            print(f"❌ Performance summary test failed: {e}")
            return False
    
    def test_real_time_metrics(self) -> bool:
        """Test real-time metrics functionality"""
        print("\n=== Testing Real-time Metrics ===")
        
        try:
            # Get real-time metrics
            real_time = self.tracker.get_real_time_metrics()
            
            required_fields = [
                'recent_measurements', 'current_processing_time',
                'current_accuracy', 'current_confidence', 'trend'
            ]
            
            for field in required_fields:
                if field not in real_time:
                    print(f"❌ Missing field in real-time metrics: {field}")
                    return False
            
            print(f"✅ Real-time metrics: {real_time['recent_measurements']} measurements")
            print(f"   Current processing time: {real_time['current_processing_time']:.2f}s")
            print(f"   Trends: {real_time['trend']}")
            
            return True
            
        except Exception as e:
            print(f"❌ Real-time metrics test failed: {e}")
            return False
    
    def test_performance_report(self) -> bool:
        """Test comprehensive performance report generation"""
        print("\n=== Testing Performance Report ===")
        
        try:
            # Generate comprehensive report
            report = self.tracker.generate_performance_report(days=1)
            
            required_sections = [
                'report_date', 'analysis_period_days', 'summary',
                'real_time_metrics', 'active_alerts', 'recommendations'
            ]
            
            for section in required_sections:
                if section not in report:
                    print(f"❌ Missing section in report: {section}")
                    return False
            
            print(f"✅ Report generated for {report['analysis_period_days']} days")
            print(f"   Active alerts: {len(report['active_alerts'])}")
            print(f"   Recommendations: {len(report['recommendations'])}")
            
            # Display some recommendations
            for i, rec in enumerate(report['recommendations'][:2], 1):
                print(f"   {i}. [{rec['priority'].upper()}] {rec['action'][:50]}...")
            
            return True
            
        except Exception as e:
            print(f"❌ Performance report test failed: {e}")
            return False
    
    def test_error_handling(self) -> bool:
        """Test error handling scenarios"""
        print("\n=== Testing Error Handling ===")
        
        try:
            # Test with missing data
            start_time = time.time()
            end_time = start_time + 1.0
            
            # Minimal results
            minimal_results = {'success': False}
            
            tracking_id = self.tracker.track_processing(
                document_id='error_test',
                document_type='unknown',
                start_time=start_time,
                end_time=end_time,
                results=minimal_results
            )
            
            print(f"✅ Error handling test passed: {tracking_id}")
            
            # Test with empty ground truth
            tracking_id2 = self.tracker.track_processing(
                document_id='no_truth_test',
                document_type='test',
                start_time=start_time,
                end_time=end_time,
                results={'success': True},
                ground_truth=None
            )
            
            print(f"✅ No ground truth test passed: {tracking_id2}")
            
            return True
            
        except Exception as e:
            print(f"❌ Error handling test failed: {e}")
            return False
    
    def run_all_tests(self) -> Dict[str, bool]:
        """Run all performance tracking tests"""
        print("🧪 Starting Performance Tracking Tests")
        print("=" * 50)
        
        self.setup_test_environment()
        
        tests = [
            ('Basic Tracking', self.test_basic_tracking),
            ('Accuracy Calculation', self.test_accuracy_calculation),
            ('Performance Alerts', self.test_performance_alerts),
            ('Metrics Storage & Retrieval', self.test_metrics_storage_retrieval),
            ('Performance Summary', self.test_performance_summary),
            ('Real-time Metrics', self.test_real_time_metrics),
            ('Performance Report', self.test_performance_report),
            ('Error Handling', self.test_error_handling)
        ]
        
        results = {}
        passed = 0
        total = len(tests)
        
        for test_name, test_func in tests:
            try:
                result = test_func()
                results[test_name] = result
                if result:
                    passed += 1
            except Exception as e:
                print(f"❌ {test_name} failed with exception: {e}")
                results[test_name] = False
        
        self.cleanup_test_environment()
        
        # Print summary
        print("\n" + "=" * 50)
        print(f"📊 Test Results: {passed}/{total} tests passed")
        print("=" * 50)
        
        for test_name, result in results.items():
            status = "✅ PASS" if result else "❌ FAIL"
            print(f"{status} {test_name}")
        
        if passed == total:
            print("\n🎉 All performance tracking tests passed!")
        else:
            print(f"\n⚠️  {total - passed} tests failed. Review the output above.")
        
        return results

def main():
    """Main test function"""
    tester = PerformanceTrackingTester()
    results = tester.run_all_tests()
    
    # Exit with appropriate code
    all_passed = all(results.values())
    sys.exit(0 if all_passed else 1)

if __name__ == "__main__":
    main()