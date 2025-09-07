#!/usr/bin/env python3
"""
Performance Testing Configuration

Configuration settings for performance testing of the Malaysian document parser.
"""

import os
from dataclasses import dataclass
from typing import Dict, Any, Optional


@dataclass
class PerformanceThresholds:
    """Performance thresholds for different test scenarios."""
    # Response time thresholds (seconds)
    max_response_time_single: float = 2.0
    max_response_time_avg: float = 1.5
    max_response_time_p95: float = 3.0
    
    # Throughput thresholds (documents per second)
    min_throughput_single_user: float = 1.0
    min_throughput_multi_user: float = 5.0
    target_throughput: float = 10.0
    
    # Error rate thresholds (percentage)
    max_error_rate_normal: float = 0.05  # 5%
    max_error_rate_stress: float = 0.10  # 10%
    
    # Resource usage thresholds
    max_memory_usage_mb: float = 1024  # 1GB
    max_cpu_usage_percent: float = 80.0
    max_memory_growth_mb: float = 100  # 100MB growth during test
    
    # Concurrency thresholds
    min_concurrent_users: int = 10
    target_concurrent_users: int = 50
    max_concurrent_users: int = 100


@dataclass
class LoadTestScenarios:
    """Predefined load test scenarios."""
    
    # Light load scenario
    light_load = {
        'concurrent_users': 5,
        'duration_seconds': 60,
        'ramp_up_seconds': 10,
        'description': 'Light load with 5 concurrent users'
    }
    
    # Normal load scenario
    normal_load = {
        'concurrent_users': 20,
        'duration_seconds': 300,  # 5 minutes
        'ramp_up_seconds': 30,
        'description': 'Normal load with 20 concurrent users'
    }
    
    # Heavy load scenario
    heavy_load = {
        'concurrent_users': 50,
        'duration_seconds': 600,  # 10 minutes
        'ramp_up_seconds': 60,
        'description': 'Heavy load with 50 concurrent users'
    }
    
    # Peak load scenario
    peak_load = {
        'concurrent_users': 100,
        'duration_seconds': 300,  # 5 minutes
        'ramp_up_seconds': 120,
        'description': 'Peak load with 100 concurrent users'
    }
    
    # Endurance test scenario
    endurance_test = {
        'concurrent_users': 30,
        'duration_seconds': 3600,  # 1 hour
        'ramp_up_seconds': 60,
        'description': 'Endurance test with 30 users for 1 hour'
    }


@dataclass
class StressTestScenarios:
    """Predefined stress test scenarios."""
    
    # Gradual stress test
    gradual_stress = {
        'max_concurrent_users': 200,
        'step_size': 10,
        'step_duration': 60,
        'description': 'Gradual stress test increasing by 10 users every minute'
    }
    
    # Rapid stress test
    rapid_stress = {
        'max_concurrent_users': 150,
        'step_size': 25,
        'step_duration': 30,
        'description': 'Rapid stress test increasing by 25 users every 30 seconds'
    }
    
    # Spike test
    spike_test = {
        'baseline_users': 20,
        'spike_users': 100,
        'spike_duration': 120,
        'description': 'Spike test from 20 to 100 users for 2 minutes'
    }


class PerformanceConfig:
    """Main performance testing configuration."""
    
    def __init__(self):
        self.thresholds = PerformanceThresholds()
        self.load_scenarios = LoadTestScenarios()
        self.stress_scenarios = StressTestScenarios()
        
        # Environment-specific settings
        self.environment = os.getenv('TEST_ENVIRONMENT', 'development')
        self.debug_mode = os.getenv('DEBUG', 'false').lower() == 'true'
        
        # Test data settings
        self.test_data_dir = os.getenv('TEST_DATA_DIR', 'tests/data')
        self.test_image_count = int(os.getenv('TEST_IMAGE_COUNT', '10'))
        
        # Monitoring settings
        self.monitoring_interval = float(os.getenv('MONITORING_INTERVAL', '0.1'))
        self.enable_detailed_logging = os.getenv('DETAILED_LOGGING', 'false').lower() == 'true'
        
        # Report settings
        self.reports_dir = os.getenv('REPORTS_DIR', 'tests/reports')
        self.generate_charts = os.getenv('GENERATE_CHARTS', 'true').lower() == 'true'
        
        # Adjust thresholds based on environment
        self._adjust_for_environment()
    
    def _adjust_for_environment(self):
        """Adjust performance thresholds based on environment."""
        if self.environment == 'ci':
            # More lenient thresholds for CI environment
            self.thresholds.max_response_time_single = 3.0
            self.thresholds.max_response_time_avg = 2.5
            self.thresholds.min_throughput_single_user = 0.5
            self.thresholds.min_throughput_multi_user = 2.0
            self.thresholds.max_error_rate_normal = 0.10
            self.thresholds.max_memory_usage_mb = 2048
            
        elif self.environment == 'production':
            # Stricter thresholds for production-like testing
            self.thresholds.max_response_time_single = 1.0
            self.thresholds.max_response_time_avg = 0.8
            self.thresholds.min_throughput_single_user = 2.0
            self.thresholds.min_throughput_multi_user = 10.0
            self.thresholds.max_error_rate_normal = 0.01
            self.thresholds.max_memory_usage_mb = 512
    
    def get_scenario_config(self, scenario_name: str) -> Optional[Dict[str, Any]]:
        """Get configuration for a specific test scenario."""
        # Load test scenarios
        load_scenarios = {
            'light_load': self.load_scenarios.light_load,
            'normal_load': self.load_scenarios.normal_load,
            'heavy_load': self.load_scenarios.heavy_load,
            'peak_load': self.load_scenarios.peak_load,
            'endurance_test': self.load_scenarios.endurance_test
        }
        
        # Stress test scenarios
        stress_scenarios = {
            'gradual_stress': self.stress_scenarios.gradual_stress,
            'rapid_stress': self.stress_scenarios.rapid_stress,
            'spike_test': self.stress_scenarios.spike_test
        }
        
        all_scenarios = {**load_scenarios, **stress_scenarios}
        return all_scenarios.get(scenario_name)
    
    def get_test_matrix(self) -> Dict[str, Any]:
        """Get test matrix for comprehensive performance testing."""
        return {
            'quick_tests': [
                'light_load'
            ],
            'standard_tests': [
                'light_load',
                'normal_load',
                'gradual_stress'
            ],
            'comprehensive_tests': [
                'light_load',
                'normal_load',
                'heavy_load',
                'gradual_stress',
                'rapid_stress'
            ],
            'full_suite': [
                'light_load',
                'normal_load',
                'heavy_load',
                'peak_load',
                'endurance_test',
                'gradual_stress',
                'rapid_stress',
                'spike_test'
            ]
        }
    
    def validate_thresholds(self, metrics: Dict[str, float]) -> Dict[str, bool]:
        """Validate performance metrics against thresholds."""
        validation_results = {}
        
        # Response time validation
        if 'avg_response_time' in metrics:
            validation_results['response_time_ok'] = (
                metrics['avg_response_time'] <= self.thresholds.max_response_time_avg
            )
        
        # Throughput validation
        if 'throughput' in metrics:
            validation_results['throughput_ok'] = (
                metrics['throughput'] >= self.thresholds.min_throughput_multi_user
            )
        
        # Error rate validation
        if 'error_rate' in metrics:
            validation_results['error_rate_ok'] = (
                metrics['error_rate'] <= self.thresholds.max_error_rate_normal
            )
        
        # Memory usage validation
        if 'peak_memory_mb' in metrics:
            validation_results['memory_ok'] = (
                metrics['peak_memory_mb'] <= self.thresholds.max_memory_usage_mb
            )
        
        # CPU usage validation
        if 'avg_cpu_percent' in metrics:
            validation_results['cpu_ok'] = (
                metrics['avg_cpu_percent'] <= self.thresholds.max_cpu_usage_percent
            )
        
        return validation_results
    
    def get_performance_report_config(self) -> Dict[str, Any]:
        """Get configuration for performance reports."""
        return {
            'include_charts': self.generate_charts,
            'detailed_metrics': self.enable_detailed_logging,
            'output_formats': ['json', 'html', 'csv'],
            'chart_types': [
                'response_time_distribution',
                'throughput_over_time',
                'resource_usage',
                'error_rate_trend'
            ],
            'thresholds': {
                'response_time': self.thresholds.max_response_time_avg,
                'throughput': self.thresholds.min_throughput_multi_user,
                'error_rate': self.thresholds.max_error_rate_normal,
                'memory_usage': self.thresholds.max_memory_usage_mb
            }
        }


# Global configuration instance
performance_config = PerformanceConfig()


# Convenience functions
def get_performance_config() -> PerformanceConfig:
    """Get the global performance configuration instance."""
    return performance_config


def get_thresholds() -> PerformanceThresholds:
    """Get performance thresholds."""
    return performance_config.thresholds


def get_scenario(scenario_name: str) -> Optional[Dict[str, Any]]:
    """Get a specific test scenario configuration."""
    return performance_config.get_scenario_config(scenario_name)


def validate_performance(metrics: Dict[str, float]) -> Dict[str, bool]:
    """Validate performance metrics against configured thresholds."""
    return performance_config.validate_thresholds(metrics)


if __name__ == '__main__':
    # Example usage
    config = get_performance_config()
    
    print("Performance Configuration:")
    print(f"Environment: {config.environment}")
    print(f"Max response time: {config.thresholds.max_response_time_avg}s")
    print(f"Min throughput: {config.thresholds.min_throughput_multi_user} docs/sec")
    
    print("\nAvailable scenarios:")
    test_matrix = config.get_test_matrix()
    for suite_name, scenarios in test_matrix.items():
        print(f"  {suite_name}: {scenarios}")
    
    print("\nExample scenario:")
    normal_load = get_scenario('normal_load')
    if normal_load:
        print(f"  {normal_load['description']}")
        print(f"  Users: {normal_load['concurrent_users']}")
        print(f"  Duration: {normal_load['duration_seconds']}s")