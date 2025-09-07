#!/usr/bin/env python3
"""
Performance Testing Module

Comprehensive performance testing framework for the Malaysian document parser.
"""

from .test_performance import (
    PerformanceMetrics,
    LoadTestConfig,
    StressTestConfig,
    PerformanceMonitor,
    PerformanceTester
)

from .performance_config import (
    PerformanceThresholds,
    LoadTestScenarios,
    StressTestScenarios,
    PerformanceConfig,
    get_performance_config,
    get_thresholds,
    get_scenario,
    validate_performance
)

__all__ = [
    # Core classes
    'PerformanceMetrics',
    'LoadTestConfig', 
    'StressTestConfig',
    'PerformanceMonitor',
    'PerformanceTester',
    
    # Configuration classes
    'PerformanceThresholds',
    'LoadTestScenarios',
    'StressTestScenarios', 
    'PerformanceConfig',
    
    # Utility functions
    'get_performance_config',
    'get_thresholds',
    'get_scenario',
    'validate_performance'
]

__version__ = '1.0.0'
__author__ = 'Malaysian Document Parser Team'
__description__ = 'Performance testing framework for document processing pipeline'