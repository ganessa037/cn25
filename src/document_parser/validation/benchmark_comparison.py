"""Benchmark Comparison System for Document Processing

This module provides a comprehensive framework for benchmarking document processing
performance against existing solutions and tracking improvements over time.
Follows the autocorrect model's organizational patterns.
"""

import json
import logging
import time
import statistics
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Union, Tuple, Callable
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from collections import defaultdict
import uuid

import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import sqlite3

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class BenchmarkType(Enum):
    """Types of benchmarks"""
    ACCURACY = "accuracy"
    SPEED = "speed"
    MEMORY = "memory"
    COST = "cost"
    RELIABILITY = "reliability"
    SCALABILITY = "scalability"

class ComparisonMethod(Enum):
    """Methods for comparison"""
    ABSOLUTE = "absolute"
    RELATIVE = "relative"
    STATISTICAL = "statistical"
    TREND_ANALYSIS = "trend_analysis"

class SolutionType(Enum):
    """Types of solutions being compared"""
    INTERNAL_MODEL = "internal_model"
    COMMERCIAL_API = "commercial_api"
    OPEN_SOURCE = "open_source"
    BASELINE = "baseline"
    HUMAN_PERFORMANCE = "human_performance"

@dataclass
class BenchmarkMetric:
    """Individual benchmark metric"""
    
    metric_name: str
    value: float
    unit: str
    timestamp: datetime = field(default_factory=datetime.now)
    
    # Context information
    test_set_size: Optional[int] = None
    document_type: Optional[str] = None
    model_version: Optional[str] = None
    
    # Statistical information
    confidence_interval: Optional[Tuple[float, float]] = None
    standard_deviation: Optional[float] = None
    sample_variance: Optional[float] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return {
            'metric_name': self.metric_name,
            'value': self.value,
            'unit': self.unit,
            'timestamp': self.timestamp.isoformat(),
            'test_set_size': self.test_set_size,
            'document_type': self.document_type,
            'model_version': self.model_version,
            'confidence_interval': self.confidence_interval,
            'standard_deviation': self.standard_deviation,
            'sample_variance': self.sample_variance
        }

@dataclass
class SolutionBenchmark:
    """Benchmark results for a solution"""
    
    solution_id: str
    solution_name: str
    solution_type: SolutionType
    version: str
    
    # Performance metrics
    metrics: Dict[str, BenchmarkMetric] = field(default_factory=dict)
    
    # Test information
    test_date: datetime = field(default_factory=datetime.now)
    test_duration: Optional[float] = None
    test_environment: Dict[str, Any] = field(default_factory=dict)
    
    # Cost and resource information
    cost_per_document: Optional[float] = None
    memory_usage_mb: Optional[float] = None
    cpu_usage_percent: Optional[float] = None
    
    # Quality metrics
    error_rate: Optional[float] = None
    failure_rate: Optional[float] = None
    
    def add_metric(self, metric: BenchmarkMetric):
        """Add a metric to the benchmark"""
        self.metrics[metric.metric_name] = metric
    
    def get_metric_value(self, metric_name: str) -> Optional[float]:
        """Get metric value by name"""
        metric = self.metrics.get(metric_name)
        return metric.value if metric else None
    
    def get_overall_score(self, weights: Dict[str, float] = None) -> float:
        """Calculate overall weighted score"""
        if not weights:
            weights = {
                'accuracy': 0.4,
                'speed': 0.2,
                'cost_efficiency': 0.2,
                'reliability': 0.2
            }
        
        score = 0.0
        total_weight = 0.0
        
        for metric_name, weight in weights.items():
            value = self.get_metric_value(metric_name)
            if value is not None:
                score += value * weight
                total_weight += weight
        
        return score / total_weight if total_weight > 0 else 0.0
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return {
            'solution_id': self.solution_id,
            'solution_name': self.solution_name,
            'solution_type': self.solution_type.value,
            'version': self.version,
            'metrics': {name: metric.to_dict() for name, metric in self.metrics.items()},
            'test_date': self.test_date.isoformat(),
            'test_duration': self.test_duration,
            'test_environment': self.test_environment,
            'cost_per_document': self.cost_per_document,
            'memory_usage_mb': self.memory_usage_mb,
            'cpu_usage_percent': self.cpu_usage_percent,
            'error_rate': self.error_rate,
            'failure_rate': self.failure_rate
        }

@dataclass
class ComparisonResult:
    """Result of benchmark comparison"""
    
    comparison_id: str
    comparison_date: datetime
    
    # Solutions being compared
    baseline_solution: str
    compared_solutions: List[str]
    
    # Comparison results
    metric_comparisons: Dict[str, Dict[str, float]] = field(default_factory=dict)
    statistical_significance: Dict[str, bool] = field(default_factory=dict)
    
    # Rankings
    overall_ranking: List[Tuple[str, float]] = field(default_factory=list)
    metric_rankings: Dict[str, List[Tuple[str, float]]] = field(default_factory=dict)
    
    # Insights
    best_performer: Optional[str] = None
    worst_performer: Optional[str] = None
    improvement_recommendations: List[str] = field(default_factory=list)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return {
            'comparison_id': self.comparison_id,
            'comparison_date': self.comparison_date.isoformat(),
            'baseline_solution': self.baseline_solution,
            'compared_solutions': self.compared_solutions,
            'metric_comparisons': self.metric_comparisons,
            'statistical_significance': self.statistical_significance,
            'overall_ranking': self.overall_ranking,
            'metric_rankings': self.metric_rankings,
            'best_performer': self.best_performer,
            'worst_performer': self.worst_performer,
            'improvement_recommendations': self.improvement_recommendations
        }

class BenchmarkDatabase:
    """Database for storing benchmark results"""
    
    def __init__(self, db_path: str = "benchmark_results.db"):
        self.db_path = db_path
        self.init_database()
    
    def init_database(self):
        """Initialize database tables"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Solutions table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS solutions (
                solution_id TEXT PRIMARY KEY,
                solution_name TEXT NOT NULL,
                solution_type TEXT NOT NULL,
                version TEXT NOT NULL,
                created_at TIMESTAMP,
                description TEXT
            )
        """)
        
        # Benchmarks table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS benchmarks (
                benchmark_id TEXT PRIMARY KEY,
                solution_id TEXT NOT NULL,
                test_date TIMESTAMP,
                test_duration REAL,
                test_environment TEXT,
                cost_per_document REAL,
                memory_usage_mb REAL,
                cpu_usage_percent REAL,
                error_rate REAL,
                failure_rate REAL,
                FOREIGN KEY (solution_id) REFERENCES solutions (solution_id)
            )
        """)
        
        # Metrics table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS metrics (
                metric_id TEXT PRIMARY KEY,
                benchmark_id TEXT NOT NULL,
                metric_name TEXT NOT NULL,
                value REAL NOT NULL,
                unit TEXT,
                timestamp TIMESTAMP,
                test_set_size INTEGER,
                document_type TEXT,
                model_version TEXT,
                confidence_interval_lower REAL,
                confidence_interval_upper REAL,
                standard_deviation REAL,
                sample_variance REAL,
                FOREIGN KEY (benchmark_id) REFERENCES benchmarks (benchmark_id)
            )
        """)
        
        # Comparisons table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS comparisons (
                comparison_id TEXT PRIMARY KEY,
                comparison_date TIMESTAMP,
                baseline_solution TEXT,
                compared_solutions TEXT,
                results TEXT,
                best_performer TEXT,
                worst_performer TEXT
            )
        """)
        
        conn.commit()
        conn.close()
        
        logger.info(f"Benchmark database initialized at {self.db_path}")
    
    def save_benchmark(self, benchmark: SolutionBenchmark) -> bool:
        """Save benchmark to database"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            # Save solution if not exists
            cursor.execute("""
                INSERT OR IGNORE INTO solutions (
                    solution_id, solution_name, solution_type, version, created_at
                ) VALUES (?, ?, ?, ?, ?)
            """, (
                benchmark.solution_id,
                benchmark.solution_name,
                benchmark.solution_type.value,
                benchmark.version,
                datetime.now()
            ))
            
            # Save benchmark
            benchmark_id = str(uuid.uuid4())
            cursor.execute("""
                INSERT INTO benchmarks (
                    benchmark_id, solution_id, test_date, test_duration,
                    test_environment, cost_per_document, memory_usage_mb,
                    cpu_usage_percent, error_rate, failure_rate
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                benchmark_id,
                benchmark.solution_id,
                benchmark.test_date,
                benchmark.test_duration,
                json.dumps(benchmark.test_environment),
                benchmark.cost_per_document,
                benchmark.memory_usage_mb,
                benchmark.cpu_usage_percent,
                benchmark.error_rate,
                benchmark.failure_rate
            ))
            
            # Save metrics
            for metric in benchmark.metrics.values():
                metric_id = str(uuid.uuid4())
                
                ci_lower = metric.confidence_interval[0] if metric.confidence_interval else None
                ci_upper = metric.confidence_interval[1] if metric.confidence_interval else None
                
                cursor.execute("""
                    INSERT INTO metrics (
                        metric_id, benchmark_id, metric_name, value, unit,
                        timestamp, test_set_size, document_type, model_version,
                        confidence_interval_lower, confidence_interval_upper,
                        standard_deviation, sample_variance
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    metric_id,
                    benchmark_id,
                    metric.metric_name,
                    metric.value,
                    metric.unit,
                    metric.timestamp,
                    metric.test_set_size,
                    metric.document_type,
                    metric.model_version,
                    ci_lower,
                    ci_upper,
                    metric.standard_deviation,
                    metric.sample_variance
                ))
            
            conn.commit()
            conn.close()
            return True
            
        except Exception as e:
            logger.error(f"Error saving benchmark: {e}")
            return False
    
    def get_benchmarks(self, solution_id: str = None, 
                      start_date: datetime = None,
                      end_date: datetime = None) -> List[SolutionBenchmark]:
        """Get benchmarks from database"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            query = """
                SELECT b.*, s.solution_name, s.solution_type, s.version
                FROM benchmarks b
                JOIN solutions s ON b.solution_id = s.solution_id
                WHERE 1=1
            """
            params = []
            
            if solution_id:
                query += " AND b.solution_id = ?"
                params.append(solution_id)
            
            if start_date:
                query += " AND b.test_date >= ?"
                params.append(start_date)
            
            if end_date:
                query += " AND b.test_date <= ?"
                params.append(end_date)
            
            query += " ORDER BY b.test_date DESC"
            
            cursor.execute(query, params)
            benchmark_rows = cursor.fetchall()
            
            benchmarks = []
            for row in benchmark_rows:
                benchmark = SolutionBenchmark(
                    solution_id=row[1],
                    solution_name=row[11],
                    solution_type=SolutionType(row[12]),
                    version=row[13],
                    test_date=datetime.fromisoformat(row[2]) if row[2] else datetime.now(),
                    test_duration=row[3],
                    test_environment=json.loads(row[4]) if row[4] else {},
                    cost_per_document=row[5],
                    memory_usage_mb=row[6],
                    cpu_usage_percent=row[7],
                    error_rate=row[8],
                    failure_rate=row[9]
                )
                
                # Get metrics for this benchmark
                cursor.execute("""
                    SELECT * FROM metrics WHERE benchmark_id = ?
                """, (row[0],))
                
                metric_rows = cursor.fetchall()
                for metric_row in metric_rows:
                    ci = None
                    if metric_row[9] is not None and metric_row[10] is not None:
                        ci = (metric_row[9], metric_row[10])
                    
                    metric = BenchmarkMetric(
                        metric_name=metric_row[2],
                        value=metric_row[3],
                        unit=metric_row[4] or "",
                        timestamp=datetime.fromisoformat(metric_row[5]) if metric_row[5] else datetime.now(),
                        test_set_size=metric_row[6],
                        document_type=metric_row[7],
                        model_version=metric_row[8],
                        confidence_interval=ci,
                        standard_deviation=metric_row[11],
                        sample_variance=metric_row[12]
                    )
                    
                    benchmark.add_metric(metric)
                
                benchmarks.append(benchmark)
            
            conn.close()
            return benchmarks
            
        except Exception as e:
            logger.error(f"Error getting benchmarks: {e}")
            return []
    
    def save_comparison(self, comparison: ComparisonResult) -> bool:
        """Save comparison result to database"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            cursor.execute("""
                INSERT INTO comparisons (
                    comparison_id, comparison_date, baseline_solution,
                    compared_solutions, results, best_performer, worst_performer
                ) VALUES (?, ?, ?, ?, ?, ?, ?)
            """, (
                comparison.comparison_id,
                comparison.comparison_date,
                comparison.baseline_solution,
                json.dumps(comparison.compared_solutions),
                json.dumps(comparison.to_dict()),
                comparison.best_performer,
                comparison.worst_performer
            ))
            
            conn.commit()
            conn.close()
            return True
            
        except Exception as e:
            logger.error(f"Error saving comparison: {e}")
            return False

class PerformanceTester:
    """Performance testing utilities"""
    
    @staticmethod
    def measure_processing_time(func: Callable, *args, **kwargs) -> Tuple[Any, float]:
        """Measure function execution time"""
        start_time = time.time()
        result = func(*args, **kwargs)
        end_time = time.time()
        
        processing_time = end_time - start_time
        return result, processing_time
    
    @staticmethod
    def measure_memory_usage(func: Callable, *args, **kwargs) -> Tuple[Any, float]:
        """Measure memory usage during function execution"""
        import psutil
        import os
        
        process = psutil.Process(os.getpid())
        
        # Get initial memory
        initial_memory = process.memory_info().rss / 1024 / 1024  # MB
        
        # Execute function
        result = func(*args, **kwargs)
        
        # Get final memory
        final_memory = process.memory_info().rss / 1024 / 1024  # MB
        
        memory_used = final_memory - initial_memory
        return result, memory_used
    
    @staticmethod
    def run_stress_test(func: Callable, test_data: List[Any], 
                       concurrent_requests: int = 10) -> Dict[str, float]:
        """Run stress test with concurrent requests"""
        import concurrent.futures
        import threading
        
        results = []
        errors = 0
        
        def run_single_test(data):
            try:
                start_time = time.time()
                func(data)
                end_time = time.time()
                return end_time - start_time
            except Exception as e:
                logger.error(f"Error in stress test: {e}")
                return None
        
        # Run concurrent tests
        with concurrent.futures.ThreadPoolExecutor(max_workers=concurrent_requests) as executor:
            futures = [executor.submit(run_single_test, data) for data in test_data]
            
            for future in concurrent.futures.as_completed(futures):
                result = future.result()
                if result is not None:
                    results.append(result)
                else:
                    errors += 1
        
        if not results:
            return {'error': 'All tests failed'}
        
        return {
            'total_requests': len(test_data),
            'successful_requests': len(results),
            'failed_requests': errors,
            'success_rate': len(results) / len(test_data),
            'avg_response_time': statistics.mean(results),
            'min_response_time': min(results),
            'max_response_time': max(results),
            'median_response_time': statistics.median(results),
            'std_response_time': statistics.stdev(results) if len(results) > 1 else 0
        }

class BenchmarkComparator:
    """Main benchmark comparison system"""
    
    def __init__(self, db_path: str = "benchmark_results.db"):
        self.db = BenchmarkDatabase(db_path)
        self.tester = PerformanceTester()
        
        logger.info("Benchmark comparator initialized")
    
    def run_benchmark(self, solution_func: Callable, solution_name: str,
                     solution_type: SolutionType, version: str,
                     test_data: List[Any], document_type: str = None) -> SolutionBenchmark:
        """Run comprehensive benchmark for a solution"""
        solution_id = f"{solution_name}_{version}_{int(time.time())}"
        
        benchmark = SolutionBenchmark(
            solution_id=solution_id,
            solution_name=solution_name,
            solution_type=solution_type,
            version=version
        )
        
        logger.info(f"Running benchmark for {solution_name} v{version}")
        
        # Measure accuracy
        accuracy_results = []
        processing_times = []
        memory_usages = []
        
        for i, data in enumerate(test_data):
            try:
                # Measure processing time and memory
                result, proc_time = self.tester.measure_processing_time(solution_func, data)
                _, memory_used = self.tester.measure_memory_usage(solution_func, data)
                
                processing_times.append(proc_time)
                memory_usages.append(memory_used)
                
                # Calculate accuracy if ground truth available
                if hasattr(data, 'ground_truth') and hasattr(result, 'extracted_data'):
                    accuracy = self._calculate_accuracy(result.extracted_data, data.ground_truth)
                    accuracy_results.append(accuracy)
                
            except Exception as e:
                logger.error(f"Error processing test item {i}: {e}")
                continue
        
        # Calculate metrics
        if accuracy_results:
            accuracy_metric = BenchmarkMetric(
                metric_name="accuracy",
                value=statistics.mean(accuracy_results),
                unit="percentage",
                test_set_size=len(accuracy_results),
                document_type=document_type,
                standard_deviation=statistics.stdev(accuracy_results) if len(accuracy_results) > 1 else 0
            )
            benchmark.add_metric(accuracy_metric)
        
        if processing_times:
            speed_metric = BenchmarkMetric(
                metric_name="avg_processing_time",
                value=statistics.mean(processing_times),
                unit="seconds",
                test_set_size=len(processing_times),
                document_type=document_type,
                standard_deviation=statistics.stdev(processing_times) if len(processing_times) > 1 else 0
            )
            benchmark.add_metric(speed_metric)
            
            throughput_metric = BenchmarkMetric(
                metric_name="throughput",
                value=len(processing_times) / sum(processing_times),
                unit="documents_per_second",
                test_set_size=len(processing_times),
                document_type=document_type
            )
            benchmark.add_metric(throughput_metric)
        
        if memory_usages:
            memory_metric = BenchmarkMetric(
                metric_name="avg_memory_usage",
                value=statistics.mean(memory_usages),
                unit="MB",
                test_set_size=len(memory_usages),
                document_type=document_type,
                standard_deviation=statistics.stdev(memory_usages) if len(memory_usages) > 1 else 0
            )
            benchmark.add_metric(memory_metric)
        
        # Run stress test
        stress_results = self.tester.run_stress_test(solution_func, test_data[:10])  # Use subset for stress test
        
        if 'success_rate' in stress_results:
            reliability_metric = BenchmarkMetric(
                metric_name="reliability",
                value=stress_results['success_rate'] * 100,
                unit="percentage",
                test_set_size=stress_results['total_requests'],
                document_type=document_type
            )
            benchmark.add_metric(reliability_metric)
        
        # Set additional benchmark properties
        benchmark.test_duration = sum(processing_times) if processing_times else 0
        benchmark.memory_usage_mb = statistics.mean(memory_usages) if memory_usages else 0
        benchmark.error_rate = (1 - stress_results.get('success_rate', 1)) * 100
        
        # Save benchmark
        self.db.save_benchmark(benchmark)
        
        logger.info(f"Benchmark completed for {solution_name}")
        return benchmark
    
    def compare_solutions(self, solution_ids: List[str], 
                         baseline_id: str = None) -> ComparisonResult:
        """Compare multiple solutions"""
        comparison_id = str(uuid.uuid4())
        
        # Get benchmarks for all solutions
        all_benchmarks = []
        for solution_id in solution_ids:
            benchmarks = self.db.get_benchmarks(solution_id=solution_id)
            if benchmarks:
                all_benchmarks.append(benchmarks[-1])  # Get latest benchmark
        
        if len(all_benchmarks) < 2:
            raise ValueError("Need at least 2 solutions to compare")
        
        # Set baseline
        if baseline_id:
            baseline_benchmark = next((b for b in all_benchmarks if b.solution_id == baseline_id), None)
        else:
            baseline_benchmark = all_benchmarks[0]
        
        comparison = ComparisonResult(
            comparison_id=comparison_id,
            comparison_date=datetime.now(),
            baseline_solution=baseline_benchmark.solution_id,
            compared_solutions=[b.solution_id for b in all_benchmarks if b.solution_id != baseline_benchmark.solution_id]
        )
        
        # Compare metrics
        metric_names = set()
        for benchmark in all_benchmarks:
            metric_names.update(benchmark.metrics.keys())
        
        for metric_name in metric_names:
            comparison.metric_comparisons[metric_name] = {}
            values = []
            
            for benchmark in all_benchmarks:
                value = benchmark.get_metric_value(metric_name)
                if value is not None:
                    comparison.metric_comparisons[metric_name][benchmark.solution_id] = value
                    values.append(value)
            
            # Statistical significance test
            if len(values) >= 2:
                try:
                    _, p_value = stats.ttest_ind(values[:len(values)//2], values[len(values)//2:])
                    comparison.statistical_significance[metric_name] = p_value < 0.05
                except:
                    comparison.statistical_significance[metric_name] = False
        
        # Calculate rankings
        overall_scores = []
        for benchmark in all_benchmarks:
            score = benchmark.get_overall_score()
            overall_scores.append((benchmark.solution_id, score))
        
        comparison.overall_ranking = sorted(overall_scores, key=lambda x: x[1], reverse=True)
        
        # Set best and worst performers
        if comparison.overall_ranking:
            comparison.best_performer = comparison.overall_ranking[0][0]
            comparison.worst_performer = comparison.overall_ranking[-1][0]
        
        # Generate recommendations
        comparison.improvement_recommendations = self._generate_recommendations(all_benchmarks, comparison)
        
        # Save comparison
        self.db.save_comparison(comparison)
        
        logger.info(f"Comparison completed: {comparison_id}")
        return comparison
    
    def generate_report(self, comparison: ComparisonResult, 
                       output_path: str = None) -> str:
        """Generate comprehensive comparison report"""
        report_lines = []
        
        report_lines.append("# Benchmark Comparison Report")
        report_lines.append(f"**Comparison ID:** {comparison.comparison_id}")
        report_lines.append(f"**Date:** {comparison.comparison_date.strftime('%Y-%m-%d %H:%M:%S')}")
        report_lines.append(f"**Baseline:** {comparison.baseline_solution}")
        report_lines.append("")
        
        # Overall ranking
        report_lines.append("## Overall Performance Ranking")
        for i, (solution_id, score) in enumerate(comparison.overall_ranking, 1):
            report_lines.append(f"{i}. {solution_id}: {score:.3f}")
        report_lines.append("")
        
        # Metric comparisons
        report_lines.append("## Detailed Metric Comparisons")
        for metric_name, values in comparison.metric_comparisons.items():
            report_lines.append(f"### {metric_name.title()}")
            
            sorted_values = sorted(values.items(), key=lambda x: x[1], reverse=True)
            for solution_id, value in sorted_values:
                significance = "*" if comparison.statistical_significance.get(metric_name, False) else ""
                report_lines.append(f"- {solution_id}: {value:.3f}{significance}")
            
            report_lines.append("")
        
        # Best performer analysis
        if comparison.best_performer:
            report_lines.append(f"## Best Performer: {comparison.best_performer}")
            report_lines.append("This solution showed the highest overall performance across all metrics.")
            report_lines.append("")
        
        # Recommendations
        if comparison.improvement_recommendations:
            report_lines.append("## Improvement Recommendations")
            for i, recommendation in enumerate(comparison.improvement_recommendations, 1):
                report_lines.append(f"{i}. {recommendation}")
            report_lines.append("")
        
        # Statistical significance note
        report_lines.append("## Notes")
        report_lines.append("- Metrics marked with * show statistically significant differences")
        report_lines.append("- Rankings are based on weighted overall scores")
        
        report_content = "\n".join(report_lines)
        
        if output_path:
            with open(output_path, 'w') as f:
                f.write(report_content)
            logger.info(f"Report saved to {output_path}")
        
        return report_content
    
    def track_performance_trends(self, solution_id: str, 
                                days: int = 30) -> Dict[str, List[Tuple[datetime, float]]]:
        """Track performance trends over time"""
        end_date = datetime.now()
        start_date = end_date - timedelta(days=days)
        
        benchmarks = self.db.get_benchmarks(
            solution_id=solution_id,
            start_date=start_date,
            end_date=end_date
        )
        
        trends = defaultdict(list)
        
        for benchmark in benchmarks:
            for metric_name, metric in benchmark.metrics.items():
                trends[metric_name].append((benchmark.test_date, metric.value))
        
        # Sort by date
        for metric_name in trends:
            trends[metric_name].sort(key=lambda x: x[0])
        
        return dict(trends)
    
    def _calculate_accuracy(self, extracted_data: Dict[str, Any], 
                          ground_truth: Dict[str, Any]) -> float:
        """Calculate accuracy between extracted and ground truth data"""
        if not ground_truth:
            return 0.0
        
        correct = 0
        total = 0
        
        for field, true_value in ground_truth.items():
            total += 1
            extracted_value = extracted_data.get(field, "")
            
            # Normalize for comparison
            true_str = str(true_value).strip().lower()
            extracted_str = str(extracted_value).strip().lower()
            
            if true_str == extracted_str:
                correct += 1
        
        return (correct / total) * 100 if total > 0 else 0.0
    
    def _generate_recommendations(self, benchmarks: List[SolutionBenchmark],
                                 comparison: ComparisonResult) -> List[str]:
        """Generate improvement recommendations"""
        recommendations = []
        
        # Analyze performance gaps
        if comparison.overall_ranking:
            best_score = comparison.overall_ranking[0][1]
            worst_score = comparison.overall_ranking[-1][1]
            
            if best_score - worst_score > 0.2:
                recommendations.append(
                    f"Significant performance gap detected. Consider optimizing lower-performing solutions."
                )
        
        # Analyze specific metrics
        for metric_name, values in comparison.metric_comparisons.items():
            if len(values) >= 2:
                max_val = max(values.values())
                min_val = min(values.values())
                
                if metric_name == "accuracy" and max_val - min_val > 10:
                    recommendations.append(
                        f"Large accuracy gap in {metric_name}. Focus on improving data quality and model training."
                    )
                elif metric_name == "avg_processing_time" and max_val / min_val > 2:
                    recommendations.append(
                        f"Processing time varies significantly. Consider optimizing slower solutions."
                    )
        
        # Memory usage recommendations
        memory_values = comparison.metric_comparisons.get("avg_memory_usage", {})
        if memory_values and max(memory_values.values()) > 1000:  # > 1GB
            recommendations.append(
                "High memory usage detected. Consider memory optimization techniques."
            )
        
        return recommendations

def main():
    """Main function for standalone execution"""
    # Example usage of the benchmark comparison system
    
    # Initialize comparator
    comparator = BenchmarkComparator()
    
    print("\n=== Benchmark Comparison System Demo ===")
    
    # Mock solution functions for demonstration
    def mock_solution_a(data):
        """Mock solution A - fast but less accurate"""
        time.sleep(0.1)  # Simulate processing
        return {
            'extracted_data': {
                'ic_number': '123456-78-9012',
                'name': 'Ahmad Ali',
                'address': '123 Jalan Merdeka'
            },
            'confidence_scores': {'ic_number': 0.85, 'name': 0.80, 'address': 0.75}
        }
    
    def mock_solution_b(data):
        """Mock solution B - slower but more accurate"""
        time.sleep(0.3)  # Simulate processing
        return {
            'extracted_data': {
                'ic_number': '123456-78-9012',
                'name': 'Ahmad bin Ali',
                'address': '123 Jalan Merdeka, KL'
            },
            'confidence_scores': {'ic_number': 0.95, 'name': 0.92, 'address': 0.88}
        }
    
    # Create test data
    class MockTestData:
        def __init__(self, ground_truth):
            self.ground_truth = ground_truth
    
    test_data = [
        MockTestData({
            'ic_number': '123456-78-9012',
            'name': 'Ahmad bin Ali',
            'address': '123 Jalan Merdeka, Kuala Lumpur'
        })
        for _ in range(5)
    ]
    
    print("\n=== Running Benchmarks ===")
    
    # Run benchmarks
    benchmark_a = comparator.run_benchmark(
        solution_func=mock_solution_a,
        solution_name="FastOCR",
        solution_type=SolutionType.INTERNAL_MODEL,
        version="1.0",
        test_data=test_data,
        document_type="identity_card"
    )
    
    benchmark_b = comparator.run_benchmark(
        solution_func=mock_solution_b,
        solution_name="AccurateOCR",
        solution_type=SolutionType.INTERNAL_MODEL,
        version="1.0",
        test_data=test_data,
        document_type="identity_card"
    )
    
    print(f"Benchmark A completed: {benchmark_a.solution_id}")
    print(f"Benchmark B completed: {benchmark_b.solution_id}")
    
    # Compare solutions
    print("\n=== Comparing Solutions ===")
    
    comparison = comparator.compare_solutions([
        benchmark_a.solution_id,
        benchmark_b.solution_id
    ])
    
    print(f"Comparison completed: {comparison.comparison_id}")
    print(f"Best performer: {comparison.best_performer}")
    
    # Generate report
    print("\n=== Generating Report ===")
    
    report = comparator.generate_report(comparison)
    print(report)
    
    # Track trends (simulated)
    print("\n=== Performance Trends ===")
    
    trends = comparator.track_performance_trends(benchmark_a.solution_id, days=7)
    
    for metric_name, trend_data in trends.items():
        if trend_data:
            latest_value = trend_data[-1][1]
            print(f"{metric_name}: {latest_value:.3f} (latest)")
    
    print("\n=== Demo Complete ===")
    print("Benchmark comparison system ready for production use.")

if __name__ == "__main__":
    main()