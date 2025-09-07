"""Performance Tracking System for Document Parser

This module implements performance monitoring and metrics tracking for document parser models,
following the autocorrect model's test_trained_models.py organizational patterns.
"""

import json
import logging
import pickle
import time
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, field
from pathlib import Path
from collections import defaultdict, deque
import statistics
import sqlite3

import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class PerformanceMetrics:
    """Performance metrics data structure"""
    timestamp: datetime = field(default_factory=datetime.now)
    document_type: str = ""
    processing_time: float = 0.0
    accuracy: float = 0.0
    precision: float = 0.0
    recall: float = 0.0
    f1_score: float = 0.0
    confidence_score: float = 0.0
    field_accuracy: Dict[str, float] = field(default_factory=dict)
    classification_accuracy: float = 0.0
    ocr_confidence: float = 0.0
    image_quality: float = 0.0
    error_count: int = 0
    success_count: int = 0
    total_documents: int = 0
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for storage"""
        return {
            'timestamp': self.timestamp.isoformat(),
            'document_type': self.document_type,
            'processing_time': self.processing_time,
            'accuracy': self.accuracy,
            'precision': self.precision,
            'recall': self.recall,
            'f1_score': self.f1_score,
            'confidence_score': self.confidence_score,
            'field_accuracy': self.field_accuracy,
            'classification_accuracy': self.classification_accuracy,
            'ocr_confidence': self.ocr_confidence,
            'image_quality': self.image_quality,
            'error_count': self.error_count,
            'success_count': self.success_count,
            'total_documents': self.total_documents
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'PerformanceMetrics':
        """Create from dictionary"""
        metrics = cls()
        metrics.timestamp = datetime.fromisoformat(data.get('timestamp', datetime.now().isoformat()))
        metrics.document_type = data.get('document_type', '')
        metrics.processing_time = data.get('processing_time', 0.0)
        metrics.accuracy = data.get('accuracy', 0.0)
        metrics.precision = data.get('precision', 0.0)
        metrics.recall = data.get('recall', 0.0)
        metrics.f1_score = data.get('f1_score', 0.0)
        metrics.confidence_score = data.get('confidence_score', 0.0)
        metrics.field_accuracy = data.get('field_accuracy', {})
        metrics.classification_accuracy = data.get('classification_accuracy', 0.0)
        metrics.ocr_confidence = data.get('ocr_confidence', 0.0)
        metrics.image_quality = data.get('image_quality', 0.0)
        metrics.error_count = data.get('error_count', 0)
        metrics.success_count = data.get('success_count', 0)
        metrics.total_documents = data.get('total_documents', 0)
        return metrics

class PerformanceStorage:
    """Storage system for performance metrics"""
    
    def __init__(self, storage_path: str = "performance_data"):
        self.storage_path = Path(storage_path)
        self.storage_path.mkdir(exist_ok=True)
        
        # Initialize SQLite database
        self.db_path = self.storage_path / "performance.db"
        self.init_database()
        
    def init_database(self):
        """Initialize SQLite database for performance storage"""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS performance_metrics (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp TEXT,
                    document_type TEXT,
                    processing_time REAL,
                    accuracy REAL,
                    precision_score REAL,
                    recall_score REAL,
                    f1_score REAL,
                    confidence_score REAL,
                    field_accuracy TEXT,
                    classification_accuracy REAL,
                    ocr_confidence REAL,
                    image_quality REAL,
                    error_count INTEGER,
                    success_count INTEGER,
                    total_documents INTEGER
                )
            """)
            
            conn.execute("""
                CREATE TABLE IF NOT EXISTS performance_alerts (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp TEXT,
                    alert_type TEXT,
                    severity TEXT,
                    message TEXT,
                    metrics TEXT,
                    resolved BOOLEAN DEFAULT FALSE
                )
            """)
            
    def store_metrics(self, metrics: PerformanceMetrics):
        """Store performance metrics in database"""
        with sqlite3.connect(self.db_path) as conn:
            data = metrics.to_dict()
            conn.execute("""
                INSERT INTO performance_metrics (
                    timestamp, document_type, processing_time, accuracy,
                    precision_score, recall_score, f1_score, confidence_score,
                    field_accuracy, classification_accuracy, ocr_confidence,
                    image_quality, error_count, success_count, total_documents
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                data['timestamp'], data['document_type'], data['processing_time'],
                data['accuracy'], data['precision'], data['recall'], data['f1_score'],
                data['confidence_score'], json.dumps(data['field_accuracy']),
                data['classification_accuracy'], data['ocr_confidence'],
                data['image_quality'], data['error_count'], data['success_count'],
                data['total_documents']
            ))
            
    def get_metrics(self, 
                   document_type: Optional[str] = None,
                   start_date: Optional[datetime] = None,
                   end_date: Optional[datetime] = None,
                   limit: int = 1000) -> List[PerformanceMetrics]:
        """Retrieve performance metrics"""
        query = "SELECT * FROM performance_metrics WHERE 1=1"
        params = []
        
        if document_type:
            query += " AND document_type = ?"
            params.append(document_type)
            
        if start_date:
            query += " AND timestamp >= ?"
            params.append(start_date.isoformat())
            
        if end_date:
            query += " AND timestamp <= ?"
            params.append(end_date.isoformat())
            
        query += " ORDER BY timestamp DESC LIMIT ?"
        params.append(limit)
        
        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            cursor = conn.execute(query, params)
            rows = cursor.fetchall()
            
            metrics_list = []
            for row in rows:
                data = dict(row)
                data['field_accuracy'] = json.loads(data['field_accuracy'] or '{}')
                metrics_list.append(PerformanceMetrics.from_dict(data))
                
        return metrics_list

class DocumentParserPerformanceTracker:
    """Main performance tracking system for document parser"""
    
    def __init__(self, storage_path: str = "performance_data", model_path: str = "models"):
        self.storage = PerformanceStorage(storage_path)
        self.model_path = Path(model_path)
        self.real_time_metrics = deque(maxlen=100)  # Keep last 100 measurements
        self.alert_thresholds = {
            'accuracy': 0.85,
            'processing_time': 5.0,
            'confidence': 0.7,
            'error_rate': 0.15
        }
        
    def track_processing(self, 
                        document_id: str,
                        document_type: str,
                        start_time: float,
                        end_time: float,
                        results: Dict[str, Any],
                        ground_truth: Optional[Dict[str, Any]] = None) -> str:
        """Track processing performance for a single document"""
        
        processing_time = end_time - start_time
        
        # Calculate metrics
        metrics = PerformanceMetrics(
            document_type=document_type,
            processing_time=processing_time,
            confidence_score=self._calculate_average_confidence(results.get('confidence', {})),
            ocr_confidence=results.get('ocr_confidence', 0.0),
            image_quality=results.get('image_quality', 0.0)
        )
        
        # Calculate accuracy if ground truth is available
        if ground_truth:
            accuracy_metrics = self._calculate_accuracy_metrics(results, ground_truth)
            metrics.accuracy = accuracy_metrics['overall_accuracy']
            metrics.precision = accuracy_metrics['precision']
            metrics.recall = accuracy_metrics['recall']
            metrics.f1_score = accuracy_metrics['f1_score']
            metrics.field_accuracy = accuracy_metrics['field_accuracy']
            metrics.classification_accuracy = accuracy_metrics['classification_accuracy']
        
        # Determine success/error
        if results.get('success', True):
            metrics.success_count = 1
            metrics.error_count = 0
        else:
            metrics.success_count = 0
            metrics.error_count = 1
            
        metrics.total_documents = 1
        
        # Store metrics
        self.storage.store_metrics(metrics)
        
        # Add to real-time tracking
        self.real_time_metrics.append(metrics)
        
        # Check for alerts
        self._check_performance_alerts(metrics)
        
        logger.info(f"Tracked performance for {document_id}: {processing_time:.2f}s, accuracy: {metrics.accuracy:.2f}")
        
        return f"performance_{document_id}_{int(time.time())}"
    
    def _calculate_average_confidence(self, confidence_dict: Dict[str, float]) -> float:
        """Calculate average confidence score"""
        if not confidence_dict:
            return 0.0
        return statistics.mean(confidence_dict.values())
    
    def _calculate_accuracy_metrics(self, results: Dict[str, Any], ground_truth: Dict[str, Any]) -> Dict[str, Any]:
        """Calculate accuracy metrics comparing results with ground truth"""
        
        # Classification accuracy
        classification_accuracy = 1.0 if results.get('classification') == ground_truth.get('classification') else 0.0
        
        # Field accuracy
        field_accuracy = {}
        predicted_fields = results.get('fields', {})
        true_fields = ground_truth.get('fields', {})
        
        all_fields = set(list(predicted_fields.keys()) + list(true_fields.keys()))
        correct_fields = 0
        
        for field in all_fields:
            predicted_value = predicted_fields.get(field, '')
            true_value = true_fields.get(field, '')
            
            # Simple string comparison (can be enhanced with fuzzy matching)
            field_correct = str(predicted_value).strip().lower() == str(true_value).strip().lower()
            field_accuracy[field] = 1.0 if field_correct else 0.0
            
            if field_correct:
                correct_fields += 1
        
        # Overall accuracy
        overall_accuracy = correct_fields / len(all_fields) if all_fields else 0.0
        
        # Calculate precision, recall, F1 (simplified for field extraction)
        true_positives = correct_fields
        false_positives = len(predicted_fields) - correct_fields
        false_negatives = len(true_fields) - correct_fields
        
        precision = true_positives / (true_positives + false_positives) if (true_positives + false_positives) > 0 else 0.0
        recall = true_positives / (true_positives + false_negatives) if (true_positives + false_negatives) > 0 else 0.0
        f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
        
        return {
            'overall_accuracy': overall_accuracy,
            'classification_accuracy': classification_accuracy,
            'field_accuracy': field_accuracy,
            'precision': precision,
            'recall': recall,
            'f1_score': f1_score
        }
    
    def _check_performance_alerts(self, metrics: PerformanceMetrics):
        """Check for performance alerts based on thresholds"""
        alerts = []
        
        # Accuracy alert
        if metrics.accuracy > 0 and metrics.accuracy < self.alert_thresholds['accuracy']:
            alerts.append({
                'type': 'low_accuracy',
                'severity': 'warning',
                'message': f'Low accuracy detected: {metrics.accuracy:.2f} < {self.alert_thresholds["accuracy"]}',
                'metrics': metrics.to_dict()
            })
        
        # Processing time alert
        if metrics.processing_time > self.alert_thresholds['processing_time']:
            alerts.append({
                'type': 'slow_processing',
                'severity': 'warning',
                'message': f'Slow processing detected: {metrics.processing_time:.2f}s > {self.alert_thresholds["processing_time"]}s',
                'metrics': metrics.to_dict()
            })
        
        # Confidence alert
        if metrics.confidence_score < self.alert_thresholds['confidence']:
            alerts.append({
                'type': 'low_confidence',
                'severity': 'info',
                'message': f'Low confidence detected: {metrics.confidence_score:.2f} < {self.alert_thresholds["confidence"]}',
                'metrics': metrics.to_dict()
            })
        
        # Store alerts
        for alert in alerts:
            self._store_alert(alert)
    
    def _store_alert(self, alert: Dict[str, Any]):
        """Store performance alert"""
        with sqlite3.connect(self.storage.db_path) as conn:
            conn.execute("""
                INSERT INTO performance_alerts (
                    timestamp, alert_type, severity, message, metrics
                ) VALUES (?, ?, ?, ?, ?)
            """, (
                datetime.now().isoformat(),
                alert['type'],
                alert['severity'],
                alert['message'],
                json.dumps(alert['metrics'])
            ))
    
    def get_performance_summary(self, days: int = 7) -> Dict[str, Any]:
        """Get performance summary for the last N days"""
        end_date = datetime.now()
        start_date = end_date - timedelta(days=days)
        
        metrics_list = self.storage.get_metrics(start_date=start_date, end_date=end_date)
        
        if not metrics_list:
            return {'message': 'No performance data available'}
        
        # Aggregate metrics
        total_documents = sum(m.total_documents for m in metrics_list)
        total_errors = sum(m.error_count for m in metrics_list)
        total_success = sum(m.success_count for m in metrics_list)
        
        processing_times = [m.processing_time for m in metrics_list if m.processing_time > 0]
        accuracies = [m.accuracy for m in metrics_list if m.accuracy > 0]
        confidences = [m.confidence_score for m in metrics_list if m.confidence_score > 0]
        
        summary = {
            'period_days': days,
            'total_documents': total_documents,
            'success_rate': total_success / total_documents if total_documents > 0 else 0.0,
            'error_rate': total_errors / total_documents if total_documents > 0 else 0.0,
            'processing_time': {
                'average': statistics.mean(processing_times) if processing_times else 0.0,
                'median': statistics.median(processing_times) if processing_times else 0.0,
                'min': min(processing_times) if processing_times else 0.0,
                'max': max(processing_times) if processing_times else 0.0
            },
            'accuracy': {
                'average': statistics.mean(accuracies) if accuracies else 0.0,
                'median': statistics.median(accuracies) if accuracies else 0.0,
                'min': min(accuracies) if accuracies else 0.0,
                'max': max(accuracies) if accuracies else 0.0
            },
            'confidence': {
                'average': statistics.mean(confidences) if confidences else 0.0,
                'median': statistics.median(confidences) if confidences else 0.0,
                'min': min(confidences) if confidences else 0.0,
                'max': max(confidences) if confidences else 0.0
            },
            'document_type_breakdown': self._get_document_type_breakdown(metrics_list)
        }
        
        return summary
    
    def _get_document_type_breakdown(self, metrics_list: List[PerformanceMetrics]) -> Dict[str, Dict[str, float]]:
        """Get performance breakdown by document type"""
        breakdown = defaultdict(lambda: {'count': 0, 'accuracy': [], 'processing_time': []})
        
        for metrics in metrics_list:
            doc_type = metrics.document_type or 'unknown'
            breakdown[doc_type]['count'] += metrics.total_documents
            
            if metrics.accuracy > 0:
                breakdown[doc_type]['accuracy'].append(metrics.accuracy)
            if metrics.processing_time > 0:
                breakdown[doc_type]['processing_time'].append(metrics.processing_time)
        
        # Calculate averages
        result = {}
        for doc_type, data in breakdown.items():
            result[doc_type] = {
                'count': data['count'],
                'average_accuracy': statistics.mean(data['accuracy']) if data['accuracy'] else 0.0,
                'average_processing_time': statistics.mean(data['processing_time']) if data['processing_time'] else 0.0
            }
        
        return result
    
    def get_real_time_metrics(self) -> Dict[str, Any]:
        """Get real-time performance metrics from recent measurements"""
        if not self.real_time_metrics:
            return {'message': 'No real-time data available'}
        
        recent_metrics = list(self.real_time_metrics)[-10:]  # Last 10 measurements
        
        processing_times = [m.processing_time for m in recent_metrics if m.processing_time > 0]
        accuracies = [m.accuracy for m in recent_metrics if m.accuracy > 0]
        confidences = [m.confidence_score for m in recent_metrics if m.confidence_score > 0]
        
        return {
            'recent_measurements': len(recent_metrics),
            'current_processing_time': statistics.mean(processing_times) if processing_times else 0.0,
            'current_accuracy': statistics.mean(accuracies) if accuracies else 0.0,
            'current_confidence': statistics.mean(confidences) if confidences else 0.0,
            'trend': self._calculate_trend(recent_metrics)
        }
    
    def _calculate_trend(self, metrics_list: List[PerformanceMetrics]) -> Dict[str, str]:
        """Calculate performance trends"""
        if len(metrics_list) < 5:
            return {'accuracy': 'insufficient_data', 'processing_time': 'insufficient_data'}
        
        # Simple trend calculation (comparing first half vs second half)
        mid_point = len(metrics_list) // 2
        first_half = metrics_list[:mid_point]
        second_half = metrics_list[mid_point:]
        
        # Accuracy trend
        first_acc = [m.accuracy for m in first_half if m.accuracy > 0]
        second_acc = [m.accuracy for m in second_half if m.accuracy > 0]
        
        if first_acc and second_acc:
            acc_trend = 'improving' if statistics.mean(second_acc) > statistics.mean(first_acc) else 'declining'
        else:
            acc_trend = 'stable'
        
        # Processing time trend
        first_time = [m.processing_time for m in first_half if m.processing_time > 0]
        second_time = [m.processing_time for m in second_half if m.processing_time > 0]
        
        if first_time and second_time:
            time_trend = 'improving' if statistics.mean(second_time) < statistics.mean(first_time) else 'declining'
        else:
            time_trend = 'stable'
        
        return {
            'accuracy': acc_trend,
            'processing_time': time_trend
        }
    
    def generate_performance_report(self, days: int = 30) -> Dict[str, Any]:
        """Generate comprehensive performance report"""
        summary = self.get_performance_summary(days)
        real_time = self.get_real_time_metrics()
        
        # Get alerts
        with sqlite3.connect(self.storage.db_path) as conn:
            conn.row_factory = sqlite3.Row
            cursor = conn.execute("""
                SELECT * FROM performance_alerts 
                WHERE timestamp >= ? AND resolved = FALSE
                ORDER BY timestamp DESC LIMIT 10
            """, ((datetime.now() - timedelta(days=days)).isoformat(),))
            
            alerts = [dict(row) for row in cursor.fetchall()]
        
        report = {
            'report_date': datetime.now().isoformat(),
            'analysis_period_days': days,
            'summary': summary,
            'real_time_metrics': real_time,
            'active_alerts': alerts,
            'recommendations': self._generate_recommendations(summary, alerts)
        }
        
        return report
    
    def _generate_recommendations(self, summary: Dict[str, Any], alerts: List[Dict[str, Any]]) -> List[Dict[str, str]]:
        """Generate performance improvement recommendations"""
        recommendations = []
        
        if summary.get('success_rate', 1.0) < 0.9:
            recommendations.append({
                'priority': 'high',
                'category': 'reliability',
                'action': 'Investigate error patterns and improve error handling',
                'reason': f"Success rate is {summary['success_rate']:.1%}, below 90% threshold"
            })
        
        avg_accuracy = summary.get('accuracy', {}).get('average', 0.0)
        if avg_accuracy > 0 and avg_accuracy < 0.85:
            recommendations.append({
                'priority': 'high',
                'category': 'accuracy',
                'action': 'Review and retrain models with recent feedback data',
                'reason': f"Average accuracy is {avg_accuracy:.1%}, below 85% threshold"
            })
        
        avg_processing_time = summary.get('processing_time', {}).get('average', 0.0)
        if avg_processing_time > 3.0:
            recommendations.append({
                'priority': 'medium',
                'category': 'performance',
                'action': 'Optimize processing pipeline and consider hardware upgrades',
                'reason': f"Average processing time is {avg_processing_time:.1f}s, above 3s threshold"
            })
        
        if len(alerts) > 5:
            recommendations.append({
                'priority': 'medium',
                'category': 'monitoring',
                'action': 'Review alert thresholds and resolve outstanding issues',
                'reason': f"{len(alerts)} active alerts require attention"
            })
        
        return recommendations

def main():
    """Demo function"""
    print("📊 Document Parser Performance Tracker Demo")
    print("=" * 50)
    
    # Initialize performance tracker
    tracker = DocumentParserPerformanceTracker()
    
    # Simulate tracking some document processing
    print("\n=== Simulating Document Processing ===")
    
    # Sample processing data
    sample_results = [
        {
            'document_id': 'mykad_001',
            'document_type': 'mykad',
            'processing_time': 2.3,
            'results': {
                'classification': 'mykad',
                'fields': {'name': 'JOHN DOE', 'ic_number': '123456-78-9012'},
                'confidence': {'name': 0.95, 'ic_number': 0.88},
                'success': True
            },
            'ground_truth': {
                'classification': 'mykad',
                'fields': {'name': 'JOHN DOE', 'ic_number': '123456-78-9012'}
            }
        },
        {
            'document_id': 'spk_001',
            'document_type': 'spk',
            'processing_time': 2.8,
            'results': {
                'classification': 'spk',
                'fields': {'vehicle_number': 'ABC1234', 'owner_name': 'JANE SMITH'},
                'confidence': {'vehicle_number': 0.92, 'owner_name': 0.85},
                'success': True
            },
            'ground_truth': {
                'classification': 'spk',
                'fields': {'vehicle_number': 'ABC1234', 'owner_name': 'JANE SMITH'}
            }
        }
    ]
    
    # Track performance for each sample
    for sample in sample_results:
        start_time = time.time()
        end_time = start_time + sample['processing_time']
        
        tracking_id = tracker.track_processing(
            document_id=sample['document_id'],
            document_type=sample['document_type'],
            start_time=start_time,
            end_time=end_time,
            results=sample['results'],
            ground_truth=sample['ground_truth']
        )
        
        print(f"✅ Tracked: {sample['document_id']} -> {tracking_id}")
    
    # Get performance summary
    print("\n=== Performance Summary ===")
    summary = tracker.get_performance_summary(days=1)
    print(f"Total documents: {summary.get('total_documents', 0)}")
    print(f"Success rate: {summary.get('success_rate', 0):.1%}")
    print(f"Average accuracy: {summary.get('accuracy', {}).get('average', 0):.1%}")
    print(f"Average processing time: {summary.get('processing_time', {}).get('average', 0):.2f}s")
    
    # Get real-time metrics
    print("\n=== Real-time Metrics ===")
    real_time = tracker.get_real_time_metrics()
    print(f"Recent measurements: {real_time.get('recent_measurements', 0)}")
    print(f"Current accuracy: {real_time.get('current_accuracy', 0):.1%}")
    print(f"Current processing time: {real_time.get('current_processing_time', 0):.2f}s")
    
    # Generate comprehensive report
    print("\n=== Performance Report ===")
    report = tracker.generate_performance_report(days=1)
    print(f"Analysis period: {report['analysis_period_days']} days")
    print(f"Active alerts: {len(report['active_alerts'])}")
    print(f"Recommendations: {len(report['recommendations'])}")
    
    for i, rec in enumerate(report['recommendations'][:3], 1):
        print(f"{i}. [{rec['priority'].upper()}] {rec['action']}")
    
    print("\n=== Demo Complete ===")
    print("Performance tracking system ready for production use.")

if __name__ == "__main__":
    main()