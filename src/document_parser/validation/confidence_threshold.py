"""Confidence Threshold Management System

This module implements dynamic confidence threshold management for auto-approval
and manual review queuing in the document processing pipeline.
Follows the autocorrect model's organizational patterns.
"""

import json
import logging
import sqlite3
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Tuple, Set
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from collections import defaultdict, deque
import uuid
import statistics

import pandas as pd
import numpy as np
from sklearn.metrics import precision_recall_curve, roc_curve, auc
from sklearn.model_selection import cross_val_score

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ThresholdType(Enum):
    """Types of confidence thresholds"""
    AUTO_APPROVE = "auto_approve"
    MANUAL_REVIEW = "manual_review"
    REJECT = "reject"
    ESCALATE = "escalate"

class DocumentType(Enum):
    """Document types for threshold management"""
    IDENTITY_CARD = "identity_card"
    PASSPORT = "passport"
    DRIVING_LICENSE = "driving_license"
    BIRTH_CERTIFICATE = "birth_certificate"
    UTILITY_BILL = "utility_bill"
    BANK_STATEMENT = "bank_statement"
    OTHER = "other"

class FieldType(Enum):
    """Field types for threshold management"""
    IC_NUMBER = "ic_number"
    NAME = "name"
    ADDRESS = "address"
    DATE_OF_BIRTH = "date_of_birth"
    PHONE_NUMBER = "phone_number"
    EMAIL = "email"
    REGISTRATION_NUMBER = "registration_number"
    OTHER = "other"

class ThresholdStrategy(Enum):
    """Strategies for threshold adjustment"""
    CONSERVATIVE = "conservative"  # Higher thresholds, more manual review
    BALANCED = "balanced"         # Balanced approach
    AGGRESSIVE = "aggressive"     # Lower thresholds, more auto-approval
    ADAPTIVE = "adaptive"         # Dynamic adjustment based on performance

class PerformanceMetric(Enum):
    """Performance metrics for threshold optimization"""
    ACCURACY = "accuracy"
    PRECISION = "precision"
    RECALL = "recall"
    F1_SCORE = "f1_score"
    THROUGHPUT = "throughput"
    REVIEW_RATE = "review_rate"

@dataclass
class ThresholdConfig:
    """Configuration for confidence thresholds"""
    
    config_id: str
    document_type: DocumentType
    field_type: FieldType
    
    # Threshold values
    auto_approve_threshold: float
    manual_review_threshold: float
    reject_threshold: float
    
    # Strategy and settings
    strategy: ThresholdStrategy = ThresholdStrategy.BALANCED
    min_samples_for_adjustment: int = 100
    adjustment_frequency_hours: int = 24
    
    # Performance targets
    target_accuracy: float = 0.95
    target_throughput: float = 0.80  # Percentage of auto-approved
    max_review_rate: float = 0.30    # Maximum manual review rate
    
    # Metadata
    created_at: datetime = field(default_factory=datetime.now)
    updated_at: datetime = field(default_factory=datetime.now)
    version: int = 1
    
    # Historical performance
    performance_history: List[Dict[str, float]] = field(default_factory=list)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return {
            'config_id': self.config_id,
            'document_type': self.document_type.value,
            'field_type': self.field_type.value,
            'auto_approve_threshold': self.auto_approve_threshold,
            'manual_review_threshold': self.manual_review_threshold,
            'reject_threshold': self.reject_threshold,
            'strategy': self.strategy.value,
            'min_samples_for_adjustment': self.min_samples_for_adjustment,
            'adjustment_frequency_hours': self.adjustment_frequency_hours,
            'target_accuracy': self.target_accuracy,
            'target_throughput': self.target_throughput,
            'max_review_rate': self.max_review_rate,
            'created_at': self.created_at.isoformat(),
            'updated_at': self.updated_at.isoformat(),
            'version': self.version,
            'performance_history': self.performance_history
        }
    
    def validate_thresholds(self) -> bool:
        """Validate threshold configuration"""
        return (
            0.0 <= self.reject_threshold <= self.manual_review_threshold <= 
            self.auto_approve_threshold <= 1.0
        )

@dataclass
class ProcessingDecision:
    """Decision made by threshold system"""
    
    decision_id: str
    document_id: str
    field_name: str
    field_value: str
    
    # Confidence and decision
    confidence_score: float
    decision_type: ThresholdType
    threshold_config_id: str
    
    # Context
    document_type: DocumentType
    field_type: FieldType
    model_version: str
    
    # Timing
    decision_timestamp: datetime = field(default_factory=datetime.now)
    
    # Outcome (filled after human review)
    actual_correct: Optional[bool] = None
    review_timestamp: Optional[datetime] = None
    reviewer_id: Optional[str] = None
    
    # Additional context
    context_data: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return {
            'decision_id': self.decision_id,
            'document_id': self.document_id,
            'field_name': self.field_name,
            'field_value': self.field_value,
            'confidence_score': self.confidence_score,
            'decision_type': self.decision_type.value,
            'threshold_config_id': self.threshold_config_id,
            'document_type': self.document_type.value,
            'field_type': self.field_type.value,
            'model_version': self.model_version,
            'decision_timestamp': self.decision_timestamp.isoformat(),
            'actual_correct': self.actual_correct,
            'review_timestamp': self.review_timestamp.isoformat() if self.review_timestamp else None,
            'reviewer_id': self.reviewer_id,
            'context_data': self.context_data
        }

@dataclass
class PerformanceMetrics:
    """Performance metrics for threshold evaluation"""
    
    config_id: str
    evaluation_period_start: datetime
    evaluation_period_end: datetime
    
    # Basic metrics
    total_decisions: int
    auto_approved: int
    manual_reviewed: int
    rejected: int
    
    # Accuracy metrics
    true_positives: int = 0
    false_positives: int = 0
    true_negatives: int = 0
    false_negatives: int = 0
    
    # Calculated metrics
    accuracy: float = 0.0
    precision: float = 0.0
    recall: float = 0.0
    f1_score: float = 0.0
    throughput_rate: float = 0.0
    review_rate: float = 0.0
    
    # Performance scores
    overall_score: float = 0.0
    
    def calculate_metrics(self):
        """Calculate derived metrics"""
        if self.total_decisions > 0:
            self.throughput_rate = self.auto_approved / self.total_decisions
            self.review_rate = self.manual_reviewed / self.total_decisions
        
        total_classified = self.true_positives + self.false_positives + self.true_negatives + self.false_negatives
        
        if total_classified > 0:
            self.accuracy = (self.true_positives + self.true_negatives) / total_classified
        
        if (self.true_positives + self.false_positives) > 0:
            self.precision = self.true_positives / (self.true_positives + self.false_positives)
        
        if (self.true_positives + self.false_negatives) > 0:
            self.recall = self.true_positives / (self.true_positives + self.false_negatives)
        
        if (self.precision + self.recall) > 0:
            self.f1_score = 2 * (self.precision * self.recall) / (self.precision + self.recall)
        
        # Calculate overall performance score
        self.overall_score = (
            self.accuracy * 0.4 +
            self.f1_score * 0.3 +
            self.throughput_rate * 0.2 +
            (1 - self.review_rate) * 0.1
        )
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return {
            'config_id': self.config_id,
            'evaluation_period_start': self.evaluation_period_start.isoformat(),
            'evaluation_period_end': self.evaluation_period_end.isoformat(),
            'total_decisions': self.total_decisions,
            'auto_approved': self.auto_approved,
            'manual_reviewed': self.manual_reviewed,
            'rejected': self.rejected,
            'true_positives': self.true_positives,
            'false_positives': self.false_positives,
            'true_negatives': self.true_negatives,
            'false_negatives': self.false_negatives,
            'accuracy': self.accuracy,
            'precision': self.precision,
            'recall': self.recall,
            'f1_score': self.f1_score,
            'throughput_rate': self.throughput_rate,
            'review_rate': self.review_rate,
            'overall_score': self.overall_score
        }

class ThresholdDatabase:
    """Database for threshold management"""
    
    def __init__(self, db_path: str = "threshold_management.db"):
        self.db_path = db_path
        self.init_database()
    
    def init_database(self):
        """Initialize database tables"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Threshold configurations table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS threshold_configs (
                config_id TEXT PRIMARY KEY,
                document_type TEXT NOT NULL,
                field_type TEXT NOT NULL,
                auto_approve_threshold REAL NOT NULL,
                manual_review_threshold REAL NOT NULL,
                reject_threshold REAL NOT NULL,
                strategy TEXT NOT NULL,
                min_samples_for_adjustment INTEGER,
                adjustment_frequency_hours INTEGER,
                target_accuracy REAL,
                target_throughput REAL,
                max_review_rate REAL,
                created_at TIMESTAMP,
                updated_at TIMESTAMP,
                version INTEGER,
                performance_history TEXT
            )
        """)
        
        # Processing decisions table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS processing_decisions (
                decision_id TEXT PRIMARY KEY,
                document_id TEXT NOT NULL,
                field_name TEXT NOT NULL,
                field_value TEXT,
                confidence_score REAL NOT NULL,
                decision_type TEXT NOT NULL,
                threshold_config_id TEXT NOT NULL,
                document_type TEXT NOT NULL,
                field_type TEXT NOT NULL,
                model_version TEXT,
                decision_timestamp TIMESTAMP,
                actual_correct BOOLEAN,
                review_timestamp TIMESTAMP,
                reviewer_id TEXT,
                context_data TEXT,
                FOREIGN KEY (threshold_config_id) REFERENCES threshold_configs (config_id)
            )
        """)
        
        # Performance metrics table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS performance_metrics (
                metric_id TEXT PRIMARY KEY,
                config_id TEXT NOT NULL,
                evaluation_period_start TIMESTAMP,
                evaluation_period_end TIMESTAMP,
                total_decisions INTEGER,
                auto_approved INTEGER,
                manual_reviewed INTEGER,
                rejected INTEGER,
                true_positives INTEGER,
                false_positives INTEGER,
                true_negatives INTEGER,
                false_negatives INTEGER,
                accuracy REAL,
                precision_score REAL,
                recall_score REAL,
                f1_score REAL,
                throughput_rate REAL,
                review_rate REAL,
                overall_score REAL,
                FOREIGN KEY (config_id) REFERENCES threshold_configs (config_id)
            )
        """)
        
        # Create indexes
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_decisions_config ON processing_decisions(threshold_config_id)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_decisions_timestamp ON processing_decisions(decision_timestamp)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_decisions_document ON processing_decisions(document_id)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_metrics_config ON performance_metrics(config_id)")
        
        conn.commit()
        conn.close()
        
        logger.info(f"Threshold database initialized at {self.db_path}")
    
    def save_threshold_config(self, config: ThresholdConfig) -> bool:
        """Save threshold configuration"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            cursor.execute("""
                INSERT OR REPLACE INTO threshold_configs (
                    config_id, document_type, field_type, auto_approve_threshold,
                    manual_review_threshold, reject_threshold, strategy,
                    min_samples_for_adjustment, adjustment_frequency_hours,
                    target_accuracy, target_throughput, max_review_rate,
                    created_at, updated_at, version, performance_history
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                config.config_id,
                config.document_type.value,
                config.field_type.value,
                config.auto_approve_threshold,
                config.manual_review_threshold,
                config.reject_threshold,
                config.strategy.value,
                config.min_samples_for_adjustment,
                config.adjustment_frequency_hours,
                config.target_accuracy,
                config.target_throughput,
                config.max_review_rate,
                config.created_at,
                config.updated_at,
                config.version,
                json.dumps(config.performance_history)
            ))
            
            conn.commit()
            conn.close()
            return True
            
        except Exception as e:
            logger.error(f"Error saving threshold config: {e}")
            return False
    
    def get_threshold_config(self, document_type: DocumentType, 
                           field_type: FieldType) -> Optional[ThresholdConfig]:
        """Get threshold configuration for document and field type"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            cursor.execute("""
                SELECT * FROM threshold_configs 
                WHERE document_type = ? AND field_type = ?
                ORDER BY version DESC LIMIT 1
            """, (document_type.value, field_type.value))
            
            row = cursor.fetchone()
            conn.close()
            
            if row:
                return self._row_to_threshold_config(row)
            return None
            
        except Exception as e:
            logger.error(f"Error getting threshold config: {e}")
            return None
    
    def save_processing_decision(self, decision: ProcessingDecision) -> bool:
        """Save processing decision"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            cursor.execute("""
                INSERT INTO processing_decisions (
                    decision_id, document_id, field_name, field_value,
                    confidence_score, decision_type, threshold_config_id,
                    document_type, field_type, model_version,
                    decision_timestamp, actual_correct, review_timestamp,
                    reviewer_id, context_data
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                decision.decision_id,
                decision.document_id,
                decision.field_name,
                decision.field_value,
                decision.confidence_score,
                decision.decision_type.value,
                decision.threshold_config_id,
                decision.document_type.value,
                decision.field_type.value,
                decision.model_version,
                decision.decision_timestamp,
                decision.actual_correct,
                decision.review_timestamp,
                decision.reviewer_id,
                json.dumps(decision.context_data)
            ))
            
            conn.commit()
            conn.close()
            return True
            
        except Exception as e:
            logger.error(f"Error saving processing decision: {e}")
            return False
    
    def update_decision_outcome(self, decision_id: str, actual_correct: bool,
                              reviewer_id: str) -> bool:
        """Update decision with actual outcome"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            cursor.execute("""
                UPDATE processing_decisions 
                SET actual_correct = ?, review_timestamp = ?, reviewer_id = ?
                WHERE decision_id = ?
            """, (actual_correct, datetime.now(), reviewer_id, decision_id))
            
            conn.commit()
            conn.close()
            return True
            
        except Exception as e:
            logger.error(f"Error updating decision outcome: {e}")
            return False
    
    def get_decisions_for_evaluation(self, config_id: str, 
                                   start_date: datetime,
                                   end_date: datetime) -> List[ProcessingDecision]:
        """Get decisions for performance evaluation"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            cursor.execute("""
                SELECT * FROM processing_decisions 
                WHERE threshold_config_id = ? 
                AND decision_timestamp BETWEEN ? AND ?
                AND actual_correct IS NOT NULL
            """, (config_id, start_date, end_date))
            
            rows = cursor.fetchall()
            conn.close()
            
            return [self._row_to_processing_decision(row) for row in rows]
            
        except Exception as e:
            logger.error(f"Error getting decisions for evaluation: {e}")
            return []
    
    def save_performance_metrics(self, metrics: PerformanceMetrics) -> bool:
        """Save performance metrics"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            metric_id = str(uuid.uuid4())
            
            cursor.execute("""
                INSERT INTO performance_metrics (
                    metric_id, config_id, evaluation_period_start, evaluation_period_end,
                    total_decisions, auto_approved, manual_reviewed, rejected,
                    true_positives, false_positives, true_negatives, false_negatives,
                    accuracy, precision_score, recall_score, f1_score,
                    throughput_rate, review_rate, overall_score
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                metric_id,
                metrics.config_id,
                metrics.evaluation_period_start,
                metrics.evaluation_period_end,
                metrics.total_decisions,
                metrics.auto_approved,
                metrics.manual_reviewed,
                metrics.rejected,
                metrics.true_positives,
                metrics.false_positives,
                metrics.true_negatives,
                metrics.false_negatives,
                metrics.accuracy,
                metrics.precision,
                metrics.recall,
                metrics.f1_score,
                metrics.throughput_rate,
                metrics.review_rate,
                metrics.overall_score
            ))
            
            conn.commit()
            conn.close()
            return True
            
        except Exception as e:
            logger.error(f"Error saving performance metrics: {e}")
            return False
    
    def _row_to_threshold_config(self, row) -> ThresholdConfig:
        """Convert database row to ThresholdConfig"""
        return ThresholdConfig(
            config_id=row[0],
            document_type=DocumentType(row[1]),
            field_type=FieldType(row[2]),
            auto_approve_threshold=row[3],
            manual_review_threshold=row[4],
            reject_threshold=row[5],
            strategy=ThresholdStrategy(row[6]),
            min_samples_for_adjustment=row[7],
            adjustment_frequency_hours=row[8],
            target_accuracy=row[9],
            target_throughput=row[10],
            max_review_rate=row[11],
            created_at=datetime.fromisoformat(row[12]) if row[12] else datetime.now(),
            updated_at=datetime.fromisoformat(row[13]) if row[13] else datetime.now(),
            version=row[14],
            performance_history=json.loads(row[15]) if row[15] else []
        )
    
    def _row_to_processing_decision(self, row) -> ProcessingDecision:
        """Convert database row to ProcessingDecision"""
        return ProcessingDecision(
            decision_id=row[0],
            document_id=row[1],
            field_name=row[2],
            field_value=row[3] or "",
            confidence_score=row[4],
            decision_type=ThresholdType(row[5]),
            threshold_config_id=row[6],
            document_type=DocumentType(row[7]),
            field_type=FieldType(row[8]),
            model_version=row[9] or "",
            decision_timestamp=datetime.fromisoformat(row[10]) if row[10] else datetime.now(),
            actual_correct=row[11],
            review_timestamp=datetime.fromisoformat(row[12]) if row[12] else None,
            reviewer_id=row[13],
            context_data=json.loads(row[14]) if row[14] else {}
        )

class ThresholdOptimizer:
    """Optimizer for confidence thresholds"""
    
    def __init__(self, db: ThresholdDatabase):
        self.db = db
    
    def optimize_thresholds(self, config: ThresholdConfig, 
                          decisions: List[ProcessingDecision]) -> ThresholdConfig:
        """Optimize thresholds based on historical performance"""
        if len(decisions) < config.min_samples_for_adjustment:
            logger.info(f"Insufficient samples for optimization: {len(decisions)} < {config.min_samples_for_adjustment}")
            return config
        
        # Extract confidence scores and outcomes
        confidences = [d.confidence_score for d in decisions]
        outcomes = [d.actual_correct for d in decisions if d.actual_correct is not None]
        
        if len(outcomes) < len(decisions) * 0.5:  # Need at least 50% labeled data
            logger.info("Insufficient labeled data for optimization")
            return config
        
        # Find optimal thresholds based on strategy
        if config.strategy == ThresholdStrategy.CONSERVATIVE:
            new_thresholds = self._optimize_conservative(confidences, outcomes, config)
        elif config.strategy == ThresholdStrategy.AGGRESSIVE:
            new_thresholds = self._optimize_aggressive(confidences, outcomes, config)
        elif config.strategy == ThresholdStrategy.ADAPTIVE:
            new_thresholds = self._optimize_adaptive(confidences, outcomes, config)
        else:  # BALANCED
            new_thresholds = self._optimize_balanced(confidences, outcomes, config)
        
        # Create new configuration
        new_config = ThresholdConfig(
            config_id=str(uuid.uuid4()),
            document_type=config.document_type,
            field_type=config.field_type,
            auto_approve_threshold=new_thresholds['auto_approve'],
            manual_review_threshold=new_thresholds['manual_review'],
            reject_threshold=new_thresholds['reject'],
            strategy=config.strategy,
            min_samples_for_adjustment=config.min_samples_for_adjustment,
            adjustment_frequency_hours=config.adjustment_frequency_hours,
            target_accuracy=config.target_accuracy,
            target_throughput=config.target_throughput,
            max_review_rate=config.max_review_rate,
            version=config.version + 1
        )
        
        # Validate new thresholds
        if not new_config.validate_thresholds():
            logger.warning("Invalid threshold configuration, keeping current")
            return config
        
        logger.info(f"Optimized thresholds: auto={new_thresholds['auto_approve']:.3f}, "
                   f"review={new_thresholds['manual_review']:.3f}, reject={new_thresholds['reject']:.3f}")
        
        return new_config
    
    def _optimize_conservative(self, confidences: List[float], outcomes: List[bool],
                             config: ThresholdConfig) -> Dict[str, float]:
        """Optimize for conservative strategy (high accuracy, more manual review)"""
        # Use precision-recall curve to find high-precision thresholds
        precision, recall, thresholds = precision_recall_curve(outcomes, confidences)
        
        # Find threshold that achieves target accuracy with high precision
        target_precision = max(config.target_accuracy, 0.95)
        
        valid_indices = precision >= target_precision
        if not any(valid_indices):
            # Fallback to highest precision
            best_idx = np.argmax(precision)
        else:
            # Among high-precision thresholds, choose one with reasonable recall
            valid_precision = precision[valid_indices]
            valid_recall = recall[valid_indices]
            valid_thresholds = thresholds[valid_indices]
            
            # Choose threshold with best balance of precision and recall
            f1_scores = 2 * (valid_precision * valid_recall) / (valid_precision + valid_recall + 1e-8)
            best_idx = np.argmax(f1_scores)
            best_threshold = valid_thresholds[best_idx]
        
        auto_approve = min(best_threshold + 0.1, 0.95)
        manual_review = max(best_threshold - 0.1, 0.3)
        reject = max(manual_review - 0.2, 0.1)
        
        return {
            'auto_approve': auto_approve,
            'manual_review': manual_review,
            'reject': reject
        }
    
    def _optimize_aggressive(self, confidences: List[float], outcomes: List[bool],
                           config: ThresholdConfig) -> Dict[str, float]:
        """Optimize for aggressive strategy (high throughput, less manual review)"""
        # Find thresholds that maximize throughput while maintaining minimum accuracy
        sorted_indices = np.argsort(confidences)[::-1]  # Sort by confidence descending
        
        min_accuracy = max(config.target_accuracy - 0.05, 0.85)
        target_throughput = min(config.target_throughput + 0.1, 0.9)
        
        # Find threshold that achieves target throughput
        auto_approve_count = int(len(confidences) * target_throughput)
        if auto_approve_count > 0:
            auto_approve = confidences[sorted_indices[auto_approve_count - 1]]
        else:
            auto_approve = 0.8
        
        # Ensure minimum accuracy
        auto_approve = max(auto_approve, 0.7)
        
        manual_review = max(auto_approve - 0.2, 0.4)
        reject = max(manual_review - 0.2, 0.1)
        
        return {
            'auto_approve': auto_approve,
            'manual_review': manual_review,
            'reject': reject
        }
    
    def _optimize_balanced(self, confidences: List[float], outcomes: List[bool],
                         config: ThresholdConfig) -> Dict[str, float]:
        """Optimize for balanced strategy"""
        # Use ROC curve to find optimal balance
        fpr, tpr, thresholds = roc_curve(outcomes, confidences)
        
        # Find threshold that maximizes Youden's J statistic (TPR - FPR)
        j_scores = tpr - fpr
        best_idx = np.argmax(j_scores)
        optimal_threshold = thresholds[best_idx]
        
        # Set thresholds around optimal point
        auto_approve = min(optimal_threshold + 0.15, 0.9)
        manual_review = max(optimal_threshold - 0.1, 0.3)
        reject = max(manual_review - 0.15, 0.1)
        
        return {
            'auto_approve': auto_approve,
            'manual_review': manual_review,
            'reject': reject
        }
    
    def _optimize_adaptive(self, confidences: List[float], outcomes: List[bool],
                         config: ThresholdConfig) -> Dict[str, float]:
        """Optimize for adaptive strategy (dynamic based on recent performance)"""
        # Calculate recent performance metrics
        recent_accuracy = sum(outcomes) / len(outcomes)
        
        # Adjust strategy based on performance
        if recent_accuracy >= config.target_accuracy:
            # Performance is good, can be more aggressive
            return self._optimize_aggressive(confidences, outcomes, config)
        else:
            # Performance needs improvement, be more conservative
            return self._optimize_conservative(confidences, outcomes, config)

class ConfidenceThresholdManager:
    """Main confidence threshold management system"""
    
    def __init__(self, db_path: str = "threshold_management.db"):
        self.db = ThresholdDatabase(db_path)
        self.optimizer = ThresholdOptimizer(self.db)
        
        # Initialize default configurations
        self._initialize_default_configs()
        
        logger.info("Confidence threshold manager initialized")
    
    def _initialize_default_configs(self):
        """Initialize default threshold configurations"""
        default_configs = [
            # Identity Card configurations
            {
                'document_type': DocumentType.IDENTITY_CARD,
                'field_type': FieldType.IC_NUMBER,
                'auto_approve': 0.9,
                'manual_review': 0.7,
                'reject': 0.3,
                'strategy': ThresholdStrategy.CONSERVATIVE
            },
            {
                'document_type': DocumentType.IDENTITY_CARD,
                'field_type': FieldType.NAME,
                'auto_approve': 0.85,
                'manual_review': 0.6,
                'reject': 0.2,
                'strategy': ThresholdStrategy.BALANCED
            },
            {
                'document_type': DocumentType.IDENTITY_CARD,
                'field_type': FieldType.ADDRESS,
                'auto_approve': 0.8,
                'manual_review': 0.5,
                'reject': 0.2,
                'strategy': ThresholdStrategy.BALANCED
            },
            # Passport configurations
            {
                'document_type': DocumentType.PASSPORT,
                'field_type': FieldType.NAME,
                'auto_approve': 0.88,
                'manual_review': 0.65,
                'reject': 0.25,
                'strategy': ThresholdStrategy.CONSERVATIVE
            },
            # Other document types can be added here
        ]
        
        for config_data in default_configs:
            existing_config = self.db.get_threshold_config(
                config_data['document_type'],
                config_data['field_type']
            )
            
            if not existing_config:
                config = ThresholdConfig(
                    config_id=str(uuid.uuid4()),
                    document_type=config_data['document_type'],
                    field_type=config_data['field_type'],
                    auto_approve_threshold=config_data['auto_approve'],
                    manual_review_threshold=config_data['manual_review'],
                    reject_threshold=config_data['reject'],
                    strategy=config_data['strategy']
                )
                
                self.db.save_threshold_config(config)
                logger.info(f"Created default config for {config_data['document_type'].value} - {config_data['field_type'].value}")
    
    def make_decision(self, document_id: str, field_name: str, field_value: str,
                     confidence_score: float, document_type: DocumentType,
                     field_type: FieldType, model_version: str,
                     context_data: Dict[str, Any] = None) -> ProcessingDecision:
        """Make processing decision based on confidence thresholds"""
        # Get threshold configuration
        config = self.db.get_threshold_config(document_type, field_type)
        
        if not config:
            # Use default conservative thresholds
            logger.warning(f"No threshold config found for {document_type.value} - {field_type.value}, using defaults")
            config = ThresholdConfig(
                config_id="default",
                document_type=document_type,
                field_type=field_type,
                auto_approve_threshold=0.85,
                manual_review_threshold=0.6,
                reject_threshold=0.3,
                strategy=ThresholdStrategy.CONSERVATIVE
            )
        
        # Determine decision type based on confidence score
        if confidence_score >= config.auto_approve_threshold:
            decision_type = ThresholdType.AUTO_APPROVE
        elif confidence_score >= config.manual_review_threshold:
            decision_type = ThresholdType.MANUAL_REVIEW
        elif confidence_score >= config.reject_threshold:
            decision_type = ThresholdType.MANUAL_REVIEW  # Low confidence still gets review
        else:
            decision_type = ThresholdType.REJECT
        
        # Create decision record
        decision = ProcessingDecision(
            decision_id=str(uuid.uuid4()),
            document_id=document_id,
            field_name=field_name,
            field_value=field_value,
            confidence_score=confidence_score,
            decision_type=decision_type,
            threshold_config_id=config.config_id,
            document_type=document_type,
            field_type=field_type,
            model_version=model_version,
            context_data=context_data or {}
        )
        
        # Save decision to database
        self.db.save_processing_decision(decision)
        
        logger.debug(f"Decision made: {decision_type.value} for {field_name} (confidence: {confidence_score:.3f})")
        
        return decision
    
    def update_decision_outcome(self, decision_id: str, actual_correct: bool,
                              reviewer_id: str) -> bool:
        """Update decision with actual outcome from human review"""
        success = self.db.update_decision_outcome(decision_id, actual_correct, reviewer_id)
        
        if success:
            logger.info(f"Updated decision outcome: {decision_id[:8]}... = {actual_correct}")
        
        return success
    
    def evaluate_performance(self, config_id: str, days_back: int = 7) -> PerformanceMetrics:
        """Evaluate performance of threshold configuration"""
        end_date = datetime.now()
        start_date = end_date - timedelta(days=days_back)
        
        # Get decisions for evaluation period
        decisions = self.db.get_decisions_for_evaluation(config_id, start_date, end_date)
        
        if not decisions:
            logger.warning(f"No decisions found for evaluation: {config_id}")
            return None
        
        # Calculate metrics
        metrics = PerformanceMetrics(
            config_id=config_id,
            evaluation_period_start=start_date,
            evaluation_period_end=end_date,
            total_decisions=len(decisions),
            auto_approved=len([d for d in decisions if d.decision_type == ThresholdType.AUTO_APPROVE]),
            manual_reviewed=len([d for d in decisions if d.decision_type == ThresholdType.MANUAL_REVIEW]),
            rejected=len([d for d in decisions if d.decision_type == ThresholdType.REJECT])
        )
        
        # Calculate accuracy metrics
        for decision in decisions:
            if decision.actual_correct is not None:
                if decision.decision_type == ThresholdType.AUTO_APPROVE:
                    if decision.actual_correct:
                        metrics.true_positives += 1
                    else:
                        metrics.false_positives += 1
                elif decision.decision_type == ThresholdType.REJECT:
                    if not decision.actual_correct:
                        metrics.true_negatives += 1
                    else:
                        metrics.false_negatives += 1
        
        # Calculate derived metrics
        metrics.calculate_metrics()
        
        # Save metrics to database
        self.db.save_performance_metrics(metrics)
        
        logger.info(f"Performance evaluation completed: accuracy={metrics.accuracy:.3f}, "
                   f"throughput={metrics.throughput_rate:.3f}")
        
        return metrics
    
    def optimize_thresholds(self, document_type: DocumentType, field_type: FieldType,
                          days_back: int = 30) -> bool:
        """Optimize thresholds based on recent performance"""
        # Get current configuration
        config = self.db.get_threshold_config(document_type, field_type)
        
        if not config:
            logger.warning(f"No configuration found for {document_type.value} - {field_type.value}")
            return False
        
        # Get recent decisions
        end_date = datetime.now()
        start_date = end_date - timedelta(days=days_back)
        decisions = self.db.get_decisions_for_evaluation(config.config_id, start_date, end_date)
        
        if len(decisions) < config.min_samples_for_adjustment:
            logger.info(f"Insufficient data for optimization: {len(decisions)} < {config.min_samples_for_adjustment}")
            return False
        
        # Optimize thresholds
        new_config = self.optimizer.optimize_thresholds(config, decisions)
        
        if new_config.config_id != config.config_id:  # New configuration created
            # Save new configuration
            success = self.db.save_threshold_config(new_config)
            
            if success:
                logger.info(f"Thresholds optimized for {document_type.value} - {field_type.value}")
                return True
        
        return False
    
    def get_threshold_summary(self) -> Dict[str, Any]:
        """Get summary of all threshold configurations"""
        try:
            conn = sqlite3.connect(self.db.db_path)
            cursor = conn.cursor()
            
            # Get all current configurations
            cursor.execute("""
                SELECT document_type, field_type, auto_approve_threshold,
                       manual_review_threshold, reject_threshold, strategy,
                       version, updated_at
                FROM threshold_configs
                WHERE (document_type, field_type, version) IN (
                    SELECT document_type, field_type, MAX(version)
                    FROM threshold_configs
                    GROUP BY document_type, field_type
                )
                ORDER BY document_type, field_type
            """)
            
            configs = []
            for row in cursor.fetchall():
                configs.append({
                    'document_type': row[0],
                    'field_type': row[1],
                    'auto_approve_threshold': row[2],
                    'manual_review_threshold': row[3],
                    'reject_threshold': row[4],
                    'strategy': row[5],
                    'version': row[6],
                    'updated_at': row[7]
                })
            
            # Get recent decision statistics
            cursor.execute("""
                SELECT decision_type, COUNT(*)
                FROM processing_decisions
                WHERE decision_timestamp >= datetime('now', '-7 days')
                GROUP BY decision_type
            """)
            
            decision_stats = dict(cursor.fetchall())
            
            conn.close()
            
            return {
                'total_configurations': len(configs),
                'configurations': configs,
                'recent_decisions': decision_stats,
                'summary_generated_at': datetime.now().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Error getting threshold summary: {e}")
            return {}
    
    def run_optimization_cycle(self) -> Dict[str, Any]:
        """Run optimization cycle for all configurations"""
        results = {
            'optimized_configs': [],
            'skipped_configs': [],
            'errors': []
        }
        
        # Get all document and field type combinations
        try:
            conn = sqlite3.connect(self.db.db_path)
            cursor = conn.cursor()
            
            cursor.execute("""
                SELECT DISTINCT document_type, field_type
                FROM threshold_configs
            """)
            
            combinations = cursor.fetchall()
            conn.close()
            
            for doc_type_str, field_type_str in combinations:
                try:
                    doc_type = DocumentType(doc_type_str)
                    field_type = FieldType(field_type_str)
                    
                    success = self.optimize_thresholds(doc_type, field_type)
                    
                    if success:
                        results['optimized_configs'].append(f"{doc_type_str} - {field_type_str}")
                    else:
                        results['skipped_configs'].append(f"{doc_type_str} - {field_type_str}")
                        
                except Exception as e:
                    error_msg = f"Error optimizing {doc_type_str} - {field_type_str}: {e}"
                    results['errors'].append(error_msg)
                    logger.error(error_msg)
            
            logger.info(f"Optimization cycle completed: {len(results['optimized_configs'])} optimized, "
                       f"{len(results['skipped_configs'])} skipped, {len(results['errors'])} errors")
            
        except Exception as e:
            error_msg = f"Error running optimization cycle: {e}"
            results['errors'].append(error_msg)
            logger.error(error_msg)
        
        return results

def main():
    """Main function for standalone execution"""
    # Example usage of the confidence threshold management system
    
    # Initialize system
    threshold_manager = ConfidenceThresholdManager()
    
    print("\n=== Confidence Threshold Management Demo ===")
    
    # Simulate processing decisions
    sample_extractions = [
        {
            'document_id': 'doc_001',
            'field_name': 'ic_number',
            'field_value': '123456-78-9012',
            'confidence_score': 0.92,
            'document_type': DocumentType.IDENTITY_CARD,
            'field_type': FieldType.IC_NUMBER,
            'model_version': 'v1.0'
        },
        {
            'document_id': 'doc_002',
            'field_name': 'name',
            'field_value': 'Ahmad bin Ali',
            'confidence_score': 0.75,
            'document_type': DocumentType.IDENTITY_CARD,
            'field_type': FieldType.NAME,
            'model_version': 'v1.0'
        },
        {
            'document_id': 'doc_003',
            'field_name': 'address',
            'field_value': '123 Jalan Merdeka',
            'confidence_score': 0.45,
            'document_type': DocumentType.IDENTITY_CARD,
            'field_type': FieldType.ADDRESS,
            'model_version': 'v1.0'
        },
        {
            'document_id': 'doc_004',
            'field_name': 'ic_number',
            'field_value': '987654-32-1098',
            'confidence_score': 0.88,
            'document_type': DocumentType.IDENTITY_CARD,
            'field_type': FieldType.IC_NUMBER,
            'model_version': 'v1.0'
        }
    ]
    
    print("\n=== Making Processing Decisions ===")
    
    decisions = []
    for extraction in sample_extractions:
        decision = threshold_manager.make_decision(**extraction)
        decisions.append(decision)
        
        print(f"Field: {extraction['field_name']}, Confidence: {extraction['confidence_score']:.3f}, "
              f"Decision: {decision.decision_type.value}")
    
    # Simulate human review outcomes
    print("\n=== Simulating Human Review Outcomes ===")
    
    review_outcomes = [
        {'decision_id': decisions[0].decision_id, 'correct': True, 'reviewer': 'reviewer_001'},
        {'decision_id': decisions[1].decision_id, 'correct': True, 'reviewer': 'reviewer_002'},
        {'decision_id': decisions[2].decision_id, 'correct': False, 'reviewer': 'reviewer_001'},
        {'decision_id': decisions[3].decision_id, 'correct': True, 'reviewer': 'reviewer_003'}
    ]
    
    for outcome in review_outcomes:
        success = threshold_manager.update_decision_outcome(
            outcome['decision_id'],
            outcome['correct'],
            outcome['reviewer']
        )
        print(f"Updated decision {outcome['decision_id'][:8]}...: {outcome['correct']} (success: {success})")
    
    # Evaluate performance
    print("\n=== Performance Evaluation ===")
    
    # Get a configuration ID for evaluation
    config = threshold_manager.db.get_threshold_config(
        DocumentType.IDENTITY_CARD,
        FieldType.IC_NUMBER
    )
    
    if config:
        metrics = threshold_manager.evaluate_performance(config.config_id, days_back=1)
        
        if metrics:
            print(f"Configuration: {config.document_type.value} - {config.field_type.value}")
            print(f"Total decisions: {metrics.total_decisions}")
            print(f"Auto-approved: {metrics.auto_approved}")
            print(f"Manual review: {metrics.manual_reviewed}")
            print(f"Accuracy: {metrics.accuracy:.3f}")
            print(f"Throughput rate: {metrics.throughput_rate:.3f}")
            print(f"Overall score: {metrics.overall_score:.3f}")
    
    # Get threshold summary
    print("\n=== Threshold Summary ===")
    
    summary = threshold_manager.get_threshold_summary()
    
    print(f"Total configurations: {summary.get('total_configurations', 0)}")
    
    if summary.get('configurations'):
        print("\nCurrent threshold configurations:")
        for config in summary['configurations'][:3]:  # Show first 3
            print(f"  {config['document_type']} - {config['field_type']}: "
                  f"auto={config['auto_approve_threshold']:.2f}, "
                  f"review={config['manual_review_threshold']:.2f}, "
                  f"reject={config['reject_threshold']:.2f}")
    
    if summary.get('recent_decisions'):
        print("\nRecent decisions (7 days):")
        for decision_type, count in summary['recent_decisions'].items():
            print(f"  {decision_type}: {count}")
    
    # Run optimization cycle
    print("\n=== Running Optimization Cycle ===")
    
    optimization_results = threshold_manager.run_optimization_cycle()
    
    print(f"Optimized configurations: {len(optimization_results['optimized_configs'])}")
    print(f"Skipped configurations: {len(optimization_results['skipped_configs'])}")
    print(f"Errors: {len(optimization_results['errors'])}")
    
    if optimization_results['optimized_configs']:
        print("\nOptimized configurations:")
        for config in optimization_results['optimized_configs']:
            print(f"  - {config}")
    
    print("\n=== Demo Complete ===")
    print("Confidence threshold management system ready for production use.")

if __name__ == "__main__":
    main()