"""Feedback Loop System for Model Improvement

This module implements a comprehensive feedback loop system that uses human corrections
to continuously improve model performance, with audit trails and modification tracking.
Follows the autocorrect model's organizational patterns.
"""

import json
import logging
import pickle
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Union, Tuple, Set
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from collections import defaultdict, Counter
import uuid
import hashlib

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
import sqlite3

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class FeedbackType(Enum):
    """Types of feedback"""
    CORRECTION = "correction"
    VALIDATION = "validation"
    REJECTION = "rejection"
    ANNOTATION = "annotation"
    QUALITY_SCORE = "quality_score"

class LearningMode(Enum):
    """Learning modes for model updates"""
    INCREMENTAL = "incremental"
    BATCH = "batch"
    ACTIVE_LEARNING = "active_learning"
    REINFORCEMENT = "reinforcement"

class FeedbackPriority(Enum):
    """Priority levels for feedback processing"""
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"
    CRITICAL = "critical"

class ModelUpdateStatus(Enum):
    """Status of model updates"""
    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    FAILED = "failed"
    SKIPPED = "skipped"

@dataclass
class FeedbackEntry:
    """Individual feedback entry"""
    
    feedback_id: str
    feedback_type: FeedbackType
    
    # Source information
    document_id: str
    field_name: str
    original_value: str
    corrected_value: str
    
    # Context
    document_type: str
    model_version: str
    confidence_score: float
    
    # Feedback metadata
    reviewer_id: str
    feedback_timestamp: datetime = field(default_factory=datetime.now)
    priority: FeedbackPriority = FeedbackPriority.MEDIUM
    
    # Quality metrics
    correction_confidence: Optional[float] = None
    validation_score: Optional[float] = None
    
    # Processing status
    processed: bool = False
    processed_timestamp: Optional[datetime] = None
    
    # Additional context
    context_data: Dict[str, Any] = field(default_factory=dict)
    tags: List[str] = field(default_factory=list)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return {
            'feedback_id': self.feedback_id,
            'feedback_type': self.feedback_type.value,
            'document_id': self.document_id,
            'field_name': self.field_name,
            'original_value': self.original_value,
            'corrected_value': self.corrected_value,
            'document_type': self.document_type,
            'model_version': self.model_version,
            'confidence_score': self.confidence_score,
            'reviewer_id': self.reviewer_id,
            'feedback_timestamp': self.feedback_timestamp.isoformat(),
            'priority': self.priority.value,
            'correction_confidence': self.correction_confidence,
            'validation_score': self.validation_score,
            'processed': self.processed,
            'processed_timestamp': self.processed_timestamp.isoformat() if self.processed_timestamp else None,
            'context_data': self.context_data,
            'tags': self.tags
        }
    
    def get_correction_hash(self) -> str:
        """Get hash of correction for deduplication"""
        correction_str = f"{self.field_name}:{self.original_value}:{self.corrected_value}"
        return hashlib.md5(correction_str.encode()).hexdigest()

@dataclass
class ModelUpdateRecord:
    """Record of model update"""
    
    update_id: str
    model_name: str
    model_version: str
    new_version: str
    
    # Update details
    update_type: LearningMode
    feedback_count: int
    training_data_size: int
    
    # Performance metrics
    previous_accuracy: Optional[float] = None
    new_accuracy: Optional[float] = None
    performance_improvement: Optional[float] = None
    
    # Timing
    start_time: datetime = field(default_factory=datetime.now)
    end_time: Optional[datetime] = None
    duration_seconds: Optional[float] = None
    
    # Status
    status: ModelUpdateStatus = ModelUpdateStatus.PENDING
    error_message: Optional[str] = None
    
    # Metadata
    feedback_ids: List[str] = field(default_factory=list)
    validation_results: Dict[str, float] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return {
            'update_id': self.update_id,
            'model_name': self.model_name,
            'model_version': self.model_version,
            'new_version': self.new_version,
            'update_type': self.update_type.value,
            'feedback_count': self.feedback_count,
            'training_data_size': self.training_data_size,
            'previous_accuracy': self.previous_accuracy,
            'new_accuracy': self.new_accuracy,
            'performance_improvement': self.performance_improvement,
            'start_time': self.start_time.isoformat(),
            'end_time': self.end_time.isoformat() if self.end_time else None,
            'duration_seconds': self.duration_seconds,
            'status': self.status.value,
            'error_message': self.error_message,
            'feedback_ids': self.feedback_ids,
            'validation_results': self.validation_results
        }

@dataclass
class LearningPattern:
    """Identified learning pattern from feedback"""
    
    pattern_id: str
    pattern_type: str
    
    # Pattern details
    field_name: str
    error_pattern: str
    correction_pattern: str
    frequency: int
    
    # Context
    document_types: Set[str] = field(default_factory=set)
    confidence_range: Tuple[float, float] = (0.0, 1.0)
    
    # Learning metrics
    learning_priority: float = 0.0
    impact_score: float = 0.0
    
    # Timestamps
    first_seen: datetime = field(default_factory=datetime.now)
    last_seen: datetime = field(default_factory=datetime.now)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return {
            'pattern_id': self.pattern_id,
            'pattern_type': self.pattern_type,
            'field_name': self.field_name,
            'error_pattern': self.error_pattern,
            'correction_pattern': self.correction_pattern,
            'frequency': self.frequency,
            'document_types': list(self.document_types),
            'confidence_range': self.confidence_range,
            'learning_priority': self.learning_priority,
            'impact_score': self.impact_score,
            'first_seen': self.first_seen.isoformat(),
            'last_seen': self.last_seen.isoformat()
        }

class FeedbackDatabase:
    """Database for storing feedback and learning data"""
    
    def __init__(self, db_path: str = "feedback_system.db"):
        self.db_path = db_path
        self.init_database()
    
    def init_database(self):
        """Initialize database tables"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Feedback entries table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS feedback_entries (
                feedback_id TEXT PRIMARY KEY,
                feedback_type TEXT NOT NULL,
                document_id TEXT NOT NULL,
                field_name TEXT NOT NULL,
                original_value TEXT,
                corrected_value TEXT,
                document_type TEXT,
                model_version TEXT,
                confidence_score REAL,
                reviewer_id TEXT,
                feedback_timestamp TIMESTAMP,
                priority TEXT,
                correction_confidence REAL,
                validation_score REAL,
                processed BOOLEAN DEFAULT 0,
                processed_timestamp TIMESTAMP,
                context_data TEXT,
                tags TEXT,
                correction_hash TEXT
            )
        """)
        
        # Model updates table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS model_updates (
                update_id TEXT PRIMARY KEY,
                model_name TEXT NOT NULL,
                model_version TEXT NOT NULL,
                new_version TEXT NOT NULL,
                update_type TEXT NOT NULL,
                feedback_count INTEGER,
                training_data_size INTEGER,
                previous_accuracy REAL,
                new_accuracy REAL,
                performance_improvement REAL,
                start_time TIMESTAMP,
                end_time TIMESTAMP,
                duration_seconds REAL,
                status TEXT,
                error_message TEXT,
                feedback_ids TEXT,
                validation_results TEXT
            )
        """)
        
        # Learning patterns table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS learning_patterns (
                pattern_id TEXT PRIMARY KEY,
                pattern_type TEXT NOT NULL,
                field_name TEXT NOT NULL,
                error_pattern TEXT,
                correction_pattern TEXT,
                frequency INTEGER DEFAULT 1,
                document_types TEXT,
                confidence_range_min REAL,
                confidence_range_max REAL,
                learning_priority REAL,
                impact_score REAL,
                first_seen TIMESTAMP,
                last_seen TIMESTAMP
            )
        """)
        
        # Audit trail table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS audit_trail (
                audit_id TEXT PRIMARY KEY,
                entity_type TEXT NOT NULL,
                entity_id TEXT NOT NULL,
                action TEXT NOT NULL,
                old_value TEXT,
                new_value TEXT,
                user_id TEXT,
                timestamp TIMESTAMP,
                metadata TEXT
            )
        """)
        
        # Create indexes for performance
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_feedback_document ON feedback_entries(document_id)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_feedback_field ON feedback_entries(field_name)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_feedback_processed ON feedback_entries(processed)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_patterns_field ON learning_patterns(field_name)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_audit_entity ON audit_trail(entity_type, entity_id)")
        
        conn.commit()
        conn.close()
        
        logger.info(f"Feedback database initialized at {self.db_path}")
    
    def add_feedback(self, feedback: FeedbackEntry) -> bool:
        """Add feedback entry to database"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            cursor.execute("""
                INSERT INTO feedback_entries (
                    feedback_id, feedback_type, document_id, field_name,
                    original_value, corrected_value, document_type, model_version,
                    confidence_score, reviewer_id, feedback_timestamp, priority,
                    correction_confidence, validation_score, processed,
                    processed_timestamp, context_data, tags, correction_hash
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                feedback.feedback_id,
                feedback.feedback_type.value,
                feedback.document_id,
                feedback.field_name,
                feedback.original_value,
                feedback.corrected_value,
                feedback.document_type,
                feedback.model_version,
                feedback.confidence_score,
                feedback.reviewer_id,
                feedback.feedback_timestamp,
                feedback.priority.value,
                feedback.correction_confidence,
                feedback.validation_score,
                feedback.processed,
                feedback.processed_timestamp,
                json.dumps(feedback.context_data),
                json.dumps(feedback.tags),
                feedback.get_correction_hash()
            ))
            
            conn.commit()
            conn.close()
            
            # Add audit trail
            self.add_audit_entry(
                entity_type="feedback",
                entity_id=feedback.feedback_id,
                action="created",
                user_id=feedback.reviewer_id,
                metadata={'feedback_type': feedback.feedback_type.value}
            )
            
            return True
            
        except Exception as e:
            logger.error(f"Error adding feedback: {e}")
            return False
    
    def get_unprocessed_feedback(self, limit: int = 100, 
                                priority: FeedbackPriority = None) -> List[FeedbackEntry]:
        """Get unprocessed feedback entries"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            query = "SELECT * FROM feedback_entries WHERE processed = 0"
            params = []
            
            if priority:
                query += " AND priority = ?"
                params.append(priority.value)
            
            query += " ORDER BY feedback_timestamp ASC LIMIT ?"
            params.append(limit)
            
            cursor.execute(query, params)
            rows = cursor.fetchall()
            
            conn.close()
            
            return [self._row_to_feedback(row) for row in rows]
            
        except Exception as e:
            logger.error(f"Error getting unprocessed feedback: {e}")
            return []
    
    def mark_feedback_processed(self, feedback_ids: List[str]) -> bool:
        """Mark feedback as processed"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            placeholders = ','.join(['?' for _ in feedback_ids])
            cursor.execute(f"""
                UPDATE feedback_entries 
                SET processed = 1, processed_timestamp = ?
                WHERE feedback_id IN ({placeholders})
            """, [datetime.now()] + feedback_ids)
            
            conn.commit()
            conn.close()
            return True
            
        except Exception as e:
            logger.error(f"Error marking feedback as processed: {e}")
            return False
    
    def add_model_update(self, update: ModelUpdateRecord) -> bool:
        """Add model update record"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            cursor.execute("""
                INSERT INTO model_updates (
                    update_id, model_name, model_version, new_version,
                    update_type, feedback_count, training_data_size,
                    previous_accuracy, new_accuracy, performance_improvement,
                    start_time, end_time, duration_seconds, status,
                    error_message, feedback_ids, validation_results
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                update.update_id,
                update.model_name,
                update.model_version,
                update.new_version,
                update.update_type.value,
                update.feedback_count,
                update.training_data_size,
                update.previous_accuracy,
                update.new_accuracy,
                update.performance_improvement,
                update.start_time,
                update.end_time,
                update.duration_seconds,
                update.status.value,
                update.error_message,
                json.dumps(update.feedback_ids),
                json.dumps(update.validation_results)
            ))
            
            conn.commit()
            conn.close()
            return True
            
        except Exception as e:
            logger.error(f"Error adding model update: {e}")
            return False
    
    def add_learning_pattern(self, pattern: LearningPattern) -> bool:
        """Add learning pattern"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            cursor.execute("""
                INSERT OR REPLACE INTO learning_patterns (
                    pattern_id, pattern_type, field_name, error_pattern,
                    correction_pattern, frequency, document_types,
                    confidence_range_min, confidence_range_max,
                    learning_priority, impact_score, first_seen, last_seen
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                pattern.pattern_id,
                pattern.pattern_type,
                pattern.field_name,
                pattern.error_pattern,
                pattern.correction_pattern,
                pattern.frequency,
                json.dumps(list(pattern.document_types)),
                pattern.confidence_range[0],
                pattern.confidence_range[1],
                pattern.learning_priority,
                pattern.impact_score,
                pattern.first_seen,
                pattern.last_seen
            ))
            
            conn.commit()
            conn.close()
            return True
            
        except Exception as e:
            logger.error(f"Error adding learning pattern: {e}")
            return False
    
    def add_audit_entry(self, entity_type: str, entity_id: str, action: str,
                       user_id: str = None, old_value: str = None,
                       new_value: str = None, metadata: Dict[str, Any] = None) -> bool:
        """Add audit trail entry"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            audit_id = str(uuid.uuid4())
            
            cursor.execute("""
                INSERT INTO audit_trail (
                    audit_id, entity_type, entity_id, action,
                    old_value, new_value, user_id, timestamp, metadata
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                audit_id,
                entity_type,
                entity_id,
                action,
                old_value,
                new_value,
                user_id,
                datetime.now(),
                json.dumps(metadata) if metadata else None
            ))
            
            conn.commit()
            conn.close()
            return True
            
        except Exception as e:
            logger.error(f"Error adding audit entry: {e}")
            return False
    
    def get_feedback_stats(self, days: int = 30) -> Dict[str, Any]:
        """Get feedback statistics"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            start_date = datetime.now() - timedelta(days=days)
            
            # Total feedback count
            cursor.execute(
                "SELECT COUNT(*) FROM feedback_entries WHERE feedback_timestamp >= ?",
                (start_date,)
            )
            total_feedback = cursor.fetchone()[0]
            
            # Feedback by type
            cursor.execute("""
                SELECT feedback_type, COUNT(*) 
                FROM feedback_entries 
                WHERE feedback_timestamp >= ?
                GROUP BY feedback_type
            """, (start_date,))
            feedback_by_type = dict(cursor.fetchall())
            
            # Feedback by field
            cursor.execute("""
                SELECT field_name, COUNT(*) 
                FROM feedback_entries 
                WHERE feedback_timestamp >= ?
                GROUP BY field_name
                ORDER BY COUNT(*) DESC
                LIMIT 10
            """, (start_date,))
            top_fields = dict(cursor.fetchall())
            
            # Processing rate
            cursor.execute(
                "SELECT COUNT(*) FROM feedback_entries WHERE processed = 1 AND feedback_timestamp >= ?",
                (start_date,)
            )
            processed_count = cursor.fetchone()[0]
            
            processing_rate = (processed_count / total_feedback * 100) if total_feedback > 0 else 0
            
            conn.close()
            
            return {
                'total_feedback': total_feedback,
                'feedback_by_type': feedback_by_type,
                'top_fields': top_fields,
                'processed_count': processed_count,
                'processing_rate': processing_rate,
                'period_days': days
            }
            
        except Exception as e:
            logger.error(f"Error getting feedback stats: {e}")
            return {}
    
    def _row_to_feedback(self, row) -> FeedbackEntry:
        """Convert database row to FeedbackEntry"""
        return FeedbackEntry(
            feedback_id=row[0],
            feedback_type=FeedbackType(row[1]),
            document_id=row[2],
            field_name=row[3],
            original_value=row[4] or "",
            corrected_value=row[5] or "",
            document_type=row[6] or "",
            model_version=row[7] or "",
            confidence_score=row[8] or 0.0,
            reviewer_id=row[9] or "",
            feedback_timestamp=datetime.fromisoformat(row[10]) if row[10] else datetime.now(),
            priority=FeedbackPriority(row[11]) if row[11] else FeedbackPriority.MEDIUM,
            correction_confidence=row[12],
            validation_score=row[13],
            processed=bool(row[14]),
            processed_timestamp=datetime.fromisoformat(row[15]) if row[15] else None,
            context_data=json.loads(row[16]) if row[16] else {},
            tags=json.loads(row[17]) if row[17] else []
        )

class PatternAnalyzer:
    """Analyzer for identifying learning patterns from feedback"""
    
    def __init__(self, db: FeedbackDatabase):
        self.db = db
    
    def analyze_correction_patterns(self, feedback_entries: List[FeedbackEntry]) -> List[LearningPattern]:
        """Analyze correction patterns from feedback"""
        patterns = []
        
        # Group by field and analyze patterns
        field_corrections = defaultdict(list)
        
        for feedback in feedback_entries:
            if feedback.feedback_type == FeedbackType.CORRECTION:
                field_corrections[feedback.field_name].append(feedback)
        
        for field_name, corrections in field_corrections.items():
            # Find common error patterns
            error_patterns = self._find_error_patterns(corrections)
            
            for pattern_data in error_patterns:
                pattern = LearningPattern(
                    pattern_id=str(uuid.uuid4()),
                    pattern_type="correction_pattern",
                    field_name=field_name,
                    error_pattern=pattern_data['error_pattern'],
                    correction_pattern=pattern_data['correction_pattern'],
                    frequency=pattern_data['frequency']
                )
                
                # Calculate priority and impact
                pattern.learning_priority = self._calculate_learning_priority(pattern_data)
                pattern.impact_score = self._calculate_impact_score(pattern_data)
                
                # Set document types and confidence range
                pattern.document_types = set(c.document_type for c in pattern_data['corrections'])
                confidences = [c.confidence_score for c in pattern_data['corrections']]
                pattern.confidence_range = (min(confidences), max(confidences))
                
                patterns.append(pattern)
        
        return patterns
    
    def _find_error_patterns(self, corrections: List[FeedbackEntry]) -> List[Dict[str, Any]]:
        """Find common error patterns in corrections"""
        pattern_groups = defaultdict(list)
        
        for correction in corrections:
            # Simple pattern matching - can be enhanced with NLP
            original = correction.original_value.lower().strip()
            corrected = correction.corrected_value.lower().strip()
            
            # Group similar corrections
            pattern_key = f"{original[:10]}...{corrected[:10]}"
            pattern_groups[pattern_key].append(correction)
        
        patterns = []
        for pattern_key, group in pattern_groups.items():
            if len(group) >= 2:  # Only patterns that occur multiple times
                patterns.append({
                    'error_pattern': group[0].original_value,
                    'correction_pattern': group[0].corrected_value,
                    'frequency': len(group),
                    'corrections': group
                })
        
        return patterns
    
    def _calculate_learning_priority(self, pattern_data: Dict[str, Any]) -> float:
        """Calculate learning priority for pattern"""
        frequency = pattern_data['frequency']
        
        # Higher frequency = higher priority
        frequency_score = min(frequency / 10.0, 1.0)
        
        # Lower confidence = higher priority
        avg_confidence = sum(c.confidence_score for c in pattern_data['corrections']) / len(pattern_data['corrections'])
        confidence_score = 1.0 - avg_confidence
        
        return (frequency_score * 0.6) + (confidence_score * 0.4)
    
    def _calculate_impact_score(self, pattern_data: Dict[str, Any]) -> float:
        """Calculate impact score for pattern"""
        # Impact based on frequency and document type diversity
        frequency = pattern_data['frequency']
        doc_types = set(c.document_type for c in pattern_data['corrections'])
        
        frequency_impact = min(frequency / 20.0, 1.0)
        diversity_impact = min(len(doc_types) / 5.0, 1.0)
        
        return (frequency_impact * 0.7) + (diversity_impact * 0.3)

class ModelUpdater:
    """Model updater using feedback data"""
    
    def __init__(self, db: FeedbackDatabase):
        self.db = db
        self.pattern_analyzer = PatternAnalyzer(db)
    
    def prepare_training_data(self, feedback_entries: List[FeedbackEntry]) -> Tuple[List[Dict], List[str]]:
        """Prepare training data from feedback"""
        training_samples = []
        labels = []
        
        for feedback in feedback_entries:
            if feedback.feedback_type == FeedbackType.CORRECTION:
                # Create training sample
                sample = {
                    'field_name': feedback.field_name,
                    'original_text': feedback.original_value,
                    'document_type': feedback.document_type,
                    'confidence_score': feedback.confidence_score,
                    'context': feedback.context_data
                }
                
                training_samples.append(sample)
                labels.append(feedback.corrected_value)
        
        return training_samples, labels
    
    def update_model_incremental(self, model_name: str, current_version: str,
                                feedback_entries: List[FeedbackEntry]) -> ModelUpdateRecord:
        """Update model incrementally with new feedback"""
        update_id = str(uuid.uuid4())
        new_version = f"{current_version}.{int(datetime.now().timestamp())}"
        
        update_record = ModelUpdateRecord(
            update_id=update_id,
            model_name=model_name,
            model_version=current_version,
            new_version=new_version,
            update_type=LearningMode.INCREMENTAL,
            feedback_count=len(feedback_entries),
            training_data_size=len(feedback_entries),
            feedback_ids=[f.feedback_id for f in feedback_entries]
        )
        
        try:
            update_record.status = ModelUpdateStatus.IN_PROGRESS
            
            # Prepare training data
            training_data, labels = self.prepare_training_data(feedback_entries)
            
            if not training_data:
                update_record.status = ModelUpdateStatus.SKIPPED
                update_record.error_message = "No valid training data"
                return update_record
            
            # Simulate model update (replace with actual model training)
            logger.info(f"Updating model {model_name} with {len(training_data)} samples")
            
            # Analyze patterns
            patterns = self.pattern_analyzer.analyze_correction_patterns(feedback_entries)
            
            # Save patterns to database
            for pattern in patterns:
                self.db.add_learning_pattern(pattern)
            
            # Simulate performance improvement
            update_record.previous_accuracy = 0.85  # Mock previous accuracy
            update_record.new_accuracy = min(0.95, update_record.previous_accuracy + (len(patterns) * 0.01))
            update_record.performance_improvement = update_record.new_accuracy - update_record.previous_accuracy
            
            # Mark as completed
            update_record.end_time = datetime.now()
            update_record.duration_seconds = (update_record.end_time - update_record.start_time).total_seconds()
            update_record.status = ModelUpdateStatus.COMPLETED
            
            # Validation results
            update_record.validation_results = {
                'patterns_identified': len(patterns),
                'high_priority_patterns': len([p for p in patterns if p.learning_priority > 0.7]),
                'field_coverage': len(set(f.field_name for f in feedback_entries))
            }
            
            logger.info(f"Model update completed: {update_id}")
            
        except Exception as e:
            update_record.status = ModelUpdateStatus.FAILED
            update_record.error_message = str(e)
            update_record.end_time = datetime.now()
            logger.error(f"Model update failed: {e}")
        
        # Save update record
        self.db.add_model_update(update_record)
        
        return update_record
    
    def update_model_batch(self, model_name: str, current_version: str,
                          days_back: int = 7) -> ModelUpdateRecord:
        """Update model with batch of recent feedback"""
        # Get recent unprocessed feedback
        feedback_entries = self.db.get_unprocessed_feedback(limit=1000)
        
        if not feedback_entries:
            logger.info("No unprocessed feedback for batch update")
            return None
        
        # Filter recent feedback
        cutoff_date = datetime.now() - timedelta(days=days_back)
        recent_feedback = [f for f in feedback_entries if f.feedback_timestamp >= cutoff_date]
        
        if not recent_feedback:
            logger.info("No recent feedback for batch update")
            return None
        
        # Perform incremental update
        update_record = self.update_model_incremental(model_name, current_version, recent_feedback)
        update_record.update_type = LearningMode.BATCH
        
        # Mark feedback as processed
        if update_record.status == ModelUpdateStatus.COMPLETED:
            feedback_ids = [f.feedback_id for f in recent_feedback]
            self.db.mark_feedback_processed(feedback_ids)
        
        return update_record

class FeedbackLoopSystem:
    """Main feedback loop system"""
    
    def __init__(self, db_path: str = "feedback_system.db"):
        self.db = FeedbackDatabase(db_path)
        self.model_updater = ModelUpdater(self.db)
        self.pattern_analyzer = PatternAnalyzer(self.db)
        
        logger.info("Feedback loop system initialized")
    
    def collect_feedback(self, document_id: str, field_name: str,
                        original_value: str, corrected_value: str,
                        document_type: str, model_version: str,
                        confidence_score: float, reviewer_id: str,
                        feedback_type: FeedbackType = FeedbackType.CORRECTION,
                        priority: FeedbackPriority = FeedbackPriority.MEDIUM,
                        context_data: Dict[str, Any] = None) -> str:
        """Collect feedback from human reviewers"""
        feedback_id = str(uuid.uuid4())
        
        feedback = FeedbackEntry(
            feedback_id=feedback_id,
            feedback_type=feedback_type,
            document_id=document_id,
            field_name=field_name,
            original_value=original_value,
            corrected_value=corrected_value,
            document_type=document_type,
            model_version=model_version,
            confidence_score=confidence_score,
            reviewer_id=reviewer_id,
            priority=priority,
            context_data=context_data or {}
        )
        
        # Add to database
        success = self.db.add_feedback(feedback)
        
        if success:
            logger.info(f"Feedback collected: {feedback_id}")
            
            # Check if immediate update is needed (high priority)
            if priority == FeedbackPriority.CRITICAL:
                self._trigger_immediate_update(model_version, [feedback])
            
            return feedback_id
        else:
            raise Exception("Failed to collect feedback")
    
    def process_feedback_batch(self, model_name: str, current_version: str,
                              batch_size: int = 100) -> Optional[ModelUpdateRecord]:
        """Process a batch of feedback for model improvement"""
        # Get unprocessed feedback
        feedback_entries = self.db.get_unprocessed_feedback(limit=batch_size)
        
        if not feedback_entries:
            logger.info("No unprocessed feedback to process")
            return None
        
        logger.info(f"Processing {len(feedback_entries)} feedback entries")
        
        # Update model
        update_record = self.model_updater.update_model_incremental(
            model_name, current_version, feedback_entries
        )
        
        # Mark feedback as processed if update was successful
        if update_record.status == ModelUpdateStatus.COMPLETED:
            feedback_ids = [f.feedback_id for f in feedback_entries]
            self.db.mark_feedback_processed(feedback_ids)
            
            logger.info(f"Processed {len(feedback_ids)} feedback entries")
        
        return update_record
    
    def schedule_regular_updates(self, model_name: str, current_version: str,
                               update_frequency_hours: int = 24) -> bool:
        """Schedule regular model updates based on feedback"""
        # This would typically be implemented with a scheduler like Celery
        # For now, we'll just log the scheduling
        
        logger.info(f"Scheduled regular updates for {model_name} every {update_frequency_hours} hours")
        
        # In a real implementation, you would:
        # 1. Set up a scheduled task
        # 2. Call process_feedback_batch periodically
        # 3. Monitor update success/failure
        
        return True
    
    def get_learning_insights(self, days: int = 30) -> Dict[str, Any]:
        """Get insights from learning patterns and feedback"""
        # Get feedback statistics
        stats = self.db.get_feedback_stats(days)
        
        # Get recent patterns
        try:
            conn = sqlite3.connect(self.db.db_path)
            cursor = conn.cursor()
            
            start_date = datetime.now() - timedelta(days=days)
            
            cursor.execute("""
                SELECT pattern_type, COUNT(*), AVG(learning_priority), AVG(impact_score)
                FROM learning_patterns 
                WHERE last_seen >= ?
                GROUP BY pattern_type
            """, (start_date,))
            
            pattern_stats = {}
            for row in cursor.fetchall():
                pattern_stats[row[0]] = {
                    'count': row[1],
                    'avg_priority': row[2],
                    'avg_impact': row[3]
                }
            
            # Get top learning opportunities
            cursor.execute("""
                SELECT field_name, COUNT(*), AVG(learning_priority)
                FROM learning_patterns 
                WHERE last_seen >= ?
                GROUP BY field_name
                ORDER BY AVG(learning_priority) DESC
                LIMIT 10
            """, (start_date,))
            
            top_opportunities = []
            for row in cursor.fetchall():
                top_opportunities.append({
                    'field_name': row[0],
                    'pattern_count': row[1],
                    'avg_priority': row[2]
                })
            
            conn.close()
            
        except Exception as e:
            logger.error(f"Error getting learning insights: {e}")
            pattern_stats = {}
            top_opportunities = []
        
        return {
            'feedback_stats': stats,
            'pattern_stats': pattern_stats,
            'top_learning_opportunities': top_opportunities,
            'period_days': days
        }
    
    def _trigger_immediate_update(self, model_version: str, 
                                 critical_feedback: List[FeedbackEntry]):
        """Trigger immediate model update for critical feedback"""
        logger.warning(f"Triggering immediate update for {len(critical_feedback)} critical feedback entries")
        
        # In a real implementation, this would:
        # 1. Queue high-priority model update
        # 2. Notify relevant teams
        # 3. Apply hotfix if needed
        
        # For now, just log the action
        for feedback in critical_feedback:
            logger.warning(f"Critical feedback: {feedback.field_name} - {feedback.original_value} -> {feedback.corrected_value}")

def main():
    """Main function for standalone execution"""
    # Example usage of the feedback loop system
    
    # Initialize system
    feedback_system = FeedbackLoopSystem()
    
    print("\n=== Feedback Loop System Demo ===")
    
    # Simulate collecting feedback
    sample_feedback = [
        {
            'document_id': 'doc_001',
            'field_name': 'ic_number',
            'original_value': '123456789012',
            'corrected_value': '123456-78-9012',
            'document_type': 'identity_card',
            'model_version': 'v1.0',
            'confidence_score': 0.75,
            'reviewer_id': 'reviewer_001',
            'priority': FeedbackPriority.HIGH
        },
        {
            'document_id': 'doc_002',
            'field_name': 'name',
            'original_value': 'Ahmad Ali',
            'corrected_value': 'Ahmad bin Ali',
            'document_type': 'identity_card',
            'model_version': 'v1.0',
            'confidence_score': 0.82,
            'reviewer_id': 'reviewer_002',
            'priority': FeedbackPriority.MEDIUM
        },
        {
            'document_id': 'doc_003',
            'field_name': 'address',
            'original_value': '123 Jalan Merdeka',
            'corrected_value': '123 Jalan Merdeka, Kuala Lumpur',
            'document_type': 'identity_card',
            'model_version': 'v1.0',
            'confidence_score': 0.68,
            'reviewer_id': 'reviewer_001',
            'priority': FeedbackPriority.HIGH
        }
    ]
    
    print("\n=== Collecting Feedback ===")
    
    feedback_ids = []
    for feedback_data in sample_feedback:
        feedback_id = feedback_system.collect_feedback(**feedback_data)
        feedback_ids.append(feedback_id)
        print(f"Collected feedback: {feedback_id[:8]}... for {feedback_data['field_name']}")
    
    # Process feedback batch
    print("\n=== Processing Feedback Batch ===")
    
    update_record = feedback_system.process_feedback_batch(
        model_name="DocumentParser",
        current_version="v1.0",
        batch_size=10
    )
    
    if update_record:
        print(f"Model update completed: {update_record.update_id[:8]}...")
        print(f"Status: {update_record.status.value}")
        print(f"Performance improvement: {update_record.performance_improvement:.3f}")
        print(f"Patterns identified: {update_record.validation_results.get('patterns_identified', 0)}")
    else:
        print("No model update performed")
    
    # Get learning insights
    print("\n=== Learning Insights ===")
    
    insights = feedback_system.get_learning_insights(days=7)
    
    print(f"Total feedback (7 days): {insights['feedback_stats'].get('total_feedback', 0)}")
    print(f"Processing rate: {insights['feedback_stats'].get('processing_rate', 0):.1f}%")
    
    if insights['top_learning_opportunities']:
        print("\nTop learning opportunities:")
        for i, opportunity in enumerate(insights['top_learning_opportunities'][:3], 1):
            print(f"{i}. {opportunity['field_name']}: {opportunity['pattern_count']} patterns (priority: {opportunity['avg_priority']:.2f})")
    
    # Schedule regular updates
    print("\n=== Scheduling Regular Updates ===")
    
    success = feedback_system.schedule_regular_updates(
        model_name="DocumentParser",
        current_version="v1.1",
        update_frequency_hours=24
    )
    
    print(f"Regular updates scheduled: {success}")
    
    print("\n=== Demo Complete ===")
    print("Feedback loop system ready for continuous learning.")

if __name__ == "__main__":
    main()