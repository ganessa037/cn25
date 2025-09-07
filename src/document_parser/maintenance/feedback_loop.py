"""Feedback Loop System for Document Parser Model Improvement

This module implements a comprehensive feedback loop system that uses human corrections
to continuously improve document parser performance, with audit trails and modification tracking.
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
    """Types of feedback for document parsing"""
    FIELD_CORRECTION = "field_correction"
    CLASSIFICATION_CORRECTION = "classification_correction"
    OCR_CORRECTION = "ocr_correction"
    VALIDATION = "validation"
    REJECTION = "rejection"
    QUALITY_SCORE = "quality_score"
    ANNOTATION_UPDATE = "annotation_update"

class LearningMode(Enum):
    """Learning modes for model updates"""
    INCREMENTAL = "incremental"
    BATCH = "batch"
    ACTIVE_LEARNING = "active_learning"
    REINFORCEMENT = "reinforcement"

class DocumentType(Enum):
    """Supported document types"""
    MYKAD = "mykad"
    SPK = "spk"
    VEHICLE_CERT = "vehicle_cert"
    GENERAL = "general"

@dataclass
class DocumentFeedback:
    """Feedback data structure for document parsing"""
    feedback_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    document_id: str = ""
    document_type: DocumentType = DocumentType.GENERAL
    feedback_type: FeedbackType = FeedbackType.VALIDATION
    
    # Original extraction results
    original_classification: Optional[str] = None
    original_fields: Dict[str, Any] = field(default_factory=dict)
    original_confidence: Dict[str, float] = field(default_factory=dict)
    
    # Corrected results
    corrected_classification: Optional[str] = None
    corrected_fields: Dict[str, Any] = field(default_factory=dict)
    field_corrections: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    
    # Metadata
    user_id: str = ""
    timestamp: datetime = field(default_factory=datetime.now)
    confidence_score: float = 0.0
    processing_time: float = 0.0
    
    # Quality metrics
    image_quality: Optional[float] = None
    ocr_confidence: Optional[float] = None
    extraction_confidence: Optional[float] = None
    
    # Additional context
    notes: str = ""
    tags: List[str] = field(default_factory=list)
    priority: int = 1  # 1-5, 5 being highest priority
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for storage"""
        data = {
            'feedback_id': self.feedback_id,
            'document_id': self.document_id,
            'document_type': self.document_type.value,
            'feedback_type': self.feedback_type.value,
            'original_classification': self.original_classification,
            'original_fields': self.original_fields,
            'original_confidence': self.original_confidence,
            'corrected_classification': self.corrected_classification,
            'corrected_fields': self.corrected_fields,
            'field_corrections': self.field_corrections,
            'user_id': self.user_id,
            'timestamp': self.timestamp.isoformat(),
            'confidence_score': self.confidence_score,
            'processing_time': self.processing_time,
            'image_quality': self.image_quality,
            'ocr_confidence': self.ocr_confidence,
            'extraction_confidence': self.extraction_confidence,
            'notes': self.notes,
            'tags': self.tags,
            'priority': self.priority
        }
        return data
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'DocumentFeedback':
        """Create from dictionary"""
        feedback = cls()
        feedback.feedback_id = data.get('feedback_id', str(uuid.uuid4()))
        feedback.document_id = data.get('document_id', '')
        feedback.document_type = DocumentType(data.get('document_type', 'general'))
        feedback.feedback_type = FeedbackType(data.get('feedback_type', 'validation'))
        feedback.original_classification = data.get('original_classification')
        feedback.original_fields = data.get('original_fields', {})
        feedback.original_confidence = data.get('original_confidence', {})
        feedback.corrected_classification = data.get('corrected_classification')
        feedback.corrected_fields = data.get('corrected_fields', {})
        feedback.field_corrections = data.get('field_corrections', {})
        feedback.user_id = data.get('user_id', '')
        feedback.timestamp = datetime.fromisoformat(data.get('timestamp', datetime.now().isoformat()))
        feedback.confidence_score = data.get('confidence_score', 0.0)
        feedback.processing_time = data.get('processing_time', 0.0)
        feedback.image_quality = data.get('image_quality')
        feedback.ocr_confidence = data.get('ocr_confidence')
        feedback.extraction_confidence = data.get('extraction_confidence')
        feedback.notes = data.get('notes', '')
        feedback.tags = data.get('tags', [])
        feedback.priority = data.get('priority', 1)
        return feedback

class FeedbackStorage:
    """Storage system for feedback data"""
    
    def __init__(self, storage_path: str = "feedback_data"):
        self.storage_path = Path(storage_path)
        self.storage_path.mkdir(exist_ok=True)
        
        # Initialize SQLite database
        self.db_path = self.storage_path / "feedback.db"
        self.init_database()
        
    def init_database(self):
        """Initialize SQLite database for feedback storage"""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS feedback (
                    feedback_id TEXT PRIMARY KEY,
                    document_id TEXT,
                    document_type TEXT,
                    feedback_type TEXT,
                    original_classification TEXT,
                    original_fields TEXT,
                    original_confidence TEXT,
                    corrected_classification TEXT,
                    corrected_fields TEXT,
                    field_corrections TEXT,
                    user_id TEXT,
                    timestamp TEXT,
                    confidence_score REAL,
                    processing_time REAL,
                    image_quality REAL,
                    ocr_confidence REAL,
                    extraction_confidence REAL,
                    notes TEXT,
                    tags TEXT,
                    priority INTEGER
                )
            """)
            
            conn.execute("""
                CREATE TABLE IF NOT EXISTS learning_sessions (
                    session_id TEXT PRIMARY KEY,
                    start_time TEXT,
                    end_time TEXT,
                    feedback_count INTEGER,
                    model_version TEXT,
                    learning_mode TEXT,
                    performance_metrics TEXT,
                    notes TEXT
                )
            """)
            
    def store_feedback(self, feedback: DocumentFeedback):
        """Store feedback in database"""
        with sqlite3.connect(self.db_path) as conn:
            data = feedback.to_dict()
            conn.execute("""
                INSERT OR REPLACE INTO feedback (
                    feedback_id, document_id, document_type, feedback_type,
                    original_classification, original_fields, original_confidence,
                    corrected_classification, corrected_fields, field_corrections,
                    user_id, timestamp, confidence_score, processing_time,
                    image_quality, ocr_confidence, extraction_confidence,
                    notes, tags, priority
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                data['feedback_id'], data['document_id'], data['document_type'], data['feedback_type'],
                data['original_classification'], json.dumps(data['original_fields']), json.dumps(data['original_confidence']),
                data['corrected_classification'], json.dumps(data['corrected_fields']), json.dumps(data['field_corrections']),
                data['user_id'], data['timestamp'], data['confidence_score'], data['processing_time'],
                data['image_quality'], data['ocr_confidence'], data['extraction_confidence'],
                data['notes'], json.dumps(data['tags']), data['priority']
            ))
            
    def get_feedback(self, feedback_id: str) -> Optional[DocumentFeedback]:
        """Retrieve feedback by ID"""
        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            cursor = conn.execute("SELECT * FROM feedback WHERE feedback_id = ?", (feedback_id,))
            row = cursor.fetchone()
            
            if row:
                data = dict(row)
                # Parse JSON fields
                data['original_fields'] = json.loads(data['original_fields'] or '{}')
                data['original_confidence'] = json.loads(data['original_confidence'] or '{}')
                data['corrected_fields'] = json.loads(data['corrected_fields'] or '{}')
                data['field_corrections'] = json.loads(data['field_corrections'] or '{}')
                data['tags'] = json.loads(data['tags'] or '[]')
                
                return DocumentFeedback.from_dict(data)
        return None
        
    def get_feedback_by_criteria(self, 
                                document_type: Optional[DocumentType] = None,
                                feedback_type: Optional[FeedbackType] = None,
                                start_date: Optional[datetime] = None,
                                end_date: Optional[datetime] = None,
                                min_priority: int = 1) -> List[DocumentFeedback]:
        """Get feedback matching criteria"""
        query = "SELECT * FROM feedback WHERE priority >= ?"
        params = [min_priority]
        
        if document_type:
            query += " AND document_type = ?"
            params.append(document_type.value)
            
        if feedback_type:
            query += " AND feedback_type = ?"
            params.append(feedback_type.value)
            
        if start_date:
            query += " AND timestamp >= ?"
            params.append(start_date.isoformat())
            
        if end_date:
            query += " AND timestamp <= ?"
            params.append(end_date.isoformat())
            
        query += " ORDER BY timestamp DESC"
        
        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            cursor = conn.execute(query, params)
            rows = cursor.fetchall()
            
            feedback_list = []
            for row in rows:
                data = dict(row)
                # Parse JSON fields
                data['original_fields'] = json.loads(data['original_fields'] or '{}')
                data['original_confidence'] = json.loads(data['original_confidence'] or '{}')
                data['corrected_fields'] = json.loads(data['corrected_fields'] or '{}')
                data['field_corrections'] = json.loads(data['field_corrections'] or '{}')
                data['tags'] = json.loads(data['tags'] or '[]')
                
                feedback_list.append(DocumentFeedback.from_dict(data))
                
        return feedback_list

class DocumentParserFeedbackSystem:
    """Main feedback system for document parser improvement"""
    
    def __init__(self, storage_path: str = "feedback_data", model_path: str = "models"):
        self.storage = FeedbackStorage(storage_path)
        self.model_path = Path(model_path)
        self.learning_history = []
        
    def collect_feedback(self, 
                        document_id: str,
                        document_type: DocumentType,
                        original_results: Dict[str, Any],
                        corrected_results: Dict[str, Any],
                        user_id: str,
                        feedback_type: FeedbackType = FeedbackType.FIELD_CORRECTION,
                        notes: str = "",
                        priority: int = 3) -> str:
        """Collect feedback from user corrections"""
        
        feedback = DocumentFeedback(
            document_id=document_id,
            document_type=document_type,
            feedback_type=feedback_type,
            original_classification=original_results.get('classification'),
            original_fields=original_results.get('fields', {}),
            original_confidence=original_results.get('confidence', {}),
            corrected_classification=corrected_results.get('classification'),
            corrected_fields=corrected_results.get('fields', {}),
            user_id=user_id,
            notes=notes,
            priority=priority
        )
        
        # Calculate field corrections
        field_corrections = {}
        original_fields = original_results.get('fields', {})
        corrected_fields = corrected_results.get('fields', {})
        
        for field_name in set(list(original_fields.keys()) + list(corrected_fields.keys())):
            original_value = original_fields.get(field_name)
            corrected_value = corrected_fields.get(field_name)
            
            if original_value != corrected_value:
                field_corrections[field_name] = {
                    'original': original_value,
                    'corrected': corrected_value,
                    'correction_type': self._determine_correction_type(original_value, corrected_value)
                }
        
        feedback.field_corrections = field_corrections
        
        # Store feedback
        self.storage.store_feedback(feedback)
        
        logger.info(f"Collected feedback for document {document_id}: {len(field_corrections)} field corrections")
        
        return feedback.feedback_id
    
    def _determine_correction_type(self, original: Any, corrected: Any) -> str:
        """Determine the type of correction made"""
        if original is None and corrected is not None:
            return "addition"
        elif original is not None and corrected is None:
            return "deletion"
        elif original != corrected:
            return "modification"
        else:
            return "no_change"
    
    def analyze_feedback_patterns(self, days: int = 30) -> Dict[str, Any]:
        """Analyze feedback patterns to identify improvement opportunities"""
        end_date = datetime.now()
        start_date = end_date - timedelta(days=days)
        
        feedback_list = self.storage.get_feedback_by_criteria(
            start_date=start_date,
            end_date=end_date
        )
        
        if not feedback_list:
            return {'message': 'No feedback data available for analysis'}
        
        analysis = {
            'total_feedback': len(feedback_list),
            'feedback_by_type': Counter([f.feedback_type.value for f in feedback_list]),
            'feedback_by_document_type': Counter([f.document_type.value for f in feedback_list]),
            'field_error_patterns': defaultdict(int),
            'classification_errors': defaultdict(int),
            'priority_distribution': Counter([f.priority for f in feedback_list]),
            'user_activity': Counter([f.user_id for f in feedback_list])
        }
        
        # Analyze field correction patterns
        for feedback in feedback_list:
            for field_name, correction in feedback.field_corrections.items():
                error_key = f"{field_name}_{correction['correction_type']}"
                analysis['field_error_patterns'][error_key] += 1
        
        # Analyze classification errors
        for feedback in feedback_list:
            if (feedback.original_classification and 
                feedback.corrected_classification and 
                feedback.original_classification != feedback.corrected_classification):
                
                error_key = f"{feedback.original_classification} -> {feedback.corrected_classification}"
                analysis['classification_errors'][error_key] += 1
        
        return analysis
    
    def identify_learning_opportunities(self, min_frequency: int = 5) -> List[Dict[str, Any]]:
        """Identify high-impact learning opportunities"""
        analysis = self.analyze_feedback_patterns()
        
        opportunities = []
        
        # Field correction opportunities
        for error_pattern, frequency in analysis['field_error_patterns'].items():
            if frequency >= min_frequency:
                field_name, correction_type = error_pattern.rsplit('_', 1)
                opportunities.append({
                    'type': 'field_extraction',
                    'field_name': field_name,
                    'correction_type': correction_type,
                    'frequency': frequency,
                    'priority': min(5, frequency // 2),
                    'suggested_action': f"Improve {field_name} extraction for {correction_type} cases"
                })
        
        # Classification opportunities
        for error_pattern, frequency in analysis['classification_errors'].items():
            if frequency >= min_frequency:
                opportunities.append({
                    'type': 'classification',
                    'error_pattern': error_pattern,
                    'frequency': frequency,
                    'priority': min(5, frequency // 2),
                    'suggested_action': f"Improve classification for pattern: {error_pattern}"
                })
        
        # Sort by priority and frequency
        opportunities.sort(key=lambda x: (x['priority'], x['frequency']), reverse=True)
        
        return opportunities
    
    def prepare_training_data(self, 
                             document_type: Optional[DocumentType] = None,
                             feedback_types: Optional[List[FeedbackType]] = None,
                             min_priority: int = 2) -> Dict[str, Any]:
        """Prepare training data from feedback"""
        
        feedback_list = self.storage.get_feedback_by_criteria(
            document_type=document_type,
            min_priority=min_priority
        )
        
        if feedback_types:
            feedback_list = [f for f in feedback_list if f.feedback_type in feedback_types]
        
        training_data = {
            'classification_corrections': [],
            'field_corrections': [],
            'metadata': {
                'total_samples': len(feedback_list),
                'document_types': list(set([f.document_type.value for f in feedback_list])),
                'feedback_types': list(set([f.feedback_type.value for f in feedback_list])),
                'date_range': {
                    'start': min([f.timestamp for f in feedback_list]).isoformat() if feedback_list else None,
                    'end': max([f.timestamp for f in feedback_list]).isoformat() if feedback_list else None
                }
            }
        }
        
        for feedback in feedback_list:
            # Classification corrections
            if (feedback.original_classification and 
                feedback.corrected_classification and 
                feedback.original_classification != feedback.corrected_classification):
                
                training_data['classification_corrections'].append({
                    'document_id': feedback.document_id,
                    'original': feedback.original_classification,
                    'corrected': feedback.corrected_classification,
                    'confidence': feedback.confidence_score,
                    'document_type': feedback.document_type.value
                })
            
            # Field corrections
            for field_name, correction in feedback.field_corrections.items():
                training_data['field_corrections'].append({
                    'document_id': feedback.document_id,
                    'document_type': feedback.document_type.value,
                    'field_name': field_name,
                    'original_value': correction['original'],
                    'corrected_value': correction['corrected'],
                    'correction_type': correction['correction_type'],
                    'confidence': feedback.original_confidence.get(field_name, 0.0)
                })
        
        return training_data
    
    def schedule_regular_updates(self, 
                                model_name: str,
                                current_version: str,
                                update_frequency_hours: int = 24) -> bool:
        """Schedule regular model updates based on feedback"""
        
        try:
            # Create update schedule configuration
            schedule_config = {
                'model_name': model_name,
                'current_version': current_version,
                'update_frequency_hours': update_frequency_hours,
                'last_update': datetime.now().isoformat(),
                'next_update': (datetime.now() + timedelta(hours=update_frequency_hours)).isoformat(),
                'min_feedback_threshold': 10,
                'min_priority_threshold': 2
            }
            
            # Save schedule configuration
            schedule_file = self.storage.storage_path / 'update_schedule.json'
            with open(schedule_file, 'w') as f:
                json.dump(schedule_config, f, indent=2)
            
            logger.info(f"Scheduled regular updates for {model_name} every {update_frequency_hours} hours")
            
            return True
            
        except Exception as e:
            logger.error(f"Failed to schedule updates: {e}")
            return False
    
    def generate_feedback_report(self, days: int = 7) -> Dict[str, Any]:
        """Generate comprehensive feedback report"""
        
        analysis = self.analyze_feedback_patterns(days)
        opportunities = self.identify_learning_opportunities()
        
        report = {
            'report_date': datetime.now().isoformat(),
            'analysis_period_days': days,
            'summary': analysis,
            'learning_opportunities': opportunities[:10],  # Top 10 opportunities
            'recommendations': []
        }
        
        # Generate recommendations
        if analysis['total_feedback'] > 0:
            success_rate = 1 - (analysis['total_feedback'] / 100)  # Simplified calculation
            
            if success_rate < 0.8:
                report['recommendations'].append({
                    'priority': 'high',
                    'action': 'Immediate model retraining recommended',
                    'reason': f'Success rate below 80%: {success_rate:.2%}'
                })
            
            if len(opportunities) > 5:
                report['recommendations'].append({
                    'priority': 'medium',
                    'action': 'Focus on top error patterns',
                    'reason': f'{len(opportunities)} improvement opportunities identified'
                })
        
        return report

def main():
    """Demo function"""
    print("🔄 Document Parser Feedback Loop System Demo")
    print("=" * 50)
    
    # Initialize feedback system
    feedback_system = DocumentParserFeedbackSystem()
    
    # Simulate collecting feedback
    print("\n=== Collecting Sample Feedback ===")
    
    # Sample feedback 1: MyKad field correction
    feedback_id1 = feedback_system.collect_feedback(
        document_id="mykad_001",
        document_type=DocumentType.MYKAD,
        original_results={
            'classification': 'mykad',
            'fields': {
                'name': 'JOHN DOE',
                'ic_number': '123456-78-9012',
                'address': '123 MAIN ST'
            },
            'confidence': {'name': 0.95, 'ic_number': 0.88, 'address': 0.72}
        },
        corrected_results={
            'classification': 'mykad',
            'fields': {
                'name': 'JOHN DOE',
                'ic_number': '123456-78-9012',
                'address': '123 MAIN STREET, KUALA LUMPUR'
            }
        },
        user_id="user_001",
        feedback_type=FeedbackType.FIELD_CORRECTION,
        notes="Address was incomplete",
        priority=3
    )
    
    print(f"✅ Collected feedback: {feedback_id1}")
    
    # Sample feedback 2: Classification correction
    feedback_id2 = feedback_system.collect_feedback(
        document_id="doc_002",
        document_type=DocumentType.SPK,
        original_results={
            'classification': 'mykad',
            'fields': {'vehicle_number': 'ABC1234'}
        },
        corrected_results={
            'classification': 'spk',
            'fields': {'vehicle_number': 'ABC1234'}
        },
        user_id="user_002",
        feedback_type=FeedbackType.CLASSIFICATION_CORRECTION,
        priority=4
    )
    
    print(f"✅ Collected feedback: {feedback_id2}")
    
    # Analyze feedback patterns
    print("\n=== Analyzing Feedback Patterns ===")
    analysis = feedback_system.analyze_feedback_patterns()
    print(f"Total feedback: {analysis['total_feedback']}")
    print(f"Feedback by type: {dict(analysis['feedback_by_type'])}")
    print(f"Field error patterns: {dict(analysis['field_error_patterns'])}")
    
    # Identify learning opportunities
    print("\n=== Learning Opportunities ===")
    opportunities = feedback_system.identify_learning_opportunities(min_frequency=1)
    for i, opportunity in enumerate(opportunities[:3], 1):
        print(f"{i}. {opportunity['suggested_action']} (frequency: {opportunity['frequency']}, priority: {opportunity['priority']})")
    
    # Generate report
    print("\n=== Feedback Report ===")
    report = feedback_system.generate_feedback_report()
    print(f"Analysis period: {report['analysis_period_days']} days")
    print(f"Learning opportunities: {len(report['learning_opportunities'])}")
    print(f"Recommendations: {len(report['recommendations'])}")
    
    # Schedule updates
    print("\n=== Scheduling Updates ===")
    success = feedback_system.schedule_regular_updates(
        model_name="DocumentParser",
        current_version="v1.0",
        update_frequency_hours=24
    )
    print(f"Update scheduling: {'✅ Success' if success else '❌ Failed'}")
    
    print("\n=== Demo Complete ===")
    print("Feedback loop system ready for continuous learning.")

if __name__ == "__main__":
    main()