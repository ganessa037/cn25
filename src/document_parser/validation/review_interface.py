"""Review Interface for Human-in-the-Loop Validation

This module provides a web interface for manual verification of document processing results,
with confidence thresholds, auto-approval mechanisms, and feedback collection.
Follows the autocorrect model's organizational patterns.
"""

import json
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Union, Tuple
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from collections import defaultdict
import uuid

import pandas as pd
from flask import Flask, render_template, request, jsonify, session, redirect, url_for
from flask_sqlalchemy import SQLAlchemy
from flask_login import LoginManager, UserMixin, login_user, logout_user, login_required, current_user
from werkzeug.security import generate_password_hash, check_password_hash
from sqlalchemy import func, and_, or_
import sqlite3

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ReviewStatus(Enum):
    """Status of review items"""
    PENDING = "pending"
    IN_REVIEW = "in_review"
    APPROVED = "approved"
    REJECTED = "rejected"
    AUTO_APPROVED = "auto_approved"
    NEEDS_REVISION = "needs_revision"

class ConfidenceLevel(Enum):
    """Confidence levels for auto-approval"""
    HIGH = "high"  # >= 0.9
    MEDIUM = "medium"  # 0.7 - 0.89
    LOW = "low"  # < 0.7

class DocumentType(Enum):
    """Document types for review"""
    IDENTITY_CARD = "identity_card"
    PASSPORT = "passport"
    DRIVING_LICENSE = "driving_license"
    VEHICLE_REGISTRATION = "vehicle_registration"
    BANK_STATEMENT = "bank_statement"
    INVOICE = "invoice"
    RECEIPT = "receipt"
    UTILITY_BILL = "utility_bill"

@dataclass
class ReviewItem:
    """Item for human review"""
    
    item_id: str
    document_type: DocumentType
    
    # Original document data
    original_image_path: Optional[str] = None
    
    # Extracted data
    extracted_data: Dict[str, Any] = field(default_factory=dict)
    confidence_scores: Dict[str, float] = field(default_factory=dict)
    
    # Review data
    status: ReviewStatus = ReviewStatus.PENDING
    assigned_reviewer: Optional[str] = None
    review_notes: str = ""
    corrected_data: Dict[str, Any] = field(default_factory=dict)
    
    # Timestamps
    created_at: datetime = field(default_factory=datetime.now)
    assigned_at: Optional[datetime] = None
    reviewed_at: Optional[datetime] = None
    
    # Metadata
    processing_time: Optional[float] = None
    model_version: Optional[str] = None
    priority: int = 1  # 1=low, 2=medium, 3=high
    
    def get_overall_confidence(self) -> float:
        """Get overall confidence score"""
        if not self.confidence_scores:
            return 0.0
        return sum(self.confidence_scores.values()) / len(self.confidence_scores)
    
    def get_confidence_level(self) -> ConfidenceLevel:
        """Get confidence level category"""
        overall_conf = self.get_overall_confidence()
        if overall_conf >= 0.9:
            return ConfidenceLevel.HIGH
        elif overall_conf >= 0.7:
            return ConfidenceLevel.MEDIUM
        else:
            return ConfidenceLevel.LOW
    
    def should_auto_approve(self, threshold: float = 0.9) -> bool:
        """Check if item should be auto-approved"""
        return self.get_overall_confidence() >= threshold
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return {
            'item_id': self.item_id,
            'document_type': self.document_type.value,
            'original_image_path': self.original_image_path,
            'extracted_data': self.extracted_data,
            'confidence_scores': self.confidence_scores,
            'status': self.status.value,
            'assigned_reviewer': self.assigned_reviewer,
            'review_notes': self.review_notes,
            'corrected_data': self.corrected_data,
            'created_at': self.created_at.isoformat(),
            'assigned_at': self.assigned_at.isoformat() if self.assigned_at else None,
            'reviewed_at': self.reviewed_at.isoformat() if self.reviewed_at else None,
            'processing_time': self.processing_time,
            'model_version': self.model_version,
            'priority': self.priority,
            'overall_confidence': self.get_overall_confidence(),
            'confidence_level': self.get_confidence_level().value
        }

@dataclass
class ReviewerStats:
    """Statistics for reviewer performance"""
    
    reviewer_id: str
    total_reviewed: int = 0
    approved_count: int = 0
    rejected_count: int = 0
    avg_review_time: float = 0.0
    accuracy_score: float = 0.0
    
    # Time-based stats
    reviews_today: int = 0
    reviews_this_week: int = 0
    reviews_this_month: int = 0
    
    def get_approval_rate(self) -> float:
        """Get approval rate"""
        if self.total_reviewed == 0:
            return 0.0
        return self.approved_count / self.total_reviewed
    
    def get_rejection_rate(self) -> float:
        """Get rejection rate"""
        if self.total_reviewed == 0:
            return 0.0
        return self.rejected_count / self.total_reviewed

class ReviewDatabase:
    """Database interface for review system"""
    
    def __init__(self, db_path: str = "review_system.db"):
        self.db_path = db_path
        self.init_database()
    
    def init_database(self):
        """Initialize database tables"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Review items table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS review_items (
                item_id TEXT PRIMARY KEY,
                document_type TEXT NOT NULL,
                original_image_path TEXT,
                extracted_data TEXT,
                confidence_scores TEXT,
                status TEXT NOT NULL,
                assigned_reviewer TEXT,
                review_notes TEXT,
                corrected_data TEXT,
                created_at TIMESTAMP,
                assigned_at TIMESTAMP,
                reviewed_at TIMESTAMP,
                processing_time REAL,
                model_version TEXT,
                priority INTEGER DEFAULT 1
            )
        """)
        
        # Users table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS users (
                user_id TEXT PRIMARY KEY,
                username TEXT UNIQUE NOT NULL,
                email TEXT UNIQUE NOT NULL,
                password_hash TEXT NOT NULL,
                role TEXT NOT NULL,
                created_at TIMESTAMP,
                last_login TIMESTAMP,
                is_active BOOLEAN DEFAULT 1
            )
        """)
        
        # Review history table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS review_history (
                history_id TEXT PRIMARY KEY,
                item_id TEXT NOT NULL,
                reviewer_id TEXT NOT NULL,
                action TEXT NOT NULL,
                old_status TEXT,
                new_status TEXT,
                changes TEXT,
                timestamp TIMESTAMP,
                FOREIGN KEY (item_id) REFERENCES review_items (item_id),
                FOREIGN KEY (reviewer_id) REFERENCES users (user_id)
            )
        """)
        
        # Feedback table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS feedback (
                feedback_id TEXT PRIMARY KEY,
                item_id TEXT NOT NULL,
                reviewer_id TEXT NOT NULL,
                field_name TEXT NOT NULL,
                original_value TEXT,
                corrected_value TEXT,
                confidence_before REAL,
                feedback_type TEXT,
                timestamp TIMESTAMP,
                FOREIGN KEY (item_id) REFERENCES review_items (item_id),
                FOREIGN KEY (reviewer_id) REFERENCES users (user_id)
            )
        """)
        
        conn.commit()
        conn.close()
        
        logger.info(f"Database initialized at {self.db_path}")
    
    def add_review_item(self, item: ReviewItem) -> bool:
        """Add review item to database"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            cursor.execute("""
                INSERT INTO review_items (
                    item_id, document_type, original_image_path, extracted_data,
                    confidence_scores, status, assigned_reviewer, review_notes,
                    corrected_data, created_at, assigned_at, reviewed_at,
                    processing_time, model_version, priority
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                item.item_id,
                item.document_type.value,
                item.original_image_path,
                json.dumps(item.extracted_data),
                json.dumps(item.confidence_scores),
                item.status.value,
                item.assigned_reviewer,
                item.review_notes,
                json.dumps(item.corrected_data),
                item.created_at,
                item.assigned_at,
                item.reviewed_at,
                item.processing_time,
                item.model_version,
                item.priority
            ))
            
            conn.commit()
            conn.close()
            return True
            
        except Exception as e:
            logger.error(f"Error adding review item: {e}")
            return False
    
    def get_review_item(self, item_id: str) -> Optional[ReviewItem]:
        """Get review item by ID"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            cursor.execute("SELECT * FROM review_items WHERE item_id = ?", (item_id,))
            row = cursor.fetchone()
            
            conn.close()
            
            if row:
                return self._row_to_review_item(row)
            return None
            
        except Exception as e:
            logger.error(f"Error getting review item: {e}")
            return None
    
    def get_pending_items(self, limit: int = 50, document_type: str = None) -> List[ReviewItem]:
        """Get pending review items"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            query = "SELECT * FROM review_items WHERE status = 'pending'"
            params = []
            
            if document_type:
                query += " AND document_type = ?"
                params.append(document_type)
            
            query += " ORDER BY priority DESC, created_at ASC LIMIT ?"
            params.append(limit)
            
            cursor.execute(query, params)
            rows = cursor.fetchall()
            
            conn.close()
            
            return [self._row_to_review_item(row) for row in rows]
            
        except Exception as e:
            logger.error(f"Error getting pending items: {e}")
            return []
    
    def update_review_item(self, item: ReviewItem) -> bool:
        """Update review item in database"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            cursor.execute("""
                UPDATE review_items SET
                    status = ?, assigned_reviewer = ?, review_notes = ?,
                    corrected_data = ?, assigned_at = ?, reviewed_at = ?
                WHERE item_id = ?
            """, (
                item.status.value,
                item.assigned_reviewer,
                item.review_notes,
                json.dumps(item.corrected_data),
                item.assigned_at,
                item.reviewed_at,
                item.item_id
            ))
            
            conn.commit()
            conn.close()
            return True
            
        except Exception as e:
            logger.error(f"Error updating review item: {e}")
            return False
    
    def add_review_history(self, item_id: str, reviewer_id: str, action: str,
                          old_status: str, new_status: str, changes: Dict[str, Any] = None):
        """Add review history entry"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            history_id = str(uuid.uuid4())
            
            cursor.execute("""
                INSERT INTO review_history (
                    history_id, item_id, reviewer_id, action, old_status,
                    new_status, changes, timestamp
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                history_id,
                item_id,
                reviewer_id,
                action,
                old_status,
                new_status,
                json.dumps(changes) if changes else None,
                datetime.now()
            ))
            
            conn.commit()
            conn.close()
            
        except Exception as e:
            logger.error(f"Error adding review history: {e}")
    
    def get_reviewer_stats(self, reviewer_id: str) -> ReviewerStats:
        """Get reviewer statistics"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            # Get basic stats
            cursor.execute("""
                SELECT 
                    COUNT(*) as total,
                    SUM(CASE WHEN status = 'approved' THEN 1 ELSE 0 END) as approved,
                    SUM(CASE WHEN status = 'rejected' THEN 1 ELSE 0 END) as rejected
                FROM review_items 
                WHERE assigned_reviewer = ? AND status IN ('approved', 'rejected')
            """, (reviewer_id,))
            
            row = cursor.fetchone()
            total, approved, rejected = row if row else (0, 0, 0)
            
            # Get time-based stats
            today = datetime.now().date()
            week_start = today - timedelta(days=today.weekday())
            month_start = today.replace(day=1)
            
            cursor.execute("""
                SELECT 
                    SUM(CASE WHEN DATE(reviewed_at) = ? THEN 1 ELSE 0 END) as today,
                    SUM(CASE WHEN DATE(reviewed_at) >= ? THEN 1 ELSE 0 END) as week,
                    SUM(CASE WHEN DATE(reviewed_at) >= ? THEN 1 ELSE 0 END) as month
                FROM review_items 
                WHERE assigned_reviewer = ? AND reviewed_at IS NOT NULL
            """, (today, week_start, month_start, reviewer_id))
            
            time_row = cursor.fetchone()
            today_count, week_count, month_count = time_row if time_row else (0, 0, 0)
            
            conn.close()
            
            return ReviewerStats(
                reviewer_id=reviewer_id,
                total_reviewed=total,
                approved_count=approved,
                rejected_count=rejected,
                reviews_today=today_count,
                reviews_this_week=week_count,
                reviews_this_month=month_count
            )
            
        except Exception as e:
            logger.error(f"Error getting reviewer stats: {e}")
            return ReviewerStats(reviewer_id=reviewer_id)
    
    def _row_to_review_item(self, row) -> ReviewItem:
        """Convert database row to ReviewItem"""
        return ReviewItem(
            item_id=row[0],
            document_type=DocumentType(row[1]),
            original_image_path=row[2],
            extracted_data=json.loads(row[3]) if row[3] else {},
            confidence_scores=json.loads(row[4]) if row[4] else {},
            status=ReviewStatus(row[5]),
            assigned_reviewer=row[6],
            review_notes=row[7] or "",
            corrected_data=json.loads(row[8]) if row[8] else {},
            created_at=datetime.fromisoformat(row[9]) if row[9] else datetime.now(),
            assigned_at=datetime.fromisoformat(row[10]) if row[10] else None,
            reviewed_at=datetime.fromisoformat(row[11]) if row[11] else None,
            processing_time=row[12],
            model_version=row[13],
            priority=row[14] or 1
        )

class ConfidenceThresholdManager:
    """Manager for confidence thresholds and auto-approval"""
    
    def __init__(self):
        self.thresholds = {
            DocumentType.IDENTITY_CARD: {
                'auto_approve': 0.95,
                'manual_review': 0.7,
                'high_priority': 0.5
            },
            DocumentType.PASSPORT: {
                'auto_approve': 0.98,
                'manual_review': 0.8,
                'high_priority': 0.6
            },
            DocumentType.DRIVING_LICENSE: {
                'auto_approve': 0.92,
                'manual_review': 0.75,
                'high_priority': 0.55
            },
            # Default thresholds
            'default': {
                'auto_approve': 0.9,
                'manual_review': 0.7,
                'high_priority': 0.5
            }
        }
    
    def get_thresholds(self, document_type: DocumentType) -> Dict[str, float]:
        """Get thresholds for document type"""
        return self.thresholds.get(document_type, self.thresholds['default'])
    
    def should_auto_approve(self, item: ReviewItem) -> bool:
        """Check if item should be auto-approved"""
        thresholds = self.get_thresholds(item.document_type)
        return item.get_overall_confidence() >= thresholds['auto_approve']
    
    def should_manual_review(self, item: ReviewItem) -> bool:
        """Check if item needs manual review"""
        thresholds = self.get_thresholds(item.document_type)
        confidence = item.get_overall_confidence()
        return thresholds['manual_review'] <= confidence < thresholds['auto_approve']
    
    def get_priority(self, item: ReviewItem) -> int:
        """Get priority level for item"""
        thresholds = self.get_thresholds(item.document_type)
        confidence = item.get_overall_confidence()
        
        if confidence < thresholds['high_priority']:
            return 3  # High priority
        elif confidence < thresholds['manual_review']:
            return 2  # Medium priority
        else:
            return 1  # Low priority
    
    def update_thresholds(self, document_type: DocumentType, 
                         new_thresholds: Dict[str, float]):
        """Update thresholds for document type"""
        self.thresholds[document_type] = new_thresholds
        logger.info(f"Updated thresholds for {document_type.value}: {new_thresholds}")

class ReviewInterface:
    """Main review interface class"""
    
    def __init__(self, db_path: str = "review_system.db"):
        self.db = ReviewDatabase(db_path)
        self.threshold_manager = ConfidenceThresholdManager()
        
        # Flask app setup
        self.app = Flask(__name__)
        self.app.secret_key = 'review_interface_secret_key'  # Change in production
        
        # Setup routes
        self._setup_routes()
        
        logger.info("Review interface initialized")
    
    def add_item_for_review(self, extracted_data: Dict[str, Any],
                           confidence_scores: Dict[str, float],
                           document_type: DocumentType,
                           original_image_path: str = None,
                           model_version: str = None) -> str:
        """Add item for review"""
        item_id = str(uuid.uuid4())
        
        item = ReviewItem(
            item_id=item_id,
            document_type=document_type,
            original_image_path=original_image_path,
            extracted_data=extracted_data,
            confidence_scores=confidence_scores,
            model_version=model_version
        )
        
        # Check if should auto-approve
        if self.threshold_manager.should_auto_approve(item):
            item.status = ReviewStatus.AUTO_APPROVED
            item.reviewed_at = datetime.now()
            logger.info(f"Item {item_id} auto-approved (confidence: {item.get_overall_confidence():.3f})")
        else:
            # Set priority
            item.priority = self.threshold_manager.get_priority(item)
            logger.info(f"Item {item_id} queued for review (confidence: {item.get_overall_confidence():.3f}, priority: {item.priority})")
        
        # Add to database
        success = self.db.add_review_item(item)
        
        if success:
            return item_id
        else:
            raise Exception("Failed to add item for review")
    
    def assign_item(self, item_id: str, reviewer_id: str) -> bool:
        """Assign item to reviewer"""
        item = self.db.get_review_item(item_id)
        if not item:
            return False
        
        old_status = item.status.value
        item.status = ReviewStatus.IN_REVIEW
        item.assigned_reviewer = reviewer_id
        item.assigned_at = datetime.now()
        
        success = self.db.update_review_item(item)
        
        if success:
            self.db.add_review_history(
                item_id, reviewer_id, "assigned",
                old_status, item.status.value
            )
        
        return success
    
    def submit_review(self, item_id: str, reviewer_id: str,
                     status: ReviewStatus, corrected_data: Dict[str, Any] = None,
                     review_notes: str = "") -> bool:
        """Submit review for item"""
        item = self.db.get_review_item(item_id)
        if not item:
            return False
        
        old_status = item.status.value
        item.status = status
        item.review_notes = review_notes
        item.reviewed_at = datetime.now()
        
        if corrected_data:
            item.corrected_data = corrected_data
        
        success = self.db.update_review_item(item)
        
        if success:
            # Add to history
            changes = {
                'corrected_data': corrected_data,
                'review_notes': review_notes
            }
            
            self.db.add_review_history(
                item_id, reviewer_id, "reviewed",
                old_status, status.value, changes
            )
            
            # Collect feedback for model improvement
            self._collect_feedback(item, reviewer_id, corrected_data)
        
        return success
    
    def _collect_feedback(self, item: ReviewItem, reviewer_id: str,
                         corrected_data: Dict[str, Any]):
        """Collect feedback for model improvement"""
        if not corrected_data:
            return
        
        try:
            conn = sqlite3.connect(self.db.db_path)
            cursor = conn.cursor()
            
            for field_name, corrected_value in corrected_data.items():
                original_value = item.extracted_data.get(field_name, "")
                confidence_before = item.confidence_scores.get(field_name, 0.0)
                
                # Only record if there's a change
                if str(original_value) != str(corrected_value):
                    feedback_id = str(uuid.uuid4())
                    
                    cursor.execute("""
                        INSERT INTO feedback (
                            feedback_id, item_id, reviewer_id, field_name,
                            original_value, corrected_value, confidence_before,
                            feedback_type, timestamp
                        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """, (
                        feedback_id,
                        item.item_id,
                        reviewer_id,
                        field_name,
                        str(original_value),
                        str(corrected_value),
                        confidence_before,
                        "correction",
                        datetime.now()
                    ))
            
            conn.commit()
            conn.close()
            
        except Exception as e:
            logger.error(f"Error collecting feedback: {e}")
    
    def get_review_queue(self, reviewer_id: str = None, limit: int = 20) -> List[ReviewItem]:
        """Get review queue for reviewer"""
        if reviewer_id:
            # Get items assigned to specific reviewer
            try:
                conn = sqlite3.connect(self.db.db_path)
                cursor = conn.cursor()
                
                cursor.execute("""
                    SELECT * FROM review_items 
                    WHERE assigned_reviewer = ? AND status = 'in_review'
                    ORDER BY priority DESC, created_at ASC
                    LIMIT ?
                """, (reviewer_id, limit))
                
                rows = cursor.fetchall()
                conn.close()
                
                return [self.db._row_to_review_item(row) for row in rows]
                
            except Exception as e:
                logger.error(f"Error getting reviewer queue: {e}")
                return []
        else:
            # Get pending items
            return self.db.get_pending_items(limit)
    
    def get_dashboard_stats(self) -> Dict[str, Any]:
        """Get dashboard statistics"""
        try:
            conn = sqlite3.connect(self.db.db_path)
            cursor = conn.cursor()
            
            # Overall stats
            cursor.execute("""
                SELECT 
                    status,
                    COUNT(*) as count
                FROM review_items 
                GROUP BY status
            """)
            
            status_counts = {row[0]: row[1] for row in cursor.fetchall()}
            
            # Today's stats
            today = datetime.now().date()
            cursor.execute("""
                SELECT COUNT(*) FROM review_items 
                WHERE DATE(created_at) = ?
            """, (today,))
            
            today_created = cursor.fetchone()[0]
            
            cursor.execute("""
                SELECT COUNT(*) FROM review_items 
                WHERE DATE(reviewed_at) = ?
            """, (today,))
            
            today_reviewed = cursor.fetchone()[0]
            
            # Average processing time
            cursor.execute("""
                SELECT AVG(julianday(reviewed_at) - julianday(assigned_at)) * 24 * 60
                FROM review_items 
                WHERE reviewed_at IS NOT NULL AND assigned_at IS NOT NULL
            """)
            
            avg_review_time = cursor.fetchone()[0] or 0.0
            
            conn.close()
            
            return {
                'status_counts': status_counts,
                'today_created': today_created,
                'today_reviewed': today_reviewed,
                'avg_review_time_minutes': avg_review_time,
                'pending_count': status_counts.get('pending', 0),
                'in_review_count': status_counts.get('in_review', 0),
                'auto_approved_count': status_counts.get('auto_approved', 0)
            }
            
        except Exception as e:
            logger.error(f"Error getting dashboard stats: {e}")
            return {}
    
    def _setup_routes(self):
        """Setup Flask routes"""
        
        @self.app.route('/')
        def dashboard():
            """Main dashboard"""
            stats = self.get_dashboard_stats()
            return render_template('dashboard.html', stats=stats)
        
        @self.app.route('/queue')
        def review_queue():
            """Review queue page"""
            reviewer_id = request.args.get('reviewer_id')
            items = self.get_review_queue(reviewer_id)
            return render_template('review_queue.html', items=items)
        
        @self.app.route('/review/<item_id>')
        def review_item(item_id):
            """Review item page"""
            item = self.db.get_review_item(item_id)
            if not item:
                return "Item not found", 404
            return render_template('review_item.html', item=item)
        
        @self.app.route('/api/assign', methods=['POST'])
        def api_assign():
            """API endpoint to assign item"""
            data = request.json
            item_id = data.get('item_id')
            reviewer_id = data.get('reviewer_id')
            
            success = self.assign_item(item_id, reviewer_id)
            return jsonify({'success': success})
        
        @self.app.route('/api/submit_review', methods=['POST'])
        def api_submit_review():
            """API endpoint to submit review"""
            data = request.json
            item_id = data.get('item_id')
            reviewer_id = data.get('reviewer_id')
            status = ReviewStatus(data.get('status'))
            corrected_data = data.get('corrected_data', {})
            review_notes = data.get('review_notes', '')
            
            success = self.submit_review(
                item_id, reviewer_id, status,
                corrected_data, review_notes
            )
            
            return jsonify({'success': success})
        
        @self.app.route('/api/stats')
        def api_stats():
            """API endpoint for statistics"""
            return jsonify(self.get_dashboard_stats())
    
    def run(self, host='127.0.0.1', port=5000, debug=False):
        """Run the Flask application"""
        logger.info(f"Starting review interface on {host}:{port}")
        self.app.run(host=host, port=port, debug=debug)

def main():
    """Main function for standalone execution"""
    # Example usage of the review interface
    
    # Initialize interface
    interface = ReviewInterface()
    
    # Add sample items for review
    sample_items = [
        {
            'extracted_data': {
                'ic_number': '123456-78-9012',
                'name': 'Ahmad bin Ali',
                'address': '123 Jalan Merdeka, KL',
                'phone_number': '0123456789'
            },
            'confidence_scores': {
                'ic_number': 0.95,
                'name': 0.92,
                'address': 0.78,
                'phone_number': 0.88
            },
            'document_type': DocumentType.IDENTITY_CARD
        },
        {
            'extracted_data': {
                'ic_number': '987654-32-1098',
                'name': 'Siti binti Hassan',
                'address': '456 Jalan Bangsar',
                'phone_number': '0198765432'
            },
            'confidence_scores': {
                'ic_number': 0.65,
                'name': 0.72,
                'address': 0.58,
                'phone_number': 0.69
            },
            'document_type': DocumentType.IDENTITY_CARD
        }
    ]
    
    print("\n=== Review Interface Demo ===")
    
    # Add items
    item_ids = []
    for i, item_data in enumerate(sample_items):
        item_id = interface.add_item_for_review(
            extracted_data=item_data['extracted_data'],
            confidence_scores=item_data['confidence_scores'],
            document_type=item_data['document_type'],
            model_version="v1.0"
        )
        item_ids.append(item_id)
        print(f"Added item {i+1}: {item_id}")
    
    # Get dashboard stats
    stats = interface.get_dashboard_stats()
    print("\n=== Dashboard Stats ===")
    print(f"Pending items: {stats.get('pending_count', 0)}")
    print(f"Auto-approved items: {stats.get('auto_approved_count', 0)}")
    print(f"Items created today: {stats.get('today_created', 0)}")
    
    # Get review queue
    queue = interface.get_review_queue(limit=10)
    print(f"\n=== Review Queue ({len(queue)} items) ===")
    
    for item in queue:
        print(f"Item: {item.item_id[:8]}...")
        print(f"  Type: {item.document_type.value}")
        print(f"  Confidence: {item.get_overall_confidence():.3f}")
        print(f"  Priority: {item.priority}")
        print(f"  Status: {item.status.value}")
        print()
    
    # Simulate review process
    if queue:
        item = queue[0]
        reviewer_id = "reviewer_001"
        
        print(f"=== Simulating Review for {item.item_id[:8]}... ===")
        
        # Assign item
        success = interface.assign_item(item.item_id, reviewer_id)
        print(f"Assignment successful: {success}")
        
        # Submit review with corrections
        corrected_data = item.extracted_data.copy()
        corrected_data['address'] = '123 Jalan Merdeka, Kuala Lumpur'  # Correction
        
        success = interface.submit_review(
            item.item_id,
            reviewer_id,
            ReviewStatus.APPROVED,
            corrected_data,
            "Corrected address format"
        )
        print(f"Review submission successful: {success}")
    
    print("\n=== Demo Complete ===")
    print("To run the web interface, call interface.run()")
    
    # Uncomment to run web interface
    # interface.run(debug=True)

if __name__ == "__main__":
    main()