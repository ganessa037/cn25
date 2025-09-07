#!/usr/bin/env python3
"""
User Acceptance Testing - User Feedback Collection

Comprehensive framework for collecting, analyzing, and managing user feedback
for the Malaysian document parser system.
"""

import os
import sys
import json
import time
import logging
import sqlite3
import hashlib
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime, timedelta
from dataclasses import dataclass, field
from enum import Enum

import pytest
import requests
from flask import Flask, request, jsonify, render_template_string

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))


class FeedbackType(Enum):
    """Types of user feedback."""
    ACCURACY = "accuracy"
    PERFORMANCE = "performance"
    USABILITY = "usability"
    FEATURE_REQUEST = "feature_request"
    BUG_REPORT = "bug_report"
    GENERAL = "general"


class SeverityLevel(Enum):
    """Severity levels for feedback."""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class FeedbackStatus(Enum):
    """Status of feedback items."""
    NEW = "new"
    ACKNOWLEDGED = "acknowledged"
    IN_PROGRESS = "in_progress"
    RESOLVED = "resolved"
    CLOSED = "closed"


@dataclass
class UserFeedback:
    """User feedback data structure."""
    feedback_id: str
    user_id: str
    session_id: str
    feedback_type: FeedbackType
    severity: SeverityLevel
    title: str
    description: str
    document_type: str
    processing_result: Dict[str, Any]
    expected_result: Dict[str, Any]
    user_rating: int  # 1-5 scale
    ease_of_use_rating: int  # 1-5 scale
    accuracy_rating: int  # 1-5 scale
    speed_rating: int  # 1-5 scale
    overall_satisfaction: int  # 1-5 scale
    suggestions: str
    contact_info: str
    browser_info: Dict[str, str]
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())
    status: FeedbackStatus = FeedbackStatus.NEW
    tags: List[str] = field(default_factory=list)
    attachments: List[str] = field(default_factory=list)
    follow_up_required: bool = False
    resolution_notes: str = ""
    resolved_at: Optional[str] = None
    resolved_by: Optional[str] = None


@dataclass
class FeedbackAnalytics:
    """Analytics data for user feedback."""
    total_feedback_count: int
    feedback_by_type: Dict[str, int]
    feedback_by_severity: Dict[str, int]
    feedback_by_status: Dict[str, int]
    average_ratings: Dict[str, float]
    common_issues: List[Dict[str, Any]]
    user_satisfaction_trend: List[Dict[str, Any]]
    response_time_metrics: Dict[str, float]
    resolution_rate: float
    top_feature_requests: List[Dict[str, Any]]
    critical_issues: List[Dict[str, Any]]
    generated_at: str = field(default_factory=lambda: datetime.now().isoformat())


class UserFeedbackCollector:
    """Comprehensive user feedback collection and management system."""
    
    def __init__(self, database_path: str = None, api_base_url: str = None):
        self.database_path = database_path or "user_feedback.db"
        self.api_base_url = api_base_url or "http://localhost:8000"
        
        # Setup logging
        self.logger = logging.getLogger(__name__)
        
        # Initialize database
        self._initialize_database()
        
        # Feedback collection configuration
        self.config = {
            'collection_methods': ['web_form', 'api', 'email', 'survey'],
            'rating_scale': {'min': 1, 'max': 5},
            'auto_categorization': True,
            'sentiment_analysis': True,
            'follow_up_threshold': 3,  # Follow up if rating <= 3
            'critical_response_time': 24,  # hours
            'survey_frequency': 30,  # days
            'retention_period': 365  # days
        }
        
        # Analytics cache
        self._analytics_cache = None
        self._cache_timestamp = None
        self._cache_duration = 3600  # 1 hour
        
        # Flask app for web interface
        self.app = None
        self._setup_web_interface()
    
    def _initialize_database(self):
        """Initialize SQLite database for feedback storage."""
        conn = sqlite3.connect(self.database_path)
        cursor = conn.cursor()
        
        # Create feedback table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS feedback (
                feedback_id TEXT PRIMARY KEY,
                user_id TEXT NOT NULL,
                session_id TEXT,
                feedback_type TEXT NOT NULL,
                severity TEXT NOT NULL,
                title TEXT NOT NULL,
                description TEXT NOT NULL,
                document_type TEXT,
                processing_result TEXT,
                expected_result TEXT,
                user_rating INTEGER,
                ease_of_use_rating INTEGER,
                accuracy_rating INTEGER,
                speed_rating INTEGER,
                overall_satisfaction INTEGER,
                suggestions TEXT,
                contact_info TEXT,
                browser_info TEXT,
                timestamp TEXT NOT NULL,
                status TEXT DEFAULT 'new',
                tags TEXT,
                attachments TEXT,
                follow_up_required BOOLEAN DEFAULT 0,
                resolution_notes TEXT,
                resolved_at TEXT,
                resolved_by TEXT
            )
        ''')
        
        # Create indexes
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_feedback_type ON feedback(feedback_type)')
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_feedback_severity ON feedback(severity)')
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_feedback_status ON feedback(status)')
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_feedback_timestamp ON feedback(timestamp)')
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_user_id ON feedback(user_id)')
        
        # Create analytics table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS feedback_analytics (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                analytics_data TEXT NOT NULL,
                generated_at TEXT NOT NULL
            )
        ''')
        
        conn.commit()
        conn.close()
    
    def _setup_web_interface(self):
        """Setup Flask web interface for feedback collection."""
        self.app = Flask(__name__)
        
        # Feedback form template
        feedback_form_template = '''
        <!DOCTYPE html>
        <html>
        <head>
            <title>Document Parser Feedback</title>
            <meta charset="utf-8">
            <meta name="viewport" content="width=device-width, initial-scale=1">
            <style>
                body { font-family: Arial, sans-serif; max-width: 800px; margin: 0 auto; padding: 20px; }
                .form-group { margin-bottom: 15px; }
                label { display: block; margin-bottom: 5px; font-weight: bold; }
                input, select, textarea { width: 100%; padding: 8px; border: 1px solid #ddd; border-radius: 4px; }
                textarea { height: 100px; resize: vertical; }
                .rating-group { display: flex; gap: 10px; align-items: center; }
                .rating-group input { width: auto; }
                button { background: #007bff; color: white; padding: 10px 20px; border: none; border-radius: 4px; cursor: pointer; }
                button:hover { background: #0056b3; }
                .success { color: green; margin-top: 10px; }
                .error { color: red; margin-top: 10px; }
            </style>
        </head>
        <body>
            <h1>Document Parser Feedback</h1>
            <p>Help us improve our Malaysian document processing system by providing your feedback.</p>
            
            <form id="feedbackForm" method="POST">
                <div class="form-group">
                    <label for="feedback_type">Feedback Type:</label>
                    <select name="feedback_type" id="feedback_type" required>
                        <option value="">Select type...</option>
                        <option value="accuracy">Accuracy Issue</option>
                        <option value="performance">Performance Issue</option>
                        <option value="usability">Usability Issue</option>
                        <option value="feature_request">Feature Request</option>
                        <option value="bug_report">Bug Report</option>
                        <option value="general">General Feedback</option>
                    </select>
                </div>
                
                <div class="form-group">
                    <label for="severity">Severity:</label>
                    <select name="severity" id="severity" required>
                        <option value="">Select severity...</option>
                        <option value="low">Low</option>
                        <option value="medium">Medium</option>
                        <option value="high">High</option>
                        <option value="critical">Critical</option>
                    </select>
                </div>
                
                <div class="form-group">
                    <label for="title">Title:</label>
                    <input type="text" name="title" id="title" required maxlength="200">
                </div>
                
                <div class="form-group">
                    <label for="description">Description:</label>
                    <textarea name="description" id="description" required placeholder="Please provide detailed information about your feedback..."></textarea>
                </div>
                
                <div class="form-group">
                    <label for="document_type">Document Type:</label>
                    <select name="document_type" id="document_type">
                        <option value="">Select document type...</option>
                        <option value="mykad">MyKad</option>
                        <option value="spk">SPK Certificate</option>
                        <option value="other">Other</option>
                    </select>
                </div>
                
                <h3>Ratings (1 = Poor, 5 = Excellent)</h3>
                
                <div class="form-group">
                    <label>Overall Satisfaction:</label>
                    <div class="rating-group">
                        <input type="radio" name="overall_satisfaction" value="1" id="overall_1"><label for="overall_1">1</label>
                        <input type="radio" name="overall_satisfaction" value="2" id="overall_2"><label for="overall_2">2</label>
                        <input type="radio" name="overall_satisfaction" value="3" id="overall_3"><label for="overall_3">3</label>
                        <input type="radio" name="overall_satisfaction" value="4" id="overall_4"><label for="overall_4">4</label>
                        <input type="radio" name="overall_satisfaction" value="5" id="overall_5"><label for="overall_5">5</label>
                    </div>
                </div>
                
                <div class="form-group">
                    <label>Accuracy Rating:</label>
                    <div class="rating-group">
                        <input type="radio" name="accuracy_rating" value="1" id="accuracy_1"><label for="accuracy_1">1</label>
                        <input type="radio" name="accuracy_rating" value="2" id="accuracy_2"><label for="accuracy_2">2</label>
                        <input type="radio" name="accuracy_rating" value="3" id="accuracy_3"><label for="accuracy_3">3</label>
                        <input type="radio" name="accuracy_rating" value="4" id="accuracy_4"><label for="accuracy_4">4</label>
                        <input type="radio" name="accuracy_rating" value="5" id="accuracy_5"><label for="accuracy_5">5</label>
                    </div>
                </div>
                
                <div class="form-group">
                    <label>Speed Rating:</label>
                    <div class="rating-group">
                        <input type="radio" name="speed_rating" value="1" id="speed_1"><label for="speed_1">1</label>
                        <input type="radio" name="speed_rating" value="2" id="speed_2"><label for="speed_2">2</label>
                        <input type="radio" name="speed_rating" value="3" id="speed_3"><label for="speed_3">3</label>
                        <input type="radio" name="speed_rating" value="4" id="speed_4"><label for="speed_4">4</label>
                        <input type="radio" name="speed_rating" value="5" id="speed_5"><label for="speed_5">5</label>
                    </div>
                </div>
                
                <div class="form-group">
                    <label>Ease of Use Rating:</label>
                    <div class="rating-group">
                        <input type="radio" name="ease_of_use_rating" value="1" id="ease_1"><label for="ease_1">1</label>
                        <input type="radio" name="ease_of_use_rating" value="2" id="ease_2"><label for="ease_2">2</label>
                        <input type="radio" name="ease_of_use_rating" value="3" id="ease_3"><label for="ease_3">3</label>
                        <input type="radio" name="ease_of_use_rating" value="4" id="ease_4"><label for="ease_4">4</label>
                        <input type="radio" name="ease_of_use_rating" value="5" id="ease_5"><label for="ease_5">5</label>
                    </div>
                </div>
                
                <div class="form-group">
                    <label for="suggestions">Suggestions for Improvement:</label>
                    <textarea name="suggestions" id="suggestions" placeholder="Any suggestions to make our system better?"></textarea>
                </div>
                
                <div class="form-group">
                    <label for="contact_info">Contact Information (Optional):</label>
                    <input type="email" name="contact_info" id="contact_info" placeholder="your.email@example.com">
                </div>
                
                <button type="submit">Submit Feedback</button>
            </form>
            
            <div id="message"></div>
            
            <script>
                document.getElementById('feedbackForm').addEventListener('submit', function(e) {
                    e.preventDefault();
                    
                    const formData = new FormData(this);
                    const data = {};
                    
                    for (let [key, value] of formData.entries()) {
                        data[key] = value;
                    }
                    
                    fetch('/submit_feedback', {
                        method: 'POST',
                        headers: {
                            'Content-Type': 'application/json',
                        },
                        body: JSON.stringify(data)
                    })
                    .then(response => response.json())
                    .then(data => {
                        const messageDiv = document.getElementById('message');
                        if (data.success) {
                            messageDiv.innerHTML = '<div class="success">Thank you for your feedback! Your feedback ID is: ' + data.feedback_id + '</div>';
                            document.getElementById('feedbackForm').reset();
                        } else {
                            messageDiv.innerHTML = '<div class="error">Error: ' + data.error + '</div>';
                        }
                    })
                    .catch(error => {
                        document.getElementById('message').innerHTML = '<div class="error">Error submitting feedback. Please try again.</div>';
                    });
                });
            </script>
        </body>
        </html>
        '''
        
        @self.app.route('/feedback')
        def feedback_form():
            return render_template_string(feedback_form_template)
        
        @self.app.route('/submit_feedback', methods=['POST'])
        def submit_feedback():
            try:
                data = request.get_json()
                feedback = self._create_feedback_from_form(data)
                feedback_id = self.collect_feedback(feedback)
                return jsonify({'success': True, 'feedback_id': feedback_id})
            except Exception as e:
                self.logger.error(f"Error submitting feedback: {e}")
                return jsonify({'success': False, 'error': str(e)}), 400
        
        @self.app.route('/feedback/analytics')
        def feedback_analytics():
            analytics = self.get_feedback_analytics()
            return jsonify(analytics.__dict__)
        
        @self.app.route('/feedback/api/submit', methods=['POST'])
        def api_submit_feedback():
            try:
                data = request.get_json()
                feedback = UserFeedback(**data)
                feedback_id = self.collect_feedback(feedback)
                return jsonify({'success': True, 'feedback_id': feedback_id})
            except Exception as e:
                self.logger.error(f"Error submitting feedback via API: {e}")
                return jsonify({'success': False, 'error': str(e)}), 400
    
    def _create_feedback_from_form(self, form_data: Dict[str, Any]) -> UserFeedback:
        """Create UserFeedback object from form data."""
        # Generate IDs
        feedback_id = self._generate_feedback_id()
        user_id = form_data.get('user_id', 'anonymous')
        session_id = form_data.get('session_id', self._generate_session_id())
        
        # Get browser info from request headers
        browser_info = {
            'user_agent': request.headers.get('User-Agent', ''),
            'accept_language': request.headers.get('Accept-Language', ''),
            'remote_addr': request.remote_addr
        }
        
        return UserFeedback(
            feedback_id=feedback_id,
            user_id=user_id,
            session_id=session_id,
            feedback_type=FeedbackType(form_data.get('feedback_type', 'general')),
            severity=SeverityLevel(form_data.get('severity', 'medium')),
            title=form_data.get('title', ''),
            description=form_data.get('description', ''),
            document_type=form_data.get('document_type', ''),
            processing_result={},
            expected_result={},
            user_rating=int(form_data.get('user_rating', 3)),
            ease_of_use_rating=int(form_data.get('ease_of_use_rating', 3)),
            accuracy_rating=int(form_data.get('accuracy_rating', 3)),
            speed_rating=int(form_data.get('speed_rating', 3)),
            overall_satisfaction=int(form_data.get('overall_satisfaction', 3)),
            suggestions=form_data.get('suggestions', ''),
            contact_info=form_data.get('contact_info', ''),
            browser_info=browser_info,
            follow_up_required=int(form_data.get('overall_satisfaction', 3)) <= self.config['follow_up_threshold']
        )
    
    def _generate_feedback_id(self) -> str:
        """Generate unique feedback ID."""
        timestamp = str(int(time.time() * 1000))
        random_part = hashlib.md5(os.urandom(16)).hexdigest()[:8]
        return f"FB_{timestamp}_{random_part}"
    
    def _generate_session_id(self) -> str:
        """Generate session ID."""
        return hashlib.md5(f"{time.time()}_{os.urandom(8)}".encode()).hexdigest()
    
    def collect_feedback(self, feedback: UserFeedback) -> str:
        """Collect and store user feedback."""
        self.logger.info(f"Collecting feedback: {feedback.feedback_id}")
        
        # Auto-categorize and analyze
        if self.config['auto_categorization']:
            feedback = self._auto_categorize_feedback(feedback)
        
        if self.config['sentiment_analysis']:
            feedback = self._analyze_sentiment(feedback)
        
        # Store in database
        self._store_feedback(feedback)
        
        # Trigger follow-up if needed
        if feedback.follow_up_required:
            self._schedule_follow_up(feedback)
        
        # Handle critical issues
        if feedback.severity == SeverityLevel.CRITICAL:
            self._handle_critical_feedback(feedback)
        
        # Invalidate analytics cache
        self._analytics_cache = None
        
        return feedback.feedback_id
    
    def _auto_categorize_feedback(self, feedback: UserFeedback) -> UserFeedback:
        """Auto-categorize feedback based on content."""
        text = f"{feedback.title} {feedback.description}".lower()
        
        # Define keywords for auto-categorization
        keywords = {
            'accuracy': ['wrong', 'incorrect', 'mistake', 'error', 'inaccurate', 'missing field'],
            'performance': ['slow', 'fast', 'speed', 'timeout', 'loading', 'response time'],
            'usability': ['difficult', 'confusing', 'user interface', 'ui', 'ux', 'navigation'],
            'bug_report': ['bug', 'crash', 'broken', 'not working', 'error message', 'exception'],
            'feature_request': ['feature', 'add', 'new', 'enhancement', 'improvement', 'suggestion']
        }
        
        # Auto-tag based on keywords
        tags = []
        for category, words in keywords.items():
            if any(word in text for word in words):
                tags.append(category)
        
        feedback.tags.extend(tags)
        
        return feedback
    
    def _analyze_sentiment(self, feedback: UserFeedback) -> UserFeedback:
        """Analyze sentiment of feedback text."""
        # Simple sentiment analysis based on keywords
        text = f"{feedback.title} {feedback.description}".lower()
        
        positive_words = ['good', 'great', 'excellent', 'amazing', 'love', 'perfect', 'helpful']
        negative_words = ['bad', 'terrible', 'awful', 'hate', 'horrible', 'useless', 'frustrating']
        
        positive_count = sum(1 for word in positive_words if word in text)
        negative_count = sum(1 for word in negative_words if word in text)
        
        if positive_count > negative_count:
            feedback.tags.append('positive_sentiment')
        elif negative_count > positive_count:
            feedback.tags.append('negative_sentiment')
        else:
            feedback.tags.append('neutral_sentiment')
        
        return feedback
    
    def _store_feedback(self, feedback: UserFeedback):
        """Store feedback in database."""
        conn = sqlite3.connect(self.database_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            INSERT INTO feedback (
                feedback_id, user_id, session_id, feedback_type, severity, title, description,
                document_type, processing_result, expected_result, user_rating, ease_of_use_rating,
                accuracy_rating, speed_rating, overall_satisfaction, suggestions, contact_info,
                browser_info, timestamp, status, tags, attachments, follow_up_required,
                resolution_notes, resolved_at, resolved_by
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        ''', (
            feedback.feedback_id, feedback.user_id, feedback.session_id,
            feedback.feedback_type.value, feedback.severity.value,
            feedback.title, feedback.description, feedback.document_type,
            json.dumps(feedback.processing_result), json.dumps(feedback.expected_result),
            feedback.user_rating, feedback.ease_of_use_rating, feedback.accuracy_rating,
            feedback.speed_rating, feedback.overall_satisfaction, feedback.suggestions,
            feedback.contact_info, json.dumps(feedback.browser_info), feedback.timestamp,
            feedback.status.value, json.dumps(feedback.tags), json.dumps(feedback.attachments),
            feedback.follow_up_required, feedback.resolution_notes, feedback.resolved_at,
            feedback.resolved_by
        ))
        
        conn.commit()
        conn.close()
    
    def _schedule_follow_up(self, feedback: UserFeedback):
        """Schedule follow-up for low-rated feedback."""
        self.logger.info(f"Scheduling follow-up for feedback: {feedback.feedback_id}")
        # In a real implementation, this would integrate with a task scheduler
        # For now, we'll just log the need for follow-up
    
    def _handle_critical_feedback(self, feedback: UserFeedback):
        """Handle critical feedback with immediate attention."""
        self.logger.critical(f"Critical feedback received: {feedback.feedback_id}")
        # In a real implementation, this would send alerts to the development team
        # For now, we'll just log the critical issue
    
    def get_feedback_by_id(self, feedback_id: str) -> Optional[UserFeedback]:
        """Retrieve feedback by ID."""
        conn = sqlite3.connect(self.database_path)
        cursor = conn.cursor()
        
        cursor.execute('SELECT * FROM feedback WHERE feedback_id = ?', (feedback_id,))
        row = cursor.fetchone()
        
        conn.close()
        
        if row:
            return self._row_to_feedback(row)
        return None
    
    def get_feedback_by_user(self, user_id: str) -> List[UserFeedback]:
        """Retrieve all feedback from a specific user."""
        conn = sqlite3.connect(self.database_path)
        cursor = conn.cursor()
        
        cursor.execute('SELECT * FROM feedback WHERE user_id = ? ORDER BY timestamp DESC', (user_id,))
        rows = cursor.fetchall()
        
        conn.close()
        
        return [self._row_to_feedback(row) for row in rows]
    
    def get_feedback_by_criteria(self, feedback_type: FeedbackType = None,
                               severity: SeverityLevel = None,
                               status: FeedbackStatus = None,
                               start_date: str = None,
                               end_date: str = None,
                               limit: int = 100) -> List[UserFeedback]:
        """Retrieve feedback based on criteria."""
        conn = sqlite3.connect(self.database_path)
        cursor = conn.cursor()
        
        query = 'SELECT * FROM feedback WHERE 1=1'
        params = []
        
        if feedback_type:
            query += ' AND feedback_type = ?'
            params.append(feedback_type.value)
        
        if severity:
            query += ' AND severity = ?'
            params.append(severity.value)
        
        if status:
            query += ' AND status = ?'
            params.append(status.value)
        
        if start_date:
            query += ' AND timestamp >= ?'
            params.append(start_date)
        
        if end_date:
            query += ' AND timestamp <= ?'
            params.append(end_date)
        
        query += ' ORDER BY timestamp DESC LIMIT ?'
        params.append(limit)
        
        cursor.execute(query, params)
        rows = cursor.fetchall()
        
        conn.close()
        
        return [self._row_to_feedback(row) for row in rows]
    
    def _row_to_feedback(self, row) -> UserFeedback:
        """Convert database row to UserFeedback object."""
        return UserFeedback(
            feedback_id=row[0],
            user_id=row[1],
            session_id=row[2],
            feedback_type=FeedbackType(row[3]),
            severity=SeverityLevel(row[4]),
            title=row[5],
            description=row[6],
            document_type=row[7],
            processing_result=json.loads(row[8]) if row[8] else {},
            expected_result=json.loads(row[9]) if row[9] else {},
            user_rating=row[10],
            ease_of_use_rating=row[11],
            accuracy_rating=row[12],
            speed_rating=row[13],
            overall_satisfaction=row[14],
            suggestions=row[15],
            contact_info=row[16],
            browser_info=json.loads(row[17]) if row[17] else {},
            timestamp=row[18],
            status=FeedbackStatus(row[19]),
            tags=json.loads(row[20]) if row[20] else [],
            attachments=json.loads(row[21]) if row[21] else [],
            follow_up_required=bool(row[22]),
            resolution_notes=row[23],
            resolved_at=row[24],
            resolved_by=row[25]
        )
    
    def update_feedback_status(self, feedback_id: str, status: FeedbackStatus,
                             resolution_notes: str = "", resolved_by: str = "") -> bool:
        """Update feedback status."""
        conn = sqlite3.connect(self.database_path)
        cursor = conn.cursor()
        
        resolved_at = datetime.now().isoformat() if status == FeedbackStatus.RESOLVED else None
        
        cursor.execute('''
            UPDATE feedback 
            SET status = ?, resolution_notes = ?, resolved_at = ?, resolved_by = ?
            WHERE feedback_id = ?
        ''', (status.value, resolution_notes, resolved_at, resolved_by, feedback_id))
        
        success = cursor.rowcount > 0
        conn.commit()
        conn.close()
        
        # Invalidate analytics cache
        if success:
            self._analytics_cache = None
        
        return success
    
    def get_feedback_analytics(self, force_refresh: bool = False) -> FeedbackAnalytics:
        """Get comprehensive feedback analytics."""
        # Check cache
        if (not force_refresh and self._analytics_cache and self._cache_timestamp and
            time.time() - self._cache_timestamp < self._cache_duration):
            return self._analytics_cache
        
        conn = sqlite3.connect(self.database_path)
        cursor = conn.cursor()
        
        # Total feedback count
        cursor.execute('SELECT COUNT(*) FROM feedback')
        total_count = cursor.fetchone()[0]
        
        # Feedback by type
        cursor.execute('SELECT feedback_type, COUNT(*) FROM feedback GROUP BY feedback_type')
        feedback_by_type = dict(cursor.fetchall())
        
        # Feedback by severity
        cursor.execute('SELECT severity, COUNT(*) FROM feedback GROUP BY severity')
        feedback_by_severity = dict(cursor.fetchall())
        
        # Feedback by status
        cursor.execute('SELECT status, COUNT(*) FROM feedback GROUP BY status')
        feedback_by_status = dict(cursor.fetchall())
        
        # Average ratings
        cursor.execute('''
            SELECT 
                AVG(user_rating) as avg_user_rating,
                AVG(ease_of_use_rating) as avg_ease_rating,
                AVG(accuracy_rating) as avg_accuracy_rating,
                AVG(speed_rating) as avg_speed_rating,
                AVG(overall_satisfaction) as avg_satisfaction
            FROM feedback
        ''')
        ratings_row = cursor.fetchone()
        average_ratings = {
            'user_rating': ratings_row[0] or 0,
            'ease_of_use': ratings_row[1] or 0,
            'accuracy': ratings_row[2] or 0,
            'speed': ratings_row[3] or 0,
            'overall_satisfaction': ratings_row[4] or 0
        }
        
        # Common issues (most frequent feedback types with low ratings)
        cursor.execute('''
            SELECT feedback_type, title, COUNT(*) as count
            FROM feedback 
            WHERE overall_satisfaction <= 3
            GROUP BY feedback_type, title
            ORDER BY count DESC
            LIMIT 10
        ''')
        common_issues = [{'type': row[0], 'title': row[1], 'count': row[2]} for row in cursor.fetchall()]
        
        # User satisfaction trend (last 30 days)
        thirty_days_ago = (datetime.now() - timedelta(days=30)).isoformat()
        cursor.execute('''
            SELECT DATE(timestamp) as date, AVG(overall_satisfaction) as avg_satisfaction
            FROM feedback 
            WHERE timestamp >= ?
            GROUP BY DATE(timestamp)
            ORDER BY date
        ''', (thirty_days_ago,))
        satisfaction_trend = [{'date': row[0], 'satisfaction': row[1]} for row in cursor.fetchall()]
        
        # Response time metrics
        cursor.execute('''
            SELECT 
                AVG(CASE WHEN resolved_at IS NOT NULL 
                    THEN (julianday(resolved_at) - julianday(timestamp)) * 24 
                    ELSE NULL END) as avg_response_time_hours,
                MIN(CASE WHEN resolved_at IS NOT NULL 
                    THEN (julianday(resolved_at) - julianday(timestamp)) * 24 
                    ELSE NULL END) as min_response_time_hours,
                MAX(CASE WHEN resolved_at IS NOT NULL 
                    THEN (julianday(resolved_at) - julianday(timestamp)) * 24 
                    ELSE NULL END) as max_response_time_hours
            FROM feedback
        ''')
        response_row = cursor.fetchone()
        response_time_metrics = {
            'average_hours': response_row[0] or 0,
            'minimum_hours': response_row[1] or 0,
            'maximum_hours': response_row[2] or 0
        }
        
        # Resolution rate
        cursor.execute('SELECT COUNT(*) FROM feedback WHERE status IN ("resolved", "closed")')
        resolved_count = cursor.fetchone()[0]
        resolution_rate = resolved_count / total_count if total_count > 0 else 0
        
        # Top feature requests
        cursor.execute('''
            SELECT title, description, COUNT(*) as count
            FROM feedback 
            WHERE feedback_type = 'feature_request'
            GROUP BY title, description
            ORDER BY count DESC
            LIMIT 5
        ''')
        top_feature_requests = [{'title': row[0], 'description': row[1], 'count': row[2]} for row in cursor.fetchall()]
        
        # Critical issues
        cursor.execute('''
            SELECT feedback_id, title, description, timestamp
            FROM feedback 
            WHERE severity = 'critical' AND status NOT IN ('resolved', 'closed')
            ORDER BY timestamp DESC
            LIMIT 10
        ''')
        critical_issues = [{
            'feedback_id': row[0], 
            'title': row[1], 
            'description': row[2], 
            'timestamp': row[3]
        } for row in cursor.fetchall()]
        
        conn.close()
        
        # Create analytics object
        analytics = FeedbackAnalytics(
            total_feedback_count=total_count,
            feedback_by_type=feedback_by_type,
            feedback_by_severity=feedback_by_severity,
            feedback_by_status=feedback_by_status,
            average_ratings=average_ratings,
            common_issues=common_issues,
            user_satisfaction_trend=satisfaction_trend,
            response_time_metrics=response_time_metrics,
            resolution_rate=resolution_rate,
            top_feature_requests=top_feature_requests,
            critical_issues=critical_issues
        )
        
        # Cache the results
        self._analytics_cache = analytics
        self._cache_timestamp = time.time()
        
        return analytics
    
    def generate_feedback_report(self, output_file: str = None) -> str:
        """Generate comprehensive feedback report."""
        analytics = self.get_feedback_analytics()
        
        output_file = output_file or f"feedback_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        
        report_data = {
            'report_metadata': {
                'generated_at': datetime.now().isoformat(),
                'report_type': 'user_feedback_analysis',
                'data_period': 'all_time'
            },
            'analytics': analytics.__dict__,
            'recommendations': self._generate_feedback_recommendations(analytics)
        }
        
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(report_data, f, indent=2, ensure_ascii=False)
        
        self.logger.info(f"Feedback report generated: {output_file}")
        return output_file
    
    def _generate_feedback_recommendations(self, analytics: FeedbackAnalytics) -> List[str]:
        """Generate recommendations based on feedback analytics."""
        recommendations = []
        
        # Low satisfaction recommendations
        if analytics.average_ratings['overall_satisfaction'] < 3.5:
            recommendations.append("Overall user satisfaction is below average. Focus on addressing common issues.")
        
        # Accuracy issues
        if analytics.average_ratings['accuracy'] < 3.5:
            recommendations.append("Accuracy ratings are low. Review and improve field extraction algorithms.")
        
        # Performance issues
        if analytics.average_ratings['speed'] < 3.5:
            recommendations.append("Speed ratings are low. Optimize processing pipeline for better performance.")
        
        # Usability issues
        if analytics.average_ratings['ease_of_use'] < 3.5:
            recommendations.append("Ease of use ratings are low. Improve user interface and user experience.")
        
        # High response time
        if analytics.response_time_metrics['average_hours'] > 48:
            recommendations.append("Average response time is high. Improve support team response times.")
        
        # Low resolution rate
        if analytics.resolution_rate < 0.8:
            recommendations.append("Resolution rate is low. Focus on resolving pending feedback items.")
        
        # Critical issues
        if len(analytics.critical_issues) > 0:
            recommendations.append(f"There are {len(analytics.critical_issues)} unresolved critical issues. Address immediately.")
        
        # Feature requests
        if len(analytics.top_feature_requests) > 0:
            top_request = analytics.top_feature_requests[0]
            recommendations.append(f"Most requested feature: '{top_request['title']}' ({top_request['count']} requests)")
        
        return recommendations
    
    def start_feedback_server(self, host: str = '0.0.0.0', port: int = 5000, debug: bool = False):
        """Start the feedback collection web server."""
        self.logger.info(f"Starting feedback server on {host}:{port}")
        self.app.run(host=host, port=port, debug=debug)
    
    def simulate_user_feedback(self, num_feedback: int = 10) -> List[str]:
        """Simulate user feedback for testing purposes."""
        feedback_ids = []
        
        sample_feedback_data = [
            {
                'feedback_type': FeedbackType.ACCURACY,
                'severity': SeverityLevel.HIGH,
                'title': 'IC number extraction incorrect',
                'description': 'The system extracted wrong IC number from my MyKad',
                'document_type': 'mykad',
                'overall_satisfaction': 2,
                'accuracy_rating': 1,
                'speed_rating': 4,
                'ease_of_use_rating': 3
            },
            {
                'feedback_type': FeedbackType.PERFORMANCE,
                'severity': SeverityLevel.MEDIUM,
                'title': 'Processing takes too long',
                'description': 'Document processing is slower than expected',
                'document_type': 'spk',
                'overall_satisfaction': 3,
                'accuracy_rating': 4,
                'speed_rating': 2,
                'ease_of_use_rating': 4
            },
            {
                'feedback_type': FeedbackType.USABILITY,
                'severity': SeverityLevel.LOW,
                'title': 'Interface could be more intuitive',
                'description': 'The upload process could be simplified',
                'document_type': 'mykad',
                'overall_satisfaction': 4,
                'accuracy_rating': 4,
                'speed_rating': 4,
                'ease_of_use_rating': 3
            },
            {
                'feedback_type': FeedbackType.FEATURE_REQUEST,
                'severity': SeverityLevel.LOW,
                'title': 'Add batch processing',
                'description': 'Would like to process multiple documents at once',
                'document_type': 'both',
                'overall_satisfaction': 4,
                'accuracy_rating': 4,
                'speed_rating': 4,
                'ease_of_use_rating': 4,
                'suggestions': 'Add drag and drop for multiple files'
            },
            {
                'feedback_type': FeedbackType.BUG_REPORT,
                'severity': SeverityLevel.CRITICAL,
                'title': 'System crashes with large files',
                'description': 'Application crashes when uploading files larger than 5MB',
                'document_type': 'mykad',
                'overall_satisfaction': 1,
                'accuracy_rating': 1,
                'speed_rating': 1,
                'ease_of_use_rating': 2
            }
        ]
        
        for i in range(num_feedback):
            # Select random feedback template
            template = sample_feedback_data[i % len(sample_feedback_data)]
            
            # Create feedback with variations
            feedback = UserFeedback(
                feedback_id=self._generate_feedback_id(),
                user_id=f"test_user_{i % 5}",  # 5 different test users
                session_id=self._generate_session_id(),
                feedback_type=template['feedback_type'],
                severity=template['severity'],
                title=f"{template['title']} (Test {i+1})",
                description=template['description'],
                document_type=template['document_type'],
                processing_result={},
                expected_result={},
                user_rating=template.get('user_rating', 3),
                ease_of_use_rating=template.get('ease_of_use_rating', 3),
                accuracy_rating=template.get('accuracy_rating', 3),
                speed_rating=template.get('speed_rating', 3),
                overall_satisfaction=template.get('overall_satisfaction', 3),
                suggestions=template.get('suggestions', ''),
                contact_info=f"test{i}@example.com",
                browser_info={'user_agent': 'Test Browser', 'remote_addr': '127.0.0.1'}
            )
            
            feedback_id = self.collect_feedback(feedback)
            feedback_ids.append(feedback_id)
        
        return feedback_ids


# Pytest fixtures and test functions
@pytest.fixture
def feedback_collector(tmp_path):
    """Fixture for feedback collector."""
    db_path = tmp_path / "test_feedback.db"
    return UserFeedbackCollector(database_path=str(db_path))


class TestUserFeedbackCollection:
    """Test class for user feedback collection."""
    
    @pytest.mark.user_acceptance
    @pytest.mark.feedback
    def test_feedback_collection(self, feedback_collector):
        """Test basic feedback collection functionality."""
        feedback = UserFeedback(
            feedback_id="test_001",
            user_id="test_user",
            session_id="test_session",
            feedback_type=FeedbackType.ACCURACY,
            severity=SeverityLevel.HIGH,
            title="Test feedback",
            description="This is a test feedback",
            document_type="mykad",
            processing_result={},
            expected_result={},
            user_rating=3,
            ease_of_use_rating=4,
            accuracy_rating=2,
            speed_rating=4,
            overall_satisfaction=3,
            suggestions="Improve accuracy",
            contact_info="test@example.com",
            browser_info={}
        )
        
        feedback_id = feedback_collector.collect_feedback(feedback)
        assert feedback_id == "test_001"
        
        # Retrieve and verify
        retrieved = feedback_collector.get_feedback_by_id(feedback_id)
        assert retrieved is not None
        assert retrieved.title == "Test feedback"
        assert retrieved.feedback_type == FeedbackType.ACCURACY
    
    @pytest.mark.user_acceptance
    @pytest.mark.feedback
    def test_feedback_analytics(self, feedback_collector):
        """Test feedback analytics generation."""
        # Generate sample feedback
        feedback_ids = feedback_collector.simulate_user_feedback(5)
        assert len(feedback_ids) == 5
        
        # Get analytics
        analytics = feedback_collector.get_feedback_analytics()
        
        # Verify analytics structure
        assert analytics.total_feedback_count == 5
        assert len(analytics.feedback_by_type) > 0
        assert len(analytics.feedback_by_severity) > 0
        assert 'overall_satisfaction' in analytics.average_ratings
    
    @pytest.mark.user_acceptance
    @pytest.mark.feedback
    def test_feedback_status_update(self, feedback_collector):
        """Test feedback status updates."""
        # Create feedback
        feedback_ids = feedback_collector.simulate_user_feedback(1)
        feedback_id = feedback_ids[0]
        
        # Update status
        success = feedback_collector.update_feedback_status(
            feedback_id, 
            FeedbackStatus.RESOLVED, 
            "Issue resolved", 
            "test_admin"
        )
        
        assert success
        
        # Verify update
        feedback = feedback_collector.get_feedback_by_id(feedback_id)
        assert feedback.status == FeedbackStatus.RESOLVED
        assert feedback.resolution_notes == "Issue resolved"
        assert feedback.resolved_by == "test_admin"
        assert feedback.resolved_at is not None
    
    @pytest.mark.user_acceptance
    @pytest.mark.feedback
    def test_feedback_filtering(self, feedback_collector):
        """Test feedback filtering by criteria."""
        # Generate sample feedback
        feedback_collector.simulate_user_feedback(10)
        
        # Test filtering by type
        accuracy_feedback = feedback_collector.get_feedback_by_criteria(
            feedback_type=FeedbackType.ACCURACY
        )
        assert len(accuracy_feedback) > 0
        assert all(f.feedback_type == FeedbackType.ACCURACY for f in accuracy_feedback)
        
        # Test filtering by severity
        critical_feedback = feedback_collector.get_feedback_by_criteria(
            severity=SeverityLevel.CRITICAL
        )
        assert all(f.severity == SeverityLevel.CRITICAL for f in critical_feedback)
    
    @pytest.mark.user_acceptance
    @pytest.mark.feedback
    def test_feedback_report_generation(self, feedback_collector, tmp_path):
        """Test feedback report generation."""
        # Generate sample feedback
        feedback_collector.simulate_user_feedback(5)
        
        # Generate report
        report_file = feedback_collector.generate_feedback_report(
            str(tmp_path / "test_report.json")
        )
        
        assert Path(report_file).exists()
        
        # Verify report content
        with open(report_file, 'r') as f:
            report_data = json.load(f)
        
        assert 'report_metadata' in report_data
        assert 'analytics' in report_data
        assert 'recommendations' in report_data
        assert report_data['analytics']['total_feedback_count'] == 5
    
    @pytest.mark.user_acceptance
    @pytest.mark.feedback
    @pytest.mark.integration
    def test_web_interface_setup(self, feedback_collector):
        """Test web interface setup."""
        # Verify Flask app is created
        assert feedback_collector.app is not None
        
        # Test client
        with feedback_collector.app.test_client() as client:
            # Test feedback form page
            response = client.get('/feedback')
            assert response.status_code == 200
            assert b'Document Parser Feedback' in response.data
            
            # Test analytics endpoint
            response = client.get('/feedback/analytics')
            assert response.status_code == 200
            
            # Test API endpoint
            test_feedback_data = {
                'feedback_id': 'api_test_001',
                'user_id': 'api_user',
                'session_id': 'api_session',
                'feedback_type': 'general',
                'severity': 'medium',
                'title': 'API Test',
                'description': 'Testing API submission',
                'document_type': 'mykad',
                'processing_result': {},
                'expected_result': {},
                'user_rating': 4,
                'ease_of_use_rating': 4,
                'accuracy_rating': 4,
                'speed_rating': 4,
                'overall_satisfaction': 4,
                'suggestions': '',
                'contact_info': 'api@test.com',
                'browser_info': {}
            }
            
            response = client.post('/feedback/api/submit', 
                                 json=test_feedback_data,
                                 content_type='application/json')
            assert response.status_code == 200
            
            response_data = response.get_json()
            assert response_data['success'] is True
            assert 'feedback_id' in response_data


if __name__ == "__main__":
    # Example usage
    collector = UserFeedbackCollector()
    
    # Simulate some feedback
    feedback_ids = collector.simulate_user_feedback(10)
    print(f"Generated {len(feedback_ids)} feedback items")
    
    # Get analytics
    analytics = collector.get_feedback_analytics()
    print(f"Total feedback: {analytics.total_feedback_count}")
    print(f"Average satisfaction: {analytics.average_ratings['overall_satisfaction']:.2f}")
    
    # Generate report
    report_file = collector.generate_feedback_report()
    print(f"Report generated: {report_file}")
    
    # Start web server (uncomment to run)
    # collector.start_feedback_server(port=5000, debug=True)