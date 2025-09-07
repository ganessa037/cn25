#!/usr/bin/env python3
"""Simple Web Interface for Document Parser Review

This module provides a simple web interface for human-in-the-loop validation
and feedback collection, following the autocorrect model's simple approach.
"""

import os
import json
import sqlite3
import base64
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Any
from urllib.parse import unquote

try:
    from flask import Flask, render_template_string, request, jsonify, redirect, url_for
except ImportError:
    print("Flask not installed. Install with: pip install flask")
    exit(1)

# Simple HTML templates
MAIN_TEMPLATE = """
<!DOCTYPE html>
<html>
<head>
    <title>Document Parser Review Interface</title>
    <meta charset="utf-8">
    <meta name="viewport" content="width=device-width, initial-scale=1">
    <style>
        body { font-family: Arial, sans-serif; margin: 20px; background-color: #f5f5f5; }
        .container { max-width: 1200px; margin: 0 auto; background: white; padding: 20px; border-radius: 8px; box-shadow: 0 2px 4px rgba(0,0,0,0.1); }
        .header { text-align: center; margin-bottom: 30px; padding-bottom: 20px; border-bottom: 2px solid #007bff; }
        .stats { display: flex; justify-content: space-around; margin-bottom: 30px; }
        .stat-box { text-align: center; padding: 15px; background: #f8f9fa; border-radius: 5px; }
        .stat-number { font-size: 24px; font-weight: bold; color: #007bff; }
        .stat-label { font-size: 14px; color: #666; }
        .document-list { margin-top: 20px; }
        .document-item { border: 1px solid #ddd; margin-bottom: 10px; padding: 15px; border-radius: 5px; background: white; }
        .document-header { display: flex; justify-content: space-between; align-items: center; margin-bottom: 10px; }
        .document-type { background: #007bff; color: white; padding: 4px 8px; border-radius: 3px; font-size: 12px; }
        .confidence-score { font-weight: bold; }
        .confidence-high { color: #28a745; }
        .confidence-medium { color: #ffc107; }
        .confidence-low { color: #dc3545; }
        .btn { padding: 8px 16px; border: none; border-radius: 4px; cursor: pointer; text-decoration: none; display: inline-block; }
        .btn-primary { background: #007bff; color: white; }
        .btn-success { background: #28a745; color: white; }
        .btn-warning { background: #ffc107; color: black; }
        .btn-danger { background: #dc3545; color: white; }
        .btn:hover { opacity: 0.8; }
        .status-pending { border-left: 4px solid #ffc107; }
        .status-reviewed { border-left: 4px solid #28a745; }
        .status-flagged { border-left: 4px solid #dc3545; }
        .filters { margin-bottom: 20px; padding: 15px; background: #f8f9fa; border-radius: 5px; }
        .filter-group { display: inline-block; margin-right: 20px; }
        .filter-group label { display: block; margin-bottom: 5px; font-weight: bold; }
        .filter-group select, .filter-group input { padding: 5px; border: 1px solid #ddd; border-radius: 3px; }
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>📄 Document Parser Review Interface</h1>
            <p>Human-in-the-loop validation and feedback collection</p>
        </div>
        
        <div class="stats">
            <div class="stat-box">
                <div class="stat-number">{{ stats.total_documents }}</div>
                <div class="stat-label">Total Documents</div>
            </div>
            <div class="stat-box">
                <div class="stat-number">{{ stats.pending_review }}</div>
                <div class="stat-label">Pending Review</div>
            </div>
            <div class="stat-box">
                <div class="stat-number">{{ stats.reviewed }}</div>
                <div class="stat-label">Reviewed</div>
            </div>
            <div class="stat-box">
                <div class="stat-number">{{ '%.1f'|format(stats.avg_confidence * 100) }}%</div>
                <div class="stat-label">Avg Confidence</div>
            </div>
        </div>
        
        <div class="filters">
            <form method="GET">
                <div class="filter-group">
                    <label>Document Type:</label>
                    <select name="doc_type">
                        <option value="">All Types</option>
                        <option value="mykad" {{ 'selected' if request.args.get('doc_type') == 'mykad' }}>MyKad</option>
                        <option value="spk" {{ 'selected' if request.args.get('doc_type') == 'spk' }}>SPK</option>
                        <option value="passport" {{ 'selected' if request.args.get('doc_type') == 'passport' }}>Passport</option>
                        <option value="license" {{ 'selected' if request.args.get('doc_type') == 'license' }}>License</option>
                    </select>
                </div>
                <div class="filter-group">
                    <label>Status:</label>
                    <select name="status">
                        <option value="">All Status</option>
                        <option value="pending" {{ 'selected' if request.args.get('status') == 'pending' }}>Pending</option>
                        <option value="reviewed" {{ 'selected' if request.args.get('status') == 'reviewed' }}>Reviewed</option>
                        <option value="flagged" {{ 'selected' if request.args.get('status') == 'flagged' }}>Flagged</option>
                    </select>
                </div>
                <div class="filter-group">
                    <label>Min Confidence:</label>
                    <input type="number" name="min_confidence" min="0" max="100" step="5" 
                           value="{{ request.args.get('min_confidence', '') }}" placeholder="0-100">
                </div>
                <div class="filter-group">
                    <button type="submit" class="btn btn-primary">Filter</button>
                    <a href="/" class="btn btn-warning">Clear</a>
                </div>
            </form>
        </div>
        
        <div class="document-list">
            {% for doc in documents %}
            <div class="document-item status-{{ doc.status }}">
                <div class="document-header">
                    <div>
                        <span class="document-type">{{ doc.document_type.upper() }}</span>
                        <strong>{{ doc.document_id }}</strong>
                        <small>({{ doc.created_at }})</small>
                    </div>
                    <div>
                        <span class="confidence-score confidence-{{ 'high' if doc.confidence > 0.8 else 'medium' if doc.confidence > 0.6 else 'low' }}">
                            {{ '%.1f'|format(doc.confidence * 100) }}% confidence
                        </span>
                    </div>
                </div>
                
                <div style="margin-bottom: 10px;">
                    <strong>Extracted Fields:</strong>
                    {% for field, value in doc.extracted_fields.items() %}
                        <span style="margin-right: 15px;"><em>{{ field }}:</em> {{ value }}</span>
                    {% endfor %}
                </div>
                
                <div>
                    <a href="/review/{{ doc.id }}" class="btn btn-primary">Review</a>
                    {% if doc.status == 'pending' %}
                        <a href="/quick_approve/{{ doc.id }}" class="btn btn-success">Quick Approve</a>
                        <a href="/flag/{{ doc.id }}" class="btn btn-danger">Flag Issue</a>
                    {% endif %}
                </div>
            </div>
            {% endfor %}
        </div>
        
        {% if not documents %}
        <div style="text-align: center; padding: 40px; color: #666;">
            <h3>No documents found</h3>
            <p>No documents match the current filters or no documents have been processed yet.</p>
        </div>
        {% endif %}
    </div>
</body>
</html>
"""

REVIEW_TEMPLATE = """
<!DOCTYPE html>
<html>
<head>
    <title>Review Document - {{ document.document_id }}</title>
    <meta charset="utf-8">
    <meta name="viewport" content="width=device-width, initial-scale=1">
    <style>
        body { font-family: Arial, sans-serif; margin: 20px; background-color: #f5f5f5; }
        .container { max-width: 1000px; margin: 0 auto; background: white; padding: 20px; border-radius: 8px; box-shadow: 0 2px 4px rgba(0,0,0,0.1); }
        .header { text-align: center; margin-bottom: 30px; padding-bottom: 20px; border-bottom: 2px solid #007bff; }
        .document-info { background: #f8f9fa; padding: 15px; border-radius: 5px; margin-bottom: 20px; }
        .image-section { text-align: center; margin-bottom: 30px; }
        .document-image { max-width: 100%; height: auto; border: 2px solid #ddd; border-radius: 5px; }
        .fields-section { margin-bottom: 30px; }
        .field-group { margin-bottom: 15px; padding: 10px; border: 1px solid #ddd; border-radius: 5px; }
        .field-label { font-weight: bold; margin-bottom: 5px; }
        .field-extracted { background: #e3f2fd; padding: 8px; border-radius: 3px; margin-bottom: 5px; }
        .field-correction { width: 100%; padding: 8px; border: 1px solid #ddd; border-radius: 3px; }
        .confidence-indicator { float: right; padding: 2px 6px; border-radius: 3px; font-size: 12px; }
        .confidence-high { background: #d4edda; color: #155724; }
        .confidence-medium { background: #fff3cd; color: #856404; }
        .confidence-low { background: #f8d7da; color: #721c24; }
        .btn { padding: 10px 20px; border: none; border-radius: 4px; cursor: pointer; text-decoration: none; display: inline-block; margin-right: 10px; }
        .btn-primary { background: #007bff; color: white; }
        .btn-success { background: #28a745; color: white; }
        .btn-danger { background: #dc3545; color: white; }
        .btn-secondary { background: #6c757d; color: white; }
        .btn:hover { opacity: 0.8; }
        .feedback-section { background: #f8f9fa; padding: 15px; border-radius: 5px; margin-bottom: 20px; }
        .feedback-textarea { width: 100%; height: 80px; padding: 8px; border: 1px solid #ddd; border-radius: 3px; resize: vertical; }
        .actions { text-align: center; padding-top: 20px; border-top: 1px solid #ddd; }
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>📄 Review Document</h1>
            <p>{{ document.document_id }} - {{ document.document_type.upper() }}</p>
        </div>
        
        <div class="document-info">
            <strong>Document ID:</strong> {{ document.document_id }}<br>
            <strong>Type:</strong> {{ document.document_type }}<br>
            <strong>Processed:</strong> {{ document.created_at }}<br>
            <strong>Overall Confidence:</strong> {{ '%.1f'|format(document.confidence * 100) }}%<br>
            <strong>Status:</strong> {{ document.status.title() }}
        </div>
        
        {% if document.image_data %}
        <div class="image-section">
            <h3>Document Image</h3>
            <img src="data:image/jpeg;base64,{{ document.image_data }}" class="document-image" alt="Document">
        </div>
        {% endif %}
        
        <form method="POST">
            <div class="fields-section">
                <h3>Extracted Fields</h3>
                {% for field, data in document.field_details.items() %}
                <div class="field-group">
                    <div class="field-label">
                        {{ field.replace('_', ' ').title() }}
                        <span class="confidence-indicator confidence-{{ 'high' if data.confidence > 0.8 else 'medium' if data.confidence > 0.6 else 'low' }}">
                            {{ '%.1f'|format(data.confidence * 100) }}%
                        </span>
                    </div>
                    <div class="field-extracted">
                        <strong>Extracted:</strong> {{ data.value or '(empty)' }}
                    </div>
                    <input type="text" name="field_{{ field }}" class="field-correction" 
                           placeholder="Enter correction if needed" value="{{ data.value or '' }}">
                </div>
                {% endfor %}
            </div>
            
            <div class="feedback-section">
                <h3>Additional Feedback</h3>
                <textarea name="feedback" class="feedback-textarea" 
                          placeholder="Provide any additional feedback about the extraction quality, missing fields, or other issues..."></textarea>
            </div>
            
            <div class="actions">
                <button type="submit" name="action" value="approve" class="btn btn-success">✅ Approve</button>
                <button type="submit" name="action" value="correct" class="btn btn-primary">✏️ Submit Corrections</button>
                <button type="submit" name="action" value="flag" class="btn btn-danger">🚩 Flag for Review</button>
                <a href="/" class="btn btn-secondary">← Back to List</a>
            </div>
        </form>
    </div>
</body>
</html>
"""

class DocumentReviewApp:
    """Simple web application for document review"""
    
    def __init__(self, storage_path: str = "review_data"):
        self.storage_path = Path(storage_path)
        self.storage_path.mkdir(exist_ok=True)
        
        # Initialize database
        self.db_path = self.storage_path / "review.db"
        self.init_database()
        
        # Initialize Flask app
        self.app = Flask(__name__)
        self.app.secret_key = 'document_review_secret_key_change_in_production'
        
        # Register routes
        self.register_routes()
    
    def init_database(self):
        """Initialize SQLite database for review storage"""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS documents (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    document_id TEXT UNIQUE,
                    document_type TEXT,
                    extracted_fields TEXT,
                    field_confidences TEXT,
                    overall_confidence REAL,
                    image_data TEXT,
                    status TEXT DEFAULT 'pending',
                    created_at TEXT,
                    reviewed_at TEXT,
                    reviewer_feedback TEXT,
                    corrected_fields TEXT
                )
            """)
            
            conn.execute("""
                CREATE TABLE IF NOT EXISTS feedback (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    document_id TEXT,
                    feedback_type TEXT,
                    feedback_data TEXT,
                    created_at TEXT,
                    FOREIGN KEY (document_id) REFERENCES documents (document_id)
                )
            """)
    
    def register_routes(self):
        """Register Flask routes"""
        
        @self.app.route('/')
        def index():
            # Get filter parameters
            doc_type = request.args.get('doc_type', '')
            status = request.args.get('status', '')
            min_confidence = request.args.get('min_confidence', '')
            
            # Build query
            query = "SELECT * FROM documents WHERE 1=1"
            params = []
            
            if doc_type:
                query += " AND document_type = ?"
                params.append(doc_type)
            
            if status:
                query += " AND status = ?"
                params.append(status)
            
            if min_confidence:
                query += " AND overall_confidence >= ?"
                params.append(float(min_confidence) / 100.0)
            
            query += " ORDER BY created_at DESC LIMIT 50"
            
            # Get documents
            with sqlite3.connect(self.db_path) as conn:
                conn.row_factory = sqlite3.Row
                cursor = conn.execute(query, params)
                rows = cursor.fetchall()
                
                documents = []
                for row in rows:
                    doc = dict(row)
                    doc['extracted_fields'] = json.loads(doc['extracted_fields'] or '{}')
                    doc['confidence'] = doc['overall_confidence'] or 0.0
                    documents.append(doc)
            
            # Get statistics
            stats = self.get_statistics()
            
            return render_template_string(MAIN_TEMPLATE, documents=documents, stats=stats)
        
        @self.app.route('/review/<int:doc_id>')
        def review_document(doc_id):
            with sqlite3.connect(self.db_path) as conn:
                conn.row_factory = sqlite3.Row
                cursor = conn.execute("SELECT * FROM documents WHERE id = ?", (doc_id,))
                row = cursor.fetchone()
                
                if not row:
                    return "Document not found", 404
                
                document = dict(row)
                document['extracted_fields'] = json.loads(document['extracted_fields'] or '{}')
                document['field_confidences'] = json.loads(document['field_confidences'] or '{}')
                
                # Prepare field details
                field_details = {}
                for field, value in document['extracted_fields'].items():
                    field_details[field] = {
                        'value': value,
                        'confidence': document['field_confidences'].get(field, 0.0)
                    }
                
                document['field_details'] = field_details
                document['confidence'] = document['overall_confidence'] or 0.0
            
            return render_template_string(REVIEW_TEMPLATE, document=document)
        
        @self.app.route('/review/<int:doc_id>', methods=['POST'])
        def submit_review(doc_id):
            action = request.form.get('action')
            feedback = request.form.get('feedback', '')
            
            # Collect field corrections
            corrected_fields = {}
            for key, value in request.form.items():
                if key.startswith('field_') and value.strip():
                    field_name = key[6:]  # Remove 'field_' prefix
                    corrected_fields[field_name] = value.strip()
            
            # Update document status
            status_map = {
                'approve': 'reviewed',
                'correct': 'reviewed',
                'flag': 'flagged'
            }
            
            new_status = status_map.get(action, 'pending')
            
            with sqlite3.connect(self.db_path) as conn:
                conn.execute("""
                    UPDATE documents 
                    SET status = ?, reviewed_at = ?, reviewer_feedback = ?, corrected_fields = ?
                    WHERE id = ?
                """, (
                    new_status,
                    datetime.now().isoformat(),
                    feedback,
                    json.dumps(corrected_fields),
                    doc_id
                ))
                
                # Store feedback
                if feedback or corrected_fields:
                    # Get document_id
                    cursor = conn.execute("SELECT document_id FROM documents WHERE id = ?", (doc_id,))
                    document_id = cursor.fetchone()[0]
                    
                    feedback_data = {
                        'action': action,
                        'feedback': feedback,
                        'corrections': corrected_fields
                    }
                    
                    conn.execute("""
                        INSERT INTO feedback (document_id, feedback_type, feedback_data, created_at)
                        VALUES (?, ?, ?, ?)
                    """, (
                        document_id,
                        action,
                        json.dumps(feedback_data),
                        datetime.now().isoformat()
                    ))
            
            return redirect(url_for('index'))
        
        @self.app.route('/quick_approve/<int:doc_id>')
        def quick_approve(doc_id):
            with sqlite3.connect(self.db_path) as conn:
                conn.execute("""
                    UPDATE documents 
                    SET status = 'reviewed', reviewed_at = ?
                    WHERE id = ?
                """, (datetime.now().isoformat(), doc_id))
            
            return redirect(url_for('index'))
        
        @self.app.route('/flag/<int:doc_id>')
        def flag_document(doc_id):
            with sqlite3.connect(self.db_path) as conn:
                conn.execute("""
                    UPDATE documents 
                    SET status = 'flagged', reviewed_at = ?
                    WHERE id = ?
                """, (datetime.now().isoformat(), doc_id))
            
            return redirect(url_for('index'))
        
        @self.app.route('/api/submit_document', methods=['POST'])
        def api_submit_document():
            """API endpoint for submitting documents for review"""
            try:
                data = request.get_json()
                
                required_fields = ['document_id', 'document_type', 'extracted_fields']
                for field in required_fields:
                    if field not in data:
                        return jsonify({'error': f'Missing required field: {field}'}), 400
                
                # Store document
                with sqlite3.connect(self.db_path) as conn:
                    conn.execute("""
                        INSERT OR REPLACE INTO documents (
                            document_id, document_type, extracted_fields, 
                            field_confidences, overall_confidence, image_data, created_at
                        ) VALUES (?, ?, ?, ?, ?, ?, ?)
                    """, (
                        data['document_id'],
                        data['document_type'],
                        json.dumps(data['extracted_fields']),
                        json.dumps(data.get('field_confidences', {})),
                        data.get('overall_confidence', 0.0),
                        data.get('image_data', ''),
                        datetime.now().isoformat()
                    ))
                
                return jsonify({'status': 'success', 'message': 'Document submitted for review'})
                
            except Exception as e:
                return jsonify({'error': str(e)}), 500
        
        @self.app.route('/api/get_feedback')
        def api_get_feedback():
            """API endpoint for retrieving feedback data"""
            try:
                with sqlite3.connect(self.db_path) as conn:
                    conn.row_factory = sqlite3.Row
                    cursor = conn.execute("""
                        SELECT f.*, d.document_type 
                        FROM feedback f
                        JOIN documents d ON f.document_id = d.document_id
                        ORDER BY f.created_at DESC
                        LIMIT 100
                    """)
                    
                    feedback_list = []
                    for row in cursor.fetchall():
                        feedback = dict(row)
                        feedback['feedback_data'] = json.loads(feedback['feedback_data'])
                        feedback_list.append(feedback)
                
                return jsonify({'feedback': feedback_list})
                
            except Exception as e:
                return jsonify({'error': str(e)}), 500
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get review statistics"""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.execute("SELECT COUNT(*) FROM documents")
            total_documents = cursor.fetchone()[0]
            
            cursor = conn.execute("SELECT COUNT(*) FROM documents WHERE status = 'pending'")
            pending_review = cursor.fetchone()[0]
            
            cursor = conn.execute("SELECT COUNT(*) FROM documents WHERE status = 'reviewed'")
            reviewed = cursor.fetchone()[0]
            
            cursor = conn.execute("SELECT AVG(overall_confidence) FROM documents WHERE overall_confidence > 0")
            avg_confidence = cursor.fetchone()[0] or 0.0
        
        return {
            'total_documents': total_documents,
            'pending_review': pending_review,
            'reviewed': reviewed,
            'avg_confidence': avg_confidence
        }
    
    def add_sample_documents(self):
        """Add sample documents for testing"""
        sample_docs = [
            {
                'document_id': 'mykad_sample_001',
                'document_type': 'mykad',
                'extracted_fields': {
                    'name': 'AHMAD BIN ALI',
                    'ic_number': '123456-78-9012',
                    'address': '123 JALAN UTAMA, KUALA LUMPUR'
                },
                'field_confidences': {
                    'name': 0.95,
                    'ic_number': 0.88,
                    'address': 0.72
                },
                'overall_confidence': 0.85
            },
            {
                'document_id': 'spk_sample_001',
                'document_type': 'spk',
                'extracted_fields': {
                    'vehicle_number': 'ABC1234',
                    'owner_name': 'SITI BINTI HASSAN',
                    'vehicle_type': 'KERETA'
                },
                'field_confidences': {
                    'vehicle_number': 0.92,
                    'owner_name': 0.78,
                    'vehicle_type': 0.95
                },
                'overall_confidence': 0.88
            },
            {
                'document_id': 'passport_sample_001',
                'document_type': 'passport',
                'extracted_fields': {
                    'passport_number': 'A12345678',
                    'name': 'JOHN DOE',
                    'nationality': 'MALAYSIAN'
                },
                'field_confidences': {
                    'passport_number': 0.65,
                    'name': 0.82,
                    'nationality': 0.90
                },
                'overall_confidence': 0.79
            }
        ]
        
        with sqlite3.connect(self.db_path) as conn:
            for doc in sample_docs:
                conn.execute("""
                    INSERT OR REPLACE INTO documents (
                        document_id, document_type, extracted_fields, 
                        field_confidences, overall_confidence, created_at
                    ) VALUES (?, ?, ?, ?, ?, ?)
                """, (
                    doc['document_id'],
                    doc['document_type'],
                    json.dumps(doc['extracted_fields']),
                    json.dumps(doc['field_confidences']),
                    doc['overall_confidence'],
                    datetime.now().isoformat()
                ))
        
        print(f"✅ Added {len(sample_docs)} sample documents")
    
    def run(self, host='127.0.0.1', port=5000, debug=True):
        """Run the Flask application"""
        print(f"🌐 Starting Document Review Interface")
        print(f"📍 URL: http://{host}:{port}")
        print(f"📁 Storage: {self.storage_path}")
        print("\n=== API Endpoints ===")
        print(f"POST /api/submit_document - Submit document for review")
        print(f"GET  /api/get_feedback - Retrieve feedback data")
        print("\n=== Web Interface ===")
        print(f"GET  / - Main review interface")
        print(f"GET  /review/<id> - Review specific document")
        print("\nPress Ctrl+C to stop the server")
        
        self.app.run(host=host, port=port, debug=debug)

def main():
    """Demo function"""
    print("🌐 Document Parser Review Interface Demo")
    print("=" * 50)
    
    # Initialize review app
    app = DocumentReviewApp()
    
    # Add sample documents
    print("\n=== Adding Sample Documents ===")
    app.add_sample_documents()
    
    # Start the web server
    print("\n=== Starting Web Server ===")
    try:
        app.run(host='127.0.0.1', port=5000, debug=False)
    except KeyboardInterrupt:
        print("\n\n👋 Server stopped by user")
    except Exception as e:
        print(f"\n❌ Server error: {e}")
        print("\nMake sure Flask is installed: pip install flask")

if __name__ == "__main__":
    main()