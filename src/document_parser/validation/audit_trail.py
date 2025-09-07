"""Audit Trail System

This module implements a comprehensive audit trail system to track all modifications,
approvals, and validation decisions in the document processing pipeline.
Follows the autocorrect model's organizational patterns.
"""

import json
import logging
import sqlite3
import hashlib
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Tuple, Set, Union
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from collections import defaultdict
import uuid
import gzip
import base64

import pandas as pd
from cryptography.fernet import Fernet
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class AuditEventType(Enum):
    """Types of audit events"""
    DOCUMENT_UPLOADED = "document_uploaded"
    EXTRACTION_PERFORMED = "extraction_performed"
    VALIDATION_APPLIED = "validation_applied"
    CONFIDENCE_EVALUATED = "confidence_evaluated"
    DECISION_MADE = "decision_made"
    HUMAN_REVIEW_STARTED = "human_review_started"
    HUMAN_REVIEW_COMPLETED = "human_review_completed"
    FIELD_MODIFIED = "field_modified"
    DOCUMENT_APPROVED = "document_approved"
    DOCUMENT_REJECTED = "document_rejected"
    THRESHOLD_UPDATED = "threshold_updated"
    MODEL_UPDATED = "model_updated"
    SYSTEM_ERROR = "system_error"
    USER_LOGIN = "user_login"
    USER_LOGOUT = "user_logout"
    PERMISSION_CHANGED = "permission_changed"
    DATA_EXPORT = "data_export"
    BACKUP_CREATED = "backup_created"
    CONFIGURATION_CHANGED = "configuration_changed"

class AuditSeverity(Enum):
    """Severity levels for audit events"""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"

class DataClassification(Enum):
    """Data classification levels"""
    PUBLIC = "public"
    INTERNAL = "internal"
    CONFIDENTIAL = "confidential"
    RESTRICTED = "restricted"

class AuditStatus(Enum):
    """Status of audit records"""
    ACTIVE = "active"
    ARCHIVED = "archived"
    DELETED = "deleted"
    ENCRYPTED = "encrypted"

@dataclass
class AuditEvent:
    """Individual audit event record"""
    
    event_id: str
    event_type: AuditEventType
    timestamp: datetime
    
    # Actor information
    user_id: Optional[str] = None
    user_role: Optional[str] = None
    session_id: Optional[str] = None
    ip_address: Optional[str] = None
    user_agent: Optional[str] = None
    
    # Resource information
    resource_type: Optional[str] = None
    resource_id: Optional[str] = None
    document_id: Optional[str] = None
    field_name: Optional[str] = None
    
    # Event details
    action_description: str = ""
    old_value: Optional[str] = None
    new_value: Optional[str] = None
    confidence_score: Optional[float] = None
    
    # Context and metadata
    severity: AuditSeverity = AuditSeverity.MEDIUM
    data_classification: DataClassification = DataClassification.INTERNAL
    tags: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    # System information
    system_version: Optional[str] = None
    model_version: Optional[str] = None
    request_id: Optional[str] = None
    
    # Integrity and security
    checksum: Optional[str] = None
    encrypted: bool = False
    retention_period_days: int = 2555  # 7 years default
    
    # Status
    status: AuditStatus = AuditStatus.ACTIVE
    
    def __post_init__(self):
        """Post-initialization processing"""
        if not self.checksum:
            self.checksum = self._calculate_checksum()
    
    def _calculate_checksum(self) -> str:
        """Calculate checksum for integrity verification"""
        # Create a string representation of key fields
        data_string = f"{self.event_id}{self.event_type.value}{self.timestamp.isoformat()}"
        data_string += f"{self.user_id or ''}{self.action_description}"
        data_string += f"{self.old_value or ''}{self.new_value or ''}"
        
        # Calculate SHA-256 hash
        return hashlib.sha256(data_string.encode()).hexdigest()
    
    def verify_integrity(self) -> bool:
        """Verify the integrity of the audit record"""
        current_checksum = self._calculate_checksum()
        return current_checksum == self.checksum
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return {
            'event_id': self.event_id,
            'event_type': self.event_type.value,
            'timestamp': self.timestamp.isoformat(),
            'user_id': self.user_id,
            'user_role': self.user_role,
            'session_id': self.session_id,
            'ip_address': self.ip_address,
            'user_agent': self.user_agent,
            'resource_type': self.resource_type,
            'resource_id': self.resource_id,
            'document_id': self.document_id,
            'field_name': self.field_name,
            'action_description': self.action_description,
            'old_value': self.old_value,
            'new_value': self.new_value,
            'confidence_score': self.confidence_score,
            'severity': self.severity.value,
            'data_classification': self.data_classification.value,
            'tags': self.tags,
            'metadata': self.metadata,
            'system_version': self.system_version,
            'model_version': self.model_version,
            'request_id': self.request_id,
            'checksum': self.checksum,
            'encrypted': self.encrypted,
            'retention_period_days': self.retention_period_days,
            'status': self.status.value
        }
    
    def should_encrypt(self) -> bool:
        """Determine if this event should be encrypted"""
        return (
            self.data_classification in [DataClassification.CONFIDENTIAL, DataClassification.RESTRICTED] or
            self.severity == AuditSeverity.CRITICAL or
            any(sensitive_field in [self.old_value, self.new_value, self.action_description] 
                for sensitive_field in ['password', 'token', 'key', 'secret'] if sensitive_field)
        )

@dataclass
class AuditQuery:
    """Query parameters for audit trail searches"""
    
    # Time range
    start_date: Optional[datetime] = None
    end_date: Optional[datetime] = None
    
    # Event filters
    event_types: List[AuditEventType] = field(default_factory=list)
    severities: List[AuditSeverity] = field(default_factory=list)
    
    # Actor filters
    user_ids: List[str] = field(default_factory=list)
    user_roles: List[str] = field(default_factory=list)
    ip_addresses: List[str] = field(default_factory=list)
    
    # Resource filters
    document_ids: List[str] = field(default_factory=list)
    resource_types: List[str] = field(default_factory=list)
    field_names: List[str] = field(default_factory=list)
    
    # Content filters
    search_text: Optional[str] = None
    tags: List[str] = field(default_factory=list)
    
    # Result options
    limit: int = 1000
    offset: int = 0
    order_by: str = "timestamp"
    order_desc: bool = True
    
    # Security options
    include_encrypted: bool = False
    decrypt_results: bool = False

@dataclass
class AuditSummary:
    """Summary statistics for audit trail"""
    
    total_events: int
    date_range_start: datetime
    date_range_end: datetime
    
    # Event type breakdown
    events_by_type: Dict[str, int] = field(default_factory=dict)
    events_by_severity: Dict[str, int] = field(default_factory=dict)
    events_by_user: Dict[str, int] = field(default_factory=dict)
    
    # Activity patterns
    events_by_hour: Dict[int, int] = field(default_factory=dict)
    events_by_day: Dict[str, int] = field(default_factory=dict)
    
    # Security metrics
    failed_login_attempts: int = 0
    permission_changes: int = 0
    critical_events: int = 0
    
    # Data integrity
    integrity_violations: int = 0
    encrypted_events: int = 0
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return {
            'total_events': self.total_events,
            'date_range_start': self.date_range_start.isoformat(),
            'date_range_end': self.date_range_end.isoformat(),
            'events_by_type': self.events_by_type,
            'events_by_severity': self.events_by_severity,
            'events_by_user': self.events_by_user,
            'events_by_hour': self.events_by_hour,
            'events_by_day': self.events_by_day,
            'failed_login_attempts': self.failed_login_attempts,
            'permission_changes': self.permission_changes,
            'critical_events': self.critical_events,
            'integrity_violations': self.integrity_violations,
            'encrypted_events': self.encrypted_events
        }

class AuditEncryption:
    """Encryption utilities for audit data"""
    
    def __init__(self, password: str = None):
        if password:
            self.key = self._derive_key(password)
            self.cipher = Fernet(self.key)
        else:
            # Generate a random key for testing
            self.key = Fernet.generate_key()
            self.cipher = Fernet(self.key)
    
    def _derive_key(self, password: str) -> bytes:
        """Derive encryption key from password"""
        password_bytes = password.encode()
        salt = b'audit_trail_salt'  # In production, use random salt per installation
        
        kdf = PBKDF2HMAC(
            algorithm=hashes.SHA256(),
            length=32,
            salt=salt,
            iterations=100000,
        )
        
        key = base64.urlsafe_b64encode(kdf.derive(password_bytes))
        return key
    
    def encrypt_data(self, data: str) -> str:
        """Encrypt sensitive data"""
        try:
            encrypted_bytes = self.cipher.encrypt(data.encode())
            return base64.urlsafe_b64encode(encrypted_bytes).decode()
        except Exception as e:
            logger.error(f"Encryption failed: {e}")
            return data  # Return original data if encryption fails
    
    def decrypt_data(self, encrypted_data: str) -> str:
        """Decrypt sensitive data"""
        try:
            encrypted_bytes = base64.urlsafe_b64decode(encrypted_data.encode())
            decrypted_bytes = self.cipher.decrypt(encrypted_bytes)
            return decrypted_bytes.decode()
        except Exception as e:
            logger.error(f"Decryption failed: {e}")
            return encrypted_data  # Return encrypted data if decryption fails

class AuditDatabase:
    """Database for audit trail storage"""
    
    def __init__(self, db_path: str = "audit_trail.db", encryption_password: str = None):
        self.db_path = db_path
        self.encryption = AuditEncryption(encryption_password) if encryption_password else None
        self.init_database()
    
    def init_database(self):
        """Initialize audit database"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Main audit events table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS audit_events (
                event_id TEXT PRIMARY KEY,
                event_type TEXT NOT NULL,
                timestamp TIMESTAMP NOT NULL,
                user_id TEXT,
                user_role TEXT,
                session_id TEXT,
                ip_address TEXT,
                user_agent TEXT,
                resource_type TEXT,
                resource_id TEXT,
                document_id TEXT,
                field_name TEXT,
                action_description TEXT,
                old_value TEXT,
                new_value TEXT,
                confidence_score REAL,
                severity TEXT NOT NULL,
                data_classification TEXT NOT NULL,
                tags TEXT,
                metadata TEXT,
                system_version TEXT,
                model_version TEXT,
                request_id TEXT,
                checksum TEXT,
                encrypted BOOLEAN DEFAULT FALSE,
                retention_period_days INTEGER DEFAULT 2555,
                status TEXT DEFAULT 'active',
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)
        
        # Audit sessions table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS audit_sessions (
                session_id TEXT PRIMARY KEY,
                user_id TEXT NOT NULL,
                start_time TIMESTAMP NOT NULL,
                end_time TIMESTAMP,
                ip_address TEXT,
                user_agent TEXT,
                events_count INTEGER DEFAULT 0,
                status TEXT DEFAULT 'active'
            )
        """)
        
        # Audit integrity table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS audit_integrity (
                check_id TEXT PRIMARY KEY,
                check_timestamp TIMESTAMP NOT NULL,
                events_checked INTEGER,
                integrity_violations INTEGER,
                check_details TEXT,
                status TEXT DEFAULT 'completed'
            )
        """)
        
        # Create indexes for performance
        indexes = [
            "CREATE INDEX IF NOT EXISTS idx_audit_timestamp ON audit_events(timestamp)",
            "CREATE INDEX IF NOT EXISTS idx_audit_user ON audit_events(user_id)",
            "CREATE INDEX IF NOT EXISTS idx_audit_document ON audit_events(document_id)",
            "CREATE INDEX IF NOT EXISTS idx_audit_event_type ON audit_events(event_type)",
            "CREATE INDEX IF NOT EXISTS idx_audit_severity ON audit_events(severity)",
            "CREATE INDEX IF NOT EXISTS idx_audit_resource ON audit_events(resource_type, resource_id)",
            "CREATE INDEX IF NOT EXISTS idx_audit_session ON audit_events(session_id)",
            "CREATE INDEX IF NOT EXISTS idx_audit_status ON audit_events(status)"
        ]
        
        for index_sql in indexes:
            cursor.execute(index_sql)
        
        conn.commit()
        conn.close()
        
        logger.info(f"Audit database initialized at {self.db_path}")
    
    def log_event(self, event: AuditEvent) -> bool:
        """Log an audit event"""
        try:
            # Encrypt sensitive data if needed
            if self.encryption and event.should_encrypt():
                if event.old_value:
                    event.old_value = self.encryption.encrypt_data(event.old_value)
                if event.new_value:
                    event.new_value = self.encryption.encrypt_data(event.new_value)
                if event.action_description:
                    event.action_description = self.encryption.encrypt_data(event.action_description)
                event.encrypted = True
            
            # Recalculate checksum after potential encryption
            event.checksum = event._calculate_checksum()
            
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            cursor.execute("""
                INSERT INTO audit_events (
                    event_id, event_type, timestamp, user_id, user_role, session_id,
                    ip_address, user_agent, resource_type, resource_id, document_id,
                    field_name, action_description, old_value, new_value, confidence_score,
                    severity, data_classification, tags, metadata, system_version,
                    model_version, request_id, checksum, encrypted, retention_period_days, status
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                event.event_id,
                event.event_type.value,
                event.timestamp,
                event.user_id,
                event.user_role,
                event.session_id,
                event.ip_address,
                event.user_agent,
                event.resource_type,
                event.resource_id,
                event.document_id,
                event.field_name,
                event.action_description,
                event.old_value,
                event.new_value,
                event.confidence_score,
                event.severity.value,
                event.data_classification.value,
                json.dumps(event.tags),
                json.dumps(event.metadata),
                event.system_version,
                event.model_version,
                event.request_id,
                event.checksum,
                event.encrypted,
                event.retention_period_days,
                event.status.value
            ))
            
            conn.commit()
            conn.close()
            
            logger.debug(f"Audit event logged: {event.event_id}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to log audit event: {e}")
            return False
    
    def query_events(self, query: AuditQuery) -> List[AuditEvent]:
        """Query audit events"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            # Build WHERE clause
            where_conditions = []
            params = []
            
            if query.start_date:
                where_conditions.append("timestamp >= ?")
                params.append(query.start_date)
            
            if query.end_date:
                where_conditions.append("timestamp <= ?")
                params.append(query.end_date)
            
            if query.event_types:
                placeholders = ','.join(['?' for _ in query.event_types])
                where_conditions.append(f"event_type IN ({placeholders})")
                params.extend([et.value for et in query.event_types])
            
            if query.severities:
                placeholders = ','.join(['?' for _ in query.severities])
                where_conditions.append(f"severity IN ({placeholders})")
                params.extend([s.value for s in query.severities])
            
            if query.user_ids:
                placeholders = ','.join(['?' for _ in query.user_ids])
                where_conditions.append(f"user_id IN ({placeholders})")
                params.extend(query.user_ids)
            
            if query.document_ids:
                placeholders = ','.join(['?' for _ in query.document_ids])
                where_conditions.append(f"document_id IN ({placeholders})")
                params.extend(query.document_ids)
            
            if query.search_text:
                where_conditions.append("(action_description LIKE ? OR old_value LIKE ? OR new_value LIKE ?)")
                search_pattern = f"%{query.search_text}%"
                params.extend([search_pattern, search_pattern, search_pattern])
            
            if not query.include_encrypted:
                where_conditions.append("encrypted = FALSE")
            
            # Build SQL query
            sql = "SELECT * FROM audit_events"
            
            if where_conditions:
                sql += " WHERE " + " AND ".join(where_conditions)
            
            sql += f" ORDER BY {query.order_by}"
            if query.order_desc:
                sql += " DESC"
            
            sql += f" LIMIT {query.limit} OFFSET {query.offset}"
            
            cursor.execute(sql, params)
            rows = cursor.fetchall()
            conn.close()
            
            # Convert rows to AuditEvent objects
            events = []
            for row in rows:
                event = self._row_to_audit_event(row)
                
                # Decrypt if requested and possible
                if query.decrypt_results and event.encrypted and self.encryption:
                    if event.old_value:
                        event.old_value = self.encryption.decrypt_data(event.old_value)
                    if event.new_value:
                        event.new_value = self.encryption.decrypt_data(event.new_value)
                    if event.action_description:
                        event.action_description = self.encryption.decrypt_data(event.action_description)
                
                events.append(event)
            
            return events
            
        except Exception as e:
            logger.error(f"Failed to query audit events: {e}")
            return []
    
    def get_audit_summary(self, start_date: datetime = None, end_date: datetime = None) -> AuditSummary:
        """Get audit trail summary"""
        try:
            if not start_date:
                start_date = datetime.now() - timedelta(days=30)
            if not end_date:
                end_date = datetime.now()
            
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            # Total events
            cursor.execute(
                "SELECT COUNT(*) FROM audit_events WHERE timestamp BETWEEN ? AND ?",
                (start_date, end_date)
            )
            total_events = cursor.fetchone()[0]
            
            summary = AuditSummary(
                total_events=total_events,
                date_range_start=start_date,
                date_range_end=end_date
            )
            
            # Events by type
            cursor.execute("""
                SELECT event_type, COUNT(*) 
                FROM audit_events 
                WHERE timestamp BETWEEN ? AND ? 
                GROUP BY event_type
            """, (start_date, end_date))
            
            summary.events_by_type = dict(cursor.fetchall())
            
            # Events by severity
            cursor.execute("""
                SELECT severity, COUNT(*) 
                FROM audit_events 
                WHERE timestamp BETWEEN ? AND ? 
                GROUP BY severity
            """, (start_date, end_date))
            
            summary.events_by_severity = dict(cursor.fetchall())
            
            # Events by user
            cursor.execute("""
                SELECT user_id, COUNT(*) 
                FROM audit_events 
                WHERE timestamp BETWEEN ? AND ? AND user_id IS NOT NULL 
                GROUP BY user_id 
                ORDER BY COUNT(*) DESC 
                LIMIT 10
            """, (start_date, end_date))
            
            summary.events_by_user = dict(cursor.fetchall())
            
            # Events by hour
            cursor.execute("""
                SELECT strftime('%H', timestamp) as hour, COUNT(*) 
                FROM audit_events 
                WHERE timestamp BETWEEN ? AND ? 
                GROUP BY hour
            """, (start_date, end_date))
            
            summary.events_by_hour = {int(hour): count for hour, count in cursor.fetchall()}
            
            # Events by day
            cursor.execute("""
                SELECT date(timestamp) as day, COUNT(*) 
                FROM audit_events 
                WHERE timestamp BETWEEN ? AND ? 
                GROUP BY day 
                ORDER BY day
            """, (start_date, end_date))
            
            summary.events_by_day = dict(cursor.fetchall())
            
            # Security metrics
            cursor.execute("""
                SELECT COUNT(*) FROM audit_events 
                WHERE event_type = 'user_login' AND severity = 'high' 
                AND timestamp BETWEEN ? AND ?
            """, (start_date, end_date))
            
            summary.failed_login_attempts = cursor.fetchone()[0]
            
            cursor.execute("""
                SELECT COUNT(*) FROM audit_events 
                WHERE event_type = 'permission_changed' 
                AND timestamp BETWEEN ? AND ?
            """, (start_date, end_date))
            
            summary.permission_changes = cursor.fetchone()[0]
            
            cursor.execute("""
                SELECT COUNT(*) FROM audit_events 
                WHERE severity = 'critical' 
                AND timestamp BETWEEN ? AND ?
            """, (start_date, end_date))
            
            summary.critical_events = cursor.fetchone()[0]
            
            cursor.execute("""
                SELECT COUNT(*) FROM audit_events 
                WHERE encrypted = TRUE 
                AND timestamp BETWEEN ? AND ?
            """, (start_date, end_date))
            
            summary.encrypted_events = cursor.fetchone()[0]
            
            conn.close()
            
            return summary
            
        except Exception as e:
            logger.error(f"Failed to generate audit summary: {e}")
            return AuditSummary(
                total_events=0,
                date_range_start=start_date or datetime.now(),
                date_range_end=end_date or datetime.now()
            )
    
    def verify_integrity(self, start_date: datetime = None, end_date: datetime = None) -> Dict[str, Any]:
        """Verify integrity of audit records"""
        try:
            if not start_date:
                start_date = datetime.now() - timedelta(days=7)
            if not end_date:
                end_date = datetime.now()
            
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            cursor.execute("""
                SELECT * FROM audit_events 
                WHERE timestamp BETWEEN ? AND ? 
                ORDER BY timestamp
            """, (start_date, end_date))
            
            rows = cursor.fetchall()
            conn.close()
            
            total_checked = len(rows)
            violations = []
            
            for row in rows:
                event = self._row_to_audit_event(row)
                
                if not event.verify_integrity():
                    violations.append({
                        'event_id': event.event_id,
                        'timestamp': event.timestamp.isoformat(),
                        'event_type': event.event_type.value,
                        'expected_checksum': event._calculate_checksum(),
                        'stored_checksum': event.checksum
                    })
            
            # Log integrity check
            check_id = str(uuid.uuid4())
            check_details = {
                'period_start': start_date.isoformat(),
                'period_end': end_date.isoformat(),
                'violations_found': len(violations)
            }
            
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            cursor.execute("""
                INSERT INTO audit_integrity (
                    check_id, check_timestamp, events_checked, 
                    integrity_violations, check_details
                ) VALUES (?, ?, ?, ?, ?)
            """, (
                check_id,
                datetime.now(),
                total_checked,
                len(violations),
                json.dumps(check_details)
            ))
            
            conn.commit()
            conn.close()
            
            result = {
                'check_id': check_id,
                'timestamp': datetime.now().isoformat(),
                'events_checked': total_checked,
                'integrity_violations': len(violations),
                'violations': violations,
                'integrity_rate': (total_checked - len(violations)) / total_checked if total_checked > 0 else 1.0
            }
            
            if violations:
                logger.warning(f"Integrity check found {len(violations)} violations out of {total_checked} events")
            else:
                logger.info(f"Integrity check passed: {total_checked} events verified")
            
            return result
            
        except Exception as e:
            logger.error(f"Failed to verify audit integrity: {e}")
            return {
                'check_id': str(uuid.uuid4()),
                'timestamp': datetime.now().isoformat(),
                'events_checked': 0,
                'integrity_violations': 0,
                'violations': [],
                'integrity_rate': 0.0,
                'error': str(e)
            }
    
    def archive_old_events(self, retention_days: int = 2555) -> Dict[str, Any]:
        """Archive old audit events"""
        try:
            cutoff_date = datetime.now() - timedelta(days=retention_days)
            
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            # Count events to be archived
            cursor.execute(
                "SELECT COUNT(*) FROM audit_events WHERE timestamp < ? AND status = 'active'",
                (cutoff_date,)
            )
            events_to_archive = cursor.fetchone()[0]
            
            if events_to_archive == 0:
                conn.close()
                return {
                    'archived_events': 0,
                    'message': 'No events to archive'
                }
            
            # Update status to archived
            cursor.execute("""
                UPDATE audit_events 
                SET status = 'archived' 
                WHERE timestamp < ? AND status = 'active'
            """, (cutoff_date,))
            
            conn.commit()
            conn.close()
            
            logger.info(f"Archived {events_to_archive} audit events older than {retention_days} days")
            
            return {
                'archived_events': events_to_archive,
                'cutoff_date': cutoff_date.isoformat(),
                'message': f'Successfully archived {events_to_archive} events'
            }
            
        except Exception as e:
            logger.error(f"Failed to archive audit events: {e}")
            return {
                'archived_events': 0,
                'error': str(e)
            }
    
    def _row_to_audit_event(self, row) -> AuditEvent:
        """Convert database row to AuditEvent"""
        return AuditEvent(
            event_id=row[0],
            event_type=AuditEventType(row[1]),
            timestamp=datetime.fromisoformat(row[2]) if row[2] else datetime.now(),
            user_id=row[3],
            user_role=row[4],
            session_id=row[5],
            ip_address=row[6],
            user_agent=row[7],
            resource_type=row[8],
            resource_id=row[9],
            document_id=row[10],
            field_name=row[11],
            action_description=row[12] or "",
            old_value=row[13],
            new_value=row[14],
            confidence_score=row[15],
            severity=AuditSeverity(row[16]),
            data_classification=DataClassification(row[17]),
            tags=json.loads(row[18]) if row[18] else [],
            metadata=json.loads(row[19]) if row[19] else {},
            system_version=row[20],
            model_version=row[21],
            request_id=row[22],
            checksum=row[23],
            encrypted=bool(row[24]),
            retention_period_days=row[25] or 2555,
            status=AuditStatus(row[26]) if row[26] else AuditStatus.ACTIVE
        )

class AuditTrailManager:
    """Main audit trail management system"""
    
    def __init__(self, db_path: str = "audit_trail.db", encryption_password: str = None):
        self.db = AuditDatabase(db_path, encryption_password)
        self.current_session_id = None
        self.current_user_id = None
        
        logger.info("Audit trail manager initialized")
    
    def start_session(self, user_id: str, ip_address: str = None, user_agent: str = None) -> str:
        """Start an audit session"""
        session_id = str(uuid.uuid4())
        self.current_session_id = session_id
        self.current_user_id = user_id
        
        # Log session start
        self.log_event(
            event_type=AuditEventType.USER_LOGIN,
            action_description=f"User session started",
            user_id=user_id,
            session_id=session_id,
            ip_address=ip_address,
            user_agent=user_agent,
            severity=AuditSeverity.LOW
        )
        
        logger.info(f"Audit session started for user {user_id}: {session_id}")
        return session_id
    
    def end_session(self, session_id: str = None) -> bool:
        """End an audit session"""
        if not session_id:
            session_id = self.current_session_id
        
        if session_id:
            # Log session end
            self.log_event(
                event_type=AuditEventType.USER_LOGOUT,
                action_description=f"User session ended",
                session_id=session_id,
                severity=AuditSeverity.LOW
            )
            
            if session_id == self.current_session_id:
                self.current_session_id = None
                self.current_user_id = None
            
            logger.info(f"Audit session ended: {session_id}")
            return True
        
        return False
    
    def log_event(self, event_type: AuditEventType, action_description: str,
                  user_id: str = None, session_id: str = None,
                  document_id: str = None, field_name: str = None,
                  old_value: str = None, new_value: str = None,
                  confidence_score: float = None, severity: AuditSeverity = AuditSeverity.MEDIUM,
                  data_classification: DataClassification = DataClassification.INTERNAL,
                  resource_type: str = None, resource_id: str = None,
                  ip_address: str = None, user_agent: str = None,
                  tags: List[str] = None, metadata: Dict[str, Any] = None,
                  system_version: str = None, model_version: str = None,
                  request_id: str = None) -> str:
        """Log an audit event"""
        
        event_id = str(uuid.uuid4())
        
        # Use current session context if not provided
        if not user_id:
            user_id = self.current_user_id
        if not session_id:
            session_id = self.current_session_id
        
        event = AuditEvent(
            event_id=event_id,
            event_type=event_type,
            timestamp=datetime.now(),
            user_id=user_id,
            session_id=session_id,
            ip_address=ip_address,
            user_agent=user_agent,
            resource_type=resource_type,
            resource_id=resource_id,
            document_id=document_id,
            field_name=field_name,
            action_description=action_description,
            old_value=old_value,
            new_value=new_value,
            confidence_score=confidence_score,
            severity=severity,
            data_classification=data_classification,
            tags=tags or [],
            metadata=metadata or {},
            system_version=system_version,
            model_version=model_version,
            request_id=request_id
        )
        
        success = self.db.log_event(event)
        
        if success:
            logger.debug(f"Audit event logged: {event_type.value} - {action_description[:50]}...")
        
        return event_id if success else None
    
    def log_document_upload(self, document_id: str, filename: str, user_id: str = None,
                          file_size: int = None, file_type: str = None) -> str:
        """Log document upload event"""
        metadata = {}
        if file_size:
            metadata['file_size'] = file_size
        if file_type:
            metadata['file_type'] = file_type
        if filename:
            metadata['filename'] = filename
        
        return self.log_event(
            event_type=AuditEventType.DOCUMENT_UPLOADED,
            action_description=f"Document uploaded: {filename}",
            user_id=user_id,
            document_id=document_id,
            resource_type="document",
            resource_id=document_id,
            severity=AuditSeverity.LOW,
            metadata=metadata
        )
    
    def log_extraction_performed(self, document_id: str, model_version: str,
                               extracted_fields: Dict[str, Any], user_id: str = None) -> str:
        """Log field extraction event"""
        metadata = {
            'extracted_fields_count': len(extracted_fields),
            'field_names': list(extracted_fields.keys())
        }
        
        return self.log_event(
            event_type=AuditEventType.EXTRACTION_PERFORMED,
            action_description=f"Field extraction completed for document",
            user_id=user_id,
            document_id=document_id,
            resource_type="document",
            resource_id=document_id,
            model_version=model_version,
            severity=AuditSeverity.LOW,
            metadata=metadata
        )
    
    def log_field_modification(self, document_id: str, field_name: str,
                             old_value: str, new_value: str, user_id: str = None,
                             confidence_score: float = None) -> str:
        """Log field modification event"""
        return self.log_event(
            event_type=AuditEventType.FIELD_MODIFIED,
            action_description=f"Field '{field_name}' modified",
            user_id=user_id,
            document_id=document_id,
            field_name=field_name,
            old_value=old_value,
            new_value=new_value,
            confidence_score=confidence_score,
            resource_type="field",
            resource_id=f"{document_id}:{field_name}",
            severity=AuditSeverity.MEDIUM,
            data_classification=DataClassification.CONFIDENTIAL
        )
    
    def log_decision_made(self, document_id: str, field_name: str, decision_type: str,
                        confidence_score: float, user_id: str = None,
                        threshold_config_id: str = None) -> str:
        """Log processing decision event"""
        metadata = {}
        if threshold_config_id:
            metadata['threshold_config_id'] = threshold_config_id
        
        return self.log_event(
            event_type=AuditEventType.DECISION_MADE,
            action_description=f"Processing decision: {decision_type} for field '{field_name}'",
            user_id=user_id,
            document_id=document_id,
            field_name=field_name,
            confidence_score=confidence_score,
            resource_type="decision",
            severity=AuditSeverity.LOW,
            metadata=metadata
        )
    
    def log_human_review(self, document_id: str, reviewer_id: str, review_result: str,
                       fields_reviewed: List[str], review_duration_seconds: int = None) -> str:
        """Log human review event"""
        metadata = {
            'fields_reviewed': fields_reviewed,
            'review_result': review_result
        }
        if review_duration_seconds:
            metadata['review_duration_seconds'] = review_duration_seconds
        
        return self.log_event(
            event_type=AuditEventType.HUMAN_REVIEW_COMPLETED,
            action_description=f"Human review completed: {review_result}",
            user_id=reviewer_id,
            document_id=document_id,
            resource_type="review",
            resource_id=document_id,
            severity=AuditSeverity.MEDIUM,
            metadata=metadata
        )
    
    def log_system_error(self, error_message: str, error_type: str = None,
                       document_id: str = None, user_id: str = None,
                       stack_trace: str = None) -> str:
        """Log system error event"""
        metadata = {}
        if error_type:
            metadata['error_type'] = error_type
        if stack_trace:
            metadata['stack_trace'] = stack_trace
        
        return self.log_event(
            event_type=AuditEventType.SYSTEM_ERROR,
            action_description=f"System error: {error_message}",
            user_id=user_id,
            document_id=document_id,
            resource_type="system",
            severity=AuditSeverity.HIGH,
            data_classification=DataClassification.INTERNAL,
            metadata=metadata
        )
    
    def search_events(self, query: AuditQuery) -> List[AuditEvent]:
        """Search audit events"""
        return self.db.query_events(query)
    
    def get_summary(self, start_date: datetime = None, end_date: datetime = None) -> AuditSummary:
        """Get audit trail summary"""
        return self.db.get_audit_summary(start_date, end_date)
    
    def verify_integrity(self, start_date: datetime = None, end_date: datetime = None) -> Dict[str, Any]:
        """Verify audit trail integrity"""
        return self.db.verify_integrity(start_date, end_date)
    
    def archive_old_events(self, retention_days: int = 2555) -> Dict[str, Any]:
        """Archive old audit events"""
        return self.db.archive_old_events(retention_days)
    
    def export_audit_trail(self, query: AuditQuery, format: str = "json") -> str:
        """Export audit trail data"""
        events = self.search_events(query)
        
        if format.lower() == "json":
            return json.dumps([event.to_dict() for event in events], indent=2)
        elif format.lower() == "csv":
            if events:
                df = pd.DataFrame([event.to_dict() for event in events])
                return df.to_csv(index=False)
            return ""
        else:
            raise ValueError(f"Unsupported export format: {format}")
    
    def get_user_activity(self, user_id: str, days_back: int = 30) -> Dict[str, Any]:
        """Get user activity summary"""
        end_date = datetime.now()
        start_date = end_date - timedelta(days=days_back)
        
        query = AuditQuery(
            start_date=start_date,
            end_date=end_date,
            user_ids=[user_id]
        )
        
        events = self.search_events(query)
        
        activity = {
            'user_id': user_id,
            'period_start': start_date.isoformat(),
            'period_end': end_date.isoformat(),
            'total_events': len(events),
            'events_by_type': {},
            'events_by_day': {},
            'documents_accessed': set(),
            'fields_modified': [],
            'login_sessions': 0
        }
        
        for event in events:
            # Count by type
            event_type = event.event_type.value
            activity['events_by_type'][event_type] = activity['events_by_type'].get(event_type, 0) + 1
            
            # Count by day
            day = event.timestamp.date().isoformat()
            activity['events_by_day'][day] = activity['events_by_day'].get(day, 0) + 1
            
            # Track documents
            if event.document_id:
                activity['documents_accessed'].add(event.document_id)
            
            # Track field modifications
            if event.event_type == AuditEventType.FIELD_MODIFIED:
                activity['fields_modified'].append({
                    'document_id': event.document_id,
                    'field_name': event.field_name,
                    'timestamp': event.timestamp.isoformat()
                })
            
            # Count login sessions
            if event.event_type == AuditEventType.USER_LOGIN:
                activity['login_sessions'] += 1
        
        activity['documents_accessed'] = list(activity['documents_accessed'])
        
        return activity

def main():
    """Main function for standalone execution"""
    # Example usage of the audit trail system
    
    # Initialize system
    audit_manager = AuditTrailManager()
    
    print("\n=== Audit Trail System Demo ===")
    
    # Start user session
    session_id = audit_manager.start_session(
        user_id="user_001",
        ip_address="192.168.1.100",
        user_agent="Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"
    )
    
    print(f"\nStarted audit session: {session_id}")
    
    # Simulate document processing events
    print("\n=== Logging Document Processing Events ===")
    
    # Document upload
    doc_id = "doc_12345"
    audit_manager.log_document_upload(
        document_id=doc_id,
        filename="identity_card_001.jpg",
        file_size=2048576,
        file_type="image/jpeg"
    )
    print("Logged: Document upload")
    
    # Field extraction
    extracted_fields = {
        "ic_number": "123456-78-9012",
        "name": "Ahmad bin Ali",
        "address": "123 Jalan Merdeka, KL"
    }
    
    audit_manager.log_extraction_performed(
        document_id=doc_id,
        model_version="v2.1",
        extracted_fields=extracted_fields
    )
    print("Logged: Field extraction")
    
    # Processing decisions
    audit_manager.log_decision_made(
        document_id=doc_id,
        field_name="ic_number",
        decision_type="auto_approve",
        confidence_score=0.92,
        threshold_config_id="config_001"
    )
    print("Logged: Processing decision")
    
    # Field modification
    audit_manager.log_field_modification(
        document_id=doc_id,
        field_name="address",
        old_value="123 Jalan Merdeka",
        new_value="123 Jalan Merdeka, Kuala Lumpur",
        confidence_score=0.75
    )
    print("Logged: Field modification")
    
    # Human review
    audit_manager.log_human_review(
        document_id=doc_id,
        reviewer_id="reviewer_001",
        review_result="approved",
        fields_reviewed=["ic_number", "name", "address"],
        review_duration_seconds=120
    )
    print("Logged: Human review")
    
    # System error
    audit_manager.log_system_error(
        error_message="Database connection timeout",
        error_type="DatabaseError",
        document_id=doc_id
    )
    print("Logged: System error")
    
    # Search audit events
    print("\n=== Searching Audit Events ===")
    
    query = AuditQuery(
        start_date=datetime.now() - timedelta(hours=1),
        end_date=datetime.now(),
        document_ids=[doc_id],
        limit=10
    )
    
    events = audit_manager.search_events(query)
    print(f"Found {len(events)} events for document {doc_id}")
    
    for event in events[:3]:  # Show first 3 events
        print(f"  - {event.timestamp.strftime('%H:%M:%S')}: {event.event_type.value} - {event.action_description}")
    
    # Get audit summary
    print("\n=== Audit Summary ===")
    
    summary = audit_manager.get_summary()
    print(f"Total events: {summary.total_events}")
    print(f"Events by type: {dict(list(summary.events_by_type.items())[:3])}")
    print(f"Events by severity: {summary.events_by_severity}")
    print(f"Critical events: {summary.critical_events}")
    print(f"Encrypted events: {summary.encrypted_events}")
    
    # Verify integrity
    print("\n=== Integrity Verification ===")
    
    integrity_result = audit_manager.verify_integrity()
    print(f"Events checked: {integrity_result['events_checked']}")
    print(f"Integrity violations: {integrity_result['integrity_violations']}")
    print(f"Integrity rate: {integrity_result['integrity_rate']:.3f}")
    
    # Get user activity
    print("\n=== User Activity ===")
    
    user_activity = audit_manager.get_user_activity("user_001", days_back=1)
    print(f"User: {user_activity['user_id']}")
    print(f"Total events: {user_activity['total_events']}")
    print(f"Documents accessed: {len(user_activity['documents_accessed'])}")
    print(f"Fields modified: {len(user_activity['fields_modified'])}")
    print(f"Login sessions: {user_activity['login_sessions']}")
    
    # Export audit trail
    print("\n=== Export Audit Trail ===")
    
    export_query = AuditQuery(
        start_date=datetime.now() - timedelta(hours=1),
        limit=5
    )
    
    json_export = audit_manager.export_audit_trail(export_query, "json")
    print(f"JSON export size: {len(json_export)} characters")
    
    # End session
    audit_manager.end_session()
    print(f"\nEnded audit session: {session_id}")
    
    print("\n=== Demo Complete ===")
    print("Audit trail system ready for production use.")
    print("\nKey features demonstrated:")
    print("- Event logging with integrity verification")
    print("- Comprehensive search and filtering")
    print("- Security and encryption support")
    print("- User activity tracking")
    print("- Data export capabilities")
    print("- Automated archival and retention")

if __name__ == "__main__":
    main()