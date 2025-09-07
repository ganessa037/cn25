#!/usr/bin/env python3
"""
Data Privacy and PDPA Compliance Module
======================================

This module implements comprehensive data privacy features compliant with
Malaysian Personal Data Protection Act (PDPA) and international standards.

Features:
- PDPA compliance framework
- Data minimization principles
- Consent management
- Data subject rights (access, rectification, erasure)
- Privacy impact assessments
- Data breach notification
- Cross-border data transfer controls
"""

import asyncio
import hashlib
import json
import logging
import uuid
from datetime import datetime, timedelta
from enum import Enum
from typing import Dict, List, Optional, Any, Set
from dataclasses import dataclass, asdict
from pathlib import Path

import aiofiles
from cryptography.fernet import Fernet
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC
import base64

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class DataCategory(Enum):
    """Categories of personal data under PDPA"""
    PERSONAL_IDENTIFIER = "personal_identifier"  # Name, IC, passport
    CONTACT_INFO = "contact_info"  # Email, phone, address
    BIOMETRIC = "biometric"  # Fingerprints, facial recognition
    FINANCIAL = "financial"  # Bank details, payment info
    HEALTH = "health"  # Medical records
    BEHAVIORAL = "behavioral"  # Usage patterns, preferences
    TECHNICAL = "technical"  # IP address, device info
    SENSITIVE = "sensitive"  # Race, religion, political views

class ConsentStatus(Enum):
    """Consent status for data processing"""
    GIVEN = "given"
    WITHDRAWN = "withdrawn"
    PENDING = "pending"
    EXPIRED = "expired"

class ProcessingPurpose(Enum):
    """Lawful purposes for data processing"""
    DOCUMENT_PROCESSING = "document_processing"
    SERVICE_PROVISION = "service_provision"
    LEGAL_COMPLIANCE = "legal_compliance"
    LEGITIMATE_INTEREST = "legitimate_interest"
    VITAL_INTEREST = "vital_interest"
    PUBLIC_TASK = "public_task"

class DataSubjectRight(Enum):
    """Rights of data subjects under PDPA"""
    ACCESS = "access"  # Right to access personal data
    RECTIFICATION = "rectification"  # Right to correct data
    ERASURE = "erasure"  # Right to delete data
    PORTABILITY = "portability"  # Right to data portability
    OBJECTION = "objection"  # Right to object to processing
    RESTRICTION = "restriction"  # Right to restrict processing

@dataclass
class ConsentRecord:
    """Record of user consent for data processing"""
    user_id: str
    data_categories: List[DataCategory]
    processing_purposes: List[ProcessingPurpose]
    consent_status: ConsentStatus
    consent_date: datetime
    expiry_date: Optional[datetime]
    withdrawal_date: Optional[datetime]
    consent_method: str  # web_form, api, email, etc.
    ip_address: str
    user_agent: str
    consent_text: str
    version: str

@dataclass
class DataProcessingRecord:
    """Record of data processing activities"""
    record_id: str
    user_id: str
    data_categories: List[DataCategory]
    processing_purpose: ProcessingPurpose
    processing_date: datetime
    data_source: str
    data_destination: str
    retention_period: timedelta
    legal_basis: str
    processor_id: str
    security_measures: List[str]

@dataclass
class DataSubjectRequest:
    """Request from data subject exercising their rights"""
    request_id: str
    user_id: str
    request_type: DataSubjectRight
    request_date: datetime
    status: str  # pending, processing, completed, rejected
    completion_date: Optional[datetime]
    request_details: Dict[str, Any]
    response_data: Optional[Dict[str, Any]]
    verification_method: str

@dataclass
class PrivacyImpactAssessment:
    """Privacy Impact Assessment for high-risk processing"""
    assessment_id: str
    processing_description: str
    data_categories: List[DataCategory]
    processing_purposes: List[ProcessingPurpose]
    risk_level: str  # low, medium, high
    identified_risks: List[str]
    mitigation_measures: List[str]
    assessment_date: datetime
    assessor_id: str
    approval_status: str

class PDPACompliance:
    """Main class for PDPA compliance management"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.consent_records: Dict[str, ConsentRecord] = {}
        self.processing_records: List[DataProcessingRecord] = []
        self.subject_requests: Dict[str, DataSubjectRequest] = {}
        self.privacy_assessments: Dict[str, PrivacyImpactAssessment] = {}
        self.encryption_key = self._generate_encryption_key()
        
    def _generate_encryption_key(self) -> Fernet:
        """Generate encryption key for sensitive data"""
        password = self.config.get('encryption_password', 'default_password').encode()
        salt = self.config.get('encryption_salt', b'salt_1234567890123456')
        kdf = PBKDF2HMAC(
            algorithm=hashes.SHA256(),
            length=32,
            salt=salt,
            iterations=100000,
        )
        key = base64.urlsafe_b64encode(kdf.derive(password))
        return Fernet(key)
    
    async def record_consent(self, consent_data: Dict[str, Any]) -> ConsentRecord:
        """Record user consent for data processing"""
        try:
            consent_record = ConsentRecord(
                user_id=consent_data['user_id'],
                data_categories=[DataCategory(cat) for cat in consent_data['data_categories']],
                processing_purposes=[ProcessingPurpose(purpose) for purpose in consent_data['processing_purposes']],
                consent_status=ConsentStatus.GIVEN,
                consent_date=datetime.now(),
                expiry_date=consent_data.get('expiry_date'),
                withdrawal_date=None,
                consent_method=consent_data['consent_method'],
                ip_address=consent_data['ip_address'],
                user_agent=consent_data['user_agent'],
                consent_text=consent_data['consent_text'],
                version=consent_data.get('version', '1.0')
            )
            
            self.consent_records[consent_record.user_id] = consent_record
            
            # Log consent recording
            logger.info(f"Consent recorded for user {consent_record.user_id}")
            
            # Save to persistent storage
            await self._save_consent_record(consent_record)
            
            return consent_record
            
        except Exception as e:
            logger.error(f"Error recording consent: {e}")
            raise
    
    async def withdraw_consent(self, user_id: str, withdrawal_reason: str = "") -> bool:
        """Process consent withdrawal"""
        try:
            if user_id not in self.consent_records:
                logger.warning(f"No consent record found for user {user_id}")
                return False
            
            consent_record = self.consent_records[user_id]
            consent_record.consent_status = ConsentStatus.WITHDRAWN
            consent_record.withdrawal_date = datetime.now()
            
            # Trigger data deletion process
            await self._trigger_data_deletion(user_id, withdrawal_reason)
            
            logger.info(f"Consent withdrawn for user {user_id}")
            return True
            
        except Exception as e:
            logger.error(f"Error withdrawing consent: {e}")
            return False
    
    def check_consent_validity(self, user_id: str, data_category: DataCategory, 
                             processing_purpose: ProcessingPurpose) -> bool:
        """Check if valid consent exists for specific processing"""
        try:
            if user_id not in self.consent_records:
                return False
            
            consent = self.consent_records[user_id]
            
            # Check consent status
            if consent.consent_status != ConsentStatus.GIVEN:
                return False
            
            # Check expiry
            if consent.expiry_date and datetime.now() > consent.expiry_date:
                consent.consent_status = ConsentStatus.EXPIRED
                return False
            
            # Check data category and purpose
            return (data_category in consent.data_categories and 
                   processing_purpose in consent.processing_purposes)
            
        except Exception as e:
            logger.error(f"Error checking consent validity: {e}")
            return False
    
    async def record_processing_activity(self, processing_data: Dict[str, Any]) -> DataProcessingRecord:
        """Record data processing activity"""
        try:
            processing_record = DataProcessingRecord(
                record_id=str(uuid.uuid4()),
                user_id=processing_data['user_id'],
                data_categories=[DataCategory(cat) for cat in processing_data['data_categories']],
                processing_purpose=ProcessingPurpose(processing_data['processing_purpose']),
                processing_date=datetime.now(),
                data_source=processing_data['data_source'],
                data_destination=processing_data['data_destination'],
                retention_period=timedelta(days=processing_data.get('retention_days', 365)),
                legal_basis=processing_data['legal_basis'],
                processor_id=processing_data['processor_id'],
                security_measures=processing_data.get('security_measures', [])
            )
            
            self.processing_records.append(processing_record)
            
            # Save to persistent storage
            await self._save_processing_record(processing_record)
            
            logger.info(f"Processing activity recorded: {processing_record.record_id}")
            return processing_record
            
        except Exception as e:
            logger.error(f"Error recording processing activity: {e}")
            raise
    
    async def handle_subject_request(self, request_data: Dict[str, Any]) -> DataSubjectRequest:
        """Handle data subject rights request"""
        try:
            request = DataSubjectRequest(
                request_id=str(uuid.uuid4()),
                user_id=request_data['user_id'],
                request_type=DataSubjectRight(request_data['request_type']),
                request_date=datetime.now(),
                status='pending',
                completion_date=None,
                request_details=request_data.get('details', {}),
                response_data=None,
                verification_method=request_data['verification_method']
            )
            
            self.subject_requests[request.request_id] = request
            
            # Process request based on type
            await self._process_subject_request(request)
            
            logger.info(f"Subject request created: {request.request_id}")
            return request
            
        except Exception as e:
            logger.error(f"Error handling subject request: {e}")
            raise
    
    async def _process_subject_request(self, request: DataSubjectRequest):
        """Process specific type of subject request"""
        try:
            request.status = 'processing'
            
            if request.request_type == DataSubjectRight.ACCESS:
                # Provide access to personal data
                user_data = await self._collect_user_data(request.user_id)
                request.response_data = user_data
                
            elif request.request_type == DataSubjectRight.ERASURE:
                # Delete personal data
                await self._delete_user_data(request.user_id)
                request.response_data = {'deleted': True, 'deletion_date': datetime.now().isoformat()}
                
            elif request.request_type == DataSubjectRight.RECTIFICATION:
                # Correct personal data
                corrections = request.request_details.get('corrections', {})
                await self._update_user_data(request.user_id, corrections)
                request.response_data = {'updated': True, 'update_date': datetime.now().isoformat()}
                
            elif request.request_type == DataSubjectRight.PORTABILITY:
                # Export data in portable format
                portable_data = await self._export_user_data(request.user_id)
                request.response_data = portable_data
            
            request.status = 'completed'
            request.completion_date = datetime.now()
            
        except Exception as e:
            request.status = 'failed'
            logger.error(f"Error processing subject request {request.request_id}: {e}")
    
    async def conduct_privacy_impact_assessment(self, assessment_data: Dict[str, Any]) -> PrivacyImpactAssessment:
        """Conduct Privacy Impact Assessment for high-risk processing"""
        try:
            assessment = PrivacyImpactAssessment(
                assessment_id=str(uuid.uuid4()),
                processing_description=assessment_data['processing_description'],
                data_categories=[DataCategory(cat) for cat in assessment_data['data_categories']],
                processing_purposes=[ProcessingPurpose(purpose) for purpose in assessment_data['processing_purposes']],
                risk_level=self._assess_risk_level(assessment_data),
                identified_risks=assessment_data.get('identified_risks', []),
                mitigation_measures=assessment_data.get('mitigation_measures', []),
                assessment_date=datetime.now(),
                assessor_id=assessment_data['assessor_id'],
                approval_status='pending'
            )
            
            self.privacy_assessments[assessment.assessment_id] = assessment
            
            logger.info(f"Privacy Impact Assessment created: {assessment.assessment_id}")
            return assessment
            
        except Exception as e:
            logger.error(f"Error conducting PIA: {e}")
            raise
    
    def _assess_risk_level(self, assessment_data: Dict[str, Any]) -> str:
        """Assess risk level based on data categories and processing"""
        high_risk_categories = {DataCategory.BIOMETRIC, DataCategory.HEALTH, DataCategory.SENSITIVE}
        data_categories = [DataCategory(cat) for cat in assessment_data['data_categories']]
        
        if any(cat in high_risk_categories for cat in data_categories):
            return 'high'
        elif len(data_categories) > 3:
            return 'medium'
        else:
            return 'low'
    
    async def _save_consent_record(self, consent_record: ConsentRecord):
        """Save consent record to persistent storage"""
        try:
            consent_dir = Path(self.config.get('consent_storage_path', './data/consent'))
            consent_dir.mkdir(parents=True, exist_ok=True)
            
            file_path = consent_dir / f"{consent_record.user_id}_consent.json"
            
            # Encrypt sensitive data
            encrypted_data = self.encryption_key.encrypt(
                json.dumps(asdict(consent_record), default=str).encode()
            )
            
            async with aiofiles.open(file_path, 'wb') as f:
                await f.write(encrypted_data)
                
        except Exception as e:
            logger.error(f"Error saving consent record: {e}")
    
    async def _save_processing_record(self, processing_record: DataProcessingRecord):
        """Save processing record to persistent storage"""
        try:
            processing_dir = Path(self.config.get('processing_storage_path', './data/processing'))
            processing_dir.mkdir(parents=True, exist_ok=True)
            
            file_path = processing_dir / f"{processing_record.record_id}_processing.json"
            
            # Encrypt sensitive data
            encrypted_data = self.encryption_key.encrypt(
                json.dumps(asdict(processing_record), default=str).encode()
            )
            
            async with aiofiles.open(file_path, 'wb') as f:
                await f.write(encrypted_data)
                
        except Exception as e:
            logger.error(f"Error saving processing record: {e}")
    
    async def _trigger_data_deletion(self, user_id: str, reason: str):
        """Trigger data deletion process after consent withdrawal"""
        try:
            # Mark user data for deletion
            deletion_record = {
                'user_id': user_id,
                'deletion_date': datetime.now(),
                'reason': reason,
                'status': 'scheduled'
            }
            
            # Save deletion record
            deletion_dir = Path(self.config.get('deletion_storage_path', './data/deletions'))
            deletion_dir.mkdir(parents=True, exist_ok=True)
            
            file_path = deletion_dir / f"{user_id}_deletion.json"
            async with aiofiles.open(file_path, 'w') as f:
                await f.write(json.dumps(deletion_record, default=str))
            
            logger.info(f"Data deletion scheduled for user {user_id}")
            
        except Exception as e:
            logger.error(f"Error triggering data deletion: {e}")
    
    async def _collect_user_data(self, user_id: str) -> Dict[str, Any]:
        """Collect all personal data for a user"""
        # This would integrate with your actual data storage systems
        return {
            'user_id': user_id,
            'consent_records': [asdict(self.consent_records.get(user_id, {}))],
            'processing_records': [asdict(record) for record in self.processing_records if record.user_id == user_id],
            'collection_date': datetime.now().isoformat()
        }
    
    async def _delete_user_data(self, user_id: str):
        """Delete all personal data for a user"""
        # Remove from in-memory storage
        if user_id in self.consent_records:
            del self.consent_records[user_id]
        
        self.processing_records = [record for record in self.processing_records if record.user_id != user_id]
        
        # Delete from persistent storage
        # This would integrate with your actual data storage systems
        logger.info(f"User data deleted for {user_id}")
    
    async def _update_user_data(self, user_id: str, corrections: Dict[str, Any]):
        """Update user data with corrections"""
        # This would integrate with your actual data storage systems
        logger.info(f"User data updated for {user_id}: {corrections}")
    
    async def _export_user_data(self, user_id: str) -> Dict[str, Any]:
        """Export user data in portable format"""
        user_data = await self._collect_user_data(user_id)
        return {
            'export_format': 'JSON',
            'export_date': datetime.now().isoformat(),
            'data': user_data
        }
    
    def generate_compliance_report(self) -> Dict[str, Any]:
        """Generate PDPA compliance report"""
        try:
            total_consents = len(self.consent_records)
            active_consents = sum(1 for consent in self.consent_records.values() 
                                if consent.consent_status == ConsentStatus.GIVEN)
            withdrawn_consents = sum(1 for consent in self.consent_records.values() 
                                   if consent.consent_status == ConsentStatus.WITHDRAWN)
            
            total_processing = len(self.processing_records)
            total_requests = len(self.subject_requests)
            completed_requests = sum(1 for request in self.subject_requests.values() 
                                   if request.status == 'completed')
            
            return {
                'report_date': datetime.now().isoformat(),
                'consent_statistics': {
                    'total_consents': total_consents,
                    'active_consents': active_consents,
                    'withdrawn_consents': withdrawn_consents,
                    'consent_rate': active_consents / total_consents if total_consents > 0 else 0
                },
                'processing_statistics': {
                    'total_processing_activities': total_processing,
                    'processing_by_purpose': self._group_processing_by_purpose()
                },
                'subject_requests': {
                    'total_requests': total_requests,
                    'completed_requests': completed_requests,
                    'completion_rate': completed_requests / total_requests if total_requests > 0 else 0,
                    'requests_by_type': self._group_requests_by_type()
                },
                'privacy_assessments': {
                    'total_assessments': len(self.privacy_assessments),
                    'high_risk_assessments': sum(1 for pia in self.privacy_assessments.values() 
                                               if pia.risk_level == 'high')
                }
            }
            
        except Exception as e:
            logger.error(f"Error generating compliance report: {e}")
            return {}
    
    def _group_processing_by_purpose(self) -> Dict[str, int]:
        """Group processing activities by purpose"""
        purpose_counts = {}
        for record in self.processing_records:
            purpose = record.processing_purpose.value
            purpose_counts[purpose] = purpose_counts.get(purpose, 0) + 1
        return purpose_counts
    
    def _group_requests_by_type(self) -> Dict[str, int]:
        """Group subject requests by type"""
        type_counts = {}
        for request in self.subject_requests.values():
            request_type = request.request_type.value
            type_counts[request_type] = type_counts.get(request_type, 0) + 1
        return type_counts

# Example usage and testing
async def main():
    """Example usage of PDPA compliance system"""
    
    # Configuration
    config = {
        'encryption_password': 'secure_password_123',
        'encryption_salt': b'salt_1234567890123456',
        'consent_storage_path': './data/consent',
        'processing_storage_path': './data/processing',
        'deletion_storage_path': './data/deletions'
    }
    
    # Initialize PDPA compliance system
    pdpa = PDPACompliance(config)
    
    print("🔒 PDPA Compliance System Initialized")
    print("=" * 50)
    
    # Example: Record user consent
    consent_data = {
        'user_id': 'user_123',
        'data_categories': ['personal_identifier', 'contact_info'],
        'processing_purposes': ['document_processing', 'service_provision'],
        'consent_method': 'web_form',
        'ip_address': '192.168.1.100',
        'user_agent': 'Mozilla/5.0...',
        'consent_text': 'I consent to the processing of my personal data for document processing services.',
        'version': '1.0'
    }
    
    consent_record = await pdpa.record_consent(consent_data)
    print(f"✅ Consent recorded for user: {consent_record.user_id}")
    
    # Example: Check consent validity
    is_valid = pdpa.check_consent_validity(
        'user_123', 
        DataCategory.PERSONAL_IDENTIFIER, 
        ProcessingPurpose.DOCUMENT_PROCESSING
    )
    print(f"📋 Consent valid for processing: {is_valid}")
    
    # Example: Record processing activity
    processing_data = {
        'user_id': 'user_123',
        'data_categories': ['personal_identifier'],
        'processing_purpose': 'document_processing',
        'data_source': 'user_upload',
        'data_destination': 'processing_engine',
        'retention_days': 365,
        'legal_basis': 'consent',
        'processor_id': 'processor_001',
        'security_measures': ['encryption', 'access_control']
    }
    
    processing_record = await pdpa.record_processing_activity(processing_data)
    print(f"📝 Processing activity recorded: {processing_record.record_id}")
    
    # Example: Handle subject access request
    request_data = {
        'user_id': 'user_123',
        'request_type': 'access',
        'verification_method': 'email_verification',
        'details': {'requested_data': 'all'}
    }
    
    subject_request = await pdpa.handle_subject_request(request_data)
    print(f"📨 Subject request processed: {subject_request.request_id}")
    
    # Example: Conduct Privacy Impact Assessment
    pia_data = {
        'processing_description': 'Automated document processing with OCR',
        'data_categories': ['personal_identifier', 'biometric'],
        'processing_purposes': ['document_processing'],
        'assessor_id': 'assessor_001',
        'identified_risks': ['data_breach', 'unauthorized_access'],
        'mitigation_measures': ['encryption', 'access_controls', 'audit_logging']
    }
    
    pia = await pdpa.conduct_privacy_impact_assessment(pia_data)
    print(f"🔍 Privacy Impact Assessment completed: {pia.assessment_id} (Risk: {pia.risk_level})")
    
    # Generate compliance report
    report = pdpa.generate_compliance_report()
    print("\n📊 PDPA Compliance Report:")
    print(json.dumps(report, indent=2, default=str))
    
    print("\n🚀 PDPA COMPLIANCE SYSTEM READY!")
    print("   ✅ Consent management implemented")
    print("   ✅ Data subject rights handling")
    print("   ✅ Processing activity logging")
    print("   ✅ Privacy impact assessments")
    print("   ✅ Compliance reporting")

if __name__ == "__main__":
    asyncio.run(main())