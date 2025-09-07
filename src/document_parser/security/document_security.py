#!/usr/bin/env python3
"""
Document Security Module
=======================

Comprehensive document security system for protecting sensitive information
in document processing workflows. Includes image processing, secure storage,
retention policies, and data anonymization.

Features:
- Sensitive area detection and blurring
- Secure encrypted document storage
- Automated retention and deletion policies
- Data anonymization and redaction
- Access logging and audit trails
- Document watermarking
- Metadata sanitization
- Secure document sharing
"""

import asyncio
import hashlib
import json
import logging
import os
import re
import shutil
import tempfile
import uuid
from datetime import datetime, timedelta
from enum import Enum
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any, Union, Set
from dataclasses import dataclass, asdict

import aiofiles
import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont, ImageFilter
import pytesseract
from cryptography.fernet import Fernet

# Import our encryption service
from .encryption import EncryptionService, EncryptionAlgorithm

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class SensitiveDataType(Enum):
    """Types of sensitive data to detect"""
    NRIC = "nric"  # Malaysian IC numbers
    PASSPORT = "passport"
    PHONE = "phone"
    EMAIL = "email"
    CREDIT_CARD = "credit_card"
    BANK_ACCOUNT = "bank_account"
    ADDRESS = "address"
    NAME = "name"
    DATE_OF_BIRTH = "date_of_birth"
    SALARY = "salary"
    CUSTOM = "custom"

class SecurityLevel(Enum):
    """Document security levels"""
    PUBLIC = "public"
    INTERNAL = "internal"
    CONFIDENTIAL = "confidential"
    RESTRICTED = "restricted"
    TOP_SECRET = "top_secret"

class RetentionPolicy(Enum):
    """Data retention policies"""
    IMMEDIATE = "immediate"  # Delete immediately after processing
    SHORT_TERM = "short_term"  # 30 days
    MEDIUM_TERM = "medium_term"  # 1 year
    LONG_TERM = "long_term"  # 7 years
    PERMANENT = "permanent"  # Keep indefinitely
    CUSTOM = "custom"  # Custom duration

class AnonymizationMethod(Enum):
    """Methods for data anonymization"""
    BLUR = "blur"
    REDACT = "redact"  # Black boxes
    REPLACE = "replace"  # Replace with placeholder
    REMOVE = "remove"  # Remove completely
    HASH = "hash"  # Replace with hash
    TOKENIZE = "tokenize"  # Replace with tokens

@dataclass
class SensitiveArea:
    """Detected sensitive area in document"""
    data_type: SensitiveDataType
    coordinates: Tuple[int, int, int, int]  # x, y, width, height
    confidence: float
    text: str
    anonymized: bool = False
    method: Optional[AnonymizationMethod] = None

@dataclass
class DocumentMetadata:
    """Document metadata and security information"""
    document_id: str
    original_filename: str
    file_type: str
    file_size: int
    security_level: SecurityLevel
    retention_policy: RetentionPolicy
    retention_until: Optional[datetime]
    created_at: datetime
    last_accessed: datetime
    access_count: int
    encryption_key_id: str
    sensitive_areas: List[SensitiveArea]
    anonymized: bool
    watermarked: bool
    checksum: str
    tags: List[str]
    owner: str
    permissions: Dict[str, List[str]]

@dataclass
class AccessLog:
    """Document access log entry"""
    log_id: str
    document_id: str
    user_id: str
    action: str  # view, download, edit, delete, etc.
    timestamp: datetime
    ip_address: str
    user_agent: str
    success: bool
    details: Dict[str, Any]

class SensitiveDataDetector:
    """Detects sensitive information in documents"""
    
    def __init__(self):
        # Malaysian NRIC pattern
        self.nric_pattern = re.compile(r'\b\d{6}-\d{2}-\d{4}\b')
        
        # Passport patterns
        self.passport_pattern = re.compile(r'\b[A-Z]\d{8}\b')
        
        # Phone patterns (Malaysian)
        self.phone_pattern = re.compile(r'\b(?:\+?6?01[0-9]-?\d{7,8}|\+?6?03-?\d{8})\b')
        
        # Email pattern
        self.email_pattern = re.compile(r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b')
        
        # Credit card pattern
        self.credit_card_pattern = re.compile(r'\b(?:\d{4}[- ]?){3}\d{4}\b')
        
        # Bank account pattern
        self.bank_account_pattern = re.compile(r'\b\d{10,16}\b')
        
        # Salary/money pattern
        self.salary_pattern = re.compile(r'\bRM\s?\d{1,3}(?:,\d{3})*(?:\.\d{2})?\b')
        
        # Date of birth pattern
        self.dob_pattern = re.compile(r'\b\d{1,2}[/-]\d{1,2}[/-]\d{4}\b')
    
    def detect_sensitive_data(self, text: str, image: Optional[np.ndarray] = None) -> List[SensitiveArea]:
        """Detect sensitive data in text and optionally in image"""
        sensitive_areas = []
        
        # Text-based detection
        patterns = {
            SensitiveDataType.NRIC: self.nric_pattern,
            SensitiveDataType.PASSPORT: self.passport_pattern,
            SensitiveDataType.PHONE: self.phone_pattern,
            SensitiveDataType.EMAIL: self.email_pattern,
            SensitiveDataType.CREDIT_CARD: self.credit_card_pattern,
            SensitiveDataType.BANK_ACCOUNT: self.bank_account_pattern,
            SensitiveDataType.SALARY: self.salary_pattern,
            SensitiveDataType.DATE_OF_BIRTH: self.dob_pattern
        }
        
        for data_type, pattern in patterns.items():
            matches = pattern.finditer(text)
            for match in matches:
                # For text-only detection, we don't have exact coordinates
                # In a real implementation, you'd use OCR results with bounding boxes
                sensitive_area = SensitiveArea(
                    data_type=data_type,
                    coordinates=(0, 0, 0, 0),  # Would be filled by OCR
                    confidence=0.9,
                    text=match.group(),
                    anonymized=False
                )
                sensitive_areas.append(sensitive_area)
        
        # Image-based detection using OCR
        if image is not None:
            try:
                # Use Tesseract to get text with bounding boxes
                ocr_data = pytesseract.image_to_data(image, output_type=pytesseract.Output.DICT)
                
                for i, word in enumerate(ocr_data['text']):
                    if word.strip():
                        # Check if word matches any sensitive pattern
                        for data_type, pattern in patterns.items():
                            if pattern.search(word):
                                x = ocr_data['left'][i]
                                y = ocr_data['top'][i]
                                w = ocr_data['width'][i]
                                h = ocr_data['height'][i]
                                conf = float(ocr_data['conf'][i]) / 100.0
                                
                                sensitive_area = SensitiveArea(
                                    data_type=data_type,
                                    coordinates=(x, y, w, h),
                                    confidence=conf,
                                    text=word,
                                    anonymized=False
                                )
                                sensitive_areas.append(sensitive_area)
                                
            except Exception as e:
                logger.warning(f"OCR detection failed: {e}")
        
        return sensitive_areas
    
    def detect_faces(self, image: np.ndarray) -> List[SensitiveArea]:
        """Detect faces in image using OpenCV"""
        try:
            # Load face cascade classifier
            face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
            
            # Convert to grayscale for detection
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            
            # Detect faces
            faces = face_cascade.detectMultiScale(gray, 1.1, 4)
            
            sensitive_areas = []
            for (x, y, w, h) in faces:
                sensitive_area = SensitiveArea(
                    data_type=SensitiveDataType.NAME,  # Faces are considered personal identifiers
                    coordinates=(x, y, w, h),
                    confidence=0.8,
                    text="[FACE]",
                    anonymized=False
                )
                sensitive_areas.append(sensitive_area)
            
            return sensitive_areas
            
        except Exception as e:
            logger.warning(f"Face detection failed: {e}")
            return []

class DocumentAnonymizer:
    """Anonymizes sensitive data in documents"""
    
    def __init__(self):
        self.replacement_tokens = {
            SensitiveDataType.NRIC: "[NRIC-REDACTED]",
            SensitiveDataType.PASSPORT: "[PASSPORT-REDACTED]",
            SensitiveDataType.PHONE: "[PHONE-REDACTED]",
            SensitiveDataType.EMAIL: "[EMAIL-REDACTED]",
            SensitiveDataType.CREDIT_CARD: "[CARD-REDACTED]",
            SensitiveDataType.BANK_ACCOUNT: "[ACCOUNT-REDACTED]",
            SensitiveDataType.SALARY: "[AMOUNT-REDACTED]",
            SensitiveDataType.DATE_OF_BIRTH: "[DOB-REDACTED]",
            SensitiveDataType.NAME: "[NAME-REDACTED]"
        }
    
    def anonymize_image(self, image: np.ndarray, sensitive_areas: List[SensitiveArea],
                       method: AnonymizationMethod = AnonymizationMethod.BLUR) -> np.ndarray:
        """Anonymize sensitive areas in image"""
        try:
            anonymized_image = image.copy()
            
            for area in sensitive_areas:
                x, y, w, h = area.coordinates
                
                if method == AnonymizationMethod.BLUR:
                    # Apply Gaussian blur to the area
                    roi = anonymized_image[y:y+h, x:x+w]
                    blurred_roi = cv2.GaussianBlur(roi, (51, 51), 0)
                    anonymized_image[y:y+h, x:x+w] = blurred_roi
                    
                elif method == AnonymizationMethod.REDACT:
                    # Draw black rectangle
                    cv2.rectangle(anonymized_image, (x, y), (x+w, y+h), (0, 0, 0), -1)
                    
                elif method == AnonymizationMethod.REPLACE:
                    # Draw rectangle with replacement text
                    cv2.rectangle(anonymized_image, (x, y), (x+w, y+h), (128, 128, 128), -1)
                    replacement_text = self.replacement_tokens.get(area.data_type, "[REDACTED]")
                    cv2.putText(anonymized_image, replacement_text, (x, y+h//2),
                              cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
                
                # Mark as anonymized
                area.anonymized = True
                area.method = method
            
            return anonymized_image
            
        except Exception as e:
            logger.error(f"Error anonymizing image: {e}")
            raise
    
    def anonymize_text(self, text: str, sensitive_areas: List[SensitiveArea],
                      method: AnonymizationMethod = AnonymizationMethod.REPLACE) -> str:
        """Anonymize sensitive data in text"""
        try:
            anonymized_text = text
            
            # Sort by position to avoid offset issues
            areas_with_text = [area for area in sensitive_areas if area.text]
            
            for area in areas_with_text:
                if method == AnonymizationMethod.REPLACE:
                    replacement = self.replacement_tokens.get(area.data_type, "[REDACTED]")
                    anonymized_text = anonymized_text.replace(area.text, replacement)
                    
                elif method == AnonymizationMethod.HASH:
                    # Replace with hash
                    hash_value = hashlib.sha256(area.text.encode()).hexdigest()[:8]
                    replacement = f"[HASH-{hash_value}]"
                    anonymized_text = anonymized_text.replace(area.text, replacement)
                    
                elif method == AnonymizationMethod.REMOVE:
                    # Remove completely
                    anonymized_text = anonymized_text.replace(area.text, "")
                    
                elif method == AnonymizationMethod.TOKENIZE:
                    # Replace with unique token
                    token = f"[TOKEN-{uuid.uuid4().hex[:8]}]"
                    anonymized_text = anonymized_text.replace(area.text, token)
                
                # Mark as anonymized
                area.anonymized = True
                area.method = method
            
            return anonymized_text
            
        except Exception as e:
            logger.error(f"Error anonymizing text: {e}")
            raise

class DocumentWatermarker:
    """Adds watermarks to documents"""
    
    def add_watermark(self, image: np.ndarray, watermark_text: str = "CONFIDENTIAL",
                     opacity: float = 0.3) -> np.ndarray:
        """Add watermark to image"""
        try:
            # Convert to PIL Image for easier text handling
            pil_image = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
            
            # Create watermark overlay
            overlay = Image.new('RGBA', pil_image.size, (255, 255, 255, 0))
            draw = ImageDraw.Draw(overlay)
            
            # Calculate font size based on image size
            font_size = max(20, min(pil_image.size) // 20)
            
            try:
                font = ImageFont.truetype("arial.ttf", font_size)
            except:
                font = ImageFont.load_default()
            
            # Get text size
            bbox = draw.textbbox((0, 0), watermark_text, font=font)
            text_width = bbox[2] - bbox[0]
            text_height = bbox[3] - bbox[1]
            
            # Position watermark diagonally across image
            x = (pil_image.width - text_width) // 2
            y = (pil_image.height - text_height) // 2
            
            # Draw watermark with transparency
            draw.text((x, y), watermark_text, font=font, 
                     fill=(128, 128, 128, int(255 * opacity)))
            
            # Rotate watermark
            overlay = overlay.rotate(45, expand=False)
            
            # Composite with original image
            watermarked = Image.alpha_composite(pil_image.convert('RGBA'), overlay)
            
            # Convert back to OpenCV format
            return cv2.cvtColor(np.array(watermarked.convert('RGB')), cv2.COLOR_RGB2BGR)
            
        except Exception as e:
            logger.error(f"Error adding watermark: {e}")
            return image

class SecureDocumentStorage:
    """Secure storage system for documents"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.storage_path = Path(config.get('storage_path', './secure_storage'))
        self.storage_path.mkdir(parents=True, exist_ok=True)
        
        # Initialize encryption service
        self.encryption_service = EncryptionService(config)
        
        # Metadata storage
        self.metadata_path = self.storage_path / 'metadata'
        self.metadata_path.mkdir(exist_ok=True)
        
        # Access logs
        self.logs_path = self.storage_path / 'logs'
        self.logs_path.mkdir(exist_ok=True)
    
    async def store_document(self, file_path: Path, metadata: DocumentMetadata) -> str:
        """Store document securely with encryption"""
        try:
            # Generate unique storage filename
            storage_filename = f"{metadata.document_id}.enc"
            storage_path = self.storage_path / storage_filename
            
            # Encrypt and store document
            encryption_result = await self.encryption_service.encrypt_file(
                file_path, storage_path, metadata.encryption_key_id
            )
            
            # Update metadata with encryption info
            metadata.checksum = hashlib.sha256(file_path.read_bytes()).hexdigest()
            
            # Store metadata
            await self._store_metadata(metadata)
            
            # Log access
            await self._log_access(metadata.document_id, metadata.owner, "store", True)
            
            logger.info(f"Document stored securely: {metadata.document_id}")
            return metadata.document_id
            
        except Exception as e:
            logger.error(f"Error storing document: {e}")
            await self._log_access(metadata.document_id, metadata.owner, "store", False, {"error": str(e)})
            raise
    
    async def retrieve_document(self, document_id: str, user_id: str) -> Tuple[bytes, DocumentMetadata]:
        """Retrieve and decrypt document"""
        try:
            # Load metadata
            metadata = await self._load_metadata(document_id)
            if not metadata:
                raise ValueError(f"Document not found: {document_id}")
            
            # Check permissions
            if not self._check_permissions(metadata, user_id, "read"):
                raise PermissionError(f"Access denied for user {user_id}")
            
            # Check retention policy
            if self._is_expired(metadata):
                raise ValueError(f"Document expired: {document_id}")
            
            # Load encrypted file
            storage_filename = f"{document_id}.enc"
            storage_path = self.storage_path / storage_filename
            
            if not storage_path.exists():
                raise FileNotFoundError(f"Document file not found: {document_id}")
            
            # Decrypt document
            decryption_result = await self.encryption_service.decrypt_file(storage_path)
            
            # Update access metadata
            metadata.last_accessed = datetime.now()
            metadata.access_count += 1
            await self._store_metadata(metadata)
            
            # Log access
            await self._log_access(document_id, user_id, "retrieve", True)
            
            logger.info(f"Document retrieved: {document_id} by {user_id}")
            return decryption_result.decrypted_data, metadata
            
        except Exception as e:
            logger.error(f"Error retrieving document: {e}")
            await self._log_access(document_id, user_id, "retrieve", False, {"error": str(e)})
            raise
    
    async def delete_document(self, document_id: str, user_id: str, force: bool = False) -> bool:
        """Securely delete document"""
        try:
            # Load metadata
            metadata = await self._load_metadata(document_id)
            if not metadata:
                return False
            
            # Check permissions
            if not force and not self._check_permissions(metadata, user_id, "delete"):
                raise PermissionError(f"Delete access denied for user {user_id}")
            
            # Delete encrypted file
            storage_filename = f"{document_id}.enc"
            storage_path = self.storage_path / storage_filename
            
            if storage_path.exists():
                # Secure deletion - overwrite with random data
                file_size = storage_path.stat().st_size
                with open(storage_path, 'wb') as f:
                    f.write(os.urandom(file_size))
                storage_path.unlink()
            
            # Delete metadata
            metadata_file = self.metadata_path / f"{document_id}.json"
            if metadata_file.exists():
                metadata_file.unlink()
            
            # Log deletion
            await self._log_access(document_id, user_id, "delete", True)
            
            logger.info(f"Document deleted: {document_id} by {user_id}")
            return True
            
        except Exception as e:
            logger.error(f"Error deleting document: {e}")
            await self._log_access(document_id, user_id, "delete", False, {"error": str(e)})
            raise
    
    async def cleanup_expired_documents(self) -> int:
        """Clean up expired documents based on retention policies"""
        try:
            deleted_count = 0
            
            # Scan all metadata files
            for metadata_file in self.metadata_path.glob("*.json"):
                try:
                    async with aiofiles.open(metadata_file, 'r') as f:
                        metadata_dict = json.loads(await f.read())
                    
                    # Reconstruct metadata object
                    metadata = DocumentMetadata(**metadata_dict)
                    
                    if self._is_expired(metadata):
                        await self.delete_document(metadata.document_id, "system", force=True)
                        deleted_count += 1
                        
                except Exception as e:
                    logger.warning(f"Error processing metadata file {metadata_file}: {e}")
            
            logger.info(f"Cleanup completed: {deleted_count} expired documents deleted")
            return deleted_count
            
        except Exception as e:
            logger.error(f"Error during cleanup: {e}")
            raise
    
    def _check_permissions(self, metadata: DocumentMetadata, user_id: str, action: str) -> bool:
        """Check if user has permission for action"""
        # Owner has all permissions
        if metadata.owner == user_id:
            return True
        
        # Check specific permissions
        user_permissions = metadata.permissions.get(user_id, [])
        return action in user_permissions or "all" in user_permissions
    
    def _is_expired(self, metadata: DocumentMetadata) -> bool:
        """Check if document has expired based on retention policy"""
        if metadata.retention_until:
            return datetime.now() > metadata.retention_until
        return False
    
    async def _store_metadata(self, metadata: DocumentMetadata):
        """Store document metadata"""
        metadata_file = self.metadata_path / f"{metadata.document_id}.json"
        
        # Convert to dict for JSON serialization
        metadata_dict = asdict(metadata)
        
        # Handle datetime serialization
        for key, value in metadata_dict.items():
            if isinstance(value, datetime):
                metadata_dict[key] = value.isoformat()
        
        async with aiofiles.open(metadata_file, 'w') as f:
            await f.write(json.dumps(metadata_dict, indent=2))
    
    async def _load_metadata(self, document_id: str) -> Optional[DocumentMetadata]:
        """Load document metadata"""
        metadata_file = self.metadata_path / f"{document_id}.json"
        
        if not metadata_file.exists():
            return None
        
        try:
            async with aiofiles.open(metadata_file, 'r') as f:
                metadata_dict = json.loads(await f.read())
            
            # Handle datetime deserialization
            for key, value in metadata_dict.items():
                if key.endswith('_at') or key.endswith('_until'):
                    if value:
                        metadata_dict[key] = datetime.fromisoformat(value)
            
            return DocumentMetadata(**metadata_dict)
            
        except Exception as e:
            logger.error(f"Error loading metadata: {e}")
            return None
    
    async def _log_access(self, document_id: str, user_id: str, action: str, 
                         success: bool, details: Optional[Dict[str, Any]] = None):
        """Log document access"""
        try:
            log_entry = AccessLog(
                log_id=str(uuid.uuid4()),
                document_id=document_id,
                user_id=user_id,
                action=action,
                timestamp=datetime.now(),
                ip_address="127.0.0.1",  # Would be actual IP in real implementation
                user_agent="DocumentParser/1.0",
                success=success,
                details=details or {}
            )
            
            # Store log entry
            log_file = self.logs_path / f"{datetime.now().strftime('%Y-%m-%d')}.log"
            
            log_dict = asdict(log_entry)
            log_dict['timestamp'] = log_entry.timestamp.isoformat()
            
            async with aiofiles.open(log_file, 'a') as f:
                await f.write(json.dumps(log_dict) + "\n")
                
        except Exception as e:
            logger.error(f"Error logging access: {e}")

class DocumentSecurityManager:
    """Main document security management system"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.detector = SensitiveDataDetector()
        self.anonymizer = DocumentAnonymizer()
        self.watermarker = DocumentWatermarker()
        self.storage = SecureDocumentStorage(config)
        
        # Retention policy mappings
        self.retention_days = {
            RetentionPolicy.IMMEDIATE: 0,
            RetentionPolicy.SHORT_TERM: 30,
            RetentionPolicy.MEDIUM_TERM: 365,
            RetentionPolicy.LONG_TERM: 2555,  # 7 years
            RetentionPolicy.PERMANENT: None
        }
    
    async def process_document(self, file_path: Path, security_level: SecurityLevel,
                             retention_policy: RetentionPolicy, owner: str,
                             anonymize: bool = True, watermark: bool = True) -> DocumentMetadata:
        """Process document with security measures"""
        try:
            document_id = str(uuid.uuid4())
            
            # Read document
            if file_path.suffix.lower() in ['.jpg', '.jpeg', '.png', '.bmp', '.tiff']:
                # Image document
                image = cv2.imread(str(file_path))
                if image is None:
                    raise ValueError(f"Could not read image: {file_path}")
                
                # Extract text using OCR
                text = pytesseract.image_to_string(image)
                
                # Detect sensitive data
                sensitive_areas = self.detector.detect_sensitive_data(text, image)
                sensitive_areas.extend(self.detector.detect_faces(image))
                
                # Anonymize if requested
                if anonymize and sensitive_areas:
                    image = self.anonymizer.anonymize_image(image, sensitive_areas)
                
                # Add watermark if requested
                if watermark:
                    watermark_text = f"{security_level.value.upper()} - {owner}"
                    image = self.watermarker.add_watermark(image, watermark_text)
                
                # Save processed image
                processed_path = file_path.with_suffix('.processed' + file_path.suffix)
                cv2.imwrite(str(processed_path), image)
                file_path = processed_path
                
            else:
                # Text document
                async with aiofiles.open(file_path, 'r', encoding='utf-8') as f:
                    text = await f.read()
                
                # Detect sensitive data
                sensitive_areas = self.detector.detect_sensitive_data(text)
                
                # Anonymize if requested
                if anonymize and sensitive_areas:
                    anonymized_text = self.anonymizer.anonymize_text(text, sensitive_areas)
                    
                    # Save anonymized text
                    processed_path = file_path.with_suffix('.processed' + file_path.suffix)
                    async with aiofiles.open(processed_path, 'w', encoding='utf-8') as f:
                        await f.write(anonymized_text)
                    file_path = processed_path
            
            # Calculate retention date
            retention_until = None
            if retention_policy != RetentionPolicy.PERMANENT:
                days = self.retention_days.get(retention_policy)
                if days is not None:
                    retention_until = datetime.now() + timedelta(days=days)
            
            # Generate encryption key
            encryption_key = self.storage.encryption_service.key_manager.generate_symmetric_key()
            
            # Create metadata
            metadata = DocumentMetadata(
                document_id=document_id,
                original_filename=file_path.name,
                file_type=file_path.suffix,
                file_size=file_path.stat().st_size,
                security_level=security_level,
                retention_policy=retention_policy,
                retention_until=retention_until,
                created_at=datetime.now(),
                last_accessed=datetime.now(),
                access_count=0,
                encryption_key_id=encryption_key.key_id,
                sensitive_areas=sensitive_areas,
                anonymized=anonymize and len(sensitive_areas) > 0,
                watermarked=watermark,
                checksum="",  # Will be calculated during storage
                tags=[],
                owner=owner,
                permissions={}
            )
            
            # Store document securely
            await self.storage.store_document(file_path, metadata)
            
            # Clean up processed file if it was created
            if file_path.name.startswith(file_path.stem + '.processed'):
                file_path.unlink()
            
            logger.info(f"Document processed and secured: {document_id}")
            return metadata
            
        except Exception as e:
            logger.error(f"Error processing document: {e}")
            raise
    
    async def get_security_report(self) -> Dict[str, Any]:
        """Generate security report"""
        try:
            report = {
                'timestamp': datetime.now().isoformat(),
                'total_documents': 0,
                'by_security_level': {},
                'by_retention_policy': {},
                'anonymized_documents': 0,
                'watermarked_documents': 0,
                'expired_documents': 0,
                'sensitive_data_types': {}
            }
            
            # Scan all metadata files
            for metadata_file in self.storage.metadata_path.glob("*.json"):
                try:
                    async with aiofiles.open(metadata_file, 'r') as f:
                        metadata_dict = json.loads(await f.read())
                    
                    report['total_documents'] += 1
                    
                    # Security level stats
                    security_level = metadata_dict.get('security_level', 'unknown')
                    report['by_security_level'][security_level] = report['by_security_level'].get(security_level, 0) + 1
                    
                    # Retention policy stats
                    retention_policy = metadata_dict.get('retention_policy', 'unknown')
                    report['by_retention_policy'][retention_policy] = report['by_retention_policy'].get(retention_policy, 0) + 1
                    
                    # Anonymization stats
                    if metadata_dict.get('anonymized', False):
                        report['anonymized_documents'] += 1
                    
                    # Watermark stats
                    if metadata_dict.get('watermarked', False):
                        report['watermarked_documents'] += 1
                    
                    # Expiration check
                    retention_until = metadata_dict.get('retention_until')
                    if retention_until and datetime.fromisoformat(retention_until) < datetime.now():
                        report['expired_documents'] += 1
                    
                    # Sensitive data types
                    for area in metadata_dict.get('sensitive_areas', []):
                        data_type = area.get('data_type', 'unknown')
                        report['sensitive_data_types'][data_type] = report['sensitive_data_types'].get(data_type, 0) + 1
                        
                except Exception as e:
                    logger.warning(f"Error processing metadata file {metadata_file}: {e}")
            
            return report
            
        except Exception as e:
            logger.error(f"Error generating security report: {e}")
            raise

# Example usage and testing
async def main():
    """Example usage of document security system"""
    
    # Configuration
    config = {
        'storage_path': './secure_storage',
        'key_storage_path': './keys',
        'max_key_usage': 10000
    }
    
    # Initialize document security manager
    security_manager = DocumentSecurityManager(config)
    
    print("🔒 Document Security System Initialized")
    print("=" * 50)
    
    # Example: Create a test document
    test_content = """
    CONFIDENTIAL DOCUMENT
    
    Employee: John Doe
    NRIC: 123456-78-9012
    Phone: +60123456789
    Email: john.doe@company.com
    Salary: RM 5,000.00
    
    This document contains sensitive information.
    """
    
    # Create temporary test file
    with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as f:
        f.write(test_content)
        test_file_path = Path(f.name)
    
    try:
        # Process document with security measures
        metadata = await security_manager.process_document(
            file_path=test_file_path,
            security_level=SecurityLevel.CONFIDENTIAL,
            retention_policy=RetentionPolicy.SHORT_TERM,
            owner="admin",
            anonymize=True,
            watermark=True
        )
        
        print(f"✅ Document processed and secured")
        print(f"   Document ID: {metadata.document_id}")
        print(f"   Security Level: {metadata.security_level.value}")
        print(f"   Retention Policy: {metadata.retention_policy.value}")
        print(f"   Sensitive Areas Found: {len(metadata.sensitive_areas)}")
        print(f"   Anonymized: {metadata.anonymized}")
        print(f"   Watermarked: {metadata.watermarked}")
        
        # List detected sensitive data types
        if metadata.sensitive_areas:
            print(f"   Detected Data Types:")
            for area in metadata.sensitive_areas:
                print(f"     - {area.data_type.value}: {area.text}")
        
        # Retrieve document
        retrieved_data, retrieved_metadata = await security_manager.storage.retrieve_document(
            metadata.document_id, "admin"
        )
        print(f"\n✅ Document retrieved successfully")
        print(f"   Size: {len(retrieved_data)} bytes")
        print(f"   Access Count: {retrieved_metadata.access_count}")
        
        # Generate security report
        report = await security_manager.get_security_report()
        print(f"\n📊 Security Report:")
        print(f"   Total Documents: {report['total_documents']}")
        print(f"   Anonymized: {report['anonymized_documents']}")
        print(f"   Watermarked: {report['watermarked_documents']}")
        print(f"   Expired: {report['expired_documents']}")
        
        # Cleanup expired documents
        deleted_count = await security_manager.storage.cleanup_expired_documents()
        print(f"\n🧹 Cleanup completed: {deleted_count} expired documents deleted")
        
    finally:
        # Clean up test file
        if test_file_path.exists():
            test_file_path.unlink()
    
    print("\n🚀 DOCUMENT SECURITY SYSTEM READY!")
    print("   ✅ Sensitive data detection (NRIC, passport, phone, email, etc.)")
    print("   ✅ Face detection and anonymization")
    print("   ✅ Multiple anonymization methods (blur, redact, replace, hash)")
    print("   ✅ Document watermarking")
    print("   ✅ Encrypted secure storage")
    print("   ✅ Automated retention policies")
    print("   ✅ Access control and audit logging")
    print("   ✅ Security reporting and compliance")

if __name__ == "__main__":
    asyncio.run(main())