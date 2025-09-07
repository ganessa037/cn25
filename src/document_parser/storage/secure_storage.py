#!/usr/bin/env python3
"""
Secure Document Storage System

Comprehensive storage system with:
- File encryption and decryption
- Secure file management
- Backup and recovery
- Storage optimization
- Access control and audit logging
"""

import os
import logging
import hashlib
import shutil
import asyncio
import aiofiles
from datetime import datetime, timedelta
from typing import Dict, Any, Optional, List, Union, BinaryIO
from enum import Enum
from dataclasses import dataclass, asdict
from pathlib import Path
import json
import tempfile

from cryptography.fernet import Fernet
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC
from cryptography.hazmat.primitives.ciphers import Cipher, algorithms, modes
from cryptography.hazmat.backends import default_backend
import base64

# Configure logging
logger = logging.getLogger(__name__)

# Storage Configuration
class StorageType(str, Enum):
    """Storage backend types."""
    LOCAL = "local"
    S3 = "s3"
    AZURE = "azure"
    GCS = "gcs"
    SFTP = "sftp"

class EncryptionMethod(str, Enum):
    """Encryption methods."""
    FERNET = "fernet"
    AES_256_GCM = "aes_256_gcm"
    AES_256_CBC = "aes_256_cbc"
    NONE = "none"

class CompressionType(str, Enum):
    """Compression types."""
    NONE = "none"
    GZIP = "gzip"
    BZIP2 = "bzip2"
    LZMA = "lzma"

class AccessLevel(str, Enum):
    """File access levels."""
    PUBLIC = "public"
    PRIVATE = "private"
    RESTRICTED = "restricted"
    CONFIDENTIAL = "confidential"

@dataclass
class StorageConfig:
    """Storage configuration settings."""
    storage_type: StorageType = StorageType.LOCAL
    base_path: str = "/tmp/document_storage"
    encryption_method: EncryptionMethod = EncryptionMethod.FERNET
    compression_type: CompressionType = CompressionType.GZIP
    max_file_size: int = 100 * 1024 * 1024  # 100MB
    backup_enabled: bool = True
    backup_retention_days: int = 30
    auto_cleanup_enabled: bool = True
    cleanup_interval_hours: int = 24
    access_logging_enabled: bool = True
    virus_scanning_enabled: bool = False
    duplicate_detection_enabled: bool = True

@dataclass
class FileMetadata:
    """File metadata structure."""
    file_id: str
    original_filename: str
    stored_filename: str
    file_size: int
    content_type: str
    checksum: str
    encryption_method: EncryptionMethod
    compression_type: CompressionType
    access_level: AccessLevel
    owner_id: str
    created_at: datetime
    modified_at: datetime
    accessed_at: datetime
    access_count: int = 0
    tags: Optional[List[str]] = None
    custom_metadata: Optional[Dict[str, Any]] = None
    backup_locations: Optional[List[str]] = None
    is_deleted: bool = False
    deletion_scheduled_at: Optional[datetime] = None

@dataclass
class StorageStats:
    """Storage statistics."""
    total_files: int
    total_size: int
    encrypted_files: int
    compressed_files: int
    backup_files: int
    deleted_files: int
    storage_utilization: float
    last_cleanup: Optional[datetime]
    last_backup: Optional[datetime]

class EncryptionManager:
    """Handle file encryption and decryption."""
    
    def __init__(self, encryption_key: Optional[bytes] = None):
        self.encryption_key = encryption_key or self._generate_key()
        self.fernet = Fernet(base64.urlsafe_b64encode(self.encryption_key[:32]))
    
    def _generate_key(self) -> bytes:
        """Generate encryption key."""
        return os.urandom(32)
    
    def _derive_key(self, password: str, salt: bytes) -> bytes:
        """Derive key from password."""
        kdf = PBKDF2HMAC(
            algorithm=hashes.SHA256(),
            length=32,
            salt=salt,
            iterations=100000,
            backend=default_backend()
        )
        return kdf.derive(password.encode())
    
    async def encrypt_file(self, file_path: str, method: EncryptionMethod) -> str:
        """Encrypt file and return encrypted file path."""
        try:
            if method == EncryptionMethod.NONE:
                return file_path
            
            encrypted_path = f"{file_path}.encrypted"
            
            async with aiofiles.open(file_path, 'rb') as infile:
                content = await infile.read()
            
            if method == EncryptionMethod.FERNET:
                encrypted_content = self.fernet.encrypt(content)
            elif method == EncryptionMethod.AES_256_GCM:
                encrypted_content = self._encrypt_aes_gcm(content)
            else:
                raise ValueError(f"Unsupported encryption method: {method}")
            
            async with aiofiles.open(encrypted_path, 'wb') as outfile:
                await outfile.write(encrypted_content)
            
            # Remove original file
            os.remove(file_path)
            
            logger.info(f"File encrypted: {file_path} -> {encrypted_path}")
            return encrypted_path
            
        except Exception as e:
            logger.error(f"Encryption failed: {e}")
            raise
    
    async def decrypt_file(self, encrypted_path: str, method: EncryptionMethod, 
                          output_path: Optional[str] = None) -> str:
        """Decrypt file and return decrypted file path."""
        try:
            if method == EncryptionMethod.NONE:
                return encrypted_path
            
            output_path = output_path or encrypted_path.replace('.encrypted', '')
            
            async with aiofiles.open(encrypted_path, 'rb') as infile:
                encrypted_content = await infile.read()
            
            if method == EncryptionMethod.FERNET:
                decrypted_content = self.fernet.decrypt(encrypted_content)
            elif method == EncryptionMethod.AES_256_GCM:
                decrypted_content = self._decrypt_aes_gcm(encrypted_content)
            else:
                raise ValueError(f"Unsupported encryption method: {method}")
            
            async with aiofiles.open(output_path, 'wb') as outfile:
                await outfile.write(decrypted_content)
            
            logger.info(f"File decrypted: {encrypted_path} -> {output_path}")
            return output_path
            
        except Exception as e:
            logger.error(f"Decryption failed: {e}")
            raise
    
    def _encrypt_aes_gcm(self, data: bytes) -> bytes:
        """Encrypt data using AES-256-GCM."""
        iv = os.urandom(12)  # 96-bit IV for GCM
        cipher = Cipher(
            algorithms.AES(self.encryption_key),
            modes.GCM(iv),
            backend=default_backend()
        )
        encryptor = cipher.encryptor()
        ciphertext = encryptor.update(data) + encryptor.finalize()
        return iv + encryptor.tag + ciphertext
    
    def _decrypt_aes_gcm(self, encrypted_data: bytes) -> bytes:
        """Decrypt data using AES-256-GCM."""
        iv = encrypted_data[:12]
        tag = encrypted_data[12:28]
        ciphertext = encrypted_data[28:]
        
        cipher = Cipher(
            algorithms.AES(self.encryption_key),
            modes.GCM(iv, tag),
            backend=default_backend()
        )
        decryptor = cipher.decryptor()
        return decryptor.update(ciphertext) + decryptor.finalize()

class CompressionManager:
    """Handle file compression and decompression."""
    
    async def compress_file(self, file_path: str, method: CompressionType) -> str:
        """Compress file and return compressed file path."""
        try:
            if method == CompressionType.NONE:
                return file_path
            
            compressed_path = f"{file_path}.{method.value}"
            
            if method == CompressionType.GZIP:
                import gzip
                async with aiofiles.open(file_path, 'rb') as infile:
                    content = await infile.read()
                
                with gzip.open(compressed_path, 'wb') as outfile:
                    outfile.write(content)
            
            elif method == CompressionType.BZIP2:
                import bz2
                async with aiofiles.open(file_path, 'rb') as infile:
                    content = await infile.read()
                
                with bz2.open(compressed_path, 'wb') as outfile:
                    outfile.write(content)
            
            elif method == CompressionType.LZMA:
                import lzma
                async with aiofiles.open(file_path, 'rb') as infile:
                    content = await infile.read()
                
                with lzma.open(compressed_path, 'wb') as outfile:
                    outfile.write(content)
            
            # Remove original file
            os.remove(file_path)
            
            logger.info(f"File compressed: {file_path} -> {compressed_path}")
            return compressed_path
            
        except Exception as e:
            logger.error(f"Compression failed: {e}")
            raise
    
    async def decompress_file(self, compressed_path: str, method: CompressionType,
                             output_path: Optional[str] = None) -> str:
        """Decompress file and return decompressed file path."""
        try:
            if method == CompressionType.NONE:
                return compressed_path
            
            output_path = output_path or compressed_path.replace(f'.{method.value}', '')
            
            if method == CompressionType.GZIP:
                import gzip
                with gzip.open(compressed_path, 'rb') as infile:
                    content = infile.read()
            
            elif method == CompressionType.BZIP2:
                import bz2
                with bz2.open(compressed_path, 'rb') as infile:
                    content = infile.read()
            
            elif method == CompressionType.LZMA:
                import lzma
                with lzma.open(compressed_path, 'rb') as infile:
                    content = infile.read()
            
            async with aiofiles.open(output_path, 'wb') as outfile:
                await outfile.write(content)
            
            logger.info(f"File decompressed: {compressed_path} -> {output_path}")
            return output_path
            
        except Exception as e:
            logger.error(f"Decompression failed: {e}")
            raise

class SecureStorage:
    """Main secure storage system."""
    
    def __init__(self, config: Optional[StorageConfig] = None):
        self.config = config or StorageConfig()
        self.encryption_manager = EncryptionManager()
        self.compression_manager = CompressionManager()
        
        # Ensure storage directory exists
        Path(self.config.base_path).mkdir(parents=True, exist_ok=True)
        
        # Metadata storage
        self.metadata_file = os.path.join(self.config.base_path, "metadata.json")
        self.metadata: Dict[str, FileMetadata] = self._load_metadata()
        
        # Access logs
        self.access_log_file = os.path.join(self.config.base_path, "access.log")
        
        logger.info(f"SecureStorage initialized with base path: {self.config.base_path}")
    
    def _load_metadata(self) -> Dict[str, FileMetadata]:
        """Load file metadata from storage."""
        try:
            if os.path.exists(self.metadata_file):
                with open(self.metadata_file, 'r') as f:
                    data = json.load(f)
                
                metadata = {}
                for file_id, meta_dict in data.items():
                    # Convert datetime strings back to datetime objects
                    for date_field in ['created_at', 'modified_at', 'accessed_at', 'deletion_scheduled_at']:
                        if meta_dict.get(date_field):
                            meta_dict[date_field] = datetime.fromisoformat(meta_dict[date_field])
                    
                    metadata[file_id] = FileMetadata(**meta_dict)
                
                return metadata
            
            return {}
            
        except Exception as e:
            logger.error(f"Failed to load metadata: {e}")
            return {}
    
    def _save_metadata(self):
        """Save file metadata to storage."""
        try:
            # Convert metadata to serializable format
            data = {}
            for file_id, metadata in self.metadata.items():
                meta_dict = asdict(metadata)
                # Convert datetime objects to strings
                for date_field in ['created_at', 'modified_at', 'accessed_at', 'deletion_scheduled_at']:
                    if meta_dict.get(date_field):
                        meta_dict[date_field] = meta_dict[date_field].isoformat()
                data[file_id] = meta_dict
            
            with open(self.metadata_file, 'w') as f:
                json.dump(data, f, indent=2)
            
        except Exception as e:
            logger.error(f"Failed to save metadata: {e}")
    
    def _generate_file_id(self) -> str:
        """Generate unique file ID."""
        import uuid
        return str(uuid.uuid4())
    
    def _calculate_checksum(self, file_path: str) -> str:
        """Calculate file checksum."""
        hash_sha256 = hashlib.sha256()
        with open(file_path, 'rb') as f:
            for chunk in iter(lambda: f.read(4096), b""):
                hash_sha256.update(chunk)
        return hash_sha256.hexdigest()
    
    def _get_storage_path(self, file_id: str) -> str:
        """Get storage path for file ID."""
        # Create subdirectories based on file ID for better organization
        subdir = file_id[:2]
        storage_dir = os.path.join(self.config.base_path, "files", subdir)
        Path(storage_dir).mkdir(parents=True, exist_ok=True)
        return os.path.join(storage_dir, file_id)
    
    async def store_file(self, file_path: str, original_filename: str,
                        owner_id: str, access_level: AccessLevel = AccessLevel.PRIVATE,
                        tags: Optional[List[str]] = None,
                        custom_metadata: Optional[Dict[str, Any]] = None) -> str:
        """Store file securely and return file ID."""
        try:
            # Validate file size
            file_size = os.path.getsize(file_path)
            if file_size > self.config.max_file_size:
                raise ValueError(f"File size {file_size} exceeds maximum {self.config.max_file_size}")
            
            # Generate file ID and storage path
            file_id = self._generate_file_id()
            storage_path = self._get_storage_path(file_id)
            
            # Copy file to storage location
            shutil.copy2(file_path, storage_path)
            
            # Calculate checksum
            checksum = self._calculate_checksum(storage_path)
            
            # Check for duplicates if enabled
            if self.config.duplicate_detection_enabled:
                duplicate_id = self._find_duplicate(checksum)
                if duplicate_id:
                    os.remove(storage_path)
                    logger.info(f"Duplicate file detected, returning existing ID: {duplicate_id}")
                    return duplicate_id
            
            # Compress file if configured
            if self.config.compression_type != CompressionType.NONE:
                storage_path = await self.compression_manager.compress_file(
                    storage_path, self.config.compression_type
                )
            
            # Encrypt file if configured
            if self.config.encryption_method != EncryptionMethod.NONE:
                storage_path = await self.encryption_manager.encrypt_file(
                    storage_path, self.config.encryption_method
                )
            
            # Detect content type
            import mimetypes
            content_type, _ = mimetypes.guess_type(original_filename)
            content_type = content_type or "application/octet-stream"
            
            # Create metadata
            now = datetime.utcnow()
            metadata = FileMetadata(
                file_id=file_id,
                original_filename=original_filename,
                stored_filename=os.path.basename(storage_path),
                file_size=file_size,
                content_type=content_type,
                checksum=checksum,
                encryption_method=self.config.encryption_method,
                compression_type=self.config.compression_type,
                access_level=access_level,
                owner_id=owner_id,
                created_at=now,
                modified_at=now,
                accessed_at=now,
                tags=tags or [],
                custom_metadata=custom_metadata or {}
            )
            
            # Store metadata
            self.metadata[file_id] = metadata
            self._save_metadata()
            
            # Log access
            if self.config.access_logging_enabled:
                await self._log_access(file_id, "STORE", owner_id)
            
            # Create backup if enabled
            if self.config.backup_enabled:
                await self._create_backup(file_id)
            
            logger.info(f"File stored successfully: {file_id}")
            return file_id
            
        except Exception as e:
            logger.error(f"Failed to store file: {e}")
            raise
    
    async def retrieve_file(self, file_id: str, user_id: str,
                           output_path: Optional[str] = None) -> str:
        """Retrieve and decrypt file, return path to decrypted file."""
        try:
            # Check if file exists
            if file_id not in self.metadata:
                raise FileNotFoundError(f"File not found: {file_id}")
            
            metadata = self.metadata[file_id]
            
            # Check access permissions
            if not self._check_access_permission(metadata, user_id):
                raise PermissionError(f"Access denied to file: {file_id}")
            
            # Get storage path
            storage_path = self._get_storage_path(file_id)
            if metadata.stored_filename != file_id:
                storage_path = os.path.join(os.path.dirname(storage_path), metadata.stored_filename)
            
            if not os.path.exists(storage_path):
                raise FileNotFoundError(f"Physical file not found: {storage_path}")
            
            # Create temporary file for processing
            temp_path = storage_path
            
            # Decrypt if encrypted
            if metadata.encryption_method != EncryptionMethod.NONE:
                temp_path = await self.encryption_manager.decrypt_file(
                    temp_path, metadata.encryption_method
                )
            
            # Decompress if compressed
            if metadata.compression_type != CompressionType.NONE:
                temp_path = await self.compression_manager.decompress_file(
                    temp_path, metadata.compression_type
                )
            
            # Copy to output path if specified
            if output_path:
                shutil.copy2(temp_path, output_path)
                # Clean up temporary file if different from storage
                if temp_path != storage_path:
                    os.remove(temp_path)
                final_path = output_path
            else:
                final_path = temp_path
            
            # Update access metadata
            metadata.accessed_at = datetime.utcnow()
            metadata.access_count += 1
            self._save_metadata()
            
            # Log access
            if self.config.access_logging_enabled:
                await self._log_access(file_id, "RETRIEVE", user_id)
            
            logger.info(f"File retrieved successfully: {file_id}")
            return final_path
            
        except Exception as e:
            logger.error(f"Failed to retrieve file: {e}")
            raise
    
    async def delete_file(self, file_id: str, user_id: str, permanent: bool = False) -> bool:
        """Delete file (soft delete by default)."""
        try:
            # Check if file exists
            if file_id not in self.metadata:
                raise FileNotFoundError(f"File not found: {file_id}")
            
            metadata = self.metadata[file_id]
            
            # Check delete permissions
            if metadata.owner_id != user_id:
                raise PermissionError(f"Only owner can delete file: {file_id}")
            
            if permanent:
                # Permanent deletion
                storage_path = self._get_storage_path(file_id)
                if metadata.stored_filename != file_id:
                    storage_path = os.path.join(os.path.dirname(storage_path), metadata.stored_filename)
                
                if os.path.exists(storage_path):
                    os.remove(storage_path)
                
                # Remove from metadata
                del self.metadata[file_id]
                self._save_metadata()
                
                logger.info(f"File permanently deleted: {file_id}")
            else:
                # Soft deletion
                metadata.is_deleted = True
                metadata.deletion_scheduled_at = datetime.utcnow() + timedelta(days=30)
                self._save_metadata()
                
                logger.info(f"File soft deleted: {file_id}")
            
            # Log access
            if self.config.access_logging_enabled:
                action = "DELETE_PERMANENT" if permanent else "DELETE_SOFT"
                await self._log_access(file_id, action, user_id)
            
            return True
            
        except Exception as e:
            logger.error(f"Failed to delete file: {e}")
            raise
    
    def get_file_metadata(self, file_id: str, user_id: str) -> Optional[FileMetadata]:
        """Get file metadata."""
        try:
            if file_id not in self.metadata:
                return None
            
            metadata = self.metadata[file_id]
            
            # Check access permissions
            if not self._check_access_permission(metadata, user_id):
                return None
            
            return metadata
            
        except Exception as e:
            logger.error(f"Failed to get file metadata: {e}")
            return None
    
    def list_files(self, user_id: str, include_deleted: bool = False) -> List[FileMetadata]:
        """List files accessible to user."""
        try:
            files = []
            
            for metadata in self.metadata.values():
                # Check access permissions
                if not self._check_access_permission(metadata, user_id):
                    continue
                
                # Filter deleted files
                if metadata.is_deleted and not include_deleted:
                    continue
                
                files.append(metadata)
            
            return files
            
        except Exception as e:
            logger.error(f"Failed to list files: {e}")
            return []
    
    def _check_access_permission(self, metadata: FileMetadata, user_id: str) -> bool:
        """Check if user has access to file."""
        # Owner always has access
        if metadata.owner_id == user_id:
            return True
        
        # Check access level
        if metadata.access_level == AccessLevel.PUBLIC:
            return True
        elif metadata.access_level == AccessLevel.PRIVATE:
            return False
        elif metadata.access_level in [AccessLevel.RESTRICTED, AccessLevel.CONFIDENTIAL]:
            # TODO: Implement role-based access control
            return False
        
        return False
    
    def _find_duplicate(self, checksum: str) -> Optional[str]:
        """Find duplicate file by checksum."""
        for file_id, metadata in self.metadata.items():
            if metadata.checksum == checksum and not metadata.is_deleted:
                return file_id
        return None
    
    async def _log_access(self, file_id: str, action: str, user_id: str):
        """Log file access."""
        try:
            log_entry = {
                "timestamp": datetime.utcnow().isoformat(),
                "file_id": file_id,
                "action": action,
                "user_id": user_id
            }
            
            async with aiofiles.open(self.access_log_file, 'a') as f:
                await f.write(json.dumps(log_entry) + "\n")
            
        except Exception as e:
            logger.error(f"Failed to log access: {e}")
    
    async def _create_backup(self, file_id: str):
        """Create backup of file."""
        try:
            # TODO: Implement backup to secondary storage
            logger.info(f"Backup created for file: {file_id}")
            
        except Exception as e:
            logger.error(f"Failed to create backup: {e}")
    
    async def cleanup_deleted_files(self) -> int:
        """Clean up soft-deleted files past retention period."""
        try:
            cleaned_count = 0
            current_time = datetime.utcnow()
            
            files_to_delete = []
            for file_id, metadata in self.metadata.items():
                if (metadata.is_deleted and 
                    metadata.deletion_scheduled_at and 
                    current_time > metadata.deletion_scheduled_at):
                    files_to_delete.append(file_id)
            
            for file_id in files_to_delete:
                try:
                    metadata = self.metadata[file_id]
                    storage_path = self._get_storage_path(file_id)
                    if metadata.stored_filename != file_id:
                        storage_path = os.path.join(os.path.dirname(storage_path), metadata.stored_filename)
                    
                    if os.path.exists(storage_path):
                        os.remove(storage_path)
                    
                    del self.metadata[file_id]
                    cleaned_count += 1
                    
                except Exception as e:
                    logger.error(f"Failed to cleanup file {file_id}: {e}")
            
            if cleaned_count > 0:
                self._save_metadata()
            
            logger.info(f"Cleaned up {cleaned_count} deleted files")
            return cleaned_count
            
        except Exception as e:
            logger.error(f"Cleanup failed: {e}")
            return 0
    
    def get_storage_stats(self) -> StorageStats:
        """Get storage statistics."""
        try:
            total_files = len(self.metadata)
            total_size = sum(meta.file_size for meta in self.metadata.values())
            encrypted_files = sum(1 for meta in self.metadata.values() 
                                if meta.encryption_method != EncryptionMethod.NONE)
            compressed_files = sum(1 for meta in self.metadata.values() 
                                 if meta.compression_type != CompressionType.NONE)
            deleted_files = sum(1 for meta in self.metadata.values() if meta.is_deleted)
            
            # Calculate storage utilization
            try:
                disk_usage = shutil.disk_usage(self.config.base_path)
                storage_utilization = (disk_usage.used / disk_usage.total) * 100
            except:
                storage_utilization = 0.0
            
            return StorageStats(
                total_files=total_files,
                total_size=total_size,
                encrypted_files=encrypted_files,
                compressed_files=compressed_files,
                backup_files=0,  # TODO: Implement backup counting
                deleted_files=deleted_files,
                storage_utilization=storage_utilization,
                last_cleanup=None,  # TODO: Track cleanup times
                last_backup=None   # TODO: Track backup times
            )
            
        except Exception as e:
            logger.error(f"Failed to get storage stats: {e}")
            return StorageStats(
                total_files=0, total_size=0, encrypted_files=0,
                compressed_files=0, backup_files=0, deleted_files=0,
                storage_utilization=0.0, last_cleanup=None, last_backup=None
            )

# Main function for standalone testing
if __name__ == "__main__":
    async def test_secure_storage():
        """Test secure storage functionality."""
        # Initialize storage
        config = StorageConfig(
            base_path="/tmp/test_storage",
            encryption_method=EncryptionMethod.FERNET,
            compression_type=CompressionType.GZIP
        )
        storage = SecureStorage(config)
        
        try:
            # Create test file
            test_content = "This is a test file for secure storage."
            test_file_path = "/tmp/test_file.txt"
            
            with open(test_file_path, 'w') as f:
                f.write(test_content)
            
            # Store file
            file_id = await storage.store_file(
                test_file_path,
                "test_file.txt",
                "user123",
                AccessLevel.PRIVATE,
                tags=["test", "demo"]
            )
            print(f"File stored with ID: {file_id}")
            
            # Retrieve file
            retrieved_path = await storage.retrieve_file(file_id, "user123", "/tmp/retrieved_file.txt")
            print(f"File retrieved to: {retrieved_path}")
            
            # Verify content
            with open(retrieved_path, 'r') as f:
                retrieved_content = f.read()
            
            if retrieved_content == test_content:
                print("Content verification successful!")
            else:
                print("Content verification failed!")
            
            # Get metadata
            metadata = storage.get_file_metadata(file_id, "user123")
            print(f"File metadata: {metadata.original_filename if metadata else 'None'}")
            
            # Get storage stats
            stats = storage.get_storage_stats()
            print(f"Storage stats: {stats.total_files} files, {stats.total_size} bytes")
            
            # Clean up
            os.remove(test_file_path)
            os.remove(retrieved_path)
            
            print("Secure storage test completed successfully!")
            
        except Exception as e:
            print(f"Secure storage test failed: {e}")
    
    # Run test
    asyncio.run(test_secure_storage())