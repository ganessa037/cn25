#!/usr/bin/env python3
"""
Document Data Models

Defines data structures for document processing, extraction results,
and validation outcomes in the document parser system.
"""

import logging
from datetime import datetime
from enum import Enum
from typing import Dict, List, Optional, Any, Union
from dataclasses import dataclass, field
from pathlib import Path
import json

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class DocumentType(Enum):
    """Supported document types."""
    MYKAD = "mykad"
    SPK = "spk"
    PASSPORT = "passport"
    DRIVING_LICENSE = "driving_license"
    BIRTH_CERTIFICATE = "birth_certificate"
    UNKNOWN = "unknown"
    
    @classmethod
    def from_string(cls, value: str) -> 'DocumentType':
        """Create DocumentType from string value."""
        try:
            return cls(value.lower())
        except ValueError:
            return cls.UNKNOWN

class ProcessingStatus(Enum):
    """Document processing status."""
    PENDING = "pending"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"
    VALIDATED = "validated"
    REJECTED = "rejected"

class ConfidenceLevel(Enum):
    """Confidence levels for extraction results."""
    VERY_LOW = "very_low"  # 0.0 - 0.3
    LOW = "low"           # 0.3 - 0.5
    MEDIUM = "medium"     # 0.5 - 0.7
    HIGH = "high"         # 0.7 - 0.9
    VERY_HIGH = "very_high"  # 0.9 - 1.0
    
    @classmethod
    def from_score(cls, score: float) -> 'ConfidenceLevel':
        """Get confidence level from numeric score."""
        if score < 0.3:
            return cls.VERY_LOW
        elif score < 0.5:
            return cls.LOW
        elif score < 0.7:
            return cls.MEDIUM
        elif score < 0.9:
            return cls.HIGH
        else:
            return cls.VERY_HIGH

@dataclass
class BoundingBox:
    """Bounding box coordinates for text regions."""
    x: float
    y: float
    width: float
    height: float
    
    @property
    def x2(self) -> float:
        """Right edge x-coordinate."""
        return self.x + self.width
    
    @property
    def y2(self) -> float:
        """Bottom edge y-coordinate."""
        return self.y + self.height
    
    @property
    def center_x(self) -> float:
        """Center x-coordinate."""
        return self.x + self.width / 2
    
    @property
    def center_y(self) -> float:
        """Center y-coordinate."""
        return self.y + self.height / 2
    
    @property
    def area(self) -> float:
        """Bounding box area."""
        return self.width * self.height
    
    def overlaps_with(self, other: 'BoundingBox') -> bool:
        """Check if this bounding box overlaps with another."""
        return not (self.x2 < other.x or other.x2 < self.x or 
                   self.y2 < other.y or other.y2 < self.y)
    
    def intersection_area(self, other: 'BoundingBox') -> float:
        """Calculate intersection area with another bounding box."""
        if not self.overlaps_with(other):
            return 0.0
        
        x_overlap = min(self.x2, other.x2) - max(self.x, other.x)
        y_overlap = min(self.y2, other.y2) - max(self.y, other.y)
        
        return x_overlap * y_overlap
    
    def to_dict(self) -> Dict:
        """Convert to dictionary."""
        return {
            "x": self.x,
            "y": self.y,
            "width": self.width,
            "height": self.height
        }
    
    @classmethod
    def from_dict(cls, data: Dict) -> 'BoundingBox':
        """Create from dictionary."""
        return cls(
            x=data["x"],
            y=data["y"],
            width=data["width"],
            height=data["height"]
        )

@dataclass
class ExtractedField:
    """Represents an extracted field from a document."""
    name: str
    value: str
    confidence: float
    bounding_box: Optional[BoundingBox] = None
    extraction_method: str = "unknown"
    is_validated: bool = False
    validation_errors: List[str] = field(default_factory=list)
    corrected_value: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    @property
    def confidence_level(self) -> ConfidenceLevel:
        """Get confidence level enum."""
        return ConfidenceLevel.from_score(self.confidence)
    
    @property
    def is_valid(self) -> bool:
        """Check if field is valid (no validation errors)."""
        return len(self.validation_errors) == 0
    
    @property
    def final_value(self) -> str:
        """Get final value (corrected if available, otherwise original)."""
        return self.corrected_value if self.corrected_value else self.value
    
    def add_validation_error(self, error: str):
        """Add a validation error."""
        if error not in self.validation_errors:
            self.validation_errors.append(error)
    
    def set_corrected_value(self, value: str):
        """Set corrected value and mark as validated."""
        self.corrected_value = value
        self.is_validated = True
    
    def to_dict(self) -> Dict:
        """Convert to dictionary."""
        return {
            "name": self.name,
            "value": self.value,
            "confidence": self.confidence,
            "confidence_level": self.confidence_level.value,
            "bounding_box": self.bounding_box.to_dict() if self.bounding_box else None,
            "extraction_method": self.extraction_method,
            "is_validated": self.is_validated,
            "validation_errors": self.validation_errors,
            "corrected_value": self.corrected_value,
            "final_value": self.final_value,
            "is_valid": self.is_valid,
            "metadata": self.metadata
        }
    
    @classmethod
    def from_dict(cls, data: Dict) -> 'ExtractedField':
        """Create from dictionary."""
        bbox_data = data.get("bounding_box")
        bbox = BoundingBox.from_dict(bbox_data) if bbox_data else None
        
        return cls(
            name=data["name"],
            value=data["value"],
            confidence=data["confidence"],
            bounding_box=bbox,
            extraction_method=data.get("extraction_method", "unknown"),
            is_validated=data.get("is_validated", False),
            validation_errors=data.get("validation_errors", []),
            corrected_value=data.get("corrected_value"),
            metadata=data.get("metadata", {})
        )

@dataclass
class DocumentMetadata:
    """Metadata for a processed document."""
    document_id: str
    file_path: Optional[str] = None
    file_name: Optional[str] = None
    file_size: Optional[int] = None
    mime_type: Optional[str] = None
    upload_timestamp: datetime = field(default_factory=datetime.now)
    processing_timestamp: Optional[datetime] = None
    completion_timestamp: Optional[datetime] = None
    processing_duration: Optional[float] = None  # seconds
    image_dimensions: Optional[Dict[str, int]] = None  # {"width": 1920, "height": 1080}
    dpi: Optional[int] = None
    color_mode: Optional[str] = None  # "RGB", "GRAYSCALE", etc.
    preprocessing_applied: List[str] = field(default_factory=list)
    ocr_engine: Optional[str] = None
    classification_model: Optional[str] = None
    extraction_model: Optional[str] = None
    validation_model: Optional[str] = None
    
    def start_processing(self):
        """Mark processing as started."""
        self.processing_timestamp = datetime.now()
    
    def complete_processing(self):
        """Mark processing as completed and calculate duration."""
        self.completion_timestamp = datetime.now()
        if self.processing_timestamp:
            self.processing_duration = (
                self.completion_timestamp - self.processing_timestamp
            ).total_seconds()
    
    def to_dict(self) -> Dict:
        """Convert to dictionary."""
        return {
            "document_id": self.document_id,
            "file_path": self.file_path,
            "file_name": self.file_name,
            "file_size": self.file_size,
            "mime_type": self.mime_type,
            "upload_timestamp": self.upload_timestamp.isoformat() if self.upload_timestamp else None,
            "processing_timestamp": self.processing_timestamp.isoformat() if self.processing_timestamp else None,
            "completion_timestamp": self.completion_timestamp.isoformat() if self.completion_timestamp else None,
            "processing_duration": self.processing_duration,
            "image_dimensions": self.image_dimensions,
            "dpi": self.dpi,
            "color_mode": self.color_mode,
            "preprocessing_applied": self.preprocessing_applied,
            "ocr_engine": self.ocr_engine,
            "classification_model": self.classification_model,
            "extraction_model": self.extraction_model,
            "validation_model": self.validation_model
        }
    
    @classmethod
    def from_dict(cls, data: Dict) -> 'DocumentMetadata':
        """Create from dictionary."""
        # Parse datetime fields
        upload_ts = None
        if data.get("upload_timestamp"):
            upload_ts = datetime.fromisoformat(data["upload_timestamp"])
        
        processing_ts = None
        if data.get("processing_timestamp"):
            processing_ts = datetime.fromisoformat(data["processing_timestamp"])
        
        completion_ts = None
        if data.get("completion_timestamp"):
            completion_ts = datetime.fromisoformat(data["completion_timestamp"])
        
        return cls(
            document_id=data["document_id"],
            file_path=data.get("file_path"),
            file_name=data.get("file_name"),
            file_size=data.get("file_size"),
            mime_type=data.get("mime_type"),
            upload_timestamp=upload_ts,
            processing_timestamp=processing_ts,
            completion_timestamp=completion_ts,
            processing_duration=data.get("processing_duration"),
            image_dimensions=data.get("image_dimensions"),
            dpi=data.get("dpi"),
            color_mode=data.get("color_mode"),
            preprocessing_applied=data.get("preprocessing_applied", []),
            ocr_engine=data.get("ocr_engine"),
            classification_model=data.get("classification_model"),
            extraction_model=data.get("extraction_model"),
            validation_model=data.get("validation_model")
        )

@dataclass
class ValidationResult:
    """Result of document validation."""
    is_valid: bool
    validation_score: float
    field_results: Dict[str, Dict] = field(default_factory=dict)
    errors: List[Dict] = field(default_factory=list)
    warnings: List[Dict] = field(default_factory=list)
    info: List[Dict] = field(default_factory=list)
    cross_validation_results: List[Dict] = field(default_factory=list)
    validation_timestamp: datetime = field(default_factory=datetime.now)
    
    @property
    def error_count(self) -> int:
        """Number of validation errors."""
        return len(self.errors)
    
    @property
    def warning_count(self) -> int:
        """Number of validation warnings."""
        return len(self.warnings)
    
    @property
    def total_issues(self) -> int:
        """Total number of validation issues."""
        return self.error_count + self.warning_count
    
    def add_error(self, field: str, message: str, details: Dict = None):
        """Add validation error."""
        self.errors.append({
            "field": field,
            "message": message,
            "details": details or {},
            "timestamp": datetime.now().isoformat()
        })
    
    def add_warning(self, field: str, message: str, details: Dict = None):
        """Add validation warning."""
        self.warnings.append({
            "field": field,
            "message": message,
            "details": details or {},
            "timestamp": datetime.now().isoformat()
        })
    
    def to_dict(self) -> Dict:
        """Convert to dictionary."""
        return {
            "is_valid": self.is_valid,
            "validation_score": self.validation_score,
            "field_results": self.field_results,
            "errors": self.errors,
            "warnings": self.warnings,
            "info": self.info,
            "cross_validation_results": self.cross_validation_results,
            "validation_timestamp": self.validation_timestamp.isoformat(),
            "summary": {
                "error_count": self.error_count,
                "warning_count": self.warning_count,
                "total_issues": self.total_issues
            }
        }

@dataclass
class ExtractionResult:
    """Complete result of document processing and extraction."""
    document_id: str
    document_type: DocumentType
    status: ProcessingStatus
    metadata: DocumentMetadata
    extracted_fields: Dict[str, ExtractedField] = field(default_factory=dict)
    validation_result: Optional[ValidationResult] = None
    raw_text: Optional[str] = None
    ocr_confidence: Optional[float] = None
    classification_confidence: Optional[float] = None
    overall_confidence: Optional[float] = None
    processing_errors: List[str] = field(default_factory=list)
    processing_warnings: List[str] = field(default_factory=list)
    created_timestamp: datetime = field(default_factory=datetime.now)
    updated_timestamp: datetime = field(default_factory=datetime.now)
    
    def add_field(self, field: ExtractedField):
        """Add an extracted field."""
        self.extracted_fields[field.name] = field
        self.updated_timestamp = datetime.now()
    
    def get_field(self, name: str) -> Optional[ExtractedField]:
        """Get an extracted field by name."""
        return self.extracted_fields.get(name)
    
    def get_field_value(self, name: str, use_corrected: bool = True) -> Optional[str]:
        """Get field value (corrected if available and requested)."""
        field = self.get_field(name)
        if not field:
            return None
        
        if use_corrected and field.corrected_value:
            return field.corrected_value
        return field.value
    
    def set_validation_result(self, validation_result: ValidationResult):
        """Set validation result."""
        self.validation_result = validation_result
        self.updated_timestamp = datetime.now()
        
        # Update status based on validation
        if validation_result.is_valid:
            self.status = ProcessingStatus.VALIDATED
        else:
            # Check if errors are critical
            critical_errors = [e for e in validation_result.errors 
                             if e.get("level") == "error"]
            if critical_errors:
                self.status = ProcessingStatus.REJECTED
            else:
                self.status = ProcessingStatus.COMPLETED
    
    def add_processing_error(self, error: str):
        """Add processing error."""
        if error not in self.processing_errors:
            self.processing_errors.append(error)
        self.updated_timestamp = datetime.now()
    
    def add_processing_warning(self, warning: str):
        """Add processing warning."""
        if warning not in self.processing_warnings:
            self.processing_warnings.append(warning)
        self.updated_timestamp = datetime.now()
    
    def calculate_overall_confidence(self) -> float:
        """Calculate overall confidence score."""
        if not self.extracted_fields:
            return 0.0
        
        # Weight different confidence scores
        weights = {
            "ocr": 0.3,
            "classification": 0.2,
            "extraction": 0.4,
            "validation": 0.1
        }
        
        total_score = 0.0
        total_weight = 0.0
        
        # OCR confidence
        if self.ocr_confidence is not None:
            total_score += weights["ocr"] * self.ocr_confidence
            total_weight += weights["ocr"]
        
        # Classification confidence
        if self.classification_confidence is not None:
            total_score += weights["classification"] * self.classification_confidence
            total_weight += weights["classification"]
        
        # Field extraction confidence (average)
        field_confidences = [f.confidence for f in self.extracted_fields.values()]
        if field_confidences:
            avg_field_confidence = sum(field_confidences) / len(field_confidences)
            total_score += weights["extraction"] * avg_field_confidence
            total_weight += weights["extraction"]
        
        # Validation score
        if self.validation_result:
            total_score += weights["validation"] * self.validation_result.validation_score
            total_weight += weights["validation"]
        
        self.overall_confidence = total_score / total_weight if total_weight > 0 else 0.0
        return self.overall_confidence
    
    @property
    def is_successful(self) -> bool:
        """Check if processing was successful."""
        return self.status in [ProcessingStatus.COMPLETED, ProcessingStatus.VALIDATED]
    
    @property
    def has_errors(self) -> bool:
        """Check if there are processing errors."""
        return len(self.processing_errors) > 0
    
    @property
    def field_count(self) -> int:
        """Number of extracted fields."""
        return len(self.extracted_fields)
    
    @property
    def valid_field_count(self) -> int:
        """Number of valid extracted fields."""
        return sum(1 for f in self.extracted_fields.values() if f.is_valid)
    
    def to_dict(self) -> Dict:
        """Convert to dictionary."""
        return {
            "document_id": self.document_id,
            "document_type": self.document_type.value,
            "status": self.status.value,
            "metadata": self.metadata.to_dict(),
            "extracted_fields": {
                name: field.to_dict() for name, field in self.extracted_fields.items()
            },
            "validation_result": self.validation_result.to_dict() if self.validation_result else None,
            "raw_text": self.raw_text,
            "ocr_confidence": self.ocr_confidence,
            "classification_confidence": self.classification_confidence,
            "overall_confidence": self.overall_confidence,
            "processing_errors": self.processing_errors,
            "processing_warnings": self.processing_warnings,
            "created_timestamp": self.created_timestamp.isoformat(),
            "updated_timestamp": self.updated_timestamp.isoformat(),
            "summary": {
                "is_successful": self.is_successful,
                "has_errors": self.has_errors,
                "field_count": self.field_count,
                "valid_field_count": self.valid_field_count
            }
        }
    
    def save_to_file(self, file_path: Union[str, Path]):
        """Save extraction result to JSON file."""
        file_path = Path(file_path)
        file_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(file_path, 'w', encoding='utf-8') as f:
            json.dump(self.to_dict(), f, indent=2, ensure_ascii=False)
        
        logger.info(f"Extraction result saved to {file_path}")
    
    @classmethod
    def load_from_file(cls, file_path: Union[str, Path]) -> 'ExtractionResult':
        """Load extraction result from JSON file."""
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        return cls.from_dict(data)
    
    @classmethod
    def from_dict(cls, data: Dict) -> 'ExtractionResult':
        """Create from dictionary."""
        # Parse enums
        document_type = DocumentType.from_string(data["document_type"])
        status = ProcessingStatus(data["status"])
        
        # Parse metadata
        metadata = DocumentMetadata.from_dict(data["metadata"])
        
        # Parse extracted fields
        extracted_fields = {}
        for name, field_data in data.get("extracted_fields", {}).items():
            extracted_fields[name] = ExtractedField.from_dict(field_data)
        
        # Parse validation result
        validation_result = None
        if data.get("validation_result"):
            val_data = data["validation_result"]
            validation_result = ValidationResult(
                is_valid=val_data["is_valid"],
                validation_score=val_data["validation_score"],
                field_results=val_data.get("field_results", {}),
                errors=val_data.get("errors", []),
                warnings=val_data.get("warnings", []),
                info=val_data.get("info", []),
                cross_validation_results=val_data.get("cross_validation_results", []),
                validation_timestamp=datetime.fromisoformat(val_data["validation_timestamp"])
            )
        
        # Parse timestamps
        created_ts = datetime.fromisoformat(data["created_timestamp"])
        updated_ts = datetime.fromisoformat(data["updated_timestamp"])
        
        return cls(
            document_id=data["document_id"],
            document_type=document_type,
            status=status,
            metadata=metadata,
            extracted_fields=extracted_fields,
            validation_result=validation_result,
            raw_text=data.get("raw_text"),
            ocr_confidence=data.get("ocr_confidence"),
            classification_confidence=data.get("classification_confidence"),
            overall_confidence=data.get("overall_confidence"),
            processing_errors=data.get("processing_errors", []),
            processing_warnings=data.get("processing_warnings", []),
            created_timestamp=created_ts,
            updated_timestamp=updated_ts
        )

# Utility functions
def create_document_id() -> str:
    """Generate a unique document ID."""
    from uuid import uuid4
    return f"doc_{uuid4().hex[:12]}"

def create_extraction_result(document_type: DocumentType, 
                           file_path: Optional[str] = None) -> ExtractionResult:
    """Create a new extraction result with basic metadata."""
    doc_id = create_document_id()
    
    metadata = DocumentMetadata(
        document_id=doc_id,
        file_path=file_path,
        file_name=Path(file_path).name if file_path else None
    )
    
    return ExtractionResult(
        document_id=doc_id,
        document_type=document_type,
        status=ProcessingStatus.PENDING,
        metadata=metadata
    )