#!/usr/bin/env python3
"""
User Acceptance Testing - Accuracy Validation System

Comprehensive framework for manual verification and accuracy validation
of the Malaysian document parser system results.
"""

import os
import sys
import json
import time
import logging
import sqlite3
import hashlib
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple, Union
from datetime import datetime, timedelta
from dataclasses import dataclass, field
from enum import Enum
import statistics

import pytest
import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))


class ValidationStatus(Enum):
    """Status of validation items."""
    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    VALIDATED = "validated"
    REJECTED = "rejected"
    NEEDS_REVIEW = "needs_review"


class AccuracyLevel(Enum):
    """Accuracy levels for validation."""
    PERFECT = "perfect"  # 100% accurate
    EXCELLENT = "excellent"  # 95-99% accurate
    GOOD = "good"  # 85-94% accurate
    FAIR = "fair"  # 70-84% accurate
    POOR = "poor"  # Below 70% accurate


class FieldType(Enum):
    """Types of fields for validation."""
    IC_NUMBER = "ic_number"
    NAME = "name"
    DATE_OF_BIRTH = "date_of_birth"
    GENDER = "gender"
    ADDRESS = "address"
    NATIONALITY = "nationality"
    RELIGION = "religion"
    CERTIFICATE_NUMBER = "certificate_number"
    CERTIFICATE_TYPE = "certificate_type"
    ISSUE_DATE = "issue_date"
    EXPIRY_DATE = "expiry_date"
    ISSUING_AUTHORITY = "issuing_authority"


@dataclass
class FieldValidation:
    """Individual field validation result."""
    field_type: FieldType
    extracted_value: str
    expected_value: str
    is_correct: bool
    confidence_score: float
    validation_notes: str = ""
    validator_id: str = ""
    validation_timestamp: str = field(default_factory=lambda: datetime.now().isoformat())
    correction_suggestion: str = ""
    difficulty_level: str = "normal"  # easy, normal, hard, very_hard


@dataclass
class DocumentValidation:
    """Complete document validation result."""
    validation_id: str
    document_id: str
    document_type: str  # mykad, spk
    document_path: str
    processing_result: Dict[str, Any]
    field_validations: List[FieldValidation]
    overall_accuracy: float
    accuracy_level: AccuracyLevel
    validation_status: ValidationStatus
    validator_id: str
    validation_start_time: str
    validation_end_time: Optional[str] = None
    validation_duration_seconds: Optional[float] = None
    validation_notes: str = ""
    quality_assessment: Dict[str, Any] = field(default_factory=dict)
    processing_time_ms: Optional[float] = None
    confidence_scores: Dict[str, float] = field(default_factory=dict)
    error_categories: List[str] = field(default_factory=list)
    improvement_suggestions: List[str] = field(default_factory=list)
    created_at: str = field(default_factory=lambda: datetime.now().isoformat())


@dataclass
class ValidationMetrics:
    """Comprehensive validation metrics."""
    total_documents_validated: int
    total_fields_validated: int
    overall_accuracy_percentage: float
    accuracy_by_document_type: Dict[str, float]
    accuracy_by_field_type: Dict[str, float]
    accuracy_by_quality_level: Dict[str, float]
    common_error_patterns: List[Dict[str, Any]]
    validation_time_metrics: Dict[str, float]
    confidence_vs_accuracy_correlation: float
    improvement_areas: List[str]
    validator_performance: Dict[str, Dict[str, float]]
    accuracy_trends: List[Dict[str, Any]]
    generated_at: str = field(default_factory=lambda: datetime.now().isoformat())


class AccuracyValidationSystem:
    """Comprehensive accuracy validation and manual verification system."""
    
    def __init__(self, database_path: str = None, validation_data_dir: str = None):
        self.database_path = database_path or "accuracy_validation.db"
        self.validation_data_dir = Path(validation_data_dir or "validation_data")
        self.validation_data_dir.mkdir(exist_ok=True)
        
        # Setup logging
        self.logger = logging.getLogger(__name__)
        
        # Initialize database
        self._initialize_database()
        
        # Validation configuration
        self.config = {
            'accuracy_thresholds': {
                'perfect': 1.0,
                'excellent': 0.95,
                'good': 0.85,
                'fair': 0.70,
                'poor': 0.0
            },
            'field_weights': {
                'ic_number': 0.25,
                'name': 0.20,
                'date_of_birth': 0.15,
                'gender': 0.10,
                'address': 0.15,
                'nationality': 0.05,
                'religion': 0.05,
                'certificate_number': 0.25,
                'certificate_type': 0.15,
                'issue_date': 0.15,
                'expiry_date': 0.15,
                'issuing_authority': 0.10
            },
            'validation_requirements': {
                'min_validators_per_document': 2,
                'consensus_threshold': 0.8,
                'max_validation_time_minutes': 30,
                'quality_check_required': True
            },
            'quality_metrics': {
                'image_resolution_min': (300, 300),
                'brightness_range': (50, 200),
                'contrast_min': 0.3,
                'blur_threshold': 100
            }
        }
        
        # Ground truth data storage
        self.ground_truth_data = {}
        self._load_ground_truth_data()
    
    def _initialize_database(self):
        """Initialize SQLite database for validation storage."""
        conn = sqlite3.connect(self.database_path)
        cursor = conn.cursor()
        
        # Create document validations table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS document_validations (
                validation_id TEXT PRIMARY KEY,
                document_id TEXT NOT NULL,
                document_type TEXT NOT NULL,
                document_path TEXT NOT NULL,
                processing_result TEXT NOT NULL,
                overall_accuracy REAL NOT NULL,
                accuracy_level TEXT NOT NULL,
                validation_status TEXT NOT NULL,
                validator_id TEXT NOT NULL,
                validation_start_time TEXT NOT NULL,
                validation_end_time TEXT,
                validation_duration_seconds REAL,
                validation_notes TEXT,
                quality_assessment TEXT,
                processing_time_ms REAL,
                confidence_scores TEXT,
                error_categories TEXT,
                improvement_suggestions TEXT,
                created_at TEXT NOT NULL
            )
        ''')
        
        # Create field validations table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS field_validations (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                validation_id TEXT NOT NULL,
                field_type TEXT NOT NULL,
                extracted_value TEXT NOT NULL,
                expected_value TEXT NOT NULL,
                is_correct BOOLEAN NOT NULL,
                confidence_score REAL NOT NULL,
                validation_notes TEXT,
                validator_id TEXT NOT NULL,
                validation_timestamp TEXT NOT NULL,
                correction_suggestion TEXT,
                difficulty_level TEXT,
                FOREIGN KEY (validation_id) REFERENCES document_validations (validation_id)
            )
        ''')
        
        # Create ground truth table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS ground_truth (
                document_id TEXT PRIMARY KEY,
                document_type TEXT NOT NULL,
                document_path TEXT NOT NULL,
                ground_truth_data TEXT NOT NULL,
                verified_by TEXT NOT NULL,
                verification_date TEXT NOT NULL,
                quality_score REAL,
                notes TEXT
            )
        ''')
        
        # Create indexes
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_validation_document_type ON document_validations(document_type)')
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_validation_accuracy ON document_validations(overall_accuracy)')
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_validation_status ON document_validations(validation_status)')
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_field_type ON field_validations(field_type)')
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_field_correct ON field_validations(is_correct)')
        
        conn.commit()
        conn.close()
    
    def _load_ground_truth_data(self):
        """Load ground truth data from database."""
        conn = sqlite3.connect(self.database_path)
        cursor = conn.cursor()
        
        cursor.execute('SELECT document_id, ground_truth_data FROM ground_truth')
        rows = cursor.fetchall()
        
        for document_id, ground_truth_json in rows:
            self.ground_truth_data[document_id] = json.loads(ground_truth_json)
        
        conn.close()
        self.logger.info(f"Loaded {len(self.ground_truth_data)} ground truth records")
    
    def add_ground_truth(self, document_id: str, document_type: str, 
                        document_path: str, ground_truth_data: Dict[str, Any],
                        verified_by: str, quality_score: float = None,
                        notes: str = "") -> bool:
        """Add ground truth data for a document."""
        try:
            conn = sqlite3.connect(self.database_path)
            cursor = conn.cursor()
            
            cursor.execute('''
                INSERT OR REPLACE INTO ground_truth 
                (document_id, document_type, document_path, ground_truth_data, 
                 verified_by, verification_date, quality_score, notes)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            ''', (
                document_id, document_type, document_path,
                json.dumps(ground_truth_data), verified_by,
                datetime.now().isoformat(), quality_score, notes
            ))
            
            conn.commit()
            conn.close()
            
            # Update in-memory cache
            self.ground_truth_data[document_id] = ground_truth_data
            
            self.logger.info(f"Added ground truth for document: {document_id}")
            return True
            
        except Exception as e:
            self.logger.error(f"Error adding ground truth: {e}")
            return False
    
    def validate_document(self, document_id: str, document_type: str,
                         document_path: str, processing_result: Dict[str, Any],
                         validator_id: str, processing_time_ms: float = None) -> str:
        """Validate a document's processing results."""
        validation_id = self._generate_validation_id()
        validation_start_time = datetime.now().isoformat()
        
        self.logger.info(f"Starting validation for document: {document_id}")
        
        # Get ground truth data
        ground_truth = self.ground_truth_data.get(document_id)
        if not ground_truth:
            raise ValueError(f"No ground truth data found for document: {document_id}")
        
        # Perform field-by-field validation
        field_validations = self._validate_fields(
            processing_result, ground_truth, validator_id
        )
        
        # Calculate overall accuracy
        overall_accuracy = self._calculate_overall_accuracy(
            field_validations, document_type
        )
        
        # Determine accuracy level
        accuracy_level = self._determine_accuracy_level(overall_accuracy)
        
        # Assess document quality
        quality_assessment = self._assess_document_quality(document_path)
        
        # Extract confidence scores
        confidence_scores = self._extract_confidence_scores(processing_result)
        
        # Identify error categories
        error_categories = self._identify_error_categories(field_validations)
        
        # Generate improvement suggestions
        improvement_suggestions = self._generate_improvement_suggestions(
            field_validations, quality_assessment, error_categories
        )
        
        # Create validation record
        validation = DocumentValidation(
            validation_id=validation_id,
            document_id=document_id,
            document_type=document_type,
            document_path=document_path,
            processing_result=processing_result,
            field_validations=field_validations,
            overall_accuracy=overall_accuracy,
            accuracy_level=accuracy_level,
            validation_status=ValidationStatus.VALIDATED,
            validator_id=validator_id,
            validation_start_time=validation_start_time,
            validation_end_time=datetime.now().isoformat(),
            quality_assessment=quality_assessment,
            processing_time_ms=processing_time_ms,
            confidence_scores=confidence_scores,
            error_categories=error_categories,
            improvement_suggestions=improvement_suggestions
        )
        
        # Calculate validation duration
        start_time = datetime.fromisoformat(validation.validation_start_time)
        end_time = datetime.fromisoformat(validation.validation_end_time)
        validation.validation_duration_seconds = (end_time - start_time).total_seconds()
        
        # Store validation
        self._store_validation(validation)
        
        self.logger.info(f"Validation completed: {validation_id} (Accuracy: {overall_accuracy:.2%})")
        return validation_id
    
    def _validate_fields(self, processing_result: Dict[str, Any],
                        ground_truth: Dict[str, Any], validator_id: str) -> List[FieldValidation]:
        """Validate individual fields."""
        field_validations = []
        
        # Define field mappings
        field_mappings = {
            'ic_number': ['ic_number', 'nric', 'identification_number'],
            'name': ['name', 'full_name'],
            'date_of_birth': ['date_of_birth', 'dob', 'birth_date'],
            'gender': ['gender', 'sex'],
            'address': ['address', 'full_address'],
            'nationality': ['nationality', 'citizenship'],
            'religion': ['religion'],
            'certificate_number': ['certificate_number', 'cert_number'],
            'certificate_type': ['certificate_type', 'cert_type'],
            'issue_date': ['issue_date', 'issued_date'],
            'expiry_date': ['expiry_date', 'expiration_date'],
            'issuing_authority': ['issuing_authority', 'authority']
        }
        
        for field_type_str, possible_keys in field_mappings.items():
            try:
                field_type = FieldType(field_type_str)
            except ValueError:
                continue
            
            # Find extracted value
            extracted_value = ""
            confidence_score = 0.0
            
            for key in possible_keys:
                if key in processing_result:
                    if isinstance(processing_result[key], dict):
                        extracted_value = processing_result[key].get('value', '')
                        confidence_score = processing_result[key].get('confidence', 0.0)
                    else:
                        extracted_value = str(processing_result[key])
                        confidence_score = processing_result.get(f"{key}_confidence", 0.0)
                    break
            
            # Find expected value
            expected_value = ""
            for key in possible_keys:
                if key in ground_truth:
                    expected_value = str(ground_truth[key])
                    break
            
            # Skip if no expected value
            if not expected_value:
                continue
            
            # Validate field
            is_correct = self._compare_field_values(
                extracted_value, expected_value, field_type
            )
            
            # Determine difficulty level
            difficulty_level = self._assess_field_difficulty(
                field_type, expected_value, extracted_value
            )
            
            # Generate correction suggestion if incorrect
            correction_suggestion = ""
            if not is_correct:
                correction_suggestion = self._generate_correction_suggestion(
                    field_type, extracted_value, expected_value
                )
            
            field_validation = FieldValidation(
                field_type=field_type,
                extracted_value=extracted_value,
                expected_value=expected_value,
                is_correct=is_correct,
                confidence_score=confidence_score,
                validator_id=validator_id,
                correction_suggestion=correction_suggestion,
                difficulty_level=difficulty_level
            )
            
            field_validations.append(field_validation)
        
        return field_validations
    
    def _compare_field_values(self, extracted: str, expected: str, field_type: FieldType) -> bool:
        """Compare extracted and expected field values."""
        # Normalize values
        extracted = extracted.strip().lower()
        expected = expected.strip().lower()
        
        # Exact match for most fields
        if extracted == expected:
            return True
        
        # Special handling for specific field types
        if field_type == FieldType.IC_NUMBER:
            # Remove hyphens and spaces for IC number comparison
            extracted_clean = ''.join(c for c in extracted if c.isdigit())
            expected_clean = ''.join(c for c in expected if c.isdigit())
            return extracted_clean == expected_clean
        
        elif field_type == FieldType.NAME:
            # Allow for minor variations in name formatting
            extracted_words = set(extracted.split())
            expected_words = set(expected.split())
            # Consider correct if 80% of words match
            if len(expected_words) > 0:
                match_ratio = len(extracted_words & expected_words) / len(expected_words)
                return match_ratio >= 0.8
        
        elif field_type in [FieldType.DATE_OF_BIRTH, FieldType.ISSUE_DATE, FieldType.EXPIRY_DATE]:
            # Normalize date formats
            extracted_date = self._normalize_date(extracted)
            expected_date = self._normalize_date(expected)
            return extracted_date == expected_date
        
        elif field_type == FieldType.ADDRESS:
            # Allow for minor variations in address formatting
            extracted_clean = ''.join(c.lower() for c in extracted if c.isalnum() or c.isspace())
            expected_clean = ''.join(c.lower() for c in expected if c.isalnum() or c.isspace())
            # Consider correct if 85% similarity
            similarity = self._calculate_string_similarity(extracted_clean, expected_clean)
            return similarity >= 0.85
        
        return False
    
    def _normalize_date(self, date_str: str) -> str:
        """Normalize date string to standard format."""
        # Remove common separators and normalize
        date_clean = ''.join(c for c in date_str if c.isdigit())
        
        # Try to parse common date formats
        if len(date_clean) == 8:  # DDMMYYYY or YYYYMMDD
            if date_clean[:2] <= '31':  # Likely DDMMYYYY
                return f"{date_clean[4:]}-{date_clean[2:4]}-{date_clean[:2]}"
            else:  # Likely YYYYMMDD
                return f"{date_clean[:4]}-{date_clean[4:6]}-{date_clean[6:]}"
        
        return date_clean
    
    def _calculate_string_similarity(self, str1: str, str2: str) -> float:
        """Calculate similarity between two strings using Levenshtein distance."""
        if len(str1) == 0 or len(str2) == 0:
            return 0.0
        
        # Simple Levenshtein distance implementation
        matrix = [[0] * (len(str2) + 1) for _ in range(len(str1) + 1)]
        
        for i in range(len(str1) + 1):
            matrix[i][0] = i
        for j in range(len(str2) + 1):
            matrix[0][j] = j
        
        for i in range(1, len(str1) + 1):
            for j in range(1, len(str2) + 1):
                if str1[i-1] == str2[j-1]:
                    matrix[i][j] = matrix[i-1][j-1]
                else:
                    matrix[i][j] = min(
                        matrix[i-1][j] + 1,    # deletion
                        matrix[i][j-1] + 1,    # insertion
                        matrix[i-1][j-1] + 1   # substitution
                    )
        
        distance = matrix[len(str1)][len(str2)]
        max_len = max(len(str1), len(str2))
        return 1.0 - (distance / max_len)
    
    def _assess_field_difficulty(self, field_type: FieldType, expected: str, extracted: str) -> str:
        """Assess the difficulty level of field extraction."""
        # Base difficulty on field type
        base_difficulty = {
            FieldType.IC_NUMBER: "normal",
            FieldType.NAME: "normal",
            FieldType.DATE_OF_BIRTH: "easy",
            FieldType.GENDER: "easy",
            FieldType.ADDRESS: "hard",
            FieldType.NATIONALITY: "normal",
            FieldType.RELIGION: "normal",
            FieldType.CERTIFICATE_NUMBER: "normal",
            FieldType.CERTIFICATE_TYPE: "easy",
            FieldType.ISSUE_DATE: "normal",
            FieldType.EXPIRY_DATE: "normal",
            FieldType.ISSUING_AUTHORITY: "normal"
        }.get(field_type, "normal")
        
        # Adjust based on content complexity
        if len(expected) > 50:  # Long text
            if base_difficulty == "easy":
                return "normal"
            elif base_difficulty == "normal":
                return "hard"
        
        # Check for special characters or formatting
        if any(c in expected for c in ".,/-()[]{}@#$%^&*"):
            if base_difficulty == "easy":
                return "normal"
            elif base_difficulty == "normal":
                return "hard"
        
        return base_difficulty
    
    def _generate_correction_suggestion(self, field_type: FieldType, 
                                      extracted: str, expected: str) -> str:
        """Generate correction suggestion for incorrect field."""
        if field_type == FieldType.IC_NUMBER:
            return f"Check IC number format. Expected: {expected}, Got: {extracted}"
        elif field_type == FieldType.NAME:
            return f"Verify name spelling and formatting. Expected: {expected}"
        elif field_type in [FieldType.DATE_OF_BIRTH, FieldType.ISSUE_DATE, FieldType.EXPIRY_DATE]:
            return f"Check date format and OCR accuracy. Expected: {expected}"
        elif field_type == FieldType.ADDRESS:
            return f"Review address extraction and line breaks. Expected format: {expected}"
        else:
            return f"Review field extraction accuracy. Expected: {expected}"
    
    def _calculate_overall_accuracy(self, field_validations: List[FieldValidation], 
                                  document_type: str) -> float:
        """Calculate overall accuracy using weighted scoring."""
        if not field_validations:
            return 0.0
        
        total_weight = 0.0
        weighted_score = 0.0
        
        for validation in field_validations:
            field_weight = self.config['field_weights'].get(
                validation.field_type.value, 0.1
            )
            total_weight += field_weight
            
            if validation.is_correct:
                weighted_score += field_weight
        
        return weighted_score / total_weight if total_weight > 0 else 0.0
    
    def _determine_accuracy_level(self, accuracy: float) -> AccuracyLevel:
        """Determine accuracy level based on score."""
        thresholds = self.config['accuracy_thresholds']
        
        if accuracy >= thresholds['perfect']:
            return AccuracyLevel.PERFECT
        elif accuracy >= thresholds['excellent']:
            return AccuracyLevel.EXCELLENT
        elif accuracy >= thresholds['good']:
            return AccuracyLevel.GOOD
        elif accuracy >= thresholds['fair']:
            return AccuracyLevel.FAIR
        else:
            return AccuracyLevel.POOR
    
    def _assess_document_quality(self, document_path: str) -> Dict[str, Any]:
        """Assess document image quality."""
        try:
            # Load image
            image = cv2.imread(document_path)
            if image is None:
                return {'error': 'Could not load image'}
            
            # Convert to grayscale
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            
            # Calculate metrics
            height, width = gray.shape
            resolution_score = min(width / 300, height / 300, 1.0)  # Normalize to 300 DPI baseline
            
            # Brightness
            brightness = np.mean(gray)
            brightness_score = 1.0 if 50 <= brightness <= 200 else 0.5
            
            # Contrast
            contrast = np.std(gray)
            contrast_score = min(contrast / 50, 1.0)  # Normalize
            
            # Blur detection (Laplacian variance)
            blur_score = cv2.Laplacian(gray, cv2.CV_64F).var()
            blur_normalized = min(blur_score / 100, 1.0)
            
            # Overall quality score
            quality_score = (resolution_score + brightness_score + contrast_score + blur_normalized) / 4
            
            return {
                'resolution': (width, height),
                'resolution_score': resolution_score,
                'brightness': brightness,
                'brightness_score': brightness_score,
                'contrast': contrast,
                'contrast_score': contrast_score,
                'blur_score': blur_score,
                'blur_normalized': blur_normalized,
                'overall_quality_score': quality_score,
                'quality_level': self._get_quality_level(quality_score)
            }
            
        except Exception as e:
            self.logger.error(f"Error assessing document quality: {e}")
            return {'error': str(e)}
    
    def _get_quality_level(self, score: float) -> str:
        """Get quality level description."""
        if score >= 0.9:
            return "excellent"
        elif score >= 0.7:
            return "good"
        elif score >= 0.5:
            return "fair"
        else:
            return "poor"
    
    def _extract_confidence_scores(self, processing_result: Dict[str, Any]) -> Dict[str, float]:
        """Extract confidence scores from processing result."""
        confidence_scores = {}
        
        for key, value in processing_result.items():
            if isinstance(value, dict) and 'confidence' in value:
                confidence_scores[key] = value['confidence']
            elif key.endswith('_confidence'):
                field_name = key.replace('_confidence', '')
                confidence_scores[field_name] = value
        
        return confidence_scores
    
    def _identify_error_categories(self, field_validations: List[FieldValidation]) -> List[str]:
        """Identify categories of errors."""
        error_categories = []
        
        incorrect_fields = [v for v in field_validations if not v.is_correct]
        
        if not incorrect_fields:
            return error_categories
        
        # Categorize errors
        ocr_errors = 0
        formatting_errors = 0
        extraction_errors = 0
        
        for validation in incorrect_fields:
            # Simple heuristics for error categorization
            if len(validation.extracted_value) == 0:
                extraction_errors += 1
            elif self._calculate_string_similarity(
                validation.extracted_value, validation.expected_value
            ) > 0.7:
                ocr_errors += 1
            else:
                formatting_errors += 1
        
        if ocr_errors > 0:
            error_categories.append("ocr_accuracy")
        if formatting_errors > 0:
            error_categories.append("field_formatting")
        if extraction_errors > 0:
            error_categories.append("field_extraction")
        
        # Check for specific field type errors
        field_type_errors = {}
        for validation in incorrect_fields:
            field_type = validation.field_type.value
            field_type_errors[field_type] = field_type_errors.get(field_type, 0) + 1
        
        # Add field-specific error categories
        for field_type, count in field_type_errors.items():
            if count > 0:
                error_categories.append(f"{field_type}_errors")
        
        return error_categories
    
    def _generate_improvement_suggestions(self, field_validations: List[FieldValidation],
                                        quality_assessment: Dict[str, Any],
                                        error_categories: List[str]) -> List[str]:
        """Generate improvement suggestions based on validation results."""
        suggestions = []
        
        # Quality-based suggestions
        if 'overall_quality_score' in quality_assessment:
            quality_score = quality_assessment['overall_quality_score']
            if quality_score < 0.5:
                suggestions.append("Improve image preprocessing to enhance quality")
            if quality_assessment.get('brightness_score', 1.0) < 0.8:
                suggestions.append("Adjust brightness normalization algorithms")
            if quality_assessment.get('contrast_score', 1.0) < 0.8:
                suggestions.append("Enhance contrast adjustment techniques")
            if quality_assessment.get('blur_normalized', 1.0) < 0.7:
                suggestions.append("Implement blur detection and sharpening")
        
        # Error category-based suggestions
        if "ocr_accuracy" in error_categories:
            suggestions.append("Fine-tune OCR engine parameters for better text recognition")
        if "field_formatting" in error_categories:
            suggestions.append("Improve field extraction and formatting rules")
        if "field_extraction" in error_categories:
            suggestions.append("Enhance field detection and boundary identification")
        
        # Field-specific suggestions
        incorrect_fields = [v for v in field_validations if not v.is_correct]
        field_error_counts = {}
        
        for validation in incorrect_fields:
            field_type = validation.field_type.value
            field_error_counts[field_type] = field_error_counts.get(field_type, 0) + 1
        
        for field_type, count in field_error_counts.items():
            if field_type == "ic_number" and count > 0:
                suggestions.append("Improve IC number pattern recognition and validation")
            elif field_type == "name" and count > 0:
                suggestions.append("Enhance name extraction with better text segmentation")
            elif field_type == "address" and count > 0:
                suggestions.append("Improve multi-line address extraction algorithms")
            elif "date" in field_type and count > 0:
                suggestions.append("Enhance date format recognition and parsing")
        
        # Confidence-based suggestions
        low_confidence_fields = [
            v for v in field_validations 
            if v.confidence_score < 0.7 and not v.is_correct
        ]
        
        if len(low_confidence_fields) > len(field_validations) * 0.3:
            suggestions.append("Review confidence calculation algorithms")
        
        return list(set(suggestions))  # Remove duplicates
    
    def _generate_validation_id(self) -> str:
        """Generate unique validation ID."""
        timestamp = str(int(time.time() * 1000))
        random_part = hashlib.md5(os.urandom(16)).hexdigest()[:8]
        return f"VAL_{timestamp}_{random_part}"
    
    def _store_validation(self, validation: DocumentValidation):
        """Store validation in database."""
        conn = sqlite3.connect(self.database_path)
        cursor = conn.cursor()
        
        # Store document validation
        cursor.execute('''
            INSERT INTO document_validations (
                validation_id, document_id, document_type, document_path, processing_result,
                overall_accuracy, accuracy_level, validation_status, validator_id,
                validation_start_time, validation_end_time, validation_duration_seconds,
                validation_notes, quality_assessment, processing_time_ms, confidence_scores,
                error_categories, improvement_suggestions, created_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        ''', (
            validation.validation_id, validation.document_id, validation.document_type,
            validation.document_path, json.dumps(validation.processing_result),
            validation.overall_accuracy, validation.accuracy_level.value,
            validation.validation_status.value, validation.validator_id,
            validation.validation_start_time, validation.validation_end_time,
            validation.validation_duration_seconds, validation.validation_notes,
            json.dumps(validation.quality_assessment), validation.processing_time_ms,
            json.dumps(validation.confidence_scores), json.dumps(validation.error_categories),
            json.dumps(validation.improvement_suggestions), validation.created_at
        ))
        
        # Store field validations
        for field_validation in validation.field_validations:
            cursor.execute('''
                INSERT INTO field_validations (
                    validation_id, field_type, extracted_value, expected_value, is_correct,
                    confidence_score, validation_notes, validator_id, validation_timestamp,
                    correction_suggestion, difficulty_level
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ''', (
                validation.validation_id, field_validation.field_type.value,
                field_validation.extracted_value, field_validation.expected_value,
                field_validation.is_correct, field_validation.confidence_score,
                field_validation.validation_notes, field_validation.validator_id,
                field_validation.validation_timestamp, field_validation.correction_suggestion,
                field_validation.difficulty_level
            ))
        
        conn.commit()
        conn.close()
    
    def get_validation_metrics(self, start_date: str = None, end_date: str = None,
                             document_type: str = None, validator_id: str = None) -> ValidationMetrics:
        """Get comprehensive validation metrics."""
        conn = sqlite3.connect(self.database_path)
        cursor = conn.cursor()
        
        # Build query conditions
        conditions = []
        params = []
        
        if start_date:
            conditions.append("created_at >= ?")
            params.append(start_date)
        
        if end_date:
            conditions.append("created_at <= ?")
            params.append(end_date)
        
        if document_type:
            conditions.append("document_type = ?")
            params.append(document_type)
        
        if validator_id:
            conditions.append("validator_id = ?")
            params.append(validator_id)
        
        where_clause = " AND ".join(conditions) if conditions else "1=1"
        
        # Total documents validated
        cursor.execute(f"SELECT COUNT(*) FROM document_validations WHERE {where_clause}", params)
        total_documents = cursor.fetchone()[0]
        
        # Total fields validated
        cursor.execute(f'''
            SELECT COUNT(*) FROM field_validations fv
            JOIN document_validations dv ON fv.validation_id = dv.validation_id
            WHERE {where_clause}
        ''', params)
        total_fields = cursor.fetchone()[0]
        
        # Overall accuracy
        cursor.execute(f"SELECT AVG(overall_accuracy) FROM document_validations WHERE {where_clause}", params)
        overall_accuracy = cursor.fetchone()[0] or 0.0
        
        # Accuracy by document type
        cursor.execute(f'''
            SELECT document_type, AVG(overall_accuracy) 
            FROM document_validations 
            WHERE {where_clause}
            GROUP BY document_type
        ''', params)
        accuracy_by_doc_type = dict(cursor.fetchall())
        
        # Accuracy by field type
        cursor.execute(f'''
            SELECT fv.field_type, AVG(CASE WHEN fv.is_correct THEN 1.0 ELSE 0.0 END)
            FROM field_validations fv
            JOIN document_validations dv ON fv.validation_id = dv.validation_id
            WHERE {where_clause}
            GROUP BY fv.field_type
        ''', params)
        accuracy_by_field = dict(cursor.fetchall())
        
        # Accuracy by quality level
        cursor.execute(f'''
            SELECT 
                CASE 
                    WHEN JSON_EXTRACT(quality_assessment, '$.overall_quality_score') >= 0.9 THEN 'excellent'
                    WHEN JSON_EXTRACT(quality_assessment, '$.overall_quality_score') >= 0.7 THEN 'good'
                    WHEN JSON_EXTRACT(quality_assessment, '$.overall_quality_score') >= 0.5 THEN 'fair'
                    ELSE 'poor'
                END as quality_level,
                AVG(overall_accuracy)
            FROM document_validations 
            WHERE {where_clause} AND quality_assessment IS NOT NULL
            GROUP BY quality_level
        ''', params)
        accuracy_by_quality = dict(cursor.fetchall())
        
        # Common error patterns
        cursor.execute(f'''
            SELECT fv.field_type, fv.correction_suggestion, COUNT(*) as count
            FROM field_validations fv
            JOIN document_validations dv ON fv.validation_id = dv.validation_id
            WHERE {where_clause} AND fv.is_correct = 0
            GROUP BY fv.field_type, fv.correction_suggestion
            ORDER BY count DESC
            LIMIT 10
        ''', params)
        common_errors = [{
            'field_type': row[0], 
            'suggestion': row[1], 
            'count': row[2]
        } for row in cursor.fetchall()]
        
        # Validation time metrics
        cursor.execute(f'''
            SELECT 
                AVG(validation_duration_seconds) as avg_time,
                MIN(validation_duration_seconds) as min_time,
                MAX(validation_duration_seconds) as max_time
            FROM document_validations 
            WHERE {where_clause} AND validation_duration_seconds IS NOT NULL
        ''', params)
        time_row = cursor.fetchone()
        time_metrics = {
            'average_seconds': time_row[0] or 0,
            'minimum_seconds': time_row[1] or 0,
            'maximum_seconds': time_row[2] or 0
        }
        
        # Confidence vs accuracy correlation
        cursor.execute(f'''
            SELECT fv.confidence_score, CASE WHEN fv.is_correct THEN 1.0 ELSE 0.0 END
            FROM field_validations fv
            JOIN document_validations dv ON fv.validation_id = dv.validation_id
            WHERE {where_clause}
        ''', params)
        confidence_accuracy_data = cursor.fetchall()
        
        correlation = 0.0
        if len(confidence_accuracy_data) > 1:
            confidences = [row[0] for row in confidence_accuracy_data]
            accuracies = [row[1] for row in confidence_accuracy_data]
            correlation = np.corrcoef(confidences, accuracies)[0, 1] if len(set(confidences)) > 1 else 0.0
        
        # Validator performance
        cursor.execute(f'''
            SELECT 
                validator_id,
                COUNT(*) as total_validations,
                AVG(overall_accuracy) as avg_accuracy,
                AVG(validation_duration_seconds) as avg_time
            FROM document_validations 
            WHERE {where_clause}
            GROUP BY validator_id
        ''', params)
        validator_performance = {
            row[0]: {
                'total_validations': row[1],
                'average_accuracy': row[2],
                'average_time_seconds': row[3]
            } for row in cursor.fetchall()
        }
        
        # Accuracy trends (last 30 days)
        thirty_days_ago = (datetime.now() - timedelta(days=30)).isoformat()
        cursor.execute(f'''
            SELECT 
                DATE(created_at) as date,
                AVG(overall_accuracy) as avg_accuracy,
                COUNT(*) as validation_count
            FROM document_validations 
            WHERE created_at >= ? AND {where_clause}
            GROUP BY DATE(created_at)
            ORDER BY date
        ''', [thirty_days_ago] + params)
        accuracy_trends = [{
            'date': row[0],
            'average_accuracy': row[1],
            'validation_count': row[2]
        } for row in cursor.fetchall()]
        
        # Improvement areas
        improvement_areas = []
        if overall_accuracy < 0.9:
            improvement_areas.append("Overall accuracy below 90%")
        
        low_accuracy_fields = [field for field, acc in accuracy_by_field.items() if acc < 0.8]
        if low_accuracy_fields:
            improvement_areas.append(f"Low accuracy fields: {', '.join(low_accuracy_fields)}")
        
        if correlation < 0.5:
            improvement_areas.append("Poor confidence-accuracy correlation")
        
        conn.close()
        
        return ValidationMetrics(
            total_documents_validated=total_documents,
            total_fields_validated=total_fields,
            overall_accuracy_percentage=overall_accuracy * 100,
            accuracy_by_document_type=accuracy_by_doc_type,
            accuracy_by_field_type=accuracy_by_field,
            accuracy_by_quality_level=accuracy_by_quality,
            common_error_patterns=common_errors,
            validation_time_metrics=time_metrics,
            confidence_vs_accuracy_correlation=correlation,
            improvement_areas=improvement_areas,
            validator_performance=validator_performance,
            accuracy_trends=accuracy_trends
        )
    
    def generate_validation_report(self, output_file: str = None, **kwargs) -> str:
        """Generate comprehensive validation report."""
        metrics = self.get_validation_metrics(**kwargs)
        
        output_file = output_file or f"validation_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        
        report_data = {
            'report_metadata': {
                'generated_at': datetime.now().isoformat(),
                'report_type': 'accuracy_validation',
                'filters': kwargs
            },
            'metrics': metrics.__dict__,
            'summary': {
                'total_documents': metrics.total_documents_validated,
                'overall_accuracy': f"{metrics.overall_accuracy_percentage:.2f}%",
                'accuracy_level': self._get_accuracy_level_description(metrics.overall_accuracy_percentage / 100),
                'top_improvement_areas': metrics.improvement_areas[:3]
            }
        }
        
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(report_data, f, indent=2, ensure_ascii=False)
        
        self.logger.info(f"Validation report generated: {output_file}")
        return output_file
    
    def _get_accuracy_level_description(self, accuracy: float) -> str:
        """Get accuracy level description."""
        if accuracy >= 0.95:
            return "Excellent"
        elif accuracy >= 0.85:
            return "Good"
        elif accuracy >= 0.70:
            return "Fair"
        else:
            return "Needs Improvement"


# Pytest fixtures and test functions
@pytest.fixture
def validation_system(tmp_path):
    """Fixture for validation system."""
    db_path = tmp_path / "test_validation.db"
    data_dir = tmp_path / "validation_data"
    return AccuracyValidationSystem(database_path=str(db_path), validation_data_dir=str(data_dir))


@pytest.fixture
def sample_ground_truth():
    """Sample ground truth data."""
    return {
        'ic_number': '123456-78-9012',
        'name': 'Ahmad Bin Abdullah',
        'date_of_birth': '15/06/1985',
        'gender': 'Lelaki',
        'address': 'No. 123, Jalan Merdeka, 50000 Kuala Lumpur',
        'nationality': 'Warganegara',
        'religion': 'Islam'
    }


@pytest.fixture
def sample_processing_result():
    """Sample processing result."""
    return {
        'ic_number': {'value': '123456-78-9012', 'confidence': 0.95},
        'name': {'value': 'Ahmad Bin Abdullah', 'confidence': 0.92},
        'date_of_birth': {'value': '15/06/1985', 'confidence': 0.88},
        'gender': {'value': 'Lelaki', 'confidence': 0.98},
        'address': {'value': 'No. 123, Jalan Merdeka, 50000 Kuala Lumpur', 'confidence': 0.85},
        'nationality': {'value': 'Warganegara', 'confidence': 0.90},
        'religion': {'value': 'Islam', 'confidence': 0.93}
    }


class TestAccuracyValidation:
    """Test class for accuracy validation system."""
    
    @pytest.mark.user_acceptance
    @pytest.mark.accuracy
    def test_ground_truth_management(self, validation_system, sample_ground_truth):
        """Test ground truth data management."""
        # Add ground truth
        success = validation_system.add_ground_truth(
            document_id="test_doc_001",
            document_type="mykad",
            document_path="/path/to/test_doc.jpg",
            ground_truth_data=sample_ground_truth,
            verified_by="test_validator",
            quality_score=0.9
        )
        
        assert success
        assert "test_doc_001" in validation_system.ground_truth_data
        assert validation_system.ground_truth_data["test_doc_001"]["ic_number"] == "123456-78-9012"
    
    @pytest.mark.user_acceptance
    @pytest.mark.accuracy
    def test_document_validation(self, validation_system, sample_ground_truth, sample_processing_result):
        """Test document validation process."""
        # Add ground truth
        validation_system.add_ground_truth(
            document_id="test_doc_002",
            document_type="mykad",
            document_path="/path/to/test_doc.jpg",
            ground_truth_data=sample_ground_truth,
            verified_by="test_validator"
        )
        
        # Validate document
        validation_id = validation_system.validate_document(
            document_id="test_doc_002",
            document_type="mykad",
            document_path="/path/to/test_doc.jpg",
            processing_result=sample_processing_result,
            validator_id="test_validator",
            processing_time_ms=1500.0
        )
        
        assert validation_id.startswith("VAL_")
    
    @pytest.mark.user_acceptance
    @pytest.mark.accuracy
    def test_field_validation_accuracy(self, validation_system):
        """Test field validation accuracy calculations."""
        # Test exact match
        assert validation_system._compare_field_values(
            "123456-78-9012", "123456-78-9012", FieldType.IC_NUMBER
        )
        
        # Test IC number normalization
        assert validation_system._compare_field_values(
            "12345678 9012", "123456-78-9012", FieldType.IC_NUMBER
        )
        
        # Test name matching
        assert validation_system._compare_field_values(
            "Ahmad Bin Abdullah", "Ahmad Bin Abdullah", FieldType.NAME
        )
        
        # Test date normalization
        assert validation_system._compare_field_values(
            "15/06/1985", "15-06-1985", FieldType.DATE_OF_BIRTH
        )
    
    @pytest.mark.user_acceptance
    @pytest.mark.accuracy
    def test_validation_metrics(self, validation_system, sample_ground_truth, sample_processing_result):
        """Test validation metrics calculation."""
        # Add ground truth and validate multiple documents
        for i in range(5):
            doc_id = f"test_doc_{i:03d}"
            validation_system.add_ground_truth(
                document_id=doc_id,
                document_type="mykad",
                document_path=f"/path/to/{doc_id}.jpg",
                ground_truth_data=sample_ground_truth,
                verified_by="test_validator"
            )
            
            # Introduce some errors for testing
            result = sample_processing_result.copy()
            if i % 2 == 0:  # Introduce errors in every other document
                result['name']['value'] = "Wrong Name"
            
            validation_system.validate_document(
                document_id=doc_id,
                document_type="mykad",
                document_path=f"/path/to/{doc_id}.jpg",
                processing_result=result,
                validator_id="test_validator"
            )
        
        # Get metrics
        metrics = validation_system.get_validation_metrics()
        
        assert metrics.total_documents_validated == 5
        assert metrics.total_fields_validated > 0
        assert 0 <= metrics.overall_accuracy_percentage <= 100
        assert "mykad" in metrics.accuracy_by_document_type
    
    @pytest.mark.user_acceptance
    @pytest.mark.accuracy
    def test_error_categorization(self, validation_system):
        """Test error categorization functionality."""
        # Create field validations with different error types
        field_validations = [
            FieldValidation(
                field_type=FieldType.IC_NUMBER,
                extracted_value="",  # Missing extraction
                expected_value="123456-78-9012",
                is_correct=False,
                confidence_score=0.1,
                validator_id="test"
            ),
            FieldValidation(
                field_type=FieldType.NAME,
                extracted_value="Ahmad Bin Abdulloh",  # OCR error
                expected_value="Ahmad Bin Abdullah",
                is_correct=False,
                confidence_score=0.8,
                validator_id="test"
            ),
            FieldValidation(
                field_type=FieldType.ADDRESS,
                extracted_value="No 123 Jalan Merdeka",  # Formatting error
                expected_value="No. 123, Jalan Merdeka, 50000 Kuala Lumpur",
                is_correct=False,
                confidence_score=0.7,
                validator_id="test"
            )
        ]
        
        error_categories = validation_system._identify_error_categories(field_validations)
        
        assert "field_extraction" in error_categories  # Missing IC number
        assert "ocr_accuracy" in error_categories  # Name OCR error
        assert "field_formatting" in error_categories  # Address formatting
    
    @pytest.mark.user_acceptance
    @pytest.mark.accuracy
    def test_improvement_suggestions(self, validation_system):
        """Test improvement suggestion generation."""
        field_validations = [
            FieldValidation(
                field_type=FieldType.IC_NUMBER,
                extracted_value="wrong_ic",
                expected_value="123456-78-9012",
                is_correct=False,
                confidence_score=0.3,
                validator_id="test"
            )
        ]
        
        quality_assessment = {
            'overall_quality_score': 0.4,
            'brightness_score': 0.3,
            'contrast_score': 0.5
        }
        
        error_categories = ["ocr_accuracy", "ic_number_errors"]
        
        suggestions = validation_system._generate_improvement_suggestions(
            field_validations, quality_assessment, error_categories
        )
        
        assert len(suggestions) > 0
        assert any("quality" in suggestion.lower() for suggestion in suggestions)
        assert any("ic number" in suggestion.lower() for suggestion in suggestions)
    
    @pytest.mark.user_acceptance
    @pytest.mark.accuracy
    def test_validation_report_generation(self, validation_system, sample_ground_truth, 
                                        sample_processing_result, tmp_path):
        """Test validation report generation."""
        # Add some validation data
        validation_system.add_ground_truth(
            document_id="report_test_001",
            document_type="mykad",
            document_path="/path/to/test.jpg",
            ground_truth_data=sample_ground_truth,
            verified_by="test_validator"
        )
        
        validation_system.validate_document(
            document_id="report_test_001",
            document_type="mykad",
            document_path="/path/to/test.jpg",
            processing_result=sample_processing_result,
            validator_id="test_validator"
        )
        
        # Generate report
        report_file = validation_system.generate_validation_report(
            str(tmp_path / "test_validation_report.json")
        )
        
        assert Path(report_file).exists()
        
        # Verify report content
        with open(report_file, 'r') as f:
            report_data = json.load(f)
        
        assert 'report_metadata' in report_data
        assert 'metrics' in report_data
        assert 'summary' in report_data
        assert report_data['metrics']['total_documents_validated'] >= 1
    
    @pytest.mark.user_acceptance
    @pytest.mark.accuracy
    @pytest.mark.performance
    def test_validation_performance(self, validation_system, sample_ground_truth, sample_processing_result):
        """Test validation system performance."""
        # Add ground truth
        validation_system.add_ground_truth(
            document_id="perf_test_001",
            document_type="mykad",
            document_path="/path/to/test.jpg",
            ground_truth_data=sample_ground_truth,
            verified_by="test_validator"
        )
        
        # Measure validation time
        start_time = time.time()
        
        validation_id = validation_system.validate_document(
            document_id="perf_test_001",
            document_type="mykad",
            document_path="/path/to/test.jpg",
            processing_result=sample_processing_result,
            validator_id="test_validator"
        )
        
        validation_time = time.time() - start_time
        
        # Validation should complete within reasonable time
        assert validation_time < 5.0  # 5 seconds max
        assert validation_id is not None


if __name__ == "__main__":
    # Example usage
    validation_system = AccuracyValidationSystem()
    
    # Add sample ground truth
    sample_ground_truth = {
        'ic_number': '123456-78-9012',
        'name': 'Ahmad Bin Abdullah',
        'date_of_birth': '15/06/1985',
        'gender': 'Lelaki',
        'address': 'No. 123, Jalan Merdeka, 50000 Kuala Lumpur',
        'nationality': 'Warganegara',
        'religion': 'Islam'
    }
    
    validation_system.add_ground_truth(
        document_id="example_001",
        document_type="mykad",
        document_path="/path/to/example.jpg",
        ground_truth_data=sample_ground_truth,
        verified_by="manual_validator",
        quality_score=0.9
    )
    
    # Sample processing result
    sample_processing_result = {
        'ic_number': {'value': '123456-78-9012', 'confidence': 0.95},
        'name': {'value': 'Ahmad Bin Abdullah', 'confidence': 0.92},
        'date_of_birth': {'value': '15/06/1985', 'confidence': 0.88},
        'gender': {'value': 'Lelaki', 'confidence': 0.98},
        'address': {'value': 'No. 123, Jalan Merdeka, 50000 Kuala Lumpur', 'confidence': 0.85},
        'nationality': {'value': 'Warganegara', 'confidence': 0.90},
        'religion': {'value': 'Islam', 'confidence': 0.93}
    }
    
    # Validate document
    validation_id = validation_system.validate_document(
        document_id="example_001",
        document_type="mykad",
        document_path="/path/to/example.jpg",
        processing_result=sample_processing_result,
        validator_id="manual_validator",
        processing_time_ms=1200.0
    )
    
    print(f"Validation completed: {validation_id}")
    
    # Generate metrics report
    metrics = validation_system.get_validation_metrics()
    print(f"Overall accuracy: {metrics.overall_accuracy_percentage:.2f}%")
    
    # Generate detailed report
    report_file = validation_system.generate_validation_report()
    print(f"Detailed report saved to: {report_file}")
    
    print("Accuracy validation system example completed successfully!")