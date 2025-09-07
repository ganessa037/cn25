"""Model Evaluation Framework

This module provides comprehensive evaluation metrics and testing framework for document processing models,
including character-level, word-level, and field-level accuracy metrics with test sets and error analysis.
Follows the autocorrect model's organizational patterns.
"""

import re
import json
import logging
import statistics
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional, Any, Union, Set, Callable
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from collections import defaultdict, Counter
import math

import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from sklearn.metrics import classification_report, mean_squared_error, mean_absolute_error
from difflib import SequenceMatcher
from Levenshtein import distance as levenshtein_distance
from fuzzywuzzy import fuzz
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class MetricType(Enum):
    """Types of evaluation metrics"""
    CHARACTER_LEVEL = "character_level"
    WORD_LEVEL = "word_level"
    FIELD_LEVEL = "field_level"
    DOCUMENT_LEVEL = "document_level"
    CONFIDENCE_LEVEL = "confidence_level"

class ErrorType(Enum):
    """Types of errors in document processing"""
    SUBSTITUTION = "substitution"
    INSERTION = "insertion"
    DELETION = "deletion"
    TRANSPOSITION = "transposition"
    MISSING_FIELD = "missing_field"
    EXTRA_FIELD = "extra_field"
    FORMAT_ERROR = "format_error"
    CONFIDENCE_ERROR = "confidence_error"

class DocumentType(Enum):
    """Document types for evaluation"""
    IDENTITY_CARD = "identity_card"
    PASSPORT = "passport"
    DRIVING_LICENSE = "driving_license"
    VEHICLE_REGISTRATION = "vehicle_registration"
    BANK_STATEMENT = "bank_statement"
    INVOICE = "invoice"
    RECEIPT = "receipt"
    UTILITY_BILL = "utility_bill"

@dataclass
class EvaluationResult:
    """Result of model evaluation"""
    
    metric_type: MetricType
    metric_name: str
    score: float
    
    # Detailed metrics
    accuracy: Optional[float] = None
    precision: Optional[float] = None
    recall: Optional[float] = None
    f1_score: Optional[float] = None
    
    # Error analysis
    error_count: int = 0
    error_rate: float = 0.0
    error_types: Dict[ErrorType, int] = field(default_factory=dict)
    
    # Confidence metrics
    confidence_accuracy: Optional[float] = None
    confidence_correlation: Optional[float] = None
    
    # Additional metadata
    sample_count: int = 0
    processing_time: Optional[float] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return {
            'metric_type': self.metric_type.value,
            'metric_name': self.metric_name,
            'score': self.score,
            'accuracy': self.accuracy,
            'precision': self.precision,
            'recall': self.recall,
            'f1_score': self.f1_score,
            'error_count': self.error_count,
            'error_rate': self.error_rate,
            'error_types': {k.value: v for k, v in self.error_types.items()},
            'confidence_accuracy': self.confidence_accuracy,
            'confidence_correlation': self.confidence_correlation,
            'sample_count': self.sample_count,
            'processing_time': self.processing_time,
            'metadata': self.metadata
        }

@dataclass
class TestCase:
    """Individual test case for evaluation"""
    
    test_id: str
    document_type: DocumentType
    
    # Ground truth data
    ground_truth: Dict[str, Any]
    
    # Model predictions
    predictions: Dict[str, Any]
    
    # Confidence scores
    confidence_scores: Dict[str, float] = field(default_factory=dict)
    
    # Processing metadata
    processing_time: Optional[float] = None
    model_version: Optional[str] = None
    
    # Additional metadata
    metadata: Dict[str, Any] = field(default_factory=dict)

@dataclass
class TestSet:
    """Collection of test cases"""
    
    name: str
    description: str
    document_type: Optional[DocumentType] = None
    
    test_cases: List[TestCase] = field(default_factory=list)
    
    # Test set metadata
    created_at: datetime = field(default_factory=datetime.now)
    version: str = "1.0"
    tags: List[str] = field(default_factory=list)
    
    def add_test_case(self, test_case: TestCase):
        """Add test case to set"""
        self.test_cases.append(test_case)
    
    def filter_by_document_type(self, document_type: DocumentType) -> 'TestSet':
        """Filter test cases by document type"""
        filtered_cases = [tc for tc in self.test_cases if tc.document_type == document_type]
        
        filtered_set = TestSet(
            name=f"{self.name}_{document_type.value}",
            description=f"{self.description} (filtered for {document_type.value})",
            document_type=document_type
        )
        filtered_set.test_cases = filtered_cases
        
        return filtered_set
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get test set statistics"""
        doc_type_counts = Counter(tc.document_type for tc in self.test_cases)
        
        return {
            'total_cases': len(self.test_cases),
            'document_types': dict(doc_type_counts),
            'avg_processing_time': np.mean([tc.processing_time for tc in self.test_cases if tc.processing_time]),
            'created_at': self.created_at.isoformat(),
            'version': self.version,
            'tags': self.tags
        }

class CharacterLevelEvaluator:
    """Evaluator for character-level accuracy"""
    
    def __init__(self):
        self.name = "Character Level Evaluator"
    
    def evaluate(self, ground_truth: str, prediction: str) -> EvaluationResult:
        """Evaluate character-level accuracy"""
        if not ground_truth or not prediction:
            return EvaluationResult(
                metric_type=MetricType.CHARACTER_LEVEL,
                metric_name="character_accuracy",
                score=0.0,
                sample_count=1
            )
        
        # Calculate character-level metrics
        char_accuracy = self._calculate_character_accuracy(ground_truth, prediction)
        edit_distance = levenshtein_distance(ground_truth, prediction)
        normalized_edit_distance = edit_distance / max(len(ground_truth), len(prediction))
        
        # Error analysis
        error_analysis = self._analyze_character_errors(ground_truth, prediction)
        
        result = EvaluationResult(
            metric_type=MetricType.CHARACTER_LEVEL,
            metric_name="character_accuracy",
            score=char_accuracy,
            accuracy=char_accuracy,
            error_count=edit_distance,
            error_rate=normalized_edit_distance,
            error_types=error_analysis['error_types'],
            sample_count=1,
            metadata={
                'edit_distance': edit_distance,
                'normalized_edit_distance': normalized_edit_distance,
                'ground_truth_length': len(ground_truth),
                'prediction_length': len(prediction),
                'error_details': error_analysis['details']
            }
        )
        
        return result
    
    def _calculate_character_accuracy(self, ground_truth: str, prediction: str) -> float:
        """Calculate character-level accuracy"""
        if not ground_truth and not prediction:
            return 1.0
        
        if not ground_truth or not prediction:
            return 0.0
        
        # Use sequence matcher for alignment
        matcher = SequenceMatcher(None, ground_truth, prediction)
        matching_chars = sum(match.size for match in matcher.get_matching_blocks())
        
        total_chars = max(len(ground_truth), len(prediction))
        return matching_chars / total_chars if total_chars > 0 else 0.0
    
    def _analyze_character_errors(self, ground_truth: str, prediction: str) -> Dict[str, Any]:
        """Analyze character-level errors"""
        error_types = defaultdict(int)
        error_details = []
        
        # Use sequence matcher to find differences
        matcher = SequenceMatcher(None, ground_truth, prediction)
        
        for tag, i1, i2, j1, j2 in matcher.get_opcodes():
            if tag == 'replace':
                error_types[ErrorType.SUBSTITUTION] += max(i2 - i1, j2 - j1)
                error_details.append({
                    'type': 'substitution',
                    'position': i1,
                    'expected': ground_truth[i1:i2],
                    'actual': prediction[j1:j2]
                })
            elif tag == 'delete':
                error_types[ErrorType.DELETION] += i2 - i1
                error_details.append({
                    'type': 'deletion',
                    'position': i1,
                    'expected': ground_truth[i1:i2],
                    'actual': ''
                })
            elif tag == 'insert':
                error_types[ErrorType.INSERTION] += j2 - j1
                error_details.append({
                    'type': 'insertion',
                    'position': i1,
                    'expected': '',
                    'actual': prediction[j1:j2]
                })
        
        return {
            'error_types': error_types,
            'details': error_details
        }

class WordLevelEvaluator:
    """Evaluator for word-level accuracy"""
    
    def __init__(self):
        self.name = "Word Level Evaluator"
    
    def evaluate(self, ground_truth: str, prediction: str) -> EvaluationResult:
        """Evaluate word-level accuracy"""
        if not ground_truth or not prediction:
            return EvaluationResult(
                metric_type=MetricType.WORD_LEVEL,
                metric_name="word_accuracy",
                score=0.0,
                sample_count=1
            )
        
        # Tokenize into words
        gt_words = self._tokenize_words(ground_truth)
        pred_words = self._tokenize_words(prediction)
        
        # Calculate word-level metrics
        word_accuracy = self._calculate_word_accuracy(gt_words, pred_words)
        word_precision, word_recall, word_f1 = self._calculate_word_prf(gt_words, pred_words)
        
        # Error analysis
        error_analysis = self._analyze_word_errors(gt_words, pred_words)
        
        result = EvaluationResult(
            metric_type=MetricType.WORD_LEVEL,
            metric_name="word_accuracy",
            score=word_accuracy,
            accuracy=word_accuracy,
            precision=word_precision,
            recall=word_recall,
            f1_score=word_f1,
            error_count=error_analysis['total_errors'],
            error_rate=error_analysis['error_rate'],
            error_types=error_analysis['error_types'],
            sample_count=1,
            metadata={
                'ground_truth_words': len(gt_words),
                'prediction_words': len(pred_words),
                'exact_matches': error_analysis['exact_matches'],
                'fuzzy_matches': error_analysis['fuzzy_matches'],
                'error_details': error_analysis['details']
            }
        )
        
        return result
    
    def _tokenize_words(self, text: str) -> List[str]:
        """Tokenize text into words"""
        # Simple word tokenization (can be enhanced with more sophisticated methods)
        words = re.findall(r'\b\w+\b', text.lower())
        return words
    
    def _calculate_word_accuracy(self, gt_words: List[str], pred_words: List[str]) -> float:
        """Calculate word-level accuracy"""
        if not gt_words and not pred_words:
            return 1.0
        
        if not gt_words or not pred_words:
            return 0.0
        
        # Use sequence matcher for word alignment
        matcher = SequenceMatcher(None, gt_words, pred_words)
        matching_words = sum(match.size for match in matcher.get_matching_blocks())
        
        total_words = max(len(gt_words), len(pred_words))
        return matching_words / total_words if total_words > 0 else 0.0
    
    def _calculate_word_prf(self, gt_words: List[str], pred_words: List[str]) -> Tuple[float, float, float]:
        """Calculate word-level precision, recall, and F1"""
        if not gt_words and not pred_words:
            return 1.0, 1.0, 1.0
        
        if not gt_words:
            return 0.0, 1.0, 0.0
        
        if not pred_words:
            return 1.0, 0.0, 0.0
        
        gt_set = set(gt_words)
        pred_set = set(pred_words)
        
        true_positives = len(gt_set & pred_set)
        false_positives = len(pred_set - gt_set)
        false_negatives = len(gt_set - pred_set)
        
        precision = true_positives / (true_positives + false_positives) if (true_positives + false_positives) > 0 else 0.0
        recall = true_positives / (true_positives + false_negatives) if (true_positives + false_negatives) > 0 else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
        
        return precision, recall, f1
    
    def _analyze_word_errors(self, gt_words: List[str], pred_words: List[str]) -> Dict[str, Any]:
        """Analyze word-level errors"""
        error_types = defaultdict(int)
        error_details = []
        exact_matches = 0
        fuzzy_matches = 0
        
        # Use sequence matcher to find differences
        matcher = SequenceMatcher(None, gt_words, pred_words)
        
        for tag, i1, i2, j1, j2 in matcher.get_opcodes():
            if tag == 'equal':
                exact_matches += i2 - i1
            elif tag == 'replace':
                # Check for fuzzy matches
                for i in range(i1, i2):
                    for j in range(j1, j2):
                        if i - i1 == j - j1:  # Aligned positions
                            similarity = fuzz.ratio(gt_words[i], pred_words[j])
                            if similarity >= 80:  # Fuzzy match threshold
                                fuzzy_matches += 1
                            else:
                                error_types[ErrorType.SUBSTITUTION] += 1
                                error_details.append({
                                    'type': 'substitution',
                                    'position': i,
                                    'expected': gt_words[i],
                                    'actual': pred_words[j],
                                    'similarity': similarity
                                })
            elif tag == 'delete':
                error_types[ErrorType.DELETION] += i2 - i1
                for i in range(i1, i2):
                    error_details.append({
                        'type': 'deletion',
                        'position': i,
                        'expected': gt_words[i],
                        'actual': ''
                    })
            elif tag == 'insert':
                error_types[ErrorType.INSERTION] += j2 - j1
                for j in range(j1, j2):
                    error_details.append({
                        'type': 'insertion',
                        'position': j,
                        'expected': '',
                        'actual': pred_words[j]
                    })
        
        total_errors = sum(error_types.values())
        total_words = max(len(gt_words), len(pred_words))
        error_rate = total_errors / total_words if total_words > 0 else 0.0
        
        return {
            'error_types': error_types,
            'total_errors': total_errors,
            'error_rate': error_rate,
            'exact_matches': exact_matches,
            'fuzzy_matches': fuzzy_matches,
            'details': error_details
        }

class FieldLevelEvaluator:
    """Evaluator for field-level accuracy"""
    
    def __init__(self):
        self.name = "Field Level Evaluator"
        
        # Field-specific evaluation strategies
        self.field_evaluators = {
            'ic_number': self._evaluate_ic_number,
            'phone_number': self._evaluate_phone_number,
            'email': self._evaluate_email,
            'date': self._evaluate_date,
            'amount': self._evaluate_amount,
            'address': self._evaluate_address,
            'default': self._evaluate_text_field
        }
    
    def evaluate(self, ground_truth: Dict[str, Any], predictions: Dict[str, Any], 
                confidence_scores: Dict[str, float] = None) -> List[EvaluationResult]:
        """Evaluate field-level accuracy"""
        results = []
        confidence_scores = confidence_scores or {}
        
        # Get all unique field names
        all_fields = set(ground_truth.keys()) | set(predictions.keys())
        
        for field_name in all_fields:
            gt_value = ground_truth.get(field_name)
            pred_value = predictions.get(field_name)
            confidence = confidence_scores.get(field_name, 0.0)
            
            # Evaluate field
            field_result = self._evaluate_field(field_name, gt_value, pred_value, confidence)
            results.append(field_result)
        
        return results
    
    def _evaluate_field(self, field_name: str, ground_truth: Any, prediction: Any, 
                       confidence: float) -> EvaluationResult:
        """Evaluate individual field"""
        # Determine evaluation strategy
        evaluator = self.field_evaluators.get('default')
        
        for pattern, eval_func in self.field_evaluators.items():
            if pattern != 'default' and pattern in field_name.lower():
                evaluator = eval_func
                break
        
        # Evaluate field
        result = evaluator(field_name, ground_truth, prediction, confidence)
        
        return result
    
    def _evaluate_ic_number(self, field_name: str, ground_truth: Any, prediction: Any, 
                           confidence: float) -> EvaluationResult:
        """Evaluate IC number field"""
        gt_str = str(ground_truth).strip() if ground_truth else ''
        pred_str = str(prediction).strip() if prediction else ''
        
        # Normalize IC numbers (remove dashes, spaces)
        gt_normalized = re.sub(r'[-\s]', '', gt_str)
        pred_normalized = re.sub(r'[-\s]', '', pred_str)
        
        # Exact match
        exact_match = gt_normalized == pred_normalized
        
        # Calculate similarity
        similarity = fuzz.ratio(gt_normalized, pred_normalized) / 100.0
        
        # Error analysis
        error_types = defaultdict(int)
        if not exact_match:
            if not pred_str:
                error_types[ErrorType.MISSING_FIELD] = 1
            elif not gt_str:
                error_types[ErrorType.EXTRA_FIELD] = 1
            else:
                error_types[ErrorType.SUBSTITUTION] = 1
        
        result = EvaluationResult(
            metric_type=MetricType.FIELD_LEVEL,
            metric_name=f"{field_name}_accuracy",
            score=1.0 if exact_match else similarity,
            accuracy=1.0 if exact_match else 0.0,
            error_count=0 if exact_match else 1,
            error_rate=0.0 if exact_match else 1.0,
            error_types=error_types,
            confidence_accuracy=1.0 if (confidence > 0.8 and exact_match) or (confidence <= 0.8 and not exact_match) else 0.0,
            sample_count=1,
            metadata={
                'exact_match': exact_match,
                'similarity': similarity,
                'ground_truth': gt_str,
                'prediction': pred_str,
                'confidence': confidence
            }
        )
        
        return result
    
    def _evaluate_phone_number(self, field_name: str, ground_truth: Any, prediction: Any, 
                              confidence: float) -> EvaluationResult:
        """Evaluate phone number field"""
        gt_str = str(ground_truth).strip() if ground_truth else ''
        pred_str = str(prediction).strip() if prediction else ''
        
        # Normalize phone numbers (remove spaces, dashes, plus signs)
        gt_normalized = re.sub(r'[\s\-\+]', '', gt_str)
        pred_normalized = re.sub(r'[\s\-\+]', '', pred_str)
        
        # Remove country code variations
        gt_normalized = re.sub(r'^(60|6)', '', gt_normalized)
        pred_normalized = re.sub(r'^(60|6)', '', pred_normalized)
        
        exact_match = gt_normalized == pred_normalized
        similarity = fuzz.ratio(gt_normalized, pred_normalized) / 100.0
        
        error_types = defaultdict(int)
        if not exact_match:
            if not pred_str:
                error_types[ErrorType.MISSING_FIELD] = 1
            elif not gt_str:
                error_types[ErrorType.EXTRA_FIELD] = 1
            else:
                error_types[ErrorType.SUBSTITUTION] = 1
        
        result = EvaluationResult(
            metric_type=MetricType.FIELD_LEVEL,
            metric_name=f"{field_name}_accuracy",
            score=1.0 if exact_match else similarity,
            accuracy=1.0 if exact_match else 0.0,
            error_count=0 if exact_match else 1,
            error_rate=0.0 if exact_match else 1.0,
            error_types=error_types,
            confidence_accuracy=1.0 if (confidence > 0.8 and exact_match) or (confidence <= 0.8 and not exact_match) else 0.0,
            sample_count=1,
            metadata={
                'exact_match': exact_match,
                'similarity': similarity,
                'ground_truth': gt_str,
                'prediction': pred_str,
                'confidence': confidence
            }
        )
        
        return result
    
    def _evaluate_email(self, field_name: str, ground_truth: Any, prediction: Any, 
                       confidence: float) -> EvaluationResult:
        """Evaluate email field"""
        gt_str = str(ground_truth).strip().lower() if ground_truth else ''
        pred_str = str(prediction).strip().lower() if prediction else ''
        
        exact_match = gt_str == pred_str
        similarity = fuzz.ratio(gt_str, pred_str) / 100.0
        
        error_types = defaultdict(int)
        if not exact_match:
            if not pred_str:
                error_types[ErrorType.MISSING_FIELD] = 1
            elif not gt_str:
                error_types[ErrorType.EXTRA_FIELD] = 1
            else:
                error_types[ErrorType.SUBSTITUTION] = 1
        
        result = EvaluationResult(
            metric_type=MetricType.FIELD_LEVEL,
            metric_name=f"{field_name}_accuracy",
            score=1.0 if exact_match else similarity,
            accuracy=1.0 if exact_match else 0.0,
            error_count=0 if exact_match else 1,
            error_rate=0.0 if exact_match else 1.0,
            error_types=error_types,
            confidence_accuracy=1.0 if (confidence > 0.8 and exact_match) or (confidence <= 0.8 and not exact_match) else 0.0,
            sample_count=1,
            metadata={
                'exact_match': exact_match,
                'similarity': similarity,
                'ground_truth': gt_str,
                'prediction': pred_str,
                'confidence': confidence
            }
        )
        
        return result
    
    def _evaluate_date(self, field_name: str, ground_truth: Any, prediction: Any, 
                      confidence: float) -> EvaluationResult:
        """Evaluate date field"""
        from dateutil import parser as date_parser
        
        gt_str = str(ground_truth).strip() if ground_truth else ''
        pred_str = str(prediction).strip() if prediction else ''
        
        # Try to parse dates
        gt_date = None
        pred_date = None
        
        try:
            gt_date = date_parser.parse(gt_str, dayfirst=True).date() if gt_str else None
        except:
            pass
        
        try:
            pred_date = date_parser.parse(pred_str, dayfirst=True).date() if pred_str else None
        except:
            pass
        
        # Compare dates
        exact_match = gt_date == pred_date if gt_date and pred_date else gt_str == pred_str
        
        # Calculate similarity based on string if date parsing fails
        if gt_date and pred_date:
            # Date-specific similarity (closer dates have higher similarity)
            if gt_date == pred_date:
                similarity = 1.0
            else:
                days_diff = abs((gt_date - pred_date).days)
                similarity = max(0.0, 1.0 - days_diff / 365.0)  # Normalize by year
        else:
            similarity = fuzz.ratio(gt_str, pred_str) / 100.0
        
        error_types = defaultdict(int)
        if not exact_match:
            if not pred_str:
                error_types[ErrorType.MISSING_FIELD] = 1
            elif not gt_str:
                error_types[ErrorType.EXTRA_FIELD] = 1
            else:
                error_types[ErrorType.SUBSTITUTION] = 1
        
        result = EvaluationResult(
            metric_type=MetricType.FIELD_LEVEL,
            metric_name=f"{field_name}_accuracy",
            score=1.0 if exact_match else similarity,
            accuracy=1.0 if exact_match else 0.0,
            error_count=0 if exact_match else 1,
            error_rate=0.0 if exact_match else 1.0,
            error_types=error_types,
            confidence_accuracy=1.0 if (confidence > 0.8 and exact_match) or (confidence <= 0.8 and not exact_match) else 0.0,
            sample_count=1,
            metadata={
                'exact_match': exact_match,
                'similarity': similarity,
                'ground_truth': gt_str,
                'prediction': pred_str,
                'ground_truth_parsed': gt_date.isoformat() if gt_date else None,
                'prediction_parsed': pred_date.isoformat() if pred_date else None,
                'confidence': confidence
            }
        )
        
        return result
    
    def _evaluate_amount(self, field_name: str, ground_truth: Any, prediction: Any, 
                        confidence: float) -> EvaluationResult:
        """Evaluate amount/monetary field"""
        gt_str = str(ground_truth).strip() if ground_truth else ''
        pred_str = str(prediction).strip() if prediction else ''
        
        # Extract numeric values
        gt_numeric = self._extract_numeric_value(gt_str)
        pred_numeric = self._extract_numeric_value(pred_str)
        
        # Compare numeric values
        exact_match = gt_numeric == pred_numeric if gt_numeric is not None and pred_numeric is not None else gt_str == pred_str
        
        # Calculate similarity
        if gt_numeric is not None and pred_numeric is not None:
            if gt_numeric == 0 and pred_numeric == 0:
                similarity = 1.0
            elif gt_numeric == 0 or pred_numeric == 0:
                similarity = 0.0
            else:
                # Relative error
                relative_error = abs(gt_numeric - pred_numeric) / max(abs(gt_numeric), abs(pred_numeric))
                similarity = max(0.0, 1.0 - relative_error)
        else:
            similarity = fuzz.ratio(gt_str, pred_str) / 100.0
        
        error_types = defaultdict(int)
        if not exact_match:
            if not pred_str:
                error_types[ErrorType.MISSING_FIELD] = 1
            elif not gt_str:
                error_types[ErrorType.EXTRA_FIELD] = 1
            else:
                error_types[ErrorType.SUBSTITUTION] = 1
        
        result = EvaluationResult(
            metric_type=MetricType.FIELD_LEVEL,
            metric_name=f"{field_name}_accuracy",
            score=1.0 if exact_match else similarity,
            accuracy=1.0 if exact_match else 0.0,
            error_count=0 if exact_match else 1,
            error_rate=0.0 if exact_match else 1.0,
            error_types=error_types,
            confidence_accuracy=1.0 if (confidence > 0.8 and exact_match) or (confidence <= 0.8 and not exact_match) else 0.0,
            sample_count=1,
            metadata={
                'exact_match': exact_match,
                'similarity': similarity,
                'ground_truth': gt_str,
                'prediction': pred_str,
                'ground_truth_numeric': gt_numeric,
                'prediction_numeric': pred_numeric,
                'confidence': confidence
            }
        )
        
        return result
    
    def _evaluate_address(self, field_name: str, ground_truth: Any, prediction: Any, 
                         confidence: float) -> EvaluationResult:
        """Evaluate address field"""
        gt_str = str(ground_truth).strip() if ground_truth else ''
        pred_str = str(prediction).strip() if prediction else ''
        
        # Normalize addresses (remove extra spaces, standardize case)
        gt_normalized = ' '.join(gt_str.lower().split())
        pred_normalized = ' '.join(pred_str.lower().split())
        
        exact_match = gt_normalized == pred_normalized
        
        # Use fuzzy matching for addresses (more lenient)
        similarity = fuzz.token_sort_ratio(gt_normalized, pred_normalized) / 100.0
        
        # Consider high similarity as acceptable for addresses
        acceptable_match = similarity >= 0.8
        
        error_types = defaultdict(int)
        if not exact_match:
            if not pred_str:
                error_types[ErrorType.MISSING_FIELD] = 1
            elif not gt_str:
                error_types[ErrorType.EXTRA_FIELD] = 1
            else:
                error_types[ErrorType.SUBSTITUTION] = 1
        
        result = EvaluationResult(
            metric_type=MetricType.FIELD_LEVEL,
            metric_name=f"{field_name}_accuracy",
            score=1.0 if exact_match else similarity,
            accuracy=1.0 if acceptable_match else 0.0,  # Use acceptable match for accuracy
            error_count=0 if acceptable_match else 1,
            error_rate=0.0 if acceptable_match else 1.0,
            error_types=error_types,
            confidence_accuracy=1.0 if (confidence > 0.8 and acceptable_match) or (confidence <= 0.8 and not acceptable_match) else 0.0,
            sample_count=1,
            metadata={
                'exact_match': exact_match,
                'acceptable_match': acceptable_match,
                'similarity': similarity,
                'ground_truth': gt_str,
                'prediction': pred_str,
                'confidence': confidence
            }
        )
        
        return result
    
    def _evaluate_text_field(self, field_name: str, ground_truth: Any, prediction: Any, 
                            confidence: float) -> EvaluationResult:
        """Evaluate generic text field"""
        gt_str = str(ground_truth).strip() if ground_truth else ''
        pred_str = str(prediction).strip() if prediction else ''
        
        exact_match = gt_str == pred_str
        similarity = fuzz.ratio(gt_str, pred_str) / 100.0
        
        error_types = defaultdict(int)
        if not exact_match:
            if not pred_str:
                error_types[ErrorType.MISSING_FIELD] = 1
            elif not gt_str:
                error_types[ErrorType.EXTRA_FIELD] = 1
            else:
                error_types[ErrorType.SUBSTITUTION] = 1
        
        result = EvaluationResult(
            metric_type=MetricType.FIELD_LEVEL,
            metric_name=f"{field_name}_accuracy",
            score=1.0 if exact_match else similarity,
            accuracy=1.0 if exact_match else 0.0,
            error_count=0 if exact_match else 1,
            error_rate=0.0 if exact_match else 1.0,
            error_types=error_types,
            confidence_accuracy=1.0 if (confidence > 0.8 and exact_match) or (confidence <= 0.8 and not exact_match) else 0.0,
            sample_count=1,
            metadata={
                'exact_match': exact_match,
                'similarity': similarity,
                'ground_truth': gt_str,
                'prediction': pred_str,
                'confidence': confidence
            }
        )
        
        return result
    
    def _extract_numeric_value(self, text: str) -> Optional[float]:
        """Extract numeric value from text"""
        if not text:
            return None
        
        # Remove currency symbols and common prefixes
        cleaned = re.sub(r'[RM$€£¥₹]', '', text)
        cleaned = re.sub(r'[^\d.,\-]', '', cleaned)
        
        if not cleaned:
            return None
        
        try:
            # Handle comma as thousand separator
            if ',' in cleaned and '.' in cleaned:
                # Assume comma is thousand separator if it appears before dot
                if cleaned.rindex(',') < cleaned.rindex('.'):
                    cleaned = cleaned.replace(',', '')
                else:
                    # Assume dot is thousand separator
                    cleaned = cleaned.replace('.', '').replace(',', '.')
            elif ',' in cleaned:
                # Check if comma is decimal separator (European style)
                if len(cleaned.split(',')[-1]) <= 2:
                    cleaned = cleaned.replace(',', '.')
                else:
                    cleaned = cleaned.replace(',', '')
            
            return float(cleaned)
        except ValueError:
            return None

class DocumentLevelEvaluator:
    """Evaluator for document-level metrics"""
    
    def __init__(self):
        self.name = "Document Level Evaluator"
        self.field_evaluator = FieldLevelEvaluator()
    
    def evaluate(self, test_case: TestCase) -> EvaluationResult:
        """Evaluate document-level accuracy"""
        # Get field-level results
        field_results = self.field_evaluator.evaluate(
            test_case.ground_truth,
            test_case.predictions,
            test_case.confidence_scores
        )
        
        # Calculate document-level metrics
        total_fields = len(field_results)
        accurate_fields = sum(1 for r in field_results if r.accuracy == 1.0)
        
        document_accuracy = accurate_fields / total_fields if total_fields > 0 else 0.0
        
        # Calculate average scores
        avg_score = np.mean([r.score for r in field_results]) if field_results else 0.0
        avg_confidence = np.mean([r.metadata.get('confidence', 0.0) for r in field_results]) if field_results else 0.0
        
        # Aggregate error types
        error_types = defaultdict(int)
        for result in field_results:
            for error_type, count in result.error_types.items():
                error_types[error_type] += count
        
        total_errors = sum(error_types.values())
        
        # Calculate confidence correlation
        confidences = [r.metadata.get('confidence', 0.0) for r in field_results]
        accuracies = [r.accuracy for r in field_results]
        
        confidence_correlation = 0.0
        if len(confidences) > 1 and len(set(confidences)) > 1 and len(set(accuracies)) > 1:
            try:
                confidence_correlation, _ = stats.pearsonr(confidences, accuracies)
                if math.isnan(confidence_correlation):
                    confidence_correlation = 0.0
            except:
                confidence_correlation = 0.0
        
        result = EvaluationResult(
            metric_type=MetricType.DOCUMENT_LEVEL,
            metric_name="document_accuracy",
            score=avg_score,
            accuracy=document_accuracy,
            error_count=total_errors,
            error_rate=total_errors / total_fields if total_fields > 0 else 0.0,
            error_types=error_types,
            confidence_correlation=confidence_correlation,
            sample_count=1,
            processing_time=test_case.processing_time,
            metadata={
                'total_fields': total_fields,
                'accurate_fields': accurate_fields,
                'average_confidence': avg_confidence,
                'field_results': [r.to_dict() for r in field_results],
                'test_id': test_case.test_id,
                'document_type': test_case.document_type.value
            }
        )
        
        return result

class ErrorAnalyzer:
    """Analyzer for error patterns and common failures"""
    
    def __init__(self):
        self.name = "Error Analyzer"
    
    def analyze_errors(self, evaluation_results: List[EvaluationResult]) -> Dict[str, Any]:
        """Analyze error patterns across evaluation results"""
        error_analysis = {
            'total_samples': len(evaluation_results),
            'error_distribution': defaultdict(int),
            'error_patterns': defaultdict(list),
            'confidence_analysis': {},
            'field_analysis': defaultdict(dict),
            'common_failures': [],
            'recommendations': []
        }
        
        # Aggregate error types
        for result in evaluation_results:
            for error_type, count in result.error_types.items():
                error_analysis['error_distribution'][error_type.value] += count
        
        # Analyze confidence vs accuracy
        confidences = []
        accuracies = []
        
        for result in evaluation_results:
            if result.metadata and 'confidence' in result.metadata:
                confidences.append(result.metadata['confidence'])
                accuracies.append(result.accuracy or 0.0)
        
        if confidences and accuracies:
            error_analysis['confidence_analysis'] = self._analyze_confidence_accuracy(confidences, accuracies)
        
        # Analyze field-specific errors
        field_errors = defaultdict(list)
        for result in evaluation_results:
            if result.metric_type == MetricType.FIELD_LEVEL:
                field_name = result.metric_name.replace('_accuracy', '')
                field_errors[field_name].append(result)
        
        for field_name, field_results in field_errors.items():
            error_analysis['field_analysis'][field_name] = self._analyze_field_errors(field_results)
        
        # Identify common failure patterns
        error_analysis['common_failures'] = self._identify_common_failures(evaluation_results)
        
        # Generate recommendations
        error_analysis['recommendations'] = self._generate_recommendations(error_analysis)
        
        return error_analysis
    
    def _analyze_confidence_accuracy(self, confidences: List[float], accuracies: List[float]) -> Dict[str, Any]:
        """Analyze relationship between confidence and accuracy"""
        analysis = {}
        
        # Calculate correlation
        if len(confidences) > 1 and len(set(confidences)) > 1 and len(set(accuracies)) > 1:
            try:
                correlation, p_value = stats.pearsonr(confidences, accuracies)
                analysis['correlation'] = correlation if not math.isnan(correlation) else 0.0
                analysis['p_value'] = p_value if not math.isnan(p_value) else 1.0
            except:
                analysis['correlation'] = 0.0
                analysis['p_value'] = 1.0
        else:
            analysis['correlation'] = 0.0
            analysis['p_value'] = 1.0
        
        # Analyze confidence bins
        confidence_bins = {
            'high_confidence_high_accuracy': 0,
            'high_confidence_low_accuracy': 0,
            'low_confidence_high_accuracy': 0,
            'low_confidence_low_accuracy': 0
        }
        
        for conf, acc in zip(confidences, accuracies):
            if conf >= 0.8:
                if acc >= 0.8:
                    confidence_bins['high_confidence_high_accuracy'] += 1
                else:
                    confidence_bins['high_confidence_low_accuracy'] += 1
            else:
                if acc >= 0.8:
                    confidence_bins['low_confidence_high_accuracy'] += 1
                else:
                    confidence_bins['low_confidence_low_accuracy'] += 1
        
        analysis['confidence_bins'] = confidence_bins
        
        # Calculate calibration metrics
        analysis['calibration'] = self._calculate_calibration(confidences, accuracies)
        
        return analysis
    
    def _calculate_calibration(self, confidences: List[float], accuracies: List[float]) -> Dict[str, float]:
        """Calculate confidence calibration metrics"""
        # Bin confidences and calculate calibration
        bins = np.linspace(0, 1, 11)  # 10 bins
        bin_boundaries = [(bins[i], bins[i+1]) for i in range(len(bins)-1)]
        
        calibration_data = []
        
        for lower, upper in bin_boundaries:
            bin_indices = [i for i, conf in enumerate(confidences) if lower <= conf < upper]
            
            if bin_indices:
                bin_confidences = [confidences[i] for i in bin_indices]
                bin_accuracies = [accuracies[i] for i in bin_indices]
                
                avg_confidence = np.mean(bin_confidences)
                avg_accuracy = np.mean(bin_accuracies)
                
                calibration_data.append({
                    'bin_range': (lower, upper),
                    'count': len(bin_indices),
                    'avg_confidence': avg_confidence,
                    'avg_accuracy': avg_accuracy,
                    'calibration_error': abs(avg_confidence - avg_accuracy)
                })
        
        # Calculate Expected Calibration Error (ECE)
        total_samples = len(confidences)
        ece = sum(data['count'] * data['calibration_error'] for data in calibration_data) / total_samples
        
        # Calculate Maximum Calibration Error (MCE)
        mce = max(data['calibration_error'] for data in calibration_data) if calibration_data else 0.0
        
        return {
            'expected_calibration_error': ece,
            'maximum_calibration_error': mce,
            'calibration_data': calibration_data
        }
    
    def _analyze_field_errors(self, field_results: List[EvaluationResult]) -> Dict[str, Any]:
        """Analyze errors for specific field"""
        total_samples = len(field_results)
        accurate_samples = sum(1 for r in field_results if r.accuracy == 1.0)
        
        error_types = defaultdict(int)
        similarities = []
        confidences = []
        
        for result in field_results:
            for error_type, count in result.error_types.items():
                error_types[error_type.value] += count
            
            if result.metadata:
                if 'similarity' in result.metadata:
                    similarities.append(result.metadata['similarity'])
                if 'confidence' in result.metadata:
                    confidences.append(result.metadata['confidence'])
        
        analysis = {
            'total_samples': total_samples,
            'accurate_samples': accurate_samples,
            'accuracy_rate': accurate_samples / total_samples if total_samples > 0 else 0.0,
            'error_types': dict(error_types),
            'avg_similarity': np.mean(similarities) if similarities else 0.0,
            'avg_confidence': np.mean(confidences) if confidences else 0.0
        }
        
        return analysis
    
    def _identify_common_failures(self, evaluation_results: List[EvaluationResult]) -> List[Dict[str, Any]]:
        """Identify common failure patterns"""
        failures = []
        
        # Group by error types
        error_groups = defaultdict(list)
        for result in evaluation_results:
            if result.error_count > 0:
                for error_type in result.error_types.keys():
                    error_groups[error_type].append(result)
        
        # Analyze each error type
        for error_type, error_results in error_groups.items():
            if len(error_results) >= 3:  # Only consider patterns with multiple occurrences
                failure_pattern = {
                    'error_type': error_type.value,
                    'frequency': len(error_results),
                    'affected_fields': list(set(r.metric_name.replace('_accuracy', '') for r in error_results)),
                    'avg_confidence': np.mean([r.metadata.get('confidence', 0.0) for r in error_results if r.metadata]),
                    'examples': []
                }
                
                # Add examples
                for result in error_results[:3]:  # First 3 examples
                    if result.metadata:
                        example = {
                            'field': result.metric_name.replace('_accuracy', ''),
                            'ground_truth': result.metadata.get('ground_truth', ''),
                            'prediction': result.metadata.get('prediction', ''),
                            'confidence': result.metadata.get('confidence', 0.0)
                        }
                        failure_pattern['examples'].append(example)
                
                failures.append(failure_pattern)
        
        # Sort by frequency
        failures.sort(key=lambda x: x['frequency'], reverse=True)
        
        return failures
    
    def _generate_recommendations(self, error_analysis: Dict[str, Any]) -> List[str]:
        """Generate recommendations based on error analysis"""
        recommendations = []
        
        # Confidence-based recommendations
        if 'confidence_analysis' in error_analysis:
            conf_analysis = error_analysis['confidence_analysis']
            
            if conf_analysis.get('correlation', 0) < 0.5:
                recommendations.append("Improve confidence calibration - confidence scores don't correlate well with accuracy")
            
            bins = conf_analysis.get('confidence_bins', {})
            if bins.get('high_confidence_low_accuracy', 0) > bins.get('high_confidence_high_accuracy', 0) * 0.2:
                recommendations.append("Reduce overconfidence - many high-confidence predictions are inaccurate")
            
            if bins.get('low_confidence_high_accuracy', 0) > bins.get('low_confidence_low_accuracy', 0) * 0.2:
                recommendations.append("Increase confidence for accurate predictions - many accurate predictions have low confidence")
        
        # Error type recommendations
        error_dist = error_analysis.get('error_distribution', {})
        total_errors = sum(error_dist.values())
        
        if total_errors > 0:
            if error_dist.get('substitution', 0) / total_errors > 0.5:
                recommendations.append("Focus on OCR accuracy - substitution errors are dominant")
            
            if error_dist.get('missing_field', 0) / total_errors > 0.3:
                recommendations.append("Improve field detection - many fields are missing from predictions")
            
            if error_dist.get('extra_field', 0) / total_errors > 0.3:
                recommendations.append("Reduce false positives - many extra fields are being detected")
        
        # Field-specific recommendations
        field_analysis = error_analysis.get('field_analysis', {})
        for field_name, field_data in field_analysis.items():
            if field_data.get('accuracy_rate', 1.0) < 0.7:
                recommendations.append(f"Improve {field_name} extraction - accuracy is below 70%")
        
        # Common failure recommendations
        common_failures = error_analysis.get('common_failures', [])
        if common_failures:
            top_failure = common_failures[0]
            recommendations.append(f"Address {top_failure['error_type']} errors - most common failure pattern affecting {len(top_failure['affected_fields'])} field types")
        
        return recommendations

class ModelEvaluationFramework:
    """Main framework for model evaluation"""
    
    def __init__(self, output_dir: str = "evaluation_results"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        
        # Initialize evaluators
        self.char_evaluator = CharacterLevelEvaluator()
        self.word_evaluator = WordLevelEvaluator()
        self.field_evaluator = FieldLevelEvaluator()
        self.document_evaluator = DocumentLevelEvaluator()
        self.error_analyzer = ErrorAnalyzer()
        
        # Test sets
        self.test_sets: Dict[str, TestSet] = {}
    
    def add_test_set(self, test_set: TestSet):
        """Add test set to framework"""
        self.test_sets[test_set.name] = test_set
        logger.info(f"Added test set '{test_set.name}' with {len(test_set.test_cases)} test cases")
    
    def evaluate_test_set(self, test_set_name: str, 
                         evaluation_types: List[MetricType] = None) -> Dict[str, Any]:
        """Evaluate a test set"""
        if test_set_name not in self.test_sets:
            raise ValueError(f"Test set '{test_set_name}' not found")
        
        test_set = self.test_sets[test_set_name]
        evaluation_types = evaluation_types or [MetricType.FIELD_LEVEL, MetricType.DOCUMENT_LEVEL]
        
        logger.info(f"Evaluating test set '{test_set_name}' with {len(test_set.test_cases)} test cases")
        
        all_results = []
        
        for test_case in test_set.test_cases:
            case_results = self._evaluate_test_case(test_case, evaluation_types)
            all_results.extend(case_results)
        
        # Aggregate results
        aggregated_results = self._aggregate_results(all_results)
        
        # Error analysis
        error_analysis = self.error_analyzer.analyze_errors(all_results)
        
        # Generate report
        evaluation_report = {
            'test_set_name': test_set_name,
            'test_set_info': test_set.get_statistics(),
            'evaluation_timestamp': datetime.now().isoformat(),
            'evaluation_types': [et.value for et in evaluation_types],
            'aggregated_results': aggregated_results,
            'error_analysis': error_analysis,
            'individual_results': [r.to_dict() for r in all_results]
        }
        
        # Save report
        self._save_evaluation_report(evaluation_report, test_set_name)
        
        return evaluation_report
    
    def _evaluate_test_case(self, test_case: TestCase, 
                           evaluation_types: List[MetricType]) -> List[EvaluationResult]:
        """Evaluate individual test case"""
        results = []
        
        # Document-level evaluation
        if MetricType.DOCUMENT_LEVEL in evaluation_types:
            doc_result = self.document_evaluator.evaluate(test_case)
            results.append(doc_result)
        
        # Field-level evaluation
        if MetricType.FIELD_LEVEL in evaluation_types:
            field_results = self.field_evaluator.evaluate(
                test_case.ground_truth,
                test_case.predictions,
                test_case.confidence_scores
            )
            results.extend(field_results)
        
        # Character and word level evaluation for text fields
        if MetricType.CHARACTER_LEVEL in evaluation_types or MetricType.WORD_LEVEL in evaluation_types:
            for field_name in test_case.ground_truth.keys():
                gt_value = test_case.ground_truth.get(field_name, '')
                pred_value = test_case.predictions.get(field_name, '')
                
                if isinstance(gt_value, str) and isinstance(pred_value, str):
                    if MetricType.CHARACTER_LEVEL in evaluation_types:
                        char_result = self.char_evaluator.evaluate(gt_value, pred_value)
                        char_result.metric_name = f"{field_name}_char_accuracy"
                        results.append(char_result)
                    
                    if MetricType.WORD_LEVEL in evaluation_types:
                        word_result = self.word_evaluator.evaluate(gt_value, pred_value)
                        word_result.metric_name = f"{field_name}_word_accuracy"
                        results.append(word_result)
        
        return results
    
    def _aggregate_results(self, results: List[EvaluationResult]) -> Dict[str, Any]:
        """Aggregate evaluation results"""
        # Group by metric type
        grouped_results = defaultdict(list)
        for result in results:
            grouped_results[result.metric_type].append(result)
        
        aggregated = {}
        
        for metric_type, type_results in grouped_results.items():
            type_stats = {
                'count': len(type_results),
                'avg_score': np.mean([r.score for r in type_results]),
                'avg_accuracy': np.mean([r.accuracy for r in type_results if r.accuracy is not None]),
                'avg_precision': np.mean([r.precision for r in type_results if r.precision is not None]),
                'avg_recall': np.mean([r.recall for r in type_results if r.recall is not None]),
                'avg_f1_score': np.mean([r.f1_score for r in type_results if r.f1_score is not None]),
                'total_errors': sum(r.error_count for r in type_results),
                'avg_error_rate': np.mean([r.error_rate for r in type_results]),
                'avg_processing_time': np.mean([r.processing_time for r in type_results if r.processing_time is not None])
            }
            
            # Remove NaN values
            for key, value in type_stats.items():
                if isinstance(value, float) and math.isnan(value):
                    type_stats[key] = 0.0
            
            aggregated[metric_type.value] = type_stats
        
        return aggregated
    
    def _save_evaluation_report(self, report: Dict[str, Any], test_set_name: str):
        """Save evaluation report to file"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        report_file = self.output_dir / f"evaluation_report_{test_set_name}_{timestamp}.json"
        
        with open(report_file, 'w') as f:
            json.dump(report, f, indent=2, default=str)
        
        logger.info(f"Evaluation report saved to {report_file}")
    
    def compare_models(self, model_results: Dict[str, Dict[str, Any]]) -> Dict[str, Any]:
        """Compare multiple model evaluation results"""
        comparison = {
            'models': list(model_results.keys()),
            'comparison_timestamp': datetime.now().isoformat(),
            'metrics_comparison': {},
            'winner_analysis': {},
            'recommendations': []
        }
        
        # Extract metrics for comparison
        all_metrics = set()
        for model_name, results in model_results.items():
            if 'aggregated_results' in results:
                for metric_type, metrics in results['aggregated_results'].items():
                    for metric_name in metrics.keys():
                        all_metrics.add(f"{metric_type}.{metric_name}")
        
        # Compare each metric
        for metric in all_metrics:
            metric_comparison = {}
            metric_type, metric_name = metric.split('.', 1)
            
            for model_name, results in model_results.items():
                value = results.get('aggregated_results', {}).get(metric_type, {}).get(metric_name, 0.0)
                metric_comparison[model_name] = value
            
            # Find best model for this metric
            if metric_comparison:
                best_model = max(metric_comparison.items(), key=lambda x: x[1])
                metric_comparison['best_model'] = best_model[0]
                metric_comparison['best_value'] = best_model[1]
            
            comparison['metrics_comparison'][metric] = metric_comparison
        
        # Overall winner analysis
        model_scores = defaultdict(int)
        for metric, metric_data in comparison['metrics_comparison'].items():
            if 'best_model' in metric_data:
                model_scores[metric_data['best_model']] += 1
        
        if model_scores:
            overall_winner = max(model_scores.items(), key=lambda x: x[1])
            comparison['winner_analysis'] = {
                'overall_winner': overall_winner[0],
                'wins_count': overall_winner[1],
                'total_metrics': len(comparison['metrics_comparison']),
                'model_scores': dict(model_scores)
            }
        
        # Generate recommendations
        comparison['recommendations'] = self._generate_model_comparison_recommendations(comparison)
        
        return comparison
    
    def _generate_model_comparison_recommendations(self, comparison: Dict[str, Any]) -> List[str]:
        """Generate recommendations from model comparison"""
        recommendations = []
        
        winner_analysis = comparison.get('winner_analysis', {})
        if winner_analysis:
            winner = winner_analysis.get('overall_winner')
            wins = winner_analysis.get('wins_count', 0)
            total = winner_analysis.get('total_metrics', 1)
            
            if wins / total > 0.7:
                recommendations.append(f"Model '{winner}' shows clear superiority, winning {wins}/{total} metrics")
            elif wins / total > 0.5:
                recommendations.append(f"Model '{winner}' has slight advantage, consider ensemble approach")
            else:
                recommendations.append("No clear winner - consider ensemble or hybrid approach")
        
        # Analyze specific metric patterns
        metrics_comparison = comparison.get('metrics_comparison', {})
        accuracy_metrics = {k: v for k, v in metrics_comparison.items() if 'accuracy' in k.lower()}
        
        if accuracy_metrics:
            # Find model with best average accuracy
            model_accuracies = defaultdict(list)
            for metric_data in accuracy_metrics.values():
                for model, value in metric_data.items():
                    if model != 'best_model' and model != 'best_value':
                        model_accuracies[model].append(value)
            
            avg_accuracies = {model: np.mean(values) for model, values in model_accuracies.items()}
            if avg_accuracies:
                best_accuracy_model = max(avg_accuracies.items(), key=lambda x: x[1])
                recommendations.append(f"For accuracy-critical applications, consider '{best_accuracy_model[0]}' (avg accuracy: {best_accuracy_model[1]:.3f})")
        
        return recommendations
    
    def generate_benchmark_report(self, test_set_name: str, 
                                 baseline_results: Dict[str, Any] = None) -> Dict[str, Any]:
        """Generate comprehensive benchmark report"""
        if test_set_name not in self.test_sets:
            raise ValueError(f"Test set '{test_set_name}' not found")
        
        # Evaluate current model
        current_results = self.evaluate_test_set(test_set_name)
        
        benchmark_report = {
            'test_set_name': test_set_name,
            'benchmark_timestamp': datetime.now().isoformat(),
            'current_results': current_results,
            'baseline_comparison': {},
            'performance_analysis': {},
            'recommendations': []
        }
        
        # Compare with baseline if provided
        if baseline_results:
            comparison = self.compare_models({
                'current': current_results,
                'baseline': baseline_results
            })
            benchmark_report['baseline_comparison'] = comparison
        
        # Performance analysis
        benchmark_report['performance_analysis'] = self._analyze_performance(current_results)
        
        # Generate recommendations
        benchmark_report['recommendations'] = self._generate_benchmark_recommendations(benchmark_report)
        
        # Save benchmark report
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        report_file = self.output_dir / f"benchmark_report_{test_set_name}_{timestamp}.json"
        
        with open(report_file, 'w') as f:
            json.dump(benchmark_report, f, indent=2, default=str)
        
        logger.info(f"Benchmark report saved to {report_file}")
        
        return benchmark_report
    
    def _analyze_performance(self, results: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze performance characteristics"""
        analysis = {
            'overall_performance': 'unknown',
            'strengths': [],
            'weaknesses': [],
            'performance_metrics': {}
        }
        
        aggregated = results.get('aggregated_results', {})
        
        # Overall performance assessment
        if 'document_level' in aggregated:
            doc_accuracy = aggregated['document_level'].get('avg_accuracy', 0.0)
            if doc_accuracy >= 0.9:
                analysis['overall_performance'] = 'excellent'
            elif doc_accuracy >= 0.8:
                analysis['overall_performance'] = 'good'
            elif doc_accuracy >= 0.7:
                analysis['overall_performance'] = 'acceptable'
            else:
                analysis['overall_performance'] = 'needs_improvement'
        
        # Identify strengths and weaknesses
        for metric_type, metrics in aggregated.items():
            avg_accuracy = metrics.get('avg_accuracy', 0.0)
            avg_score = metrics.get('avg_score', 0.0)
            
            if avg_accuracy >= 0.85:
                analysis['strengths'].append(f"Strong {metric_type} accuracy ({avg_accuracy:.3f})")
            elif avg_accuracy < 0.7:
                analysis['weaknesses'].append(f"Weak {metric_type} accuracy ({avg_accuracy:.3f})")
            
            analysis['performance_metrics'][f"{metric_type}_accuracy"] = avg_accuracy
            analysis['performance_metrics'][f"{metric_type}_score"] = avg_score
        
        return analysis
    
    def _generate_benchmark_recommendations(self, benchmark_report: Dict[str, Any]) -> List[str]:
        """Generate recommendations from benchmark analysis"""
        recommendations = []
        
        performance = benchmark_report.get('performance_analysis', {})
        overall_perf = performance.get('overall_performance', 'unknown')
        
        if overall_perf == 'needs_improvement':
            recommendations.append("Overall performance needs significant improvement - consider model retraining")
        elif overall_perf == 'acceptable':
            recommendations.append("Performance is acceptable but has room for improvement")
        elif overall_perf == 'excellent':
            recommendations.append("Excellent performance - model is production-ready")
        
        # Add specific recommendations based on weaknesses
        weaknesses = performance.get('weaknesses', [])
        for weakness in weaknesses:
            if 'field_level' in weakness:
                recommendations.append("Focus on improving field extraction accuracy")
            elif 'character_level' in weakness:
                recommendations.append("Improve OCR accuracy and character recognition")
            elif 'word_level' in weakness:
                recommendations.append("Enhance word-level processing and tokenization")
        
        # Baseline comparison recommendations
        baseline_comparison = benchmark_report.get('baseline_comparison', {})
        if baseline_comparison:
            winner_analysis = baseline_comparison.get('winner_analysis', {})
            if winner_analysis.get('overall_winner') == 'baseline':
                recommendations.append("Current model underperforms baseline - investigate regression")
            elif winner_analysis.get('overall_winner') == 'current':
                recommendations.append("Model shows improvement over baseline - consider deployment")
        
        return recommendations

def create_test_set_from_data(name: str, data_file: str, 
                             document_type: DocumentType = None) -> TestSet:
    """Create test set from data file"""
    test_set = TestSet(
        name=name,
        description=f"Test set created from {data_file}",
        document_type=document_type
    )
    
    # Load data (assuming JSON format)
    try:
        with open(data_file, 'r') as f:
            data = json.load(f)
        
        for i, item in enumerate(data):
            test_case = TestCase(
                test_id=f"{name}_{i:04d}",
                document_type=document_type or DocumentType.IDENTITY_CARD,
                ground_truth=item.get('ground_truth', {}),
                predictions=item.get('predictions', {}),
                confidence_scores=item.get('confidence_scores', {}),
                processing_time=item.get('processing_time'),
                model_version=item.get('model_version'),
                metadata=item.get('metadata', {})
            )
            test_set.add_test_case(test_case)
        
        logger.info(f"Created test set '{name}' with {len(test_set.test_cases)} test cases")
        
    except Exception as e:
        logger.error(f"Error creating test set from {data_file}: {e}")
        raise
    
    return test_set

def main():
    """Main function for standalone execution"""
    # Example usage of the model evaluation framework
    
    # Initialize framework
    framework = ModelEvaluationFramework("evaluation_results")
    
    # Create sample test data
    sample_test_cases = [
        TestCase(
            test_id="test_001",
            document_type=DocumentType.IDENTITY_CARD,
            ground_truth={
                "ic_number": "123456-78-9012",
                "name": "Ahmad bin Ali",
                "address": "123 Jalan Merdeka, Kuala Lumpur",
                "phone_number": "012-3456789"
            },
            predictions={
                "ic_number": "123456-78-9012",
                "name": "Ahmad bin Ali",
                "address": "123 Jalan Merdeka, KL",
                "phone_number": "0123456789"
            },
            confidence_scores={
                "ic_number": 0.95,
                "name": 0.92,
                "address": 0.78,
                "phone_number": 0.88
            },
            processing_time=1.2
        ),
        TestCase(
            test_id="test_002",
            document_type=DocumentType.IDENTITY_CARD,
            ground_truth={
                "ic_number": "987654-32-1098",
                "name": "Siti binti Hassan",
                "address": "456 Jalan Bangsar, Kuala Lumpur",
                "phone_number": "019-8765432"
            },
            predictions={
                "ic_number": "987654-32-1098",
                "name": "Siti binti Hassan",
                "address": "456 Jalan Bangsar, Kuala Lumpur",
                "phone_number": "0198765432"
            },
            confidence_scores={
                "ic_number": 0.98,
                "name": 0.94,
                "address": 0.91,
                "phone_number": 0.85
            },
            processing_time=0.9
        )
    ]
    
    # Create test set
    test_set = TestSet(
        name="sample_ic_test",
        description="Sample test set for IC document evaluation",
        document_type=DocumentType.IDENTITY_CARD
    )
    
    for test_case in sample_test_cases:
        test_set.add_test_case(test_case)
    
    # Add test set to framework
    framework.add_test_set(test_set)
    
    # Evaluate test set
    print("\n=== Model Evaluation Framework Demo ===")
    print(f"Evaluating test set: {test_set.name}")
    
    evaluation_results = framework.evaluate_test_set(
        "sample_ic_test",
        [MetricType.FIELD_LEVEL, MetricType.DOCUMENT_LEVEL]
    )
    
    # Display results
    print("\n=== Evaluation Results ===")
    aggregated = evaluation_results['aggregated_results']
    
    for metric_type, metrics in aggregated.items():
        print(f"\n{metric_type.upper()} METRICS:")
        print(f"  Average Accuracy: {metrics.get('avg_accuracy', 0.0):.3f}")
        print(f"  Average Score: {metrics.get('avg_score', 0.0):.3f}")
        print(f"  Total Errors: {metrics.get('total_errors', 0)}")
        print(f"  Error Rate: {metrics.get('avg_error_rate', 0.0):.3f}")
    
    # Display error analysis
    print("\n=== Error Analysis ===")
    error_analysis = evaluation_results['error_analysis']
    
    print(f"Total Samples: {error_analysis['total_samples']}")
    print("Error Distribution:")
    for error_type, count in error_analysis['error_distribution'].items():
        print(f"  {error_type}: {count}")
    
    print("\nRecommendations:")
    for i, recommendation in enumerate(error_analysis['recommendations'], 1):
        print(f"  {i}. {recommendation}")
    
    # Generate benchmark report
    print("\n=== Generating Benchmark Report ===")
    benchmark_report = framework.generate_benchmark_report("sample_ic_test")
    
    performance = benchmark_report['performance_analysis']
    print(f"Overall Performance: {performance['overall_performance']}")
    
    if performance['strengths']:
        print("\nStrengths:")
        for strength in performance['strengths']:
            print(f"  • {strength}")
    
    if performance['weaknesses']:
        print("\nWeaknesses:")
        for weakness in performance['weaknesses']:
            print(f"  • {weakness}")
    
    print("\n=== Evaluation Complete ===")
    print(f"Results saved to: {framework.output_dir}")

if __name__ == "__main__":
    main()