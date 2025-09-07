"""Model Integration Pipeline Module

This module provides comprehensive model integration pipeline functionality including
sequential and parallel processing pipelines with error handling and quality gates,
following the autocorrect model's organizational patterns.
"""

import os
import json
import time
import asyncio
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any, Union, Callable
from dataclasses import dataclass, field
from datetime import datetime
import logging
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor, as_completed
from enum import Enum
import traceback

import cv2
import numpy as np
import pandas as pd
from PIL import Image
import torch

# Import from our modules
from ..models.document_classifier import DocumentClassifier, ClassificationResult
from ..models.ocr_engines import OCRManager, OCRResult
from ..models.field_extractor import FieldExtractor, FieldExtractionResult
from .data_preparation import DataPreparationPipeline
from .classifier_training import ClassifierTrainer
from .ocr_integration import OCRIntegrationManager
from .field_extraction_training import FieldExtractionTrainer

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ProcessingMode(Enum):
    """Processing mode enumeration"""
    SEQUENTIAL = "sequential"
    PARALLEL = "parallel"
    HYBRID = "hybrid"

class QualityGate(Enum):
    """Quality gate types"""
    CONFIDENCE_THRESHOLD = "confidence_threshold"
    CONSENSUS_VALIDATION = "consensus_validation"
    FIELD_COMPLETENESS = "field_completeness"
    PROCESSING_TIME = "processing_time"
    ERROR_RATE = "error_rate"

class PipelineStage(Enum):
    """Pipeline stage enumeration"""
    PREPROCESSING = "preprocessing"
    CLASSIFICATION = "classification"
    OCR_EXTRACTION = "ocr_extraction"
    FIELD_EXTRACTION = "field_extraction"
    POSTPROCESSING = "postprocessing"
    VALIDATION = "validation"

@dataclass
class PipelineConfig:
    """Configuration for model integration pipeline"""
    
    # Processing configuration
    processing_mode: ProcessingMode = ProcessingMode.HYBRID
    max_workers: int = 4
    timeout_seconds: int = 300
    
    # Quality gates
    quality_gates: Dict[QualityGate, Any] = field(default_factory=lambda: {
        QualityGate.CONFIDENCE_THRESHOLD: 0.7,
        QualityGate.CONSENSUS_VALIDATION: True,
        QualityGate.FIELD_COMPLETENESS: 0.8,
        QualityGate.PROCESSING_TIME: 60.0,
        QualityGate.ERROR_RATE: 0.1
    })
    
    # Pipeline stages configuration
    enabled_stages: List[PipelineStage] = field(default_factory=lambda: [
        PipelineStage.PREPROCESSING,
        PipelineStage.CLASSIFICATION,
        PipelineStage.OCR_EXTRACTION,
        PipelineStage.FIELD_EXTRACTION,
        PipelineStage.POSTPROCESSING,
        PipelineStage.VALIDATION
    ])
    
    # Retry configuration
    max_retries: int = 3
    retry_delay: float = 1.0
    exponential_backoff: bool = True
    
    # Error handling
    fail_fast: bool = False
    continue_on_error: bool = True
    error_threshold: float = 0.2
    
    # Caching
    enable_caching: bool = True
    cache_size: int = 1000
    cache_ttl: int = 3600  # seconds
    
    # Monitoring
    enable_monitoring: bool = True
    log_level: str = "INFO"
    metrics_collection: bool = True
    
    # Output configuration
    output_path: str = "model_artifacts/document_parser/pipeline_results"
    checkpoint_path: str = "model_artifacts/document_parser/pipeline_checkpoints"
    
    # Model paths
    classifier_model_path: str = "model_artifacts/document_parser/classifier"
    ocr_models_path: str = "model_artifacts/document_parser/ocr_integration"
    field_extraction_models_path: str = "model_artifacts/document_parser/field_extraction"
    
    # Performance optimization
    batch_processing: bool = True
    batch_size: int = 8
    memory_optimization: bool = True
    
    # Random seed
    random_seed: int = 42

@dataclass
class PipelineResult:
    """Result from pipeline processing"""
    
    # Input information
    input_id: str
    input_path: Optional[str] = None
    
    # Processing results
    classification_result: Optional[ClassificationResult] = None
    ocr_result: Optional[OCRResult] = None
    field_extraction_result: Optional[FieldExtractionResult] = None
    
    # Pipeline metadata
    processing_time: float = 0.0
    stages_completed: List[PipelineStage] = field(default_factory=list)
    stages_failed: List[PipelineStage] = field(default_factory=list)
    
    # Quality metrics
    overall_confidence: float = 0.0
    quality_gates_passed: List[QualityGate] = field(default_factory=list)
    quality_gates_failed: List[QualityGate] = field(default_factory=list)
    
    # Error information
    errors: List[Dict[str, Any]] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    
    # Success flag
    success: bool = False
    
    # Additional metadata
    metadata: Dict[str, Any] = field(default_factory=dict)

class PipelineStageProcessor:
    """Base class for pipeline stage processors"""
    
    def __init__(self, config: PipelineConfig):
        self.config = config
        self.stage_name = "base"
    
    def process(self, input_data: Any, context: Dict[str, Any]) -> Tuple[Any, Dict[str, Any]]:
        """Process input data and return result with updated context"""
        raise NotImplementedError
    
    def validate_input(self, input_data: Any) -> bool:
        """Validate input data for this stage"""
        return input_data is not None
    
    def handle_error(self, error: Exception, input_data: Any) -> Tuple[Any, Dict[str, Any]]:
        """Handle errors in processing"""
        logger.error(f"Error in {self.stage_name}: {error}")
        return None, {'error': str(error), 'stage': self.stage_name}

class PreprocessingStage(PipelineStageProcessor):
    """Preprocessing stage for image preparation"""
    
    def __init__(self, config: PipelineConfig):
        super().__init__(config)
        self.stage_name = "preprocessing"
    
    def process(self, input_data: Union[str, np.ndarray, Image.Image], 
               context: Dict[str, Any]) -> Tuple[np.ndarray, Dict[str, Any]]:
        """Preprocess input image"""
        try:
            # Load image if path provided
            if isinstance(input_data, str):
                if not os.path.exists(input_data):
                    raise FileNotFoundError(f"Image file not found: {input_data}")
                image = cv2.imread(input_data)
                context['input_path'] = input_data
            elif isinstance(input_data, Image.Image):
                image = cv2.cvtColor(np.array(input_data), cv2.COLOR_RGB2BGR)
            elif isinstance(input_data, np.ndarray):
                image = input_data.copy()
            else:
                raise ValueError(f"Unsupported input type: {type(input_data)}")
            
            if image is None:
                raise ValueError("Failed to load image")
            
            # Basic preprocessing
            original_shape = image.shape
            
            # Resize if too large
            max_dimension = 2048
            height, width = image.shape[:2]
            if max(height, width) > max_dimension:
                scale = max_dimension / max(height, width)
                new_width = int(width * scale)
                new_height = int(height * scale)
                image = cv2.resize(image, (new_width, new_height), interpolation=cv2.INTER_AREA)
            
            # Enhance image quality
            # Convert to grayscale for some operations
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            
            # Noise reduction
            denoised = cv2.fastNlMeansDenoising(gray)
            
            # Convert back to BGR
            processed_image = cv2.cvtColor(denoised, cv2.COLOR_GRAY2BGR)
            
            # Update context
            context.update({
                'original_shape': original_shape,
                'processed_shape': processed_image.shape,
                'preprocessing_applied': ['resize', 'denoise']
            })
            
            return processed_image, context
            
        except Exception as e:
            return self.handle_error(e, input_data)

class ClassificationStage(PipelineStageProcessor):
    """Document classification stage"""
    
    def __init__(self, config: PipelineConfig):
        super().__init__(config)
        self.stage_name = "classification"
        self.classifier = None
        self.load_classifier()
    
    def load_classifier(self):
        """Load document classifier"""
        try:
            # Initialize classifier (placeholder - would load actual trained model)
            self.classifier = DocumentClassifier()
            logger.info("Document classifier loaded")
        except Exception as e:
            logger.warning(f"Failed to load classifier: {e}")
    
    def process(self, input_data: np.ndarray, 
               context: Dict[str, Any]) -> Tuple[ClassificationResult, Dict[str, Any]]:
        """Classify document type"""
        try:
            if self.classifier is None:
                raise RuntimeError("Classifier not loaded")
            
            # Perform classification
            result = self.classifier.classify_document(input_data)
            
            # Update context
            context.update({
                'document_type': result.document_type,
                'classification_confidence': result.confidence
            })
            
            return result, context
            
        except Exception as e:
            return self.handle_error(e, input_data)

class OCRExtractionStage(PipelineStageProcessor):
    """OCR text extraction stage"""
    
    def __init__(self, config: PipelineConfig):
        super().__init__(config)
        self.stage_name = "ocr_extraction"
        self.ocr_manager = None
        self.load_ocr_manager()
    
    def load_ocr_manager(self):
        """Load OCR manager"""
        try:
            # Initialize OCR manager (placeholder - would load actual configuration)
            self.ocr_manager = OCRManager()
            logger.info("OCR manager loaded")
        except Exception as e:
            logger.warning(f"Failed to load OCR manager: {e}")
    
    def process(self, input_data: np.ndarray, 
               context: Dict[str, Any]) -> Tuple[OCRResult, Dict[str, Any]]:
        """Extract text using OCR"""
        try:
            if self.ocr_manager is None:
                raise RuntimeError("OCR manager not loaded")
            
            # Perform OCR extraction
            result = self.ocr_manager.extract_text(input_data)
            
            # Update context
            context.update({
                'extracted_text': result.text,
                'ocr_confidence': result.confidence,
                'text_blocks_count': len(result.bounding_boxes)
            })
            
            return result, context
            
        except Exception as e:
            return self.handle_error(e, input_data)

class FieldExtractionStage(PipelineStageProcessor):
    """Field extraction stage"""
    
    def __init__(self, config: PipelineConfig):
        super().__init__(config)
        self.stage_name = "field_extraction"
        self.field_extractor = None
        self.load_field_extractor()
    
    def load_field_extractor(self):
        """Load field extractor"""
        try:
            # Initialize field extractor (placeholder - would load actual trained models)
            self.field_extractor = FieldExtractor()
            logger.info("Field extractor loaded")
        except Exception as e:
            logger.warning(f"Failed to load field extractor: {e}")
    
    def process(self, input_data: Tuple[OCRResult, str], 
               context: Dict[str, Any]) -> Tuple[FieldExtractionResult, Dict[str, Any]]:
        """Extract fields from OCR result"""
        try:
            if self.field_extractor is None:
                raise RuntimeError("Field extractor not loaded")
            
            ocr_result, document_type = input_data
            
            # Perform field extraction
            result = self.field_extractor.extract_fields(
                ocr_result, 
                document_type=document_type
            )
            
            # Update context
            context.update({
                'extracted_fields': result.fields,
                'field_extraction_confidence': result.confidence,
                'fields_count': len(result.fields)
            })
            
            return result, context
            
        except Exception as e:
            return self.handle_error(e, input_data)

class PostprocessingStage(PipelineStageProcessor):
    """Postprocessing stage for result refinement"""
    
    def __init__(self, config: PipelineConfig):
        super().__init__(config)
        self.stage_name = "postprocessing"
    
    def process(self, input_data: FieldExtractionResult, 
               context: Dict[str, Any]) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        """Postprocess extraction results"""
        try:
            # Clean and validate extracted fields
            cleaned_fields = {}
            
            for field_name, field_value in input_data.fields.items():
                # Basic cleaning
                if isinstance(field_value, str):
                    cleaned_value = field_value.strip()
                    
                    # Remove extra whitespace
                    cleaned_value = ' '.join(cleaned_value.split())
                    
                    # Field-specific cleaning
                    if 'amount' in field_name.lower():
                        # Clean monetary amounts
                        cleaned_value = self.clean_monetary_amount(cleaned_value)
                    elif 'date' in field_name.lower():
                        # Standardize date format
                        cleaned_value = self.standardize_date(cleaned_value)
                    elif 'number' in field_name.lower():
                        # Clean alphanumeric identifiers
                        cleaned_value = self.clean_identifier(cleaned_value)
                    
                    cleaned_fields[field_name] = cleaned_value
                else:
                    cleaned_fields[field_name] = field_value
            
            # Create final result
            final_result = {
                'document_type': context.get('document_type', 'unknown'),
                'extracted_fields': cleaned_fields,
                'confidence_scores': {
                    'classification': context.get('classification_confidence', 0.0),
                    'ocr': context.get('ocr_confidence', 0.0),
                    'field_extraction': context.get('field_extraction_confidence', 0.0)
                },
                'processing_metadata': {
                    'processing_time': context.get('processing_time', 0.0),
                    'stages_completed': context.get('stages_completed', []),
                    'text_blocks_count': context.get('text_blocks_count', 0),
                    'fields_count': context.get('fields_count', 0)
                }
            }
            
            # Update context
            context.update({
                'final_result': final_result,
                'postprocessing_applied': True
            })
            
            return final_result, context
            
        except Exception as e:
            return self.handle_error(e, input_data)
    
    def clean_monetary_amount(self, amount_str: str) -> str:
        """Clean monetary amount string"""
        # Remove currency symbols and extra characters
        import re
        cleaned = re.sub(r'[^\d.,]', '', amount_str)
        return cleaned
    
    def standardize_date(self, date_str: str) -> str:
        """Standardize date format"""
        # Basic date standardization (would be more sophisticated in practice)
        import re
        
        # Try to match common date patterns
        patterns = [
            r'(\d{1,2})[/-](\d{1,2})[/-](\d{2,4})',
            r'(\d{2,4})[/-](\d{1,2})[/-](\d{1,2})'
        ]
        
        for pattern in patterns:
            match = re.search(pattern, date_str)
            if match:
                return match.group(0)
        
        return date_str
    
    def clean_identifier(self, identifier_str: str) -> str:
        """Clean alphanumeric identifier"""
        # Remove extra spaces and special characters
        import re
        cleaned = re.sub(r'[^A-Za-z0-9-]', '', identifier_str)
        return cleaned.upper()

class ValidationStage(PipelineStageProcessor):
    """Validation stage for quality gates"""
    
    def __init__(self, config: PipelineConfig):
        super().__init__(config)
        self.stage_name = "validation"
    
    def process(self, input_data: Dict[str, Any], 
               context: Dict[str, Any]) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        """Validate processing results against quality gates"""
        try:
            validation_results = {
                'passed_gates': [],
                'failed_gates': [],
                'overall_score': 0.0,
                'validation_passed': True
            }
            
            # Check confidence threshold
            if QualityGate.CONFIDENCE_THRESHOLD in self.config.quality_gates:
                threshold = self.config.quality_gates[QualityGate.CONFIDENCE_THRESHOLD]
                confidence_scores = input_data.get('confidence_scores', {})
                avg_confidence = np.mean(list(confidence_scores.values())) if confidence_scores else 0.0
                
                if avg_confidence >= threshold:
                    validation_results['passed_gates'].append(QualityGate.CONFIDENCE_THRESHOLD)
                else:
                    validation_results['failed_gates'].append(QualityGate.CONFIDENCE_THRESHOLD)
                    validation_results['validation_passed'] = False
            
            # Check field completeness
            if QualityGate.FIELD_COMPLETENESS in self.config.quality_gates:
                completeness_threshold = self.config.quality_gates[QualityGate.FIELD_COMPLETENESS]
                extracted_fields = input_data.get('extracted_fields', {})
                document_type = input_data.get('document_type', '')
                
                # Get expected fields for document type (simplified)
                expected_fields = self.get_expected_fields(document_type)
                if expected_fields:
                    completeness = len(extracted_fields) / len(expected_fields)
                    
                    if completeness >= completeness_threshold:
                        validation_results['passed_gates'].append(QualityGate.FIELD_COMPLETENESS)
                    else:
                        validation_results['failed_gates'].append(QualityGate.FIELD_COMPLETENESS)
                        validation_results['validation_passed'] = False
            
            # Check processing time
            if QualityGate.PROCESSING_TIME in self.config.quality_gates:
                time_threshold = self.config.quality_gates[QualityGate.PROCESSING_TIME]
                processing_time = context.get('processing_time', 0.0)
                
                if processing_time <= time_threshold:
                    validation_results['passed_gates'].append(QualityGate.PROCESSING_TIME)
                else:
                    validation_results['failed_gates'].append(QualityGate.PROCESSING_TIME)
                    # Don't fail validation for time threshold, just log
            
            # Calculate overall score
            total_gates = len(self.config.quality_gates)
            passed_gates = len(validation_results['passed_gates'])
            validation_results['overall_score'] = passed_gates / total_gates if total_gates > 0 else 1.0
            
            # Update context
            context.update({
                'validation_results': validation_results,
                'quality_gates_passed': validation_results['passed_gates'],
                'quality_gates_failed': validation_results['failed_gates']
            })
            
            return validation_results, context
            
        except Exception as e:
            return self.handle_error(e, input_data)
    
    def get_expected_fields(self, document_type: str) -> List[str]:
        """Get expected fields for document type"""
        field_mapping = {
            'invoice': ['invoice_number', 'date', 'total_amount', 'vendor_name'],
            'receipt': ['receipt_number', 'date', 'total_amount', 'merchant_name'],
            'identity_card': ['ic_number', 'name', 'date_of_birth'],
            'passport': ['passport_number', 'name', 'date_of_birth', 'nationality'],
            'bank_statement': ['account_number', 'statement_date', 'balance']
        }
        
        return field_mapping.get(document_type, [])

class ModelIntegrationPipeline:
    """Main model integration pipeline"""
    
    def __init__(self, config: PipelineConfig):
        self.config = config
        self.setup_directories()
        self.setup_stages()
        
        # Performance tracking
        self.processing_stats = {
            'total_processed': 0,
            'successful_processed': 0,
            'failed_processed': 0,
            'average_processing_time': 0.0,
            'stage_performance': {}
        }
        
        # Cache for results
        self.result_cache = {} if config.enable_caching else None
    
    def setup_directories(self):
        """Create necessary directories"""
        for path in [self.config.output_path, self.config.checkpoint_path]:
            Path(path).mkdir(parents=True, exist_ok=True)
    
    def setup_stages(self):
        """Initialize pipeline stages"""
        self.stages = {}
        
        if PipelineStage.PREPROCESSING in self.config.enabled_stages:
            self.stages[PipelineStage.PREPROCESSING] = PreprocessingStage(self.config)
        
        if PipelineStage.CLASSIFICATION in self.config.enabled_stages:
            self.stages[PipelineStage.CLASSIFICATION] = ClassificationStage(self.config)
        
        if PipelineStage.OCR_EXTRACTION in self.config.enabled_stages:
            self.stages[PipelineStage.OCR_EXTRACTION] = OCRExtractionStage(self.config)
        
        if PipelineStage.FIELD_EXTRACTION in self.config.enabled_stages:
            self.stages[PipelineStage.FIELD_EXTRACTION] = FieldExtractionStage(self.config)
        
        if PipelineStage.POSTPROCESSING in self.config.enabled_stages:
            self.stages[PipelineStage.POSTPROCESSING] = PostprocessingStage(self.config)
        
        if PipelineStage.VALIDATION in self.config.enabled_stages:
            self.stages[PipelineStage.VALIDATION] = ValidationStage(self.config)
        
        logger.info(f"Initialized {len(self.stages)} pipeline stages")
    
    def process_single(self, input_data: Any, input_id: str = None) -> PipelineResult:
        """Process single input through the pipeline"""
        start_time = time.time()
        
        if input_id is None:
            input_id = f"input_{int(time.time() * 1000)}"
        
        # Initialize result
        result = PipelineResult(input_id=input_id)
        
        # Initialize context
        context = {
            'input_id': input_id,
            'start_time': start_time,
            'stages_completed': [],
            'stages_failed': [],
            'processing_time': 0.0
        }
        
        try:
            # Check cache
            if self.result_cache and input_id in self.result_cache:
                logger.info(f"Returning cached result for {input_id}")
                return self.result_cache[input_id]
            
            current_data = input_data
            
            # Process through stages sequentially
            for stage_enum in self.config.enabled_stages:
                if stage_enum not in self.stages:
                    continue
                
                stage = self.stages[stage_enum]
                stage_start_time = time.time()
                
                try:
                    logger.info(f"Processing stage: {stage.stage_name} for {input_id}")
                    
                    # Validate input for this stage
                    if not stage.validate_input(current_data):
                        raise ValueError(f"Invalid input for stage {stage.stage_name}")
                    
                    # Process stage with retry logic
                    stage_result, context = self.process_stage_with_retry(
                        stage, current_data, context
                    )
                    
                    if stage_result is None:
                        raise RuntimeError(f"Stage {stage.stage_name} returned None")
                    
                    # Update current data for next stage
                    if stage_enum == PipelineStage.PREPROCESSING:
                        current_data = stage_result
                    elif stage_enum == PipelineStage.CLASSIFICATION:
                        result.classification_result = stage_result
                        current_data = (current_data, stage_result.document_type)
                    elif stage_enum == PipelineStage.OCR_EXTRACTION:
                        result.ocr_result = stage_result
                        if hasattr(result, 'classification_result') and result.classification_result:
                            current_data = (stage_result, result.classification_result.document_type)
                        else:
                            current_data = (stage_result, 'unknown')
                    elif stage_enum == PipelineStage.FIELD_EXTRACTION:
                        result.field_extraction_result = stage_result
                        current_data = stage_result
                    elif stage_enum == PipelineStage.POSTPROCESSING:
                        current_data = stage_result
                    elif stage_enum == PipelineStage.VALIDATION:
                        # Validation is the final stage
                        pass
                    
                    # Track stage completion
                    context['stages_completed'].append(stage_enum)
                    result.stages_completed.append(stage_enum)
                    
                    # Update stage performance
                    stage_time = time.time() - stage_start_time
                    if stage.stage_name not in self.processing_stats['stage_performance']:
                        self.processing_stats['stage_performance'][stage.stage_name] = []
                    self.processing_stats['stage_performance'][stage.stage_name].append(stage_time)
                    
                except Exception as e:
                    logger.error(f"Error in stage {stage.stage_name}: {e}")
                    context['stages_failed'].append(stage_enum)
                    result.stages_failed.append(stage_enum)
                    result.errors.append({
                        'stage': stage.stage_name,
                        'error': str(e),
                        'traceback': traceback.format_exc()
                    })
                    
                    if self.config.fail_fast:
                        raise
                    
                    if not self.config.continue_on_error:
                        break
            
            # Calculate final metrics
            processing_time = time.time() - start_time
            result.processing_time = processing_time
            context['processing_time'] = processing_time
            
            # Determine success
            error_rate = len(result.stages_failed) / len(self.config.enabled_stages)
            result.success = error_rate <= self.config.error_threshold
            
            # Calculate overall confidence
            confidences = []
            if result.classification_result:
                confidences.append(result.classification_result.confidence)
            if result.ocr_result:
                confidences.append(result.ocr_result.confidence)
            if result.field_extraction_result:
                confidences.append(result.field_extraction_result.confidence)
            
            result.overall_confidence = np.mean(confidences) if confidences else 0.0
            
            # Extract quality gate results from context
            if 'quality_gates_passed' in context:
                result.quality_gates_passed = context['quality_gates_passed']
            if 'quality_gates_failed' in context:
                result.quality_gates_failed = context['quality_gates_failed']
            
            # Store additional metadata
            result.metadata = {
                'processing_mode': self.config.processing_mode.value,
                'enabled_stages': [stage.value for stage in self.config.enabled_stages],
                'context': context
            }
            
            # Cache result
            if self.result_cache:
                self.result_cache[input_id] = result
            
            # Update statistics
            self.processing_stats['total_processed'] += 1
            if result.success:
                self.processing_stats['successful_processed'] += 1
            else:
                self.processing_stats['failed_processed'] += 1
            
            # Update average processing time
            total = self.processing_stats['total_processed']
            current_avg = self.processing_stats['average_processing_time']
            self.processing_stats['average_processing_time'] = (
                (current_avg * (total - 1) + processing_time) / total
            )
            
            logger.info(f"Pipeline processing completed for {input_id} in {processing_time:.2f}s")
            
            return result
            
        except Exception as e:
            logger.error(f"Pipeline processing failed for {input_id}: {e}")
            result.success = False
            result.errors.append({
                'stage': 'pipeline',
                'error': str(e),
                'traceback': traceback.format_exc()
            })
            result.processing_time = time.time() - start_time
            
            return result
    
    def process_stage_with_retry(self, stage: PipelineStageProcessor, 
                               input_data: Any, context: Dict[str, Any]) -> Tuple[Any, Dict[str, Any]]:
        """Process stage with retry logic"""
        last_error = None
        
        for attempt in range(self.config.max_retries + 1):
            try:
                return stage.process(input_data, context)
            except Exception as e:
                last_error = e
                
                if attempt < self.config.max_retries:
                    delay = self.config.retry_delay
                    if self.config.exponential_backoff:
                        delay *= (2 ** attempt)
                    
                    logger.warning(f"Stage {stage.stage_name} failed (attempt {attempt + 1}), retrying in {delay}s: {e}")
                    time.sleep(delay)
                else:
                    logger.error(f"Stage {stage.stage_name} failed after {self.config.max_retries + 1} attempts: {e}")
        
        raise last_error
    
    def process_batch(self, input_batch: List[Tuple[Any, str]]) -> List[PipelineResult]:
        """Process batch of inputs"""
        logger.info(f"Processing batch of {len(input_batch)} inputs")
        
        if self.config.processing_mode == ProcessingMode.SEQUENTIAL:
            return self.process_batch_sequential(input_batch)
        elif self.config.processing_mode == ProcessingMode.PARALLEL:
            return self.process_batch_parallel(input_batch)
        else:  # HYBRID
            return self.process_batch_hybrid(input_batch)
    
    def process_batch_sequential(self, input_batch: List[Tuple[Any, str]]) -> List[PipelineResult]:
        """Process batch sequentially"""
        results = []
        for input_data, input_id in input_batch:
            result = self.process_single(input_data, input_id)
            results.append(result)
        return results
    
    def process_batch_parallel(self, input_batch: List[Tuple[Any, str]]) -> List[PipelineResult]:
        """Process batch in parallel"""
        results = [None] * len(input_batch)
        
        with ThreadPoolExecutor(max_workers=self.config.max_workers) as executor:
            future_to_index = {
                executor.submit(self.process_single, input_data, input_id): i
                for i, (input_data, input_id) in enumerate(input_batch)
            }
            
            for future in as_completed(future_to_index, timeout=self.config.timeout_seconds):
                index = future_to_index[future]
                try:
                    result = future.result()
                    results[index] = result
                except Exception as e:
                    logger.error(f"Parallel processing error for index {index}: {e}")
                    # Create error result
                    input_data, input_id = input_batch[index]
                    error_result = PipelineResult(input_id=input_id)
                    error_result.success = False
                    error_result.errors.append({
                        'stage': 'parallel_processing',
                        'error': str(e),
                        'traceback': traceback.format_exc()
                    })
                    results[index] = error_result
        
        return results
    
    def process_batch_hybrid(self, input_batch: List[Tuple[Any, str]]) -> List[PipelineResult]:
        """Process batch using hybrid approach"""
        # Use parallel processing for I/O intensive stages and sequential for CPU intensive
        # This is a simplified implementation
        return self.process_batch_parallel(input_batch)
    
    def save_checkpoint(self, checkpoint_name: str, data: Dict[str, Any]):
        """Save processing checkpoint"""
        checkpoint_path = Path(self.config.checkpoint_path) / f"{checkpoint_name}.json"
        with open(checkpoint_path, 'w') as f:
            json.dump(data, f, indent=2, default=str)
        logger.info(f"Checkpoint saved: {checkpoint_path}")
    
    def load_checkpoint(self, checkpoint_name: str) -> Optional[Dict[str, Any]]:
        """Load processing checkpoint"""
        checkpoint_path = Path(self.config.checkpoint_path) / f"{checkpoint_name}.json"
        if checkpoint_path.exists():
            with open(checkpoint_path, 'r') as f:
                data = json.load(f)
            logger.info(f"Checkpoint loaded: {checkpoint_path}")
            return data
        return None
    
    def get_performance_report(self) -> Dict[str, Any]:
        """Get performance report"""
        report = {
            'processing_statistics': self.processing_stats.copy(),
            'configuration': self.config.__dict__,
            'timestamp': datetime.now().isoformat()
        }
        
        # Calculate stage averages
        for stage_name, times in self.processing_stats['stage_performance'].items():
            report['processing_statistics']['stage_performance'][stage_name] = {
                'average_time': np.mean(times),
                'min_time': np.min(times),
                'max_time': np.max(times),
                'total_calls': len(times)
            }
        
        return report
    
    def save_performance_report(self):
        """Save performance report to file"""
        report = self.get_performance_report()
        report_path = Path(self.config.output_path) / 'performance_report.json'
        with open(report_path, 'w') as f:
            json.dump(report, f, indent=2, default=str)
        logger.info(f"Performance report saved: {report_path}")

def main():
    """Main function for standalone execution"""
    # Initialize configuration
    config = PipelineConfig()
    
    # Create pipeline
    pipeline = ModelIntegrationPipeline(config)
    
    # Test with sample data (if available)
    test_image_path = "data_collection/real/sample_document.jpg"
    if os.path.exists(test_image_path):
        logger.info(f"Testing pipeline with {test_image_path}")
        
        # Process single image
        result = pipeline.process_single(test_image_path, "test_sample")
        
        # Print results
        print("\n" + "="*50)
        print("PIPELINE INTEGRATION TEST RESULTS")
        print("="*50)
        print(f"Input ID: {result.input_id}")
        print(f"Success: {result.success}")
        print(f"Processing Time: {result.processing_time:.2f}s")
        print(f"Overall Confidence: {result.overall_confidence:.3f}")
        print(f"Stages Completed: {[stage.value for stage in result.stages_completed]}")
        print(f"Stages Failed: {[stage.value for stage in result.stages_failed]}")
        
        if result.classification_result:
            print(f"\nDocument Type: {result.classification_result.document_type}")
            print(f"Classification Confidence: {result.classification_result.confidence:.3f}")
        
        if result.field_extraction_result:
            print(f"\nExtracted Fields: {len(result.field_extraction_result.fields)}")
            for field, value in result.field_extraction_result.fields.items():
                print(f"  {field}: {value}")
        
        if result.errors:
            print(f"\nErrors ({len(result.errors)}):")
            for error in result.errors:
                print(f"  {error['stage']}: {error['error']}")
        
        print("="*50)
    
    # Save performance report
    pipeline.save_performance_report()
    
    logger.info("Pipeline integration setup completed")

if __name__ == "__main__":
    main()