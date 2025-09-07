"""OCR Model Integration Module

This module provides comprehensive OCR model integration functionality including
Tesseract fine-tuning, Cloud APIs, and Custom CRNN with Malaysian language support,
following the autocorrect model's organizational patterns.
"""

import os
import json
import time
import pickle
import asyncio
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any, Union
from dataclasses import dataclass, field
from datetime import datetime
import logging
from concurrent.futures import ThreadPoolExecutor, as_completed

import cv2
import numpy as np
import pandas as pd
from PIL import Image, ImageEnhance, ImageFilter
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset

# OCR Libraries
import pytesseract
import easyocr
from paddleocr import PaddleOCR

# Google Cloud Vision (optional)
try:
    from google.cloud import vision
    GOOGLE_VISION_AVAILABLE = True
except ImportError:
    GOOGLE_VISION_AVAILABLE = False
    vision = None

# Azure Cognitive Services (optional)
try:
    from azure.cognitiveservices.vision.computervision import ComputerVisionClient
    from azure.cognitiveservices.vision.computervision.models import OperationStatusCodes
    from msrest.authentication import CognitiveServicesCredentials
    AZURE_VISION_AVAILABLE = True
except ImportError:
    AZURE_VISION_AVAILABLE = False

# Import from our modules
from ..models.ocr_engines import OCRConfig, OCRResult, OCRManager

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class OCRIntegrationConfig:
    """Configuration for OCR model integration"""
    
    # Supported languages (Malaysian focus)
    languages: List[str] = field(default_factory=lambda: ['en', 'ms', 'chi_sim', 'tam'])
    primary_language: str = 'en'
    
    # Tesseract configuration
    tesseract_path: Optional[str] = None
    tesseract_config: str = '--oem 3 --psm 6'
    tesseract_models_path: str = "model_artifacts/document_parser/tesseract_models"
    
    # Cloud API configurations
    google_credentials_path: Optional[str] = None
    azure_subscription_key: Optional[str] = None
    azure_endpoint: Optional[str] = None
    
    # EasyOCR configuration
    easyocr_gpu: bool = True
    easyocr_models_path: str = "model_artifacts/document_parser/easyocr_models"
    
    # PaddleOCR configuration
    paddleocr_use_gpu: bool = True
    paddleocr_models_path: str = "model_artifacts/document_parser/paddleocr_models"
    
    # Custom CRNN configuration
    crnn_model_path: str = "model_artifacts/document_parser/crnn_models"
    crnn_vocab_path: str = "model_artifacts/document_parser/crnn_vocab"
    
    # Processing configuration
    batch_size: int = 8
    max_workers: int = 4
    timeout_seconds: int = 30
    
    # Quality control
    min_confidence: float = 0.5
    consensus_threshold: float = 0.7
    max_retries: int = 3
    
    # Output paths
    output_path: str = "model_artifacts/document_parser/ocr_integration"
    training_data_path: str = "model_artifacts/document_parser/ocr_training_data"
    
    # Performance optimization
    cache_enabled: bool = True
    cache_size: int = 1000
    parallel_processing: bool = True
    
    # Evaluation
    benchmark_dataset_path: str = "data_collection/ocr_benchmark"
    evaluation_metrics: List[str] = field(default_factory=lambda: ['accuracy', 'edit_distance', 'bleu_score'])
    
    # Random seed
    random_seed: int = 42

class TesseractIntegrator:
    """Tesseract OCR integration with fine-tuning capabilities"""
    
    def __init__(self, config: OCRIntegrationConfig):
        self.config = config
        self.setup_tesseract()
        self.training_data = []
        
    def setup_tesseract(self):
        """Setup Tesseract configuration"""
        if self.config.tesseract_path:
            pytesseract.pytesseract.tesseract_cmd = self.config.tesseract_path
        
        # Create models directory
        Path(self.config.tesseract_models_path).mkdir(parents=True, exist_ok=True)
        
        logger.info("Tesseract integrator initialized")
    
    def extract_text(self, image: np.ndarray, language: str = None) -> OCRResult:
        """Extract text using Tesseract"""
        if language is None:
            language = self.config.primary_language
        
        try:
            # Convert numpy array to PIL Image
            if isinstance(image, np.ndarray):
                image_pil = Image.fromarray(image)
            else:
                image_pil = image
            
            # Extract text with confidence scores
            data = pytesseract.image_to_data(
                image_pil, 
                lang=language,
                config=self.config.tesseract_config,
                output_type=pytesseract.Output.DICT
            )
            
            # Process results
            text_blocks = []
            confidences = []
            bounding_boxes = []
            
            for i in range(len(data['text'])):
                if int(data['conf'][i]) > 0:  # Filter out low confidence
                    text = data['text'][i].strip()
                    if text:
                        text_blocks.append(text)
                        confidences.append(float(data['conf'][i]) / 100.0)
                        
                        # Bounding box coordinates
                        x, y, w, h = data['left'][i], data['top'][i], data['width'][i], data['height'][i]
                        bounding_boxes.append([x, y, x + w, y + h])
            
            # Combine text
            full_text = ' '.join(text_blocks)
            avg_confidence = np.mean(confidences) if confidences else 0.0
            
            return OCRResult(
                text=full_text,
                confidence=avg_confidence,
                bounding_boxes=bounding_boxes,
                language=language,
                engine='tesseract',
                processing_time=0.0,
                metadata={'word_confidences': confidences, 'word_texts': text_blocks}
            )
            
        except Exception as e:
            logger.error(f"Tesseract OCR error: {e}")
            return OCRResult(
                text="",
                confidence=0.0,
                bounding_boxes=[],
                language=language,
                engine='tesseract',
                processing_time=0.0,
                metadata={'error': str(e)}
            )
    
    def prepare_training_data(self, image_text_pairs: List[Tuple[np.ndarray, str]]):
        """Prepare training data for Tesseract fine-tuning"""
        logger.info(f"Preparing {len(image_text_pairs)} training samples for Tesseract")
        
        training_dir = Path(self.config.training_data_path) / 'tesseract'
        training_dir.mkdir(parents=True, exist_ok=True)
        
        for idx, (image, ground_truth) in enumerate(image_text_pairs):
            # Save image
            image_path = training_dir / f'sample_{idx:06d}.png'
            cv2.imwrite(str(image_path), image)
            
            # Save ground truth
            gt_path = training_dir / f'sample_{idx:06d}.gt.txt'
            with open(gt_path, 'w', encoding='utf-8') as f:
                f.write(ground_truth)
            
            self.training_data.append({
                'image_path': str(image_path),
                'ground_truth_path': str(gt_path),
                'text': ground_truth
            })
        
        logger.info(f"Training data prepared in {training_dir}")
    
    def fine_tune_model(self, language: str = 'eng'):
        """Fine-tune Tesseract model (placeholder for actual fine-tuning)"""
        logger.info(f"Fine-tuning Tesseract model for language: {language}")
        
        # Note: Actual Tesseract fine-tuning requires tesstrain toolkit
        # This is a placeholder for the fine-tuning process
        
        fine_tuned_model_path = Path(self.config.tesseract_models_path) / f'{language}_finetuned.traineddata'
        
        # Placeholder: In real implementation, this would involve:
        # 1. Converting training data to Tesseract format
        # 2. Running tesstrain with custom data
        # 3. Generating new .traineddata file
        
        logger.info(f"Fine-tuned model would be saved to: {fine_tuned_model_path}")
        
        return str(fine_tuned_model_path)

class CloudOCRIntegrator:
    """Cloud OCR services integration"""
    
    def __init__(self, config: OCRIntegrationConfig):
        self.config = config
        self.setup_cloud_clients()
    
    def setup_cloud_clients(self):
        """Setup cloud OCR clients"""
        self.google_client = None
        self.azure_client = None
        
        # Google Cloud Vision
        if GOOGLE_VISION_AVAILABLE and self.config.google_credentials_path:
            try:
                os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = self.config.google_credentials_path
                self.google_client = vision.ImageAnnotatorClient()
                logger.info("Google Cloud Vision client initialized")
            except Exception as e:
                logger.warning(f"Failed to initialize Google Cloud Vision: {e}")
        
        # Azure Cognitive Services
        if AZURE_VISION_AVAILABLE and self.config.azure_subscription_key:
            try:
                self.azure_client = ComputerVisionClient(
                    self.config.azure_endpoint,
                    CognitiveServicesCredentials(self.config.azure_subscription_key)
                )
                logger.info("Azure Cognitive Services client initialized")
            except Exception as e:
                logger.warning(f"Failed to initialize Azure Cognitive Services: {e}")
    
    def extract_text_google(self, image: np.ndarray) -> OCRResult:
        """Extract text using Google Cloud Vision"""
        if not self.google_client:
            return OCRResult(
                text="", confidence=0.0, bounding_boxes=[], 
                language="", engine='google_vision',
                processing_time=0.0, metadata={'error': 'Client not available'}
            )
        
        try:
            start_time = time.time()
            
            # Convert image to bytes
            _, buffer = cv2.imencode('.png', image)
            image_bytes = buffer.tobytes()
            
            # Create Vision API image object
            vision_image = vision.Image(content=image_bytes)
            
            # Perform text detection
            response = self.google_client.text_detection(image=vision_image)
            texts = response.text_annotations
            
            processing_time = time.time() - start_time
            
            if texts:
                # First annotation contains the full text
                full_text = texts[0].description
                
                # Extract bounding boxes for individual words
                bounding_boxes = []
                for text in texts[1:]:  # Skip the first one (full text)
                    vertices = text.bounding_poly.vertices
                    if vertices:
                        x_coords = [v.x for v in vertices]
                        y_coords = [v.y for v in vertices]
                        bbox = [min(x_coords), min(y_coords), max(x_coords), max(y_coords)]
                        bounding_boxes.append(bbox)
                
                return OCRResult(
                    text=full_text,
                    confidence=0.9,  # Google doesn't provide confidence scores
                    bounding_boxes=bounding_boxes,
                    language='auto',
                    engine='google_vision',
                    processing_time=processing_time,
                    metadata={'num_detections': len(texts) - 1}
                )
            else:
                return OCRResult(
                    text="", confidence=0.0, bounding_boxes=[],
                    language='auto', engine='google_vision',
                    processing_time=processing_time,
                    metadata={'num_detections': 0}
                )
                
        except Exception as e:
            logger.error(f"Google Vision OCR error: {e}")
            return OCRResult(
                text="", confidence=0.0, bounding_boxes=[],
                language='auto', engine='google_vision',
                processing_time=0.0, metadata={'error': str(e)}
            )
    
    def extract_text_azure(self, image: np.ndarray) -> OCRResult:
        """Extract text using Azure Cognitive Services"""
        if not self.azure_client:
            return OCRResult(
                text="", confidence=0.0, bounding_boxes=[],
                language="", engine='azure_vision',
                processing_time=0.0, metadata={'error': 'Client not available'}
            )
        
        try:
            start_time = time.time()
            
            # Convert image to bytes
            _, buffer = cv2.imencode('.png', image)
            image_stream = buffer.tobytes()
            
            # Perform OCR
            read_response = self.azure_client.read_in_stream(
                image_stream, raw=True
            )
            
            # Get operation ID
            operation_id = read_response.headers['Operation-Location'].split('/')[-1]
            
            # Wait for operation to complete
            while True:
                read_result = self.azure_client.get_read_result(operation_id)
                if read_result.status not in ['notStarted', 'running']:
                    break
                time.sleep(0.1)
            
            processing_time = time.time() - start_time
            
            # Extract text and bounding boxes
            text_blocks = []
            bounding_boxes = []
            
            if read_result.status == OperationStatusCodes.succeeded:
                for text_result in read_result.analyze_result.read_results:
                    for line in text_result.lines:
                        text_blocks.append(line.text)
                        
                        # Convert bounding box
                        bbox_points = line.bounding_box
                        x_coords = [bbox_points[i] for i in range(0, len(bbox_points), 2)]
                        y_coords = [bbox_points[i] for i in range(1, len(bbox_points), 2)]
                        bbox = [min(x_coords), min(y_coords), max(x_coords), max(y_coords)]
                        bounding_boxes.append(bbox)
                
                full_text = ' '.join(text_blocks)
                
                return OCRResult(
                    text=full_text,
                    confidence=0.9,  # Azure doesn't provide detailed confidence
                    bounding_boxes=bounding_boxes,
                    language='auto',
                    engine='azure_vision',
                    processing_time=processing_time,
                    metadata={'num_lines': len(text_blocks)}
                )
            else:
                return OCRResult(
                    text="", confidence=0.0, bounding_boxes=[],
                    language='auto', engine='azure_vision',
                    processing_time=processing_time,
                    metadata={'status': read_result.status}
                )
                
        except Exception as e:
            logger.error(f"Azure Vision OCR error: {e}")
            return OCRResult(
                text="", confidence=0.0, bounding_boxes=[],
                language='auto', engine='azure_vision',
                processing_time=0.0, metadata={'error': str(e)}
            )

class LocalOCRIntegrator:
    """Local OCR engines integration (EasyOCR, PaddleOCR)"""
    
    def __init__(self, config: OCRIntegrationConfig):
        self.config = config
        self.setup_local_engines()
    
    def setup_local_engines(self):
        """Setup local OCR engines"""
        # EasyOCR
        try:
            self.easyocr_reader = easyocr.Reader(
                self.config.languages,
                gpu=self.config.easyocr_gpu,
                model_storage_directory=self.config.easyocr_models_path
            )
            logger.info("EasyOCR reader initialized")
        except Exception as e:
            logger.warning(f"Failed to initialize EasyOCR: {e}")
            self.easyocr_reader = None
        
        # PaddleOCR
        try:
            self.paddleocr_reader = PaddleOCR(
                use_angle_cls=True,
                lang='en',  # Primary language
                use_gpu=self.config.paddleocr_use_gpu,
                show_log=False
            )
            logger.info("PaddleOCR reader initialized")
        except Exception as e:
            logger.warning(f"Failed to initialize PaddleOCR: {e}")
            self.paddleocr_reader = None
    
    def extract_text_easyocr(self, image: np.ndarray) -> OCRResult:
        """Extract text using EasyOCR"""
        if not self.easyocr_reader:
            return OCRResult(
                text="", confidence=0.0, bounding_boxes=[],
                language="", engine='easyocr',
                processing_time=0.0, metadata={'error': 'Reader not available'}
            )
        
        try:
            start_time = time.time()
            
            # Perform OCR
            results = self.easyocr_reader.readtext(image)
            
            processing_time = time.time() - start_time
            
            # Process results
            text_blocks = []
            confidences = []
            bounding_boxes = []
            
            for (bbox, text, confidence) in results:
                if confidence > self.config.min_confidence:
                    text_blocks.append(text)
                    confidences.append(confidence)
                    
                    # Convert bbox format
                    x_coords = [point[0] for point in bbox]
                    y_coords = [point[1] for point in bbox]
                    bbox_rect = [min(x_coords), min(y_coords), max(x_coords), max(y_coords)]
                    bounding_boxes.append(bbox_rect)
            
            full_text = ' '.join(text_blocks)
            avg_confidence = np.mean(confidences) if confidences else 0.0
            
            return OCRResult(
                text=full_text,
                confidence=avg_confidence,
                bounding_boxes=bounding_boxes,
                language='auto',
                engine='easyocr',
                processing_time=processing_time,
                metadata={'num_detections': len(text_blocks)}
            )
            
        except Exception as e:
            logger.error(f"EasyOCR error: {e}")
            return OCRResult(
                text="", confidence=0.0, bounding_boxes=[],
                language='auto', engine='easyocr',
                processing_time=0.0, metadata={'error': str(e)}
            )
    
    def extract_text_paddleocr(self, image: np.ndarray) -> OCRResult:
        """Extract text using PaddleOCR"""
        if not self.paddleocr_reader:
            return OCRResult(
                text="", confidence=0.0, bounding_boxes=[],
                language="", engine='paddleocr',
                processing_time=0.0, metadata={'error': 'Reader not available'}
            )
        
        try:
            start_time = time.time()
            
            # Perform OCR
            results = self.paddleocr_reader.ocr(image, cls=True)
            
            processing_time = time.time() - start_time
            
            # Process results
            text_blocks = []
            confidences = []
            bounding_boxes = []
            
            if results and results[0]:
                for line in results[0]:
                    if len(line) >= 2:
                        bbox, (text, confidence) = line[0], line[1]
                        
                        if confidence > self.config.min_confidence:
                            text_blocks.append(text)
                            confidences.append(confidence)
                            
                            # Convert bbox format
                            x_coords = [point[0] for point in bbox]
                            y_coords = [point[1] for point in bbox]
                            bbox_rect = [min(x_coords), min(y_coords), max(x_coords), max(y_coords)]
                            bounding_boxes.append(bbox_rect)
            
            full_text = ' '.join(text_blocks)
            avg_confidence = np.mean(confidences) if confidences else 0.0
            
            return OCRResult(
                text=full_text,
                confidence=avg_confidence,
                bounding_boxes=bounding_boxes,
                language='auto',
                engine='paddleocr',
                processing_time=processing_time,
                metadata={'num_detections': len(text_blocks)}
            )
            
        except Exception as e:
            logger.error(f"PaddleOCR error: {e}")
            return OCRResult(
                text="", confidence=0.0, bounding_boxes=[],
                language='auto', engine='paddleocr',
                processing_time=0.0, metadata={'error': str(e)}
            )

class OCRIntegrationManager:
    """Main OCR integration manager"""
    
    def __init__(self, config: OCRIntegrationConfig):
        self.config = config
        self.setup_directories()
        
        # Initialize integrators
        self.tesseract_integrator = TesseractIntegrator(config)
        self.cloud_integrator = CloudOCRIntegrator(config)
        self.local_integrator = LocalOCRIntegrator(config)
        
        # Performance tracking
        self.performance_metrics = {
            'tesseract': {'total_time': 0, 'total_calls': 0},
            'google_vision': {'total_time': 0, 'total_calls': 0},
            'azure_vision': {'total_time': 0, 'total_calls': 0},
            'easyocr': {'total_time': 0, 'total_calls': 0},
            'paddleocr': {'total_time': 0, 'total_calls': 0}
        }
    
    def setup_directories(self):
        """Create necessary directories"""
        for path in [self.config.output_path, self.config.training_data_path]:
            Path(path).mkdir(parents=True, exist_ok=True)
    
    def extract_text_ensemble(self, image: np.ndarray, 
                             engines: List[str] = None) -> Dict[str, OCRResult]:
        """Extract text using multiple OCR engines"""
        if engines is None:
            engines = ['tesseract', 'easyocr', 'paddleocr']
        
        results = {}
        
        # Run OCR engines in parallel if enabled
        if self.config.parallel_processing:
            with ThreadPoolExecutor(max_workers=self.config.max_workers) as executor:
                future_to_engine = {}
                
                for engine in engines:
                    if engine == 'tesseract':
                        future = executor.submit(self.tesseract_integrator.extract_text, image)
                    elif engine == 'google_vision':
                        future = executor.submit(self.cloud_integrator.extract_text_google, image)
                    elif engine == 'azure_vision':
                        future = executor.submit(self.cloud_integrator.extract_text_azure, image)
                    elif engine == 'easyocr':
                        future = executor.submit(self.local_integrator.extract_text_easyocr, image)
                    elif engine == 'paddleocr':
                        future = executor.submit(self.local_integrator.extract_text_paddleocr, image)
                    else:
                        continue
                    
                    future_to_engine[future] = engine
                
                # Collect results
                for future in as_completed(future_to_engine, timeout=self.config.timeout_seconds):
                    engine = future_to_engine[future]
                    try:
                        result = future.result()
                        results[engine] = result
                        
                        # Update performance metrics
                        self.performance_metrics[engine]['total_time'] += result.processing_time
                        self.performance_metrics[engine]['total_calls'] += 1
                        
                    except Exception as e:
                        logger.error(f"Error in {engine}: {e}")
                        results[engine] = OCRResult(
                            text="", confidence=0.0, bounding_boxes=[],
                            language="", engine=engine,
                            processing_time=0.0, metadata={'error': str(e)}
                        )
        else:
            # Sequential processing
            for engine in engines:
                try:
                    if engine == 'tesseract':
                        result = self.tesseract_integrator.extract_text(image)
                    elif engine == 'google_vision':
                        result = self.cloud_integrator.extract_text_google(image)
                    elif engine == 'azure_vision':
                        result = self.cloud_integrator.extract_text_azure(image)
                    elif engine == 'easyocr':
                        result = self.local_integrator.extract_text_easyocr(image)
                    elif engine == 'paddleocr':
                        result = self.local_integrator.extract_text_paddleocr(image)
                    else:
                        continue
                    
                    results[engine] = result
                    
                    # Update performance metrics
                    self.performance_metrics[engine]['total_time'] += result.processing_time
                    self.performance_metrics[engine]['total_calls'] += 1
                    
                except Exception as e:
                    logger.error(f"Error in {engine}: {e}")
                    results[engine] = OCRResult(
                        text="", confidence=0.0, bounding_boxes=[],
                        language="", engine=engine,
                        processing_time=0.0, metadata={'error': str(e)}
                    )
        
        return results
    
    def create_consensus_result(self, results: Dict[str, OCRResult]) -> OCRResult:
        """Create consensus result from multiple OCR engines"""
        valid_results = {k: v for k, v in results.items() 
                        if v.confidence > self.config.min_confidence and v.text.strip()}
        
        if not valid_results:
            return OCRResult(
                text="", confidence=0.0, bounding_boxes=[],
                language="", engine='consensus',
                processing_time=0.0, metadata={'num_engines': 0}
            )
        
        # Simple consensus: use result with highest confidence
        best_engine = max(valid_results.keys(), key=lambda k: valid_results[k].confidence)
        best_result = valid_results[best_engine]
        
        # Calculate average processing time
        avg_processing_time = np.mean([r.processing_time for r in results.values()])
        
        return OCRResult(
            text=best_result.text,
            confidence=best_result.confidence,
            bounding_boxes=best_result.bounding_boxes,
            language=best_result.language,
            engine='consensus',
            processing_time=avg_processing_time,
            metadata={
                'best_engine': best_engine,
                'num_engines': len(valid_results),
                'all_results': {k: v.text for k, v in valid_results.items()}
            }
        )
    
    def benchmark_engines(self, test_images: List[np.ndarray], 
                         ground_truths: List[str]) -> Dict[str, Any]:
        """Benchmark OCR engines performance"""
        logger.info(f"Benchmarking OCR engines on {len(test_images)} images")
        
        engines = ['tesseract', 'easyocr', 'paddleocr']
        if self.cloud_integrator.google_client:
            engines.append('google_vision')
        if self.cloud_integrator.azure_client:
            engines.append('azure_vision')
        
        benchmark_results = {}
        
        for engine in engines:
            logger.info(f"Benchmarking {engine}...")
            
            predictions = []
            processing_times = []
            confidences = []
            
            for image in test_images:
                results = self.extract_text_ensemble(image, engines=[engine])
                result = results.get(engine)
                
                if result:
                    predictions.append(result.text)
                    processing_times.append(result.processing_time)
                    confidences.append(result.confidence)
                else:
                    predictions.append("")
                    processing_times.append(0.0)
                    confidences.append(0.0)
            
            # Calculate metrics
            from difflib import SequenceMatcher
            
            accuracies = []
            edit_distances = []
            
            for pred, gt in zip(predictions, ground_truths):
                # Character-level accuracy
                accuracy = SequenceMatcher(None, pred.lower(), gt.lower()).ratio()
                accuracies.append(accuracy)
                
                # Edit distance
                import editdistance
                edit_dist = editdistance.eval(pred.lower(), gt.lower())
                edit_distances.append(edit_dist)
            
            benchmark_results[engine] = {
                'accuracy': np.mean(accuracies),
                'avg_edit_distance': np.mean(edit_distances),
                'avg_processing_time': np.mean(processing_times),
                'avg_confidence': np.mean(confidences),
                'predictions': predictions,
                'processing_times': processing_times,
                'confidences': confidences
            }
        
        # Save benchmark results
        benchmark_path = Path(self.config.output_path) / 'benchmark_results.json'
        with open(benchmark_path, 'w') as f:
            json.dump(benchmark_results, f, indent=2, default=str)
        
        logger.info(f"Benchmark results saved to {benchmark_path}")
        
        return benchmark_results
    
    def save_integration_metadata(self):
        """Save integration metadata and performance metrics"""
        metadata = {
            'config': self.config.__dict__,
            'performance_metrics': self.performance_metrics,
            'integration_timestamp': datetime.now().isoformat(),
            'available_engines': {
                'tesseract': True,
                'google_vision': self.cloud_integrator.google_client is not None,
                'azure_vision': self.cloud_integrator.azure_client is not None,
                'easyocr': self.local_integrator.easyocr_reader is not None,
                'paddleocr': self.local_integrator.paddleocr_reader is not None
            }
        }
        
        metadata_path = Path(self.config.output_path) / 'integration_metadata.json'
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2, default=str)
        
        logger.info(f"Integration metadata saved to {metadata_path}")

def main():
    """Main function for standalone execution"""
    # Initialize configuration
    config = OCRIntegrationConfig()
    
    # Create integration manager
    manager = OCRIntegrationManager(config)
    
    # Test with a sample image (if available)
    test_image_path = "data_collection/real/sample_document.jpg"
    if os.path.exists(test_image_path):
        logger.info(f"Testing OCR integration with {test_image_path}")
        
        # Load test image
        image = cv2.imread(test_image_path)
        
        # Extract text using ensemble
        results = manager.extract_text_ensemble(image)
        
        # Create consensus result
        consensus = manager.create_consensus_result(results)
        
        # Print results
        print("\n" + "="*50)
        print("OCR INTEGRATION TEST RESULTS")
        print("="*50)
        
        for engine, result in results.items():
            print(f"\n{engine.upper()}:")
            print(f"  Text: {result.text[:100]}..." if len(result.text) > 100 else f"  Text: {result.text}")
            print(f"  Confidence: {result.confidence:.3f}")
            print(f"  Processing time: {result.processing_time:.3f}s")
        
        print(f"\nCONSENSUS:")
        print(f"  Text: {consensus.text[:100]}..." if len(consensus.text) > 100 else f"  Text: {consensus.text}")
        print(f"  Confidence: {consensus.confidence:.3f}")
        print(f"  Best engine: {consensus.metadata.get('best_engine', 'unknown')}")
        print("="*50)
    
    # Save metadata
    manager.save_integration_metadata()
    
    logger.info("OCR integration setup completed")

if __name__ == "__main__":
    main()