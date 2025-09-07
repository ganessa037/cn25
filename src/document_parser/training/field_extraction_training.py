"""Field Extraction Models Training Module

This module provides comprehensive field extraction model training functionality including
template-based extraction, ML-based extraction with NER models, and coordinate-based extraction,
following the autocorrect model's organizational patterns.
"""

import os
import json
import pickle
import re
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any, Union
from dataclasses import dataclass, field
from datetime import datetime
import logging
from collections import defaultdict

import cv2
import numpy as np
import pandas as pd
from PIL import Image
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.preprocessing import LabelEncoder

# NLP libraries
import spacy
from transformers import (
    AutoTokenizer, AutoModelForTokenClassification,
    TrainingArguments, Trainer, DataCollatorForTokenClassification
)
from datasets import Dataset as HFDataset

# Template matching
from fuzzywuzzy import fuzz, process
import jellyfish

# Import from our modules
from ..models.field_extractor import FieldExtractionResult, FieldExtractor
from ..models.ocr_engines import OCRResult

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class FieldExtractionConfig:
    """Configuration for field extraction model training"""
    
    # Document types and fields
    document_types: List[str] = field(default_factory=lambda: [
        'invoice', 'receipt', 'identity_card', 'passport', 'bank_statement'
    ])
    
    # Common fields for Malaysian documents
    field_types: Dict[str, List[str]] = field(default_factory=lambda: {
        'invoice': ['invoice_number', 'date', 'total_amount', 'vendor_name', 'tax_amount'],
        'receipt': ['receipt_number', 'date', 'total_amount', 'merchant_name', 'items'],
        'identity_card': ['ic_number', 'name', 'date_of_birth', 'address', 'nationality'],
        'passport': ['passport_number', 'name', 'date_of_birth', 'nationality', 'expiry_date'],
        'bank_statement': ['account_number', 'statement_date', 'balance', 'bank_name', 'transactions']
    })
    
    # Template-based extraction
    template_patterns: Dict[str, Dict[str, str]] = field(default_factory=lambda: {
        'invoice': {
            'invoice_number': r'(?:invoice|inv)\s*#?\s*:?\s*([A-Z0-9-]+)',
            'date': r'(?:date|dated)\s*:?\s*(\d{1,2}[/-]\d{1,2}[/-]\d{2,4})',
            'total_amount': r'(?:total|amount)\s*:?\s*(?:rm|\$)?\s*(\d+\.\d{2})',
            'vendor_name': r'(?:from|vendor|company)\s*:?\s*([A-Za-z\s&.,]+)'
        },
        'receipt': {
            'receipt_number': r'(?:receipt|rcpt)\s*#?\s*:?\s*([A-Z0-9-]+)',
            'date': r'(?:date|time)\s*:?\s*(\d{1,2}[/-]\d{1,2}[/-]\d{2,4})',
            'total_amount': r'(?:total|amount)\s*:?\s*(?:rm|\$)?\s*(\d+\.\d{2})',
            'merchant_name': r'^([A-Za-z\s&.,]+)(?:\n|$)'
        }
    })
    
    # NER model configuration
    ner_model_name: str = 'bert-base-multilingual-cased'
    ner_max_length: int = 512
    ner_learning_rate: float = 2e-5
    ner_batch_size: int = 16
    ner_num_epochs: int = 10
    
    # Coordinate-based extraction
    coordinate_features: List[str] = field(default_factory=lambda: [
        'x_position', 'y_position', 'width', 'height', 'area', 'aspect_ratio'
    ])
    
    # Training configuration
    train_split: float = 0.8
    val_split: float = 0.1
    test_split: float = 0.1
    random_seed: int = 42
    
    # Output paths
    output_path: str = "model_artifacts/document_parser/field_extraction"
    training_data_path: str = "model_artifacts/document_parser/field_training_data"
    templates_path: str = "model_artifacts/document_parser/templates"
    
    # Performance thresholds
    min_confidence: float = 0.7
    fuzzy_match_threshold: int = 80
    coordinate_distance_threshold: float = 50.0
    
    # Evaluation metrics
    evaluation_metrics: List[str] = field(default_factory=lambda: [
        'precision', 'recall', 'f1_score', 'exact_match', 'partial_match'
    ])

class TemplateBasedExtractor:
    """Template-based field extraction using regex patterns and fuzzy matching"""
    
    def __init__(self, config: FieldExtractionConfig):
        self.config = config
        self.templates = {}
        self.load_templates()
    
    def load_templates(self):
        """Load document templates"""
        templates_dir = Path(self.config.templates_path)
        templates_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize with default patterns
        self.templates = self.config.template_patterns.copy()
        
        # Load custom templates if available
        for doc_type in self.config.document_types:
            template_file = templates_dir / f'{doc_type}_template.json'
            if template_file.exists():
                with open(template_file, 'r') as f:
                    custom_template = json.load(f)
                    self.templates[doc_type] = {**self.templates.get(doc_type, {}), **custom_template}
        
        logger.info(f"Loaded templates for {len(self.templates)} document types")
    
    def extract_fields_regex(self, text: str, document_type: str) -> Dict[str, Any]:
        """Extract fields using regex patterns"""
        if document_type not in self.templates:
            return {}
        
        extracted_fields = {}
        patterns = self.templates[document_type]
        
        for field_name, pattern in patterns.items():
            try:
                matches = re.finditer(pattern, text, re.IGNORECASE | re.MULTILINE)
                field_values = []
                
                for match in matches:
                    if match.groups():
                        value = match.group(1).strip()
                        if value:
                            field_values.append(value)
                
                if field_values:
                    # Take the first match or most common value
                    extracted_fields[field_name] = field_values[0]
                    
            except re.error as e:
                logger.warning(f"Invalid regex pattern for {field_name}: {e}")
        
        return extracted_fields
    
    def extract_fields_fuzzy(self, text: str, document_type: str, 
                           known_values: Dict[str, List[str]] = None) -> Dict[str, Any]:
        """Extract fields using fuzzy string matching"""
        if not known_values:
            return {}
        
        extracted_fields = {}
        text_lines = text.split('\n')
        
        for field_name, possible_values in known_values.items():
            best_match = None
            best_score = 0
            
            for line in text_lines:
                line = line.strip()
                if not line:
                    continue
                
                # Find best fuzzy match
                match, score = process.extractOne(line, possible_values, scorer=fuzz.ratio)
                
                if score > self.config.fuzzy_match_threshold and score > best_score:
                    best_match = line
                    best_score = score
            
            if best_match:
                extracted_fields[field_name] = best_match
        
        return extracted_fields
    
    def create_template_from_samples(self, samples: List[Dict[str, Any]], 
                                   document_type: str) -> Dict[str, str]:
        """Create template patterns from annotated samples"""
        field_patterns = {}
        
        for field_name in self.config.field_types.get(document_type, []):
            patterns = []
            
            for sample in samples:
                if field_name in sample.get('fields', {}):
                    field_value = sample['fields'][field_name]
                    text = sample.get('text', '')
                    
                    # Find context around the field value
                    field_index = text.lower().find(field_value.lower())
                    if field_index != -1:
                        # Extract context before and after
                        start = max(0, field_index - 20)
                        end = min(len(text), field_index + len(field_value) + 20)
                        context = text[start:end]
                        
                        # Create pattern (simplified)
                        escaped_value = re.escape(field_value)
                        pattern = context.replace(field_value, f'({escaped_value})')
                        patterns.append(pattern)
            
            if patterns:
                # Use the most common pattern or create a generalized one
                field_patterns[field_name] = patterns[0]  # Simplified
        
        return field_patterns

class NERBasedExtractor:
    """NER-based field extraction using transformer models"""
    
    def __init__(self, config: FieldExtractionConfig):
        self.config = config
        self.tokenizer = None
        self.model = None
        self.label_encoder = LabelEncoder()
        self.setup_model()
    
    def setup_model(self):
        """Setup NER model and tokenizer"""
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(self.config.ner_model_name)
            logger.info(f"Loaded tokenizer: {self.config.ner_model_name}")
        except Exception as e:
            logger.error(f"Failed to load tokenizer: {e}")
    
    def prepare_ner_data(self, training_samples: List[Dict[str, Any]]) -> Tuple[List, List]:
        """Prepare data for NER training"""
        texts = []
        labels = []
        
        for sample in training_samples:
            text = sample.get('text', '')
            annotations = sample.get('annotations', [])
            
            # Tokenize text
            tokens = self.tokenizer.tokenize(text)
            token_labels = ['O'] * len(tokens)
            
            # Map annotations to tokens
            for annotation in annotations:
                field_name = annotation.get('field')
                start_char = annotation.get('start')
                end_char = annotation.get('end')
                
                if field_name and start_char is not None and end_char is not None:
                    # Find corresponding tokens
                    token_start = None
                    token_end = None
                    
                    char_to_token = []
                    current_pos = 0
                    
                    for i, token in enumerate(tokens):
                        token_text = self.tokenizer.convert_tokens_to_string([token])
                        token_len = len(token_text)
                        
                        for _ in range(token_len):
                            char_to_token.append(i)
                        current_pos += token_len
                    
                    if start_char < len(char_to_token):
                        token_start = char_to_token[start_char]
                        token_end = char_to_token[min(end_char - 1, len(char_to_token) - 1)]
                        
                        # Apply BIO tagging
                        if token_start is not None and token_end is not None:
                            token_labels[token_start] = f'B-{field_name}'
                            for i in range(token_start + 1, token_end + 1):
                                if i < len(token_labels):
                                    token_labels[i] = f'I-{field_name}'
            
            texts.append(tokens)
            labels.append(token_labels)
        
        return texts, labels
    
    def train_ner_model(self, training_samples: List[Dict[str, Any]]):
        """Train NER model"""
        logger.info(f"Training NER model on {len(training_samples)} samples")
        
        # Prepare data
        texts, labels = self.prepare_ner_data(training_samples)
        
        # Create label mapping
        all_labels = set()
        for label_seq in labels:
            all_labels.update(label_seq)
        
        label_list = sorted(list(all_labels))
        label_to_id = {label: i for i, label in enumerate(label_list)}
        id_to_label = {i: label for label, i in label_to_id.items()}
        
        # Convert labels to IDs
        label_ids = []
        for label_seq in labels:
            label_ids.append([label_to_id[label] for label in label_seq])
        
        # Create dataset
        def tokenize_and_align_labels(examples):
            tokenized_inputs = self.tokenizer(
                examples['tokens'],
                truncation=True,
                is_split_into_words=True,
                padding=True,
                max_length=self.config.ner_max_length
            )
            
            labels = []
            for i, label in enumerate(examples['labels']):
                word_ids = tokenized_inputs.word_ids(batch_index=i)
                previous_word_idx = None
                label_ids = []
                
                for word_idx in word_ids:
                    if word_idx is None:
                        label_ids.append(-100)
                    elif word_idx != previous_word_idx:
                        if word_idx < len(label):
                            label_ids.append(label[word_idx])
                        else:
                            label_ids.append(0)  # 'O' label
                    else:
                        label_ids.append(-100)
                    previous_word_idx = word_idx
                
                labels.append(label_ids)
            
            tokenized_inputs['labels'] = labels
            return tokenized_inputs
        
        # Split data
        train_texts, val_texts, train_labels, val_labels = train_test_split(
            texts, label_ids, test_size=0.2, random_state=self.config.random_seed
        )
        
        # Create HuggingFace datasets
        train_dataset = HFDataset.from_dict({
            'tokens': train_texts,
            'labels': train_labels
        })
        
        val_dataset = HFDataset.from_dict({
            'tokens': val_texts,
            'labels': val_labels
        })
        
        # Tokenize datasets
        train_dataset = train_dataset.map(tokenize_and_align_labels, batched=True)
        val_dataset = val_dataset.map(tokenize_and_align_labels, batched=True)
        
        # Initialize model
        self.model = AutoModelForTokenClassification.from_pretrained(
            self.config.ner_model_name,
            num_labels=len(label_list),
            id2label=id_to_label,
            label2id=label_to_id
        )
        
        # Training arguments
        training_args = TrainingArguments(
            output_dir=str(Path(self.config.output_path) / 'ner_model'),
            num_train_epochs=self.config.ner_num_epochs,
            per_device_train_batch_size=self.config.ner_batch_size,
            per_device_eval_batch_size=self.config.ner_batch_size,
            learning_rate=self.config.ner_learning_rate,
            weight_decay=0.01,
            logging_dir=str(Path(self.config.output_path) / 'logs'),
            logging_steps=100,
            evaluation_strategy='epoch',
            save_strategy='epoch',
            load_best_model_at_end=True,
            metric_for_best_model='eval_loss',
            greater_is_better=False
        )
        
        # Data collator
        data_collator = DataCollatorForTokenClassification(
            tokenizer=self.tokenizer,
            padding=True
        )
        
        # Trainer
        trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=val_dataset,
            data_collator=data_collator,
            tokenizer=self.tokenizer
        )
        
        # Train model
        trainer.train()
        
        # Save model
        model_path = Path(self.config.output_path) / 'ner_model'
        trainer.save_model(str(model_path))
        
        # Save label mapping
        label_mapping_path = model_path / 'label_mapping.json'
        with open(label_mapping_path, 'w') as f:
            json.dump({'id2label': id_to_label, 'label2id': label_to_id}, f, indent=2)
        
        logger.info(f"NER model saved to {model_path}")
    
    def extract_fields_ner(self, text: str) -> Dict[str, Any]:
        """Extract fields using trained NER model"""
        if not self.model or not self.tokenizer:
            return {}
        
        try:
            # Tokenize input
            inputs = self.tokenizer(
                text,
                return_tensors='pt',
                truncation=True,
                max_length=self.config.ner_max_length,
                padding=True
            )
            
            # Get predictions
            with torch.no_grad():
                outputs = self.model(**inputs)
                predictions = torch.argmax(outputs.logits, dim=-1)
            
            # Convert predictions to labels
            tokens = self.tokenizer.convert_ids_to_tokens(inputs['input_ids'][0])
            predicted_labels = [self.model.config.id2label[pred.item()] for pred in predictions[0]]
            
            # Extract fields from predictions
            extracted_fields = {}
            current_field = None
            current_tokens = []
            
            for token, label in zip(tokens, predicted_labels):
                if label.startswith('B-'):
                    # Save previous field
                    if current_field and current_tokens:
                        field_text = self.tokenizer.convert_tokens_to_string(current_tokens)
                        extracted_fields[current_field] = field_text.strip()
                    
                    # Start new field
                    current_field = label[2:]
                    current_tokens = [token]
                    
                elif label.startswith('I-') and current_field:
                    current_tokens.append(token)
                    
                else:
                    # Save current field if exists
                    if current_field and current_tokens:
                        field_text = self.tokenizer.convert_tokens_to_string(current_tokens)
                        extracted_fields[current_field] = field_text.strip()
                    
                    current_field = None
                    current_tokens = []
            
            # Save last field
            if current_field and current_tokens:
                field_text = self.tokenizer.convert_tokens_to_string(current_tokens)
                extracted_fields[current_field] = field_text.strip()
            
            return extracted_fields
            
        except Exception as e:
            logger.error(f"NER extraction error: {e}")
            return {}

class CoordinateBasedExtractor:
    """Coordinate-based field extraction using spatial relationships"""
    
    def __init__(self, config: FieldExtractionConfig):
        self.config = config
        self.spatial_models = {}
    
    def extract_spatial_features(self, ocr_result: OCRResult) -> List[Dict[str, Any]]:
        """Extract spatial features from OCR results"""
        features = []
        
        for i, bbox in enumerate(ocr_result.bounding_boxes):
            if len(bbox) >= 4:
                x1, y1, x2, y2 = bbox[:4]
                width = x2 - x1
                height = y2 - y1
                area = width * height
                aspect_ratio = width / height if height > 0 else 0
                
                # Get corresponding text
                text = ""
                if hasattr(ocr_result, 'metadata') and 'word_texts' in ocr_result.metadata:
                    if i < len(ocr_result.metadata['word_texts']):
                        text = ocr_result.metadata['word_texts'][i]
                
                feature = {
                    'text': text,
                    'x_position': x1,
                    'y_position': y1,
                    'width': width,
                    'height': height,
                    'area': area,
                    'aspect_ratio': aspect_ratio,
                    'center_x': (x1 + x2) / 2,
                    'center_y': (y1 + y2) / 2
                }
                
                features.append(feature)
        
        return features
    
    def find_fields_by_position(self, features: List[Dict[str, Any]], 
                              field_templates: Dict[str, Dict[str, Any]]) -> Dict[str, Any]:
        """Find fields based on spatial position templates"""
        extracted_fields = {}
        
        for field_name, template in field_templates.items():
            expected_x = template.get('x_position')
            expected_y = template.get('y_position')
            tolerance = template.get('tolerance', self.config.coordinate_distance_threshold)
            
            if expected_x is not None and expected_y is not None:
                best_match = None
                best_distance = float('inf')
                
                for feature in features:
                    # Calculate distance
                    distance = np.sqrt(
                        (feature['center_x'] - expected_x) ** 2 + 
                        (feature['center_y'] - expected_y) ** 2
                    )
                    
                    if distance < tolerance and distance < best_distance:
                        best_match = feature
                        best_distance = distance
                
                if best_match:
                    extracted_fields[field_name] = best_match['text']
        
        return extracted_fields
    
    def create_spatial_template(self, training_samples: List[Dict[str, Any]], 
                              document_type: str) -> Dict[str, Dict[str, Any]]:
        """Create spatial template from training samples"""
        field_positions = defaultdict(list)
        
        for sample in training_samples:
            ocr_result = sample.get('ocr_result')
            annotations = sample.get('annotations', [])
            
            if not ocr_result:
                continue
            
            features = self.extract_spatial_features(ocr_result)
            
            for annotation in annotations:
                field_name = annotation.get('field')
                bbox = annotation.get('bbox')
                
                if field_name and bbox and len(bbox) >= 4:
                    x1, y1, x2, y2 = bbox[:4]
                    center_x = (x1 + x2) / 2
                    center_y = (y1 + y2) / 2
                    
                    field_positions[field_name].append({
                        'center_x': center_x,
                        'center_y': center_y,
                        'width': x2 - x1,
                        'height': y2 - y1
                    })
        
        # Calculate average positions
        spatial_template = {}
        for field_name, positions in field_positions.items():
            if positions:
                avg_x = np.mean([pos['center_x'] for pos in positions])
                avg_y = np.mean([pos['center_y'] for pos in positions])
                std_x = np.std([pos['center_x'] for pos in positions])
                std_y = np.std([pos['center_y'] for pos in positions])
                
                spatial_template[field_name] = {
                    'x_position': avg_x,
                    'y_position': avg_y,
                    'tolerance': max(std_x, std_y, self.config.coordinate_distance_threshold)
                }
        
        return spatial_template

class FieldExtractionTrainer:
    """Main field extraction training manager"""
    
    def __init__(self, config: FieldExtractionConfig):
        self.config = config
        self.setup_directories()
        
        # Initialize extractors
        self.template_extractor = TemplateBasedExtractor(config)
        self.ner_extractor = NERBasedExtractor(config)
        self.coordinate_extractor = CoordinateBasedExtractor(config)
        
        # Training data
        self.training_data = []
        self.validation_data = []
        self.test_data = []
    
    def setup_directories(self):
        """Create necessary directories"""
        for path in [self.config.output_path, self.config.training_data_path, self.config.templates_path]:
            Path(path).mkdir(parents=True, exist_ok=True)
    
    def load_training_data(self, data_path: str):
        """Load training data from file"""
        data_file = Path(data_path)
        if data_file.exists():
            with open(data_file, 'r') as f:
                data = json.load(f)
            
            # Split data
            train_size = int(len(data) * self.config.train_split)
            val_size = int(len(data) * self.config.val_split)
            
            self.training_data = data[:train_size]
            self.validation_data = data[train_size:train_size + val_size]
            self.test_data = data[train_size + val_size:]
            
            logger.info(f"Loaded {len(self.training_data)} training, {len(self.validation_data)} validation, {len(self.test_data)} test samples")
        else:
            logger.warning(f"Training data file not found: {data_path}")
    
    def train_all_models(self):
        """Train all field extraction models"""
        logger.info("Starting field extraction model training")
        
        # Train template-based models
        for doc_type in self.config.document_types:
            doc_samples = [s for s in self.training_data if s.get('document_type') == doc_type]
            if doc_samples:
                logger.info(f"Training template model for {doc_type}")
                template_patterns = self.template_extractor.create_template_from_samples(doc_samples, doc_type)
                
                # Save template
                template_file = Path(self.config.templates_path) / f'{doc_type}_template.json'
                with open(template_file, 'w') as f:
                    json.dump(template_patterns, f, indent=2)
                
                # Create spatial template
                spatial_template = self.coordinate_extractor.create_spatial_template(doc_samples, doc_type)
                spatial_file = Path(self.config.templates_path) / f'{doc_type}_spatial.json'
                with open(spatial_file, 'w') as f:
                    json.dump(spatial_template, f, indent=2)
        
        # Train NER model
        if self.training_data:
            self.ner_extractor.train_ner_model(self.training_data)
        
        logger.info("Field extraction model training completed")
    
    def evaluate_models(self) -> Dict[str, Any]:
        """Evaluate all field extraction models"""
        logger.info("Evaluating field extraction models")
        
        evaluation_results = {
            'template_based': {},
            'ner_based': {},
            'coordinate_based': {},
            'ensemble': {}
        }
        
        for sample in self.test_data:
            text = sample.get('text', '')
            document_type = sample.get('document_type', '')
            ground_truth = sample.get('fields', {})
            ocr_result = sample.get('ocr_result')
            
            # Template-based extraction
            template_results = self.template_extractor.extract_fields_regex(text, document_type)
            
            # NER-based extraction
            ner_results = self.ner_extractor.extract_fields_ner(text)
            
            # Coordinate-based extraction
            coordinate_results = {}
            if ocr_result:
                features = self.coordinate_extractor.extract_spatial_features(ocr_result)
                spatial_template_file = Path(self.config.templates_path) / f'{document_type}_spatial.json'
                if spatial_template_file.exists():
                    with open(spatial_template_file, 'r') as f:
                        spatial_template = json.load(f)
                    coordinate_results = self.coordinate_extractor.find_fields_by_position(features, spatial_template)
            
            # Calculate metrics for each method
            for method, results in [('template_based', template_results), 
                                  ('ner_based', ner_results), 
                                  ('coordinate_based', coordinate_results)]:
                
                if method not in evaluation_results:
                    evaluation_results[method] = {'exact_matches': 0, 'partial_matches': 0, 'total_fields': 0}
                
                for field_name, true_value in ground_truth.items():
                    evaluation_results[method]['total_fields'] += 1
                    
                    if field_name in results:
                        predicted_value = results[field_name]
                        
                        # Exact match
                        if predicted_value.lower().strip() == true_value.lower().strip():
                            evaluation_results[method]['exact_matches'] += 1
                            evaluation_results[method]['partial_matches'] += 1
                        
                        # Partial match (fuzzy)
                        elif fuzz.ratio(predicted_value.lower(), true_value.lower()) > self.config.fuzzy_match_threshold:
                            evaluation_results[method]['partial_matches'] += 1
        
        # Calculate final metrics
        for method in evaluation_results:
            if evaluation_results[method].get('total_fields', 0) > 0:
                total = evaluation_results[method]['total_fields']
                exact = evaluation_results[method]['exact_matches']
                partial = evaluation_results[method]['partial_matches']
                
                evaluation_results[method]['exact_accuracy'] = exact / total
                evaluation_results[method]['partial_accuracy'] = partial / total
        
        # Save evaluation results
        eval_file = Path(self.config.output_path) / 'evaluation_results.json'
        with open(eval_file, 'w') as f:
            json.dump(evaluation_results, f, indent=2)
        
        logger.info(f"Evaluation results saved to {eval_file}")
        
        return evaluation_results
    
    def save_training_metadata(self):
        """Save training metadata"""
        metadata = {
            'config': self.config.__dict__,
            'training_timestamp': datetime.now().isoformat(),
            'training_data_size': len(self.training_data),
            'validation_data_size': len(self.validation_data),
            'test_data_size': len(self.test_data),
            'document_types': self.config.document_types,
            'field_types': self.config.field_types
        }
        
        metadata_path = Path(self.config.output_path) / 'training_metadata.json'
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2, default=str)
        
        logger.info(f"Training metadata saved to {metadata_path}")

def main():
    """Main function for standalone execution"""
    # Initialize configuration
    config = FieldExtractionConfig()
    
    # Create trainer
    trainer = FieldExtractionTrainer(config)
    
    # Load training data (if available)
    training_data_file = "data_collection/field_extraction_training.json"
    if os.path.exists(training_data_file):
        trainer.load_training_data(training_data_file)
        
        # Train models
        trainer.train_all_models()
        
        # Evaluate models
        evaluation_results = trainer.evaluate_models()
        
        # Print results
        print("\n" + "="*50)
        print("FIELD EXTRACTION TRAINING RESULTS")
        print("="*50)
        
        for method, metrics in evaluation_results.items():
            if 'exact_accuracy' in metrics:
                print(f"\n{method.upper()}:")
                print(f"  Exact Accuracy: {metrics['exact_accuracy']:.3f}")
                print(f"  Partial Accuracy: {metrics['partial_accuracy']:.3f}")
                print(f"  Total Fields: {metrics['total_fields']}")
        
        print("="*50)
    else:
        logger.info(f"Training data file not found: {training_data_file}")
        logger.info("Creating sample configuration and templates...")
    
    # Save metadata
    trainer.save_training_metadata()
    
    logger.info("Field extraction training setup completed")

if __name__ == "__main__":
    main()