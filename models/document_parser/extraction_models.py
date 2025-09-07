#!/usr/bin/env python3
"""
Extraction Models

Machine learning models and utilities for document classification,
field extraction, and OCR processing.
"""

import logging
import pickle
import json
from abc import ABC, abstractmethod
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Any, Union, Tuple
from dataclasses import dataclass, field
import numpy as np

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class ModelMetrics:
    """Model performance metrics."""
    accuracy: float = 0.0
    precision: float = 0.0
    recall: float = 0.0
    f1_score: float = 0.0
    confusion_matrix: Optional[List[List[int]]] = None
    classification_report: Optional[Dict] = None
    training_time: Optional[float] = None  # seconds
    inference_time: Optional[float] = None  # seconds per sample
    model_size: Optional[int] = None  # bytes
    training_samples: Optional[int] = None
    validation_samples: Optional[int] = None
    test_samples: Optional[int] = None
    created_timestamp: datetime = field(default_factory=datetime.now)
    
    def to_dict(self) -> Dict:
        """Convert to dictionary."""
        return {
            "accuracy": self.accuracy,
            "precision": self.precision,
            "recall": self.recall,
            "f1_score": self.f1_score,
            "confusion_matrix": self.confusion_matrix,
            "classification_report": self.classification_report,
            "training_time": self.training_time,
            "inference_time": self.inference_time,
            "model_size": self.model_size,
            "training_samples": self.training_samples,
            "validation_samples": self.validation_samples,
            "test_samples": self.test_samples,
            "created_timestamp": self.created_timestamp.isoformat()
        }
    
    @classmethod
    def from_dict(cls, data: Dict) -> 'ModelMetrics':
        """Create from dictionary."""
        created_ts = datetime.fromisoformat(data["created_timestamp"]) if data.get("created_timestamp") else datetime.now()
        
        return cls(
            accuracy=data.get("accuracy", 0.0),
            precision=data.get("precision", 0.0),
            recall=data.get("recall", 0.0),
            f1_score=data.get("f1_score", 0.0),
            confusion_matrix=data.get("confusion_matrix"),
            classification_report=data.get("classification_report"),
            training_time=data.get("training_time"),
            inference_time=data.get("inference_time"),
            model_size=data.get("model_size"),
            training_samples=data.get("training_samples"),
            validation_samples=data.get("validation_samples"),
            test_samples=data.get("test_samples"),
            created_timestamp=created_ts
        )

@dataclass
class ModelConfig:
    """Model configuration parameters."""
    model_name: str
    model_type: str
    version: str = "1.0.0"
    parameters: Dict[str, Any] = field(default_factory=dict)
    hyperparameters: Dict[str, Any] = field(default_factory=dict)
    preprocessing_steps: List[str] = field(default_factory=list)
    feature_columns: List[str] = field(default_factory=list)
    target_column: Optional[str] = None
    class_labels: List[str] = field(default_factory=list)
    input_shape: Optional[Tuple] = None
    output_shape: Optional[Tuple] = None
    created_timestamp: datetime = field(default_factory=datetime.now)
    
    def to_dict(self) -> Dict:
        """Convert to dictionary."""
        return {
            "model_name": self.model_name,
            "model_type": self.model_type,
            "version": self.version,
            "parameters": self.parameters,
            "hyperparameters": self.hyperparameters,
            "preprocessing_steps": self.preprocessing_steps,
            "feature_columns": self.feature_columns,
            "target_column": self.target_column,
            "class_labels": self.class_labels,
            "input_shape": list(self.input_shape) if self.input_shape else None,
            "output_shape": list(self.output_shape) if self.output_shape else None,
            "created_timestamp": self.created_timestamp.isoformat()
        }
    
    @classmethod
    def from_dict(cls, data: Dict) -> 'ModelConfig':
        """Create from dictionary."""
        created_ts = datetime.fromisoformat(data["created_timestamp"]) if data.get("created_timestamp") else datetime.now()
        
        input_shape = tuple(data["input_shape"]) if data.get("input_shape") else None
        output_shape = tuple(data["output_shape"]) if data.get("output_shape") else None
        
        return cls(
            model_name=data["model_name"],
            model_type=data["model_type"],
            version=data.get("version", "1.0.0"),
            parameters=data.get("parameters", {}),
            hyperparameters=data.get("hyperparameters", {}),
            preprocessing_steps=data.get("preprocessing_steps", []),
            feature_columns=data.get("feature_columns", []),
            target_column=data.get("target_column"),
            class_labels=data.get("class_labels", []),
            input_shape=input_shape,
            output_shape=output_shape,
            created_timestamp=created_ts
        )

class BaseModel(ABC):
    """Base class for all models."""
    
    def __init__(self, config: ModelConfig):
        self.config = config
        self.model = None
        self.is_trained = False
        self.metrics = None
        self.feature_names = []
        self.preprocessing_pipeline = None
        
        logger.info(f"Initialized {config.model_name} ({config.model_type})")
    
    @abstractmethod
    def train(self, X: Any, y: Any, **kwargs) -> ModelMetrics:
        """Train the model."""
        pass
    
    @abstractmethod
    def predict(self, X: Any) -> Any:
        """Make predictions."""
        pass
    
    @abstractmethod
    def predict_proba(self, X: Any) -> Any:
        """Get prediction probabilities."""
        pass
    
    def save_model(self, file_path: Union[str, Path]):
        """Save model to file."""
        file_path = Path(file_path)
        file_path.parent.mkdir(parents=True, exist_ok=True)
        
        model_data = {
            "config": self.config.to_dict(),
            "model": self.model,
            "is_trained": self.is_trained,
            "metrics": self.metrics.to_dict() if self.metrics else None,
            "feature_names": self.feature_names,
            "preprocessing_pipeline": self.preprocessing_pipeline
        }
        
        with open(file_path, 'wb') as f:
            pickle.dump(model_data, f)
        
        logger.info(f"Model saved to {file_path}")
    
    @classmethod
    def load_model(cls, file_path: Union[str, Path]) -> 'BaseModel':
        """Load model from file."""
        with open(file_path, 'rb') as f:
            model_data = pickle.load(f)
        
        config = ModelConfig.from_dict(model_data["config"])
        instance = cls(config)
        
        instance.model = model_data["model"]
        instance.is_trained = model_data["is_trained"]
        instance.feature_names = model_data.get("feature_names", [])
        instance.preprocessing_pipeline = model_data.get("preprocessing_pipeline")
        
        if model_data.get("metrics"):
            instance.metrics = ModelMetrics.from_dict(model_data["metrics"])
        
        logger.info(f"Model loaded from {file_path}")
        return instance
    
    def get_model_info(self) -> Dict:
        """Get model information."""
        return {
            "config": self.config.to_dict(),
            "is_trained": self.is_trained,
            "metrics": self.metrics.to_dict() if self.metrics else None,
            "feature_count": len(self.feature_names),
            "feature_names": self.feature_names[:10],  # First 10 features
            "has_preprocessing": self.preprocessing_pipeline is not None
        }

class ClassificationModel(BaseModel):
    """Document classification model."""
    
    def __init__(self, config: ModelConfig):
        super().__init__(config)
        self.label_encoder = None
        self.vectorizer = None
    
    def train(self, X: List[str], y: List[str], **kwargs) -> ModelMetrics:
        """Train the classification model."""
        try:
            from sklearn.feature_extraction.text import TfidfVectorizer
            from sklearn.preprocessing import LabelEncoder
            from sklearn.ensemble import RandomForestClassifier
            from sklearn.model_selection import train_test_split
            from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix, classification_report
            import time
            
            start_time = time.time()
            
            # Encode labels
            self.label_encoder = LabelEncoder()
            y_encoded = self.label_encoder.fit_transform(y)
            
            # Vectorize text
            self.vectorizer = TfidfVectorizer(
                max_features=self.config.parameters.get("max_features", 5000),
                ngram_range=self.config.parameters.get("ngram_range", (1, 2)),
                stop_words='english'
            )
            X_vectorized = self.vectorizer.fit_transform(X)
            
            # Split data
            test_size = kwargs.get("test_size", 0.2)
            X_train, X_test, y_train, y_test = train_test_split(
                X_vectorized, y_encoded, test_size=test_size, random_state=42, stratify=y_encoded
            )
            
            # Train model
            self.model = RandomForestClassifier(
                n_estimators=self.config.hyperparameters.get("n_estimators", 100),
                max_depth=self.config.hyperparameters.get("max_depth", None),
                random_state=42
            )
            
            self.model.fit(X_train, y_train)
            self.is_trained = True
            
            # Calculate metrics
            y_pred = self.model.predict(X_test)
            
            accuracy = accuracy_score(y_test, y_pred)
            precision, recall, f1, _ = precision_recall_fscore_support(y_test, y_pred, average='weighted')
            conf_matrix = confusion_matrix(y_test, y_pred).tolist()
            class_report = classification_report(y_test, y_pred, target_names=self.label_encoder.classes_, output_dict=True)
            
            training_time = time.time() - start_time
            
            self.metrics = ModelMetrics(
                accuracy=accuracy,
                precision=precision,
                recall=recall,
                f1_score=f1,
                confusion_matrix=conf_matrix,
                classification_report=class_report,
                training_time=training_time,
                training_samples=len(X_train),
                test_samples=len(X_test)
            )
            
            # Store feature names
            self.feature_names = self.vectorizer.get_feature_names_out().tolist()
            
            logger.info(f"Classification model trained - Accuracy: {accuracy:.3f}, F1: {f1:.3f}")
            return self.metrics
            
        except Exception as e:
            logger.error(f"Training error: {e}")
            raise
    
    def predict(self, X: List[str]) -> List[str]:
        """Predict document types."""
        if not self.is_trained:
            raise ValueError("Model must be trained before prediction")
        
        X_vectorized = self.vectorizer.transform(X)
        y_pred_encoded = self.model.predict(X_vectorized)
        return self.label_encoder.inverse_transform(y_pred_encoded).tolist()
    
    def predict_proba(self, X: List[str]) -> np.ndarray:
        """Get prediction probabilities."""
        if not self.is_trained:
            raise ValueError("Model must be trained before prediction")
        
        X_vectorized = self.vectorizer.transform(X)
        return self.model.predict_proba(X_vectorized)
    
    def predict_single(self, text: str) -> Tuple[str, float]:
        """Predict single document with confidence."""
        predictions = self.predict([text])
        probabilities = self.predict_proba([text])
        
        predicted_class = predictions[0]
        confidence = np.max(probabilities[0])
        
        return predicted_class, confidence

class FieldExtractionModel(BaseModel):
    """Field extraction model using NER and pattern matching."""
    
    def __init__(self, config: ModelConfig):
        super().__init__(config)
        self.field_patterns = {}
        self.field_extractors = {}
    
    def train(self, training_data: Dict[str, List], **kwargs) -> ModelMetrics:
        """Train field extraction patterns."""
        try:
            import re
            from collections import defaultdict
            import time
            
            start_time = time.time()
            
            # Extract patterns for each field type
            field_examples = defaultdict(list)
            
            for document in training_data.get("documents", []):
                for field_name, field_value in document.get("fields", {}).items():
                    if field_value:
                        field_examples[field_name].append(field_value)
            
            # Generate patterns for each field
            for field_name, examples in field_examples.items():
                patterns = self._generate_patterns(field_name, examples)
                self.field_patterns[field_name] = patterns
            
            self.is_trained = True
            training_time = time.time() - start_time
            
            # Calculate basic metrics
            total_fields = len(field_examples)
            total_examples = sum(len(examples) for examples in field_examples.values())
            
            self.metrics = ModelMetrics(
                accuracy=0.85,  # Placeholder - would need validation data
                training_time=training_time,
                training_samples=total_examples
            )
            
            logger.info(f"Field extraction model trained - {total_fields} field types, {total_examples} examples")
            return self.metrics
            
        except Exception as e:
            logger.error(f"Training error: {e}")
            raise
    
    def _generate_patterns(self, field_name: str, examples: List[str]) -> List[str]:
        """Generate regex patterns for a field type."""
        patterns = []
        
        if field_name == "ic_number":
            patterns = [
                r"\b(\d{6}-\d{2}-\d{4})\b",
                r"\b(\d{6}\s\d{2}\s\d{4})\b",
                r"\b(\d{12})\b"
            ]
        elif field_name == "name":
            patterns = [
                r"(?i)nama[\s:]+([A-Z][A-Z\s]+)",
                r"(?i)name[\s:]+([A-Z][A-Z\s]+)",
                r"\b([A-Z][A-Z\s]{2,30})\b"
            ]
        elif field_name == "phone":
            patterns = [
                r"\b(\+?6?0?1[0-9]-?\d{7,8})\b",
                r"\b(\+?6?0?[2-9]\d{1}-?\d{7,8})\b"
            ]
        elif field_name == "email":
            patterns = [
                r"\b([a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,})\b"
            ]
        else:
            # Generate generic patterns based on examples
            patterns = self._generate_generic_patterns(examples)
        
        return patterns
    
    def _generate_generic_patterns(self, examples: List[str]) -> List[str]:
        """Generate generic patterns from examples."""
        patterns = []
        
        # Analyze common characteristics
        if examples:
            # Check if all examples are numeric
            if all(ex.replace("-", "").replace(" ", "").isdigit() for ex in examples):
                avg_length = sum(len(ex.replace("-", "").replace(" ", "")) for ex in examples) // len(examples)
                patterns.append(f"\\b(\\d{{{avg_length-1},{avg_length+1}}})\\b")
            
            # Check if all examples are alphabetic
            elif all(ex.replace(" ", "").isalpha() for ex in examples):
                patterns.append(r"\b([A-Za-z\s]+)\b")
            
            # Mixed alphanumeric
            else:
                patterns.append(r"\b([A-Za-z0-9\s\-]+)\b")
        
        return patterns
    
    def predict(self, text: str) -> Dict[str, Any]:
        """Extract fields from text."""
        if not self.is_trained:
            raise ValueError("Model must be trained before prediction")
        
        extracted_fields = {}
        
        for field_name, patterns in self.field_patterns.items():
            for pattern in patterns:
                matches = re.findall(pattern, text, re.IGNORECASE)
                if matches:
                    # Take the first match
                    value = matches[0] if isinstance(matches[0], str) else matches[0][0]
                    extracted_fields[field_name] = {
                        "value": value.strip(),
                        "confidence": 0.8,  # Pattern-based confidence
                        "method": "pattern_matching"
                    }
                    break
        
        return extracted_fields
    
    def predict_proba(self, text: str) -> Dict[str, float]:
        """Get field extraction confidence scores."""
        fields = self.predict(text)
        return {name: data["confidence"] for name, data in fields.items()}

class OCRModel(BaseModel):
    """OCR model wrapper for text extraction."""
    
    def __init__(self, config: ModelConfig):
        super().__init__(config)
        self.ocr_engine = None
        self.preprocessing_pipeline = None
    
    def train(self, training_data: Any = None, **kwargs) -> ModelMetrics:
        """Initialize OCR engine (no training required)."""
        try:
            import time
            start_time = time.time()
            
            # Initialize OCR engines based on config
            engine_type = self.config.parameters.get("engine", "tesseract")
            
            if engine_type == "tesseract":
                try:
                    import pytesseract
                    self.ocr_engine = pytesseract
                    logger.info("Tesseract OCR engine initialized")
                except ImportError:
                    logger.warning("Tesseract not available")
            
            elif engine_type == "easyocr":
                try:
                    import easyocr
                    languages = self.config.parameters.get("languages", ['en'])
                    self.ocr_engine = easyocr.Reader(languages)
                    logger.info(f"EasyOCR engine initialized with languages: {languages}")
                except ImportError:
                    logger.warning("EasyOCR not available")
            
            self.is_trained = True
            training_time = time.time() - start_time
            
            self.metrics = ModelMetrics(
                accuracy=0.95,  # Typical OCR accuracy
                training_time=training_time
            )
            
            return self.metrics
            
        except Exception as e:
            logger.error(f"OCR initialization error: {e}")
            raise
    
    def predict(self, image: Any) -> str:
        """Extract text from image."""
        if not self.is_trained:
            raise ValueError("OCR engine must be initialized before prediction")
        
        try:
            engine_type = self.config.parameters.get("engine", "tesseract")
            
            if engine_type == "tesseract" and self.ocr_engine:
                # Tesseract OCR
                text = self.ocr_engine.image_to_string(image)
                return text.strip()
            
            elif engine_type == "easyocr" and self.ocr_engine:
                # EasyOCR
                results = self.ocr_engine.readtext(image)
                text = " ".join([result[1] for result in results])
                return text.strip()
            
            else:
                raise ValueError(f"OCR engine '{engine_type}' not available")
                
        except Exception as e:
            logger.error(f"OCR prediction error: {e}")
            return ""
    
    def predict_proba(self, image: Any) -> float:
        """Get OCR confidence score."""
        try:
            engine_type = self.config.parameters.get("engine", "tesseract")
            
            if engine_type == "tesseract" and self.ocr_engine:
                # Get confidence from Tesseract
                data = self.ocr_engine.image_to_data(image, output_type=self.ocr_engine.Output.DICT)
                confidences = [int(conf) for conf in data['conf'] if int(conf) > 0]
                return np.mean(confidences) / 100.0 if confidences else 0.0
            
            elif engine_type == "easyocr" and self.ocr_engine:
                # Get confidence from EasyOCR
                results = self.ocr_engine.readtext(image)
                confidences = [result[2] for result in results]
                return np.mean(confidences) if confidences else 0.0
            
            else:
                return 0.5  # Default confidence
                
        except Exception as e:
            logger.warning(f"Could not get OCR confidence: {e}")
            return 0.5

# Model factory functions
def create_classification_model(model_name: str = "document_classifier") -> ClassificationModel:
    """Create a document classification model."""
    config = ModelConfig(
        model_name=model_name,
        model_type="classification",
        parameters={
            "max_features": 5000,
            "ngram_range": (1, 2)
        },
        hyperparameters={
            "n_estimators": 100,
            "max_depth": None
        },
        class_labels=["mykad", "spk", "passport", "unknown"]
    )
    
    return ClassificationModel(config)

def create_field_extraction_model(model_name: str = "field_extractor") -> FieldExtractionModel:
    """Create a field extraction model."""
    config = ModelConfig(
        model_name=model_name,
        model_type="field_extraction",
        parameters={
            "supported_fields": ["ic_number", "name", "address", "phone", "email"]
        }
    )
    
    return FieldExtractionModel(config)

def create_ocr_model(engine: str = "tesseract", languages: List[str] = None) -> OCRModel:
    """Create an OCR model."""
    if languages is None:
        languages = ['en']
    
    config = ModelConfig(
        model_name=f"ocr_{engine}",
        model_type="ocr",
        parameters={
            "engine": engine,
            "languages": languages
        }
    )
    
    return OCRModel(config)

# Model management utilities
class ModelRegistry:
    """Registry for managing multiple models."""
    
    def __init__(self, models_dir: Union[str, Path]):
        self.models_dir = Path(models_dir)
        self.models_dir.mkdir(parents=True, exist_ok=True)
        self.models = {}
        
        logger.info(f"Model registry initialized at {self.models_dir}")
    
    def register_model(self, model: BaseModel, model_id: str = None):
        """Register a model in the registry."""
        if model_id is None:
            model_id = model.config.model_name
        
        self.models[model_id] = model
        logger.info(f"Model '{model_id}' registered")
    
    def get_model(self, model_id: str) -> Optional[BaseModel]:
        """Get a model from the registry."""
        return self.models.get(model_id)
    
    def save_model(self, model_id: str, filename: str = None):
        """Save a model to disk."""
        if model_id not in self.models:
            raise ValueError(f"Model '{model_id}' not found in registry")
        
        if filename is None:
            filename = f"{model_id}.pkl"
        
        file_path = self.models_dir / filename
        self.models[model_id].save_model(file_path)
    
    def load_model(self, model_id: str, filename: str = None, model_class: type = None):
        """Load a model from disk."""
        if filename is None:
            filename = f"{model_id}.pkl"
        
        file_path = self.models_dir / filename
        
        if not file_path.exists():
            raise FileNotFoundError(f"Model file not found: {file_path}")
        
        if model_class is None:
            # Try to determine model class from file
            with open(file_path, 'rb') as f:
                model_data = pickle.load(f)
            
            model_type = model_data["config"]["model_type"]
            if model_type == "classification":
                model_class = ClassificationModel
            elif model_type == "field_extraction":
                model_class = FieldExtractionModel
            elif model_type == "ocr":
                model_class = OCRModel
            else:
                raise ValueError(f"Unknown model type: {model_type}")
        
        model = model_class.load_model(file_path)
        self.register_model(model, model_id)
        
        logger.info(f"Model '{model_id}' loaded from {file_path}")
    
    def list_models(self) -> Dict[str, Dict]:
        """List all registered models."""
        return {
            model_id: model.get_model_info() 
            for model_id, model in self.models.items()
        }
    
    def get_registry_info(self) -> Dict:
        """Get registry information."""
        return {
            "models_dir": str(self.models_dir),
            "registered_models": list(self.models.keys()),
            "model_count": len(self.models),
            "available_files": [f.name for f in self.models_dir.glob("*.pkl")]
        }