#!/usr/bin/env python3
"""Training Pipeline for Document Parser Models

This module implements a basic training pipeline for model updates using collected feedback data,
following the autocorrect model's simple approach to model training and updates.
"""

import os
import json
import pickle
import sqlite3
import logging
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, field
import tempfile
import shutil

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import LabelEncoder

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class TrainingData:
    """Training data structure"""
    document_id: str = ""
    document_type: str = ""
    original_fields: Dict[str, str] = field(default_factory=dict)
    corrected_fields: Dict[str, str] = field(default_factory=dict)
    feedback_type: str = ""
    confidence_scores: Dict[str, float] = field(default_factory=dict)
    image_features: Optional[List[float]] = None
    text_features: Optional[str] = None
    created_at: datetime = field(default_factory=datetime.now)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return {
            'document_id': self.document_id,
            'document_type': self.document_type,
            'original_fields': self.original_fields,
            'corrected_fields': self.corrected_fields,
            'feedback_type': self.feedback_type,
            'confidence_scores': self.confidence_scores,
            'image_features': self.image_features,
            'text_features': self.text_features,
            'created_at': self.created_at.isoformat()
        }

@dataclass
class ModelVersion:
    """Model version information"""
    version: str = ""
    model_type: str = ""
    training_date: datetime = field(default_factory=datetime.now)
    training_data_count: int = 0
    validation_accuracy: float = 0.0
    model_path: str = ""
    metadata: Dict[str, Any] = field(default_factory=dict)
    
class TrainingDataCollector:
    """Collect and prepare training data from feedback"""
    
    def __init__(self, feedback_db_path: str, review_db_path: str):
        self.feedback_db_path = Path(feedback_db_path)
        self.review_db_path = Path(review_db_path)
        
    def collect_feedback_data(self, days: int = 30) -> List[TrainingData]:
        """Collect training data from feedback systems"""
        training_data = []
        
        # Collect from feedback loop system
        if self.feedback_db_path.exists():
            training_data.extend(self._collect_from_feedback_db(days))
        
        # Collect from review interface
        if self.review_db_path.exists():
            training_data.extend(self._collect_from_review_db(days))
        
        logger.info(f"Collected {len(training_data)} training samples")
        return training_data
    
    def _collect_from_feedback_db(self, days: int) -> List[TrainingData]:
        """Collect data from feedback loop database"""
        training_data = []
        cutoff_date = datetime.now() - timedelta(days=days)
        
        try:
            with sqlite3.connect(self.feedback_db_path) as conn:
                conn.row_factory = sqlite3.Row
                cursor = conn.execute("""
                    SELECT * FROM document_feedback 
                    WHERE timestamp >= ?
                    ORDER BY timestamp DESC
                """, (cutoff_date.isoformat(),))
                
                for row in cursor.fetchall():
                    data = dict(row)
                    
                    training_sample = TrainingData(
                        document_id=data['document_id'],
                        document_type=data['document_type'],
                        original_fields=json.loads(data.get('original_extraction', '{}')),
                        corrected_fields=json.loads(data.get('corrected_extraction', '{}')),
                        feedback_type=data.get('feedback_type', 'correction'),
                        confidence_scores=json.loads(data.get('confidence_scores', '{}')),
                        created_at=datetime.fromisoformat(data['timestamp'])
                    )
                    
                    training_data.append(training_sample)
                    
        except Exception as e:
            logger.warning(f"Could not collect from feedback DB: {e}")
        
        return training_data
    
    def _collect_from_review_db(self, days: int) -> List[TrainingData]:
        """Collect data from review interface database"""
        training_data = []
        cutoff_date = datetime.now() - timedelta(days=days)
        
        try:
            with sqlite3.connect(self.review_db_path) as conn:
                conn.row_factory = sqlite3.Row
                cursor = conn.execute("""
                    SELECT d.*, f.feedback_data 
                    FROM documents d
                    LEFT JOIN feedback f ON d.document_id = f.document_id
                    WHERE d.reviewed_at >= ? AND d.status = 'reviewed'
                    ORDER BY d.reviewed_at DESC
                """, (cutoff_date.isoformat(),))
                
                for row in cursor.fetchall():
                    data = dict(row)
                    
                    original_fields = json.loads(data.get('extracted_fields', '{}'))
                    corrected_fields = json.loads(data.get('corrected_fields', '{}'))
                    
                    # Only include if there are corrections
                    if corrected_fields:
                        training_sample = TrainingData(
                            document_id=data['document_id'],
                            document_type=data['document_type'],
                            original_fields=original_fields,
                            corrected_fields=corrected_fields,
                            feedback_type='review_correction',
                            confidence_scores=json.loads(data.get('field_confidences', '{}')),
                            created_at=datetime.fromisoformat(data['reviewed_at'])
                        )
                        
                        training_data.append(training_sample)
                        
        except Exception as e:
            logger.warning(f"Could not collect from review DB: {e}")
        
        return training_data

class SimpleDocumentClassifier:
    """Simple document type classifier"""
    
    def __init__(self):
        self.vectorizer = TfidfVectorizer(max_features=1000, stop_words='english')
        self.classifier = RandomForestClassifier(n_estimators=100, random_state=42)
        self.label_encoder = LabelEncoder()
        self.is_trained = False
        
    def prepare_features(self, training_data: List[TrainingData]) -> Tuple[np.ndarray, np.ndarray]:
        """Prepare features for training"""
        texts = []
        labels = []
        
        for sample in training_data:
            # Combine all text fields
            text_content = ' '.join([
                str(value) for value in sample.corrected_fields.values() 
                if value and isinstance(value, str)
            ])
            
            if text_content.strip():
                texts.append(text_content)
                labels.append(sample.document_type)
        
        if not texts:
            raise ValueError("No valid text data found for training")
        
        # Vectorize text
        X = self.vectorizer.fit_transform(texts)
        
        # Encode labels
        y = self.label_encoder.fit_transform(labels)
        
        return X.toarray(), y
    
    def train(self, training_data: List[TrainingData]) -> Dict[str, float]:
        """Train the classifier"""
        logger.info(f"Training classifier with {len(training_data)} samples")
        
        X, y = self.prepare_features(training_data)
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        
        # Train classifier
        self.classifier.fit(X_train, y_train)
        self.is_trained = True
        
        # Evaluate
        y_pred = self.classifier.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        
        logger.info(f"Training completed. Validation accuracy: {accuracy:.3f}")
        
        return {
            'accuracy': accuracy,
            'training_samples': len(X_train),
            'validation_samples': len(X_test)
        }
    
    def predict(self, text: str) -> Tuple[str, float]:
        """Predict document type"""
        if not self.is_trained:
            raise ValueError("Model not trained yet")
        
        X = self.vectorizer.transform([text])
        prediction = self.classifier.predict(X.toarray())[0]
        probabilities = self.classifier.predict_proba(X.toarray())[0]
        
        document_type = self.label_encoder.inverse_transform([prediction])[0]
        confidence = max(probabilities)
        
        return document_type, confidence
    
    def save(self, model_path: str):
        """Save the trained model"""
        model_data = {
            'vectorizer': self.vectorizer,
            'classifier': self.classifier,
            'label_encoder': self.label_encoder,
            'is_trained': self.is_trained
        }
        
        with open(model_path, 'wb') as f:
            pickle.dump(model_data, f)
        
        logger.info(f"Model saved to {model_path}")
    
    def load(self, model_path: str):
        """Load a trained model"""
        with open(model_path, 'rb') as f:
            model_data = pickle.load(f)
        
        self.vectorizer = model_data['vectorizer']
        self.classifier = model_data['classifier']
        self.label_encoder = model_data['label_encoder']
        self.is_trained = model_data['is_trained']
        
        logger.info(f"Model loaded from {model_path}")

class DocumentParserTrainingPipeline:
    """Main training pipeline for document parser models"""
    
    def __init__(self, 
                 models_path: str = "models",
                 feedback_db_path: str = "performance_data/feedback.db",
                 review_db_path: str = "review_data/review.db"):
        
        self.models_path = Path(models_path)
        self.models_path.mkdir(exist_ok=True)
        
        self.feedback_db_path = feedback_db_path
        self.review_db_path = review_db_path
        
        # Initialize components
        self.data_collector = TrainingDataCollector(feedback_db_path, review_db_path)
        self.classifier = SimpleDocumentClassifier()
        
        # Training history
        self.training_history_path = self.models_path / "training_history.json"
        self.training_history = self._load_training_history()
    
    def _load_training_history(self) -> List[Dict[str, Any]]:
        """Load training history"""
        if self.training_history_path.exists():
            try:
                with open(self.training_history_path, 'r') as f:
                    return json.load(f)
            except Exception as e:
                logger.warning(f"Could not load training history: {e}")
        
        return []
    
    def _save_training_history(self):
        """Save training history"""
        with open(self.training_history_path, 'w') as f:
            json.dump(self.training_history, f, indent=2)
    
    def check_training_readiness(self, min_samples: int = 50) -> Dict[str, Any]:
        """Check if enough data is available for training"""
        training_data = self.data_collector.collect_feedback_data(days=30)
        
        # Analyze data by document type
        type_counts = {}
        for sample in training_data:
            doc_type = sample.document_type
            type_counts[doc_type] = type_counts.get(doc_type, 0) + 1
        
        total_samples = len(training_data)
        ready_for_training = total_samples >= min_samples
        
        readiness_report = {
            'ready_for_training': ready_for_training,
            'total_samples': total_samples,
            'min_required': min_samples,
            'samples_by_type': type_counts,
            'recommendation': self._get_training_recommendation(total_samples, type_counts, min_samples)
        }
        
        logger.info(f"Training readiness: {ready_for_training} ({total_samples}/{min_samples} samples)")
        
        return readiness_report
    
    def _get_training_recommendation(self, total_samples: int, type_counts: Dict[str, int], min_samples: int) -> str:
        """Get training recommendation based on data analysis"""
        if total_samples < min_samples:
            return f"Need {min_samples - total_samples} more samples before training"
        
        # Check for class imbalance
        if type_counts:
            max_count = max(type_counts.values())
            min_count = min(type_counts.values())
            
            if max_count > min_count * 3:  # Significant imbalance
                return "Ready for training, but consider collecting more data for underrepresented document types"
        
        return "Ready for training with good data distribution"
    
    def train_models(self, 
                    days: int = 30, 
                    min_samples: int = 50,
                    create_backup: bool = True) -> Dict[str, Any]:
        """Train models using collected feedback data"""
        logger.info(f"Starting model training with data from last {days} days")
        
        # Check readiness
        readiness = self.check_training_readiness(min_samples)
        if not readiness['ready_for_training']:
            return {
                'success': False,
                'message': readiness['recommendation'],
                'readiness_report': readiness
            }
        
        # Collect training data
        training_data = self.data_collector.collect_feedback_data(days)
        
        if not training_data:
            return {
                'success': False,
                'message': 'No training data available',
                'readiness_report': readiness
            }
        
        # Create backup of current model if requested
        if create_backup:
            self._backup_current_models()
        
        # Train classifier
        try:
            training_results = self.classifier.train(training_data)
            
            # Generate version info
            version = datetime.now().strftime("%Y%m%d_%H%M%S")
            model_version = ModelVersion(
                version=version,
                model_type="document_classifier",
                training_date=datetime.now(),
                training_data_count=len(training_data),
                validation_accuracy=training_results['accuracy'],
                model_path=str(self.models_path / f"classifier_{version}.pkl"),
                metadata={
                    'training_results': training_results,
                    'data_collection_days': days,
                    'samples_by_type': readiness['samples_by_type']
                }
            )
            
            # Save model
            self.classifier.save(model_version.model_path)
            
            # Update training history
            self.training_history.append({
                'version': version,
                'timestamp': datetime.now().isoformat(),
                'model_type': 'document_classifier',
                'training_data_count': len(training_data),
                'validation_accuracy': training_results['accuracy'],
                'model_path': model_version.model_path,
                'metadata': model_version.metadata
            })
            
            self._save_training_history()
            
            # Create symlink to latest model
            latest_model_path = self.models_path / "latest_classifier.pkl"
            if latest_model_path.exists():
                latest_model_path.unlink()
            
            try:
                latest_model_path.symlink_to(Path(model_version.model_path).name)
            except OSError:
                # Fallback for systems that don't support symlinks
                shutil.copy2(model_version.model_path, latest_model_path)
            
            logger.info(f"Model training completed successfully. Version: {version}")
            
            return {
                'success': True,
                'version': version,
                'model_path': model_version.model_path,
                'training_results': training_results,
                'readiness_report': readiness
            }
            
        except Exception as e:
            logger.error(f"Model training failed: {e}")
            return {
                'success': False,
                'message': f"Training failed: {str(e)}",
                'readiness_report': readiness
            }
    
    def _backup_current_models(self):
        """Create backup of current models"""
        backup_dir = self.models_path / "backups" / datetime.now().strftime("%Y%m%d_%H%M%S")
        backup_dir.mkdir(parents=True, exist_ok=True)
        
        # Backup latest model if it exists
        latest_model = self.models_path / "latest_classifier.pkl"
        if latest_model.exists():
            shutil.copy2(latest_model, backup_dir / "classifier.pkl")
            logger.info(f"Current model backed up to {backup_dir}")
    
    def evaluate_model(self, model_path: Optional[str] = None) -> Dict[str, Any]:
        """Evaluate a trained model"""
        if model_path is None:
            model_path = self.models_path / "latest_classifier.pkl"
        
        if not Path(model_path).exists():
            return {'error': 'Model file not found'}
        
        # Load model
        test_classifier = SimpleDocumentClassifier()
        test_classifier.load(model_path)
        
        # Get test data
        test_data = self.data_collector.collect_feedback_data(days=7)  # Recent data for testing
        
        if not test_data:
            return {'error': 'No test data available'}
        
        # Evaluate
        correct_predictions = 0
        total_predictions = 0
        type_accuracy = {}
        
        for sample in test_data:
            if sample.corrected_fields:
                text_content = ' '.join([
                    str(value) for value in sample.corrected_fields.values() 
                    if value and isinstance(value, str)
                ])
                
                if text_content.strip():
                    try:
                        predicted_type, confidence = test_classifier.predict(text_content)
                        actual_type = sample.document_type
                        
                        is_correct = predicted_type == actual_type
                        
                        if actual_type not in type_accuracy:
                            type_accuracy[actual_type] = {'correct': 0, 'total': 0}
                        
                        type_accuracy[actual_type]['total'] += 1
                        if is_correct:
                            correct_predictions += 1
                            type_accuracy[actual_type]['correct'] += 1
                        
                        total_predictions += 1
                        
                    except Exception as e:
                        logger.warning(f"Prediction failed for sample: {e}")
        
        if total_predictions == 0:
            return {'error': 'No valid predictions could be made'}
        
        # Calculate accuracy by type
        type_accuracies = {}
        for doc_type, stats in type_accuracy.items():
            type_accuracies[doc_type] = stats['correct'] / stats['total'] if stats['total'] > 0 else 0.0
        
        overall_accuracy = correct_predictions / total_predictions
        
        evaluation_results = {
            'overall_accuracy': overall_accuracy,
            'total_predictions': total_predictions,
            'correct_predictions': correct_predictions,
            'accuracy_by_type': type_accuracies,
            'model_path': str(model_path)
        }
        
        logger.info(f"Model evaluation completed. Overall accuracy: {overall_accuracy:.3f}")
        
        return evaluation_results
    
    def get_training_status(self) -> Dict[str, Any]:
        """Get current training status and history"""
        latest_training = self.training_history[-1] if self.training_history else None
        
        readiness = self.check_training_readiness()
        
        status = {
            'last_training': latest_training,
            'training_history_count': len(self.training_history),
            'current_readiness': readiness,
            'models_directory': str(self.models_path),
            'available_models': self._list_available_models()
        }
        
        return status
    
    def _list_available_models(self) -> List[Dict[str, Any]]:
        """List available trained models"""
        models = []
        
        for model_file in self.models_path.glob("classifier_*.pkl"):
            stat = model_file.stat()
            models.append({
                'filename': model_file.name,
                'path': str(model_file),
                'size_mb': stat.st_size / (1024 * 1024),
                'created': datetime.fromtimestamp(stat.st_ctime).isoformat()
            })
        
        return sorted(models, key=lambda x: x['created'], reverse=True)
    
    def schedule_training(self, 
                         interval_days: int = 7,
                         min_new_samples: int = 20) -> Dict[str, Any]:
        """Check if scheduled training should be triggered"""
        
        # Check if enough time has passed since last training
        if self.training_history:
            last_training = datetime.fromisoformat(self.training_history[-1]['timestamp'])
            days_since_training = (datetime.now() - last_training).days
            
            if days_since_training < interval_days:
                return {
                    'should_train': False,
                    'reason': f'Only {days_since_training} days since last training (minimum: {interval_days})'
                }
        
        # Check if enough new samples are available
        readiness = self.check_training_readiness(min_new_samples)
        
        if not readiness['ready_for_training']:
            return {
                'should_train': False,
                'reason': readiness['recommendation']
            }
        
        return {
            'should_train': True,
            'reason': 'Sufficient time passed and enough new samples available',
            'readiness_report': readiness
        }

def main():
    """Demo function"""
    print("🚂 Document Parser Training Pipeline Demo")
    print("=" * 50)
    
    # Initialize training pipeline
    pipeline = DocumentParserTrainingPipeline()
    
    # Check training readiness
    print("\n=== Checking Training Readiness ===")
    readiness = pipeline.check_training_readiness(min_samples=10)  # Lower threshold for demo
    print(f"Ready for training: {readiness['ready_for_training']}")
    print(f"Total samples: {readiness['total_samples']}")
    print(f"Samples by type: {readiness['samples_by_type']}")
    print(f"Recommendation: {readiness['recommendation']}")
    
    # Get training status
    print("\n=== Training Status ===")
    status = pipeline.get_training_status()
    print(f"Training history: {status['training_history_count']} sessions")
    print(f"Available models: {len(status['available_models'])}")
    
    if status['last_training']:
        last = status['last_training']
        print(f"Last training: {last['timestamp']} (accuracy: {last['validation_accuracy']:.3f})")
    
    # Check scheduled training
    print("\n=== Scheduled Training Check ===")
    schedule_check = pipeline.schedule_training(interval_days=1, min_new_samples=5)
    print(f"Should train: {schedule_check['should_train']}")
    print(f"Reason: {schedule_check['reason']}")
    
    # Attempt training if ready
    if readiness['ready_for_training'] and readiness['total_samples'] > 0:
        print("\n=== Starting Training ===")
        training_result = pipeline.train_models(days=30, min_samples=5)
        
        if training_result['success']:
            print(f"✅ Training successful!")
            print(f"Version: {training_result['version']}")
            print(f"Model path: {training_result['model_path']}")
            print(f"Validation accuracy: {training_result['training_results']['accuracy']:.3f}")
            
            # Evaluate the trained model
            print("\n=== Model Evaluation ===")
            evaluation = pipeline.evaluate_model()
            
            if 'error' not in evaluation:
                print(f"Overall accuracy: {evaluation['overall_accuracy']:.3f}")
                print(f"Total predictions: {evaluation['total_predictions']}")
                print(f"Accuracy by type: {evaluation['accuracy_by_type']}")
            else:
                print(f"Evaluation error: {evaluation['error']}")
        else:
            print(f"❌ Training failed: {training_result['message']}")
    else:
        print("\n⏳ Not enough data for training. Collect more feedback first.")
    
    print("\n=== Demo Complete ===")
    print("Training pipeline ready for production use.")

if __name__ == "__main__":
    main()