#!/usr/bin/env python3
"""
Document Parser Training Script

Trains document classification and field extraction models
for MyKad and SPK document processing.
"""

import sys
import os
import argparse
import json
import time
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Any, Optional

import numpy as np
import cv2
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent))

from src.document_parser.core.classifier import DocumentClassifier
from src.document_parser.core.field_extractor import FieldExtractor
from src.document_parser.models.document_models import DocumentType
from src.document_parser.utils import ImageProcessor
from src.document_parser.config import load_config


class DocumentParserTrainer:
    """Main trainer class for document parser models."""
    
    def __init__(self, config_path: Optional[str] = None):
        """Initialize the trainer."""
        self.config = load_config(config_path)
        self.data_dir = Path(self.config.get('training', {}).get('data_dir', 'data/document_parser'))
        self.output_dir = Path(self.config.get('training', {}).get('output_dir', 'models/document_parser'))
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize components
        self.image_processor = ImageProcessor()
        self.classifier = None
        self.field_extractor = None
        
        # Training metrics
        self.training_history = {
            'classifier': {},
            'field_extractor': {}
        }
    
    def load_training_data(self) -> Dict[str, List[Dict]]:
        """Load training data from the data directory."""
        print("📂 Loading training data...")
        
        training_data = {
            'mykad': [],
            'spk': [],
            'unknown': []
        }
        
        # Load data for each document type
        for doc_type in training_data.keys():
            type_dir = self.data_dir / doc_type
            if not type_dir.exists():
                print(f"⚠️ Warning: No training data found for {doc_type} at {type_dir}")
                continue
            
            # Load images and annotations
            for img_path in type_dir.glob('*.jpg'):
                annotation_path = img_path.with_suffix('.json')
                
                if annotation_path.exists():
                    try:
                        # Load image
                        image = cv2.imread(str(img_path))
                        if image is None:
                            continue
                        
                        # Load annotations
                        with open(annotation_path, 'r', encoding='utf-8') as f:
                            annotations = json.load(f)
                        
                        training_data[doc_type].append({
                            'image_path': str(img_path),
                            'image': image,
                            'annotations': annotations,
                            'document_type': doc_type
                        })
                        
                    except Exception as e:
                        print(f"❌ Error loading {img_path}: {e}")
        
        # Print data statistics
        total_samples = sum(len(samples) for samples in training_data.values())
        print(f"\n📊 Training Data Statistics:")
        print(f"   Total samples: {total_samples}")
        for doc_type, samples in training_data.items():
            print(f"   {doc_type.upper()}: {len(samples)} samples")
        
        return training_data
    
    def prepare_classification_data(self, training_data: Dict[str, List[Dict]]) -> tuple:
        """Prepare data for document classification training."""
        print("\n🔄 Preparing classification data...")
        
        X, y = [], []
        
        for doc_type, samples in training_data.items():
            for sample in samples:
                # Preprocess image for classification
                processed_image = self.image_processor.preprocess_for_classification(
                    sample['image']
                )
                
                # Extract features (this would be more sophisticated in practice)
                features = self._extract_classification_features(processed_image)
                
                X.append(features)
                y.append(doc_type)
        
        X = np.array(X)
        y = np.array(y)
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        
        print(f"   Training samples: {len(X_train)}")
        print(f"   Testing samples: {len(X_test)}")
        
        return X_train, X_test, y_train, y_test
    
    def _extract_classification_features(self, image: np.ndarray) -> np.ndarray:
        """Extract features for document classification."""
        # This is a simplified feature extraction
        # In practice, you'd use more sophisticated methods
        
        # Resize image to standard size
        resized = cv2.resize(image, (224, 224))
        
        # Convert to grayscale
        gray = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)
        
        # Extract basic features
        features = []
        
        # Histogram features
        hist = cv2.calcHist([gray], [0], None, [256], [0, 256])
        features.extend(hist.flatten())
        
        # Edge features
        edges = cv2.Canny(gray, 50, 150)
        edge_density = np.sum(edges > 0) / edges.size
        features.append(edge_density)
        
        # Text region features (simplified)
        # This would use more sophisticated text detection in practice
        text_regions = self._detect_text_regions(gray)
        features.append(len(text_regions))
        features.append(np.mean([r['area'] for r in text_regions]) if text_regions else 0)
        
        return np.array(features)
    
    def _detect_text_regions(self, gray_image: np.ndarray) -> List[Dict]:
        """Detect text regions in the image (simplified)."""
        # This is a very basic text region detection
        # In practice, you'd use EAST, CRAFT, or similar methods
        
        # Find contours
        contours, _ = cv2.findContours(gray_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        text_regions = []
        for contour in contours:
            area = cv2.contourArea(contour)
            if 100 < area < 10000:  # Filter by area
                x, y, w, h = cv2.boundingRect(contour)
                aspect_ratio = w / h
                if 0.1 < aspect_ratio < 10:  # Filter by aspect ratio
                    text_regions.append({
                        'bbox': [x, y, x + w, y + h],
                        'area': area,
                        'aspect_ratio': aspect_ratio
                    })
        
        return text_regions
    
    def train_classifier(self, X_train: np.ndarray, y_train: np.ndarray) -> DocumentClassifier:
        """Train the document classifier."""
        print("\n🤖 Training document classifier...")
        
        # Initialize classifier
        self.classifier = DocumentClassifier()
        
        # Train the classifier
        start_time = time.time()
        history = self.classifier.train(X_train, y_train)
        training_time = time.time() - start_time
        
        self.training_history['classifier'] = {
            'training_time': training_time,
            'history': history,
            'timestamp': datetime.now().isoformat()
        }
        
        print(f"   ✅ Classifier training completed in {training_time:.2f}s")
        
        return self.classifier
    
    def evaluate_classifier(self, X_test: np.ndarray, y_test: np.ndarray) -> Dict[str, Any]:
        """Evaluate the trained classifier."""
        print("\n📊 Evaluating classifier...")
        
        if self.classifier is None:
            raise ValueError("Classifier not trained yet")
        
        # Make predictions
        y_pred = self.classifier.predict(X_test)
        
        # Calculate metrics
        report = classification_report(y_test, y_pred, output_dict=True)
        cm = confusion_matrix(y_test, y_pred)
        
        # Print results
        print("\n📈 Classification Results:")
        print(f"   Accuracy: {report['accuracy']:.3f}")
        print(f"   Macro Avg F1: {report['macro avg']['f1-score']:.3f}")
        print(f"   Weighted Avg F1: {report['weighted avg']['f1-score']:.3f}")
        
        # Per-class results
        print("\n📋 Per-class Results:")
        for class_name in ['mykad', 'spk', 'unknown']:
            if class_name in report:
                metrics = report[class_name]
                print(f"   {class_name.upper()}:")
                print(f"     Precision: {metrics['precision']:.3f}")
                print(f"     Recall: {metrics['recall']:.3f}")
                print(f"     F1-score: {metrics['f1-score']:.3f}")
        
        return {
            'classification_report': report,
            'confusion_matrix': cm.tolist(),
            'predictions': y_pred.tolist(),
            'ground_truth': y_test.tolist()
        }
    
    def prepare_field_extraction_data(self, training_data: Dict[str, List[Dict]]) -> Dict[str, Any]:
        """Prepare data for field extraction training."""
        print("\n🔄 Preparing field extraction data...")
        
        extraction_data = {
            'mykad': {'images': [], 'annotations': []},
            'spk': {'images': [], 'annotations': []}
        }
        
        for doc_type in ['mykad', 'spk']:
            if doc_type in training_data:
                for sample in training_data[doc_type]:
                    # Preprocess image for field extraction
                    processed_image = self.image_processor.preprocess_for_ocr(
                        sample['image']
                    )
                    
                    extraction_data[doc_type]['images'].append(processed_image)
                    extraction_data[doc_type]['annotations'].append(
                        sample['annotations'].get('fields', {})
                    )
        
        print(f"   MyKad field extraction samples: {len(extraction_data['mykad']['images'])}")
        print(f"   SPK field extraction samples: {len(extraction_data['spk']['images'])}")
        
        return extraction_data
    
    def train_field_extractor(self, extraction_data: Dict[str, Any]) -> FieldExtractor:
        """Train the field extraction models."""
        print("\n🤖 Training field extractor...")
        
        # Initialize field extractor
        self.field_extractor = FieldExtractor()
        
        start_time = time.time()
        
        # Train for each document type
        for doc_type in ['mykad', 'spk']:
            if extraction_data[doc_type]['images']:
                print(f"   Training {doc_type.upper()} field extractor...")
                
                history = self.field_extractor.train_for_document_type(
                    doc_type,
                    extraction_data[doc_type]['images'],
                    extraction_data[doc_type]['annotations']
                )
                
                self.training_history['field_extractor'][doc_type] = {
                    'history': history,
                    'timestamp': datetime.now().isoformat()
                }
        
        training_time = time.time() - start_time
        self.training_history['field_extractor']['total_training_time'] = training_time
        
        print(f"   ✅ Field extractor training completed in {training_time:.2f}s")
        
        return self.field_extractor
    
    def save_models(self) -> None:
        """Save trained models and training history."""
        print("\n💾 Saving models...")
        
        # Create model directories
        classifier_dir = self.output_dir / 'classifier'
        extractor_dir = self.output_dir / 'field_extractor'
        
        classifier_dir.mkdir(parents=True, exist_ok=True)
        extractor_dir.mkdir(parents=True, exist_ok=True)
        
        # Save classifier
        if self.classifier:
            classifier_path = classifier_dir / 'document_classifier.pkl'
            self.classifier.save(str(classifier_path))
            print(f"   ✅ Classifier saved to {classifier_path}")
        
        # Save field extractor
        if self.field_extractor:
            extractor_path = extractor_dir / 'field_extractor.pkl'
            self.field_extractor.save(str(extractor_path))
            print(f"   ✅ Field extractor saved to {extractor_path}")
        
        # Save training history
        history_path = self.output_dir / 'training_history.json'
        with open(history_path, 'w', encoding='utf-8') as f:
            json.dump(self.training_history, f, indent=2, ensure_ascii=False)
        print(f"   ✅ Training history saved to {history_path}")
    
    def create_visualizations(self, evaluation_results: Dict[str, Any]) -> None:
        """Create training and evaluation visualizations."""
        print("\n📊 Creating visualizations...")
        
        viz_dir = self.output_dir / 'visualizations'
        viz_dir.mkdir(parents=True, exist_ok=True)
        
        # Confusion matrix
        if 'confusion_matrix' in evaluation_results:
            plt.figure(figsize=(8, 6))
            cm = np.array(evaluation_results['confusion_matrix'])
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                       xticklabels=['MyKad', 'SPK', 'Unknown'],
                       yticklabels=['MyKad', 'SPK', 'Unknown'])
            plt.title('Document Classification Confusion Matrix')
            plt.ylabel('True Label')
            plt.xlabel('Predicted Label')
            plt.tight_layout()
            plt.savefig(viz_dir / 'confusion_matrix.png', dpi=300, bbox_inches='tight')
            plt.close()
            print(f"   ✅ Confusion matrix saved")
        
        # Classification report visualization
        if 'classification_report' in evaluation_results:
            report = evaluation_results['classification_report']
            
            # Extract metrics for visualization
            classes = ['mykad', 'spk', 'unknown']
            metrics = ['precision', 'recall', 'f1-score']
            
            data = []
            for class_name in classes:
                if class_name in report:
                    for metric in metrics:
                        data.append({
                            'Class': class_name.upper(),
                            'Metric': metric.title(),
                            'Score': report[class_name][metric]
                        })
            
            if data:
                import pandas as pd
                df = pd.DataFrame(data)
                
                plt.figure(figsize=(10, 6))
                sns.barplot(data=df, x='Class', y='Score', hue='Metric')
                plt.title('Classification Performance by Class')
                plt.ylabel('Score')
                plt.ylim(0, 1)
                plt.legend(title='Metric')
                plt.tight_layout()
                plt.savefig(viz_dir / 'classification_performance.png', dpi=300, bbox_inches='tight')
                plt.close()
                print(f"   ✅ Classification performance chart saved")
        
        print(f"   📁 All visualizations saved to {viz_dir}")


def main():
    """Main training function."""
    parser = argparse.ArgumentParser(description='Train Document Parser Models')
    parser.add_argument('--config', type=str, help='Path to configuration file')
    parser.add_argument('--data-dir', type=str, help='Path to training data directory')
    parser.add_argument('--output-dir', type=str, help='Path to output directory')
    parser.add_argument('--skip-classifier', action='store_true', help='Skip classifier training')
    parser.add_argument('--skip-extractor', action='store_true', help='Skip field extractor training')
    parser.add_argument('--visualize', action='store_true', default=True, help='Create visualizations')
    
    args = parser.parse_args()
    
    print("🚀 Document Parser Training Started")
    print("=" * 50)
    
    try:
        # Initialize trainer
        trainer = DocumentParserTrainer(args.config)
        
        # Override config with command line arguments
        if args.data_dir:
            trainer.data_dir = Path(args.data_dir)
        if args.output_dir:
            trainer.output_dir = Path(args.output_dir)
            trainer.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Load training data
        training_data = trainer.load_training_data()
        
        if not any(training_data.values()):
            print("❌ No training data found. Please check your data directory.")
            return 1
        
        evaluation_results = {}
        
        # Train classifier
        if not args.skip_classifier:
            X_train, X_test, y_train, y_test = trainer.prepare_classification_data(training_data)
            trainer.train_classifier(X_train, y_train)
            evaluation_results = trainer.evaluate_classifier(X_test, y_test)
        
        # Train field extractor
        if not args.skip_extractor:
            extraction_data = trainer.prepare_field_extraction_data(training_data)
            trainer.train_field_extractor(extraction_data)
        
        # Save models
        trainer.save_models()
        
        # Create visualizations
        if args.visualize and evaluation_results:
            trainer.create_visualizations(evaluation_results)
        
        print("\n🎉 Training completed successfully!")
        print(f"📁 Models saved to: {trainer.output_dir}")
        
        return 0
        
    except Exception as e:
        print(f"\n❌ Training failed: {str(e)}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == '__main__':
    exit(main())