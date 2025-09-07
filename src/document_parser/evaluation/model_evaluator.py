#!/usr/bin/env python3
"""
Model Evaluation and Performance Monitoring for Document Parser

This module provides comprehensive evaluation and monitoring capabilities
for document parser models, following the organizational patterns established
by the autocorrect model.

Author: AI Assistant
Date: 2024
"""

import os
import sys
import json
import yaml
from pathlib import Path
from typing import Dict, List, Tuple, Any, Optional, Union, Callable
from datetime import datetime
import logging
from dataclasses import dataclass, asdict
from abc import ABC, abstractmethod
import numpy as np
import pandas as pd
from collections import defaultdict, Counter
import pickle
import joblib
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns
from io import BytesIO
import base64

# Add src directory to path for imports
sys.path.append(str(Path(__file__).parent.parent.parent))

try:
    import torch
    import torch.nn as nn
    from torch.utils.data import DataLoader
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

try:
    from sklearn.metrics import (
        accuracy_score, precision_score, recall_score, f1_score,
        confusion_matrix, classification_report, roc_auc_score,
        precision_recall_curve, roc_curve, average_precision_score
    )
    from sklearn.preprocessing import label_binarize
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False

try:
    import cv2
    CV2_AVAILABLE = True
except ImportError:
    CV2_AVAILABLE = False

@dataclass
class EvaluationConfig:
    """Configuration for model evaluation"""
    # General settings
    model_type: str = 'document_classifier'  # 'document_classifier', 'text_detection', 'ocr', 'information_extraction'
    evaluation_mode: str = 'comprehensive'  # 'quick', 'comprehensive', 'production'
    
    # Data settings
    test_data_path: str = ''
    batch_size: int = 32
    num_workers: int = 4
    
    # Metrics settings
    metrics: List[str] = None
    confidence_thresholds: List[float] = None
    iou_thresholds: List[float] = None  # For detection tasks
    
    # Visualization settings
    generate_plots: bool = True
    save_predictions: bool = True
    save_misclassifications: bool = True
    max_samples_to_save: int = 100
    
    # Performance monitoring
    benchmark_mode: bool = False
    memory_profiling: bool = False
    inference_speed_test: bool = True
    
    # Output settings
    output_dir: str = 'evaluation_results'
    report_format: str = 'html'  # 'html', 'pdf', 'json'
    
    def __post_init__(self):
        if self.metrics is None:
            if self.model_type == 'document_classifier':
                self.metrics = ['accuracy', 'precision', 'recall', 'f1', 'auc']
            elif self.model_type == 'text_detection':
                self.metrics = ['precision', 'recall', 'f1', 'map']
            elif self.model_type == 'ocr':
                self.metrics = ['character_accuracy', 'word_accuracy', 'edit_distance']
            else:
                self.metrics = ['accuracy', 'precision', 'recall', 'f1']
        
        if self.confidence_thresholds is None:
            self.confidence_thresholds = [0.1, 0.3, 0.5, 0.7, 0.9]
        
        if self.iou_thresholds is None:
            self.iou_thresholds = [0.3, 0.5, 0.7, 0.9]
    
    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'EvaluationConfig':
        return cls(**data)
    
    @classmethod
    def from_yaml(cls, yaml_path: str) -> 'EvaluationConfig':
        """Load configuration from YAML file"""
        with open(yaml_path, 'r') as f:
            data = yaml.safe_load(f)
        return cls.from_dict(data)

@dataclass
class EvaluationResult:
    """Evaluation result container"""
    model_type: str
    metrics: Dict[str, float]
    confusion_matrix: Optional[np.ndarray]
    classification_report: Optional[str]
    predictions: Optional[List[Dict]]
    misclassifications: Optional[List[Dict]]
    performance_stats: Dict[str, float]
    evaluation_time: float
    timestamp: str
    
    def to_dict(self) -> Dict[str, Any]:
        result = asdict(self)
        # Convert numpy arrays to lists for JSON serialization
        if self.confusion_matrix is not None:
            result['confusion_matrix'] = self.confusion_matrix.tolist()
        return result
    
    def save_json(self, filepath: str):
        """Save results to JSON file"""
        with open(filepath, 'w') as f:
            json.dump(self.to_dict(), f, indent=2)

class BaseEvaluator(ABC):
    """Abstract base class for model evaluators"""
    
    def __init__(self, config: EvaluationConfig):
        self.config = config
        self.logger = self._setup_logging()
        self.device = self._setup_device()
        
        # Create output directory
        os.makedirs(config.output_dir, exist_ok=True)
        
        self.logger.info(f"Evaluator initialized for {config.model_type}")
    
    def _setup_logging(self) -> logging.Logger:
        """Setup logging configuration"""
        logger = logging.getLogger(f'{self.__class__.__name__}')
        logger.setLevel(logging.INFO)
        
        # Create file handler
        log_file = os.path.join(self.config.output_dir, f'evaluation_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log')
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(logging.INFO)
        
        # Create console handler
        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.INFO)
        
        # Create formatter
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        file_handler.setFormatter(formatter)
        console_handler.setFormatter(formatter)
        
        # Add handlers to logger
        logger.addHandler(file_handler)
        logger.addHandler(console_handler)
        
        return logger
    
    def _setup_device(self) -> str:
        """Setup evaluation device"""
        if TORCH_AVAILABLE:
            if torch.cuda.is_available():
                return 'cuda'
            elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
                return 'mps'
        return 'cpu'
    
    @abstractmethod
    def load_model(self, model_path: str) -> Any:
        """Load trained model"""
        pass
    
    @abstractmethod
    def prepare_data(self) -> DataLoader:
        """Prepare evaluation dataset"""
        pass
    
    @abstractmethod
    def evaluate_model(self, model: Any, data_loader: DataLoader) -> EvaluationResult:
        """Evaluate model performance"""
        pass
    
    def benchmark_inference_speed(self, model: Any, data_loader: DataLoader, num_iterations: int = 100) -> Dict[str, float]:
        """Benchmark model inference speed"""
        if not TORCH_AVAILABLE:
            return {'avg_inference_time': 0.0, 'throughput': 0.0}
        
        model.eval()
        times = []
        
        # Warmup
        with torch.no_grad():
            for i, batch in enumerate(data_loader):
                if i >= 10:  # 10 warmup iterations
                    break
                if isinstance(batch, (list, tuple)):
                    inputs = batch[0].to(self.device)
                else:
                    inputs = batch.to(self.device)
                _ = model(inputs)
        
        # Actual benchmarking
        with torch.no_grad():
            for i, batch in enumerate(data_loader):
                if i >= num_iterations:
                    break
                
                if isinstance(batch, (list, tuple)):
                    inputs = batch[0].to(self.device)
                    batch_size = inputs.size(0)
                else:
                    inputs = batch.to(self.device)
                    batch_size = inputs.size(0)
                
                start_time = datetime.now()
                _ = model(inputs)
                end_time = datetime.now()
                
                inference_time = (end_time - start_time).total_seconds()
                times.append(inference_time / batch_size)  # Per sample time
        
        avg_time = np.mean(times)
        throughput = 1.0 / avg_time if avg_time > 0 else 0.0
        
        return {
            'avg_inference_time': avg_time,
            'min_inference_time': np.min(times),
            'max_inference_time': np.max(times),
            'std_inference_time': np.std(times),
            'throughput': throughput
        }
    
    def generate_evaluation_report(self, result: EvaluationResult) -> str:
        """Generate comprehensive evaluation report"""
        report_lines = []
        report_lines.append(f"# Model Evaluation Report")
        report_lines.append(f"**Model Type:** {result.model_type}")
        report_lines.append(f"**Evaluation Date:** {result.timestamp}")
        report_lines.append(f"**Evaluation Time:** {result.evaluation_time:.2f} seconds")
        report_lines.append("")
        
        # Metrics section
        report_lines.append("## Performance Metrics")
        for metric, value in result.metrics.items():
            if isinstance(value, float):
                report_lines.append(f"- **{metric.title()}:** {value:.4f}")
            else:
                report_lines.append(f"- **{metric.title()}:** {value}")
        report_lines.append("")
        
        # Performance stats
        if result.performance_stats:
            report_lines.append("## Performance Statistics")
            for stat, value in result.performance_stats.items():
                if isinstance(value, float):
                    report_lines.append(f"- **{stat.replace('_', ' ').title()}:** {value:.6f}")
                else:
                    report_lines.append(f"- **{stat.replace('_', ' ').title()}:** {value}")
            report_lines.append("")
        
        # Classification report
        if result.classification_report:
            report_lines.append("## Detailed Classification Report")
            report_lines.append("```")
            report_lines.append(result.classification_report)
            report_lines.append("```")
            report_lines.append("")
        
        return "\n".join(report_lines)
    
    def save_evaluation_plots(self, result: EvaluationResult, class_names: Optional[List[str]] = None):
        """Save evaluation plots"""
        if not self.config.generate_plots:
            return
        
        # Confusion matrix plot
        if result.confusion_matrix is not None:
            self._plot_confusion_matrix(result.confusion_matrix, class_names)
        
        # Metrics plot
        self._plot_metrics(result.metrics)
    
    def _plot_confusion_matrix(self, cm: np.ndarray, class_names: Optional[List[str]] = None):
        """Plot confusion matrix"""
        plt.figure(figsize=(10, 8))
        
        if class_names is None:
            class_names = [f'Class {i}' for i in range(cm.shape[0])]
        
        sns.heatmap(
            cm, 
            annot=True, 
            fmt='d', 
            cmap='Blues',
            xticklabels=class_names,
            yticklabels=class_names
        )
        
        plt.title('Confusion Matrix')
        plt.xlabel('Predicted Label')
        plt.ylabel('True Label')
        plt.tight_layout()
        
        save_path = os.path.join(self.config.output_dir, 'confusion_matrix.png')
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        self.logger.info(f"Confusion matrix saved to {save_path}")
    
    def _plot_metrics(self, metrics: Dict[str, float]):
        """Plot metrics bar chart"""
        plt.figure(figsize=(12, 6))
        
        metric_names = list(metrics.keys())
        metric_values = list(metrics.values())
        
        # Filter numeric metrics
        numeric_metrics = [(name, value) for name, value in zip(metric_names, metric_values) 
                          if isinstance(value, (int, float))]
        
        if not numeric_metrics:
            return
        
        names, values = zip(*numeric_metrics)
        
        bars = plt.bar(names, values, color='skyblue', alpha=0.7)
        
        # Add value labels on bars
        for bar, value in zip(bars, values):
            plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                    f'{value:.3f}', ha='center', va='bottom')
        
        plt.title('Model Performance Metrics')
        plt.xlabel('Metrics')
        plt.ylabel('Score')
        plt.xticks(rotation=45)
        plt.ylim(0, 1.1)
        plt.grid(axis='y', alpha=0.3)
        plt.tight_layout()
        
        save_path = os.path.join(self.config.output_dir, 'metrics.png')
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        self.logger.info(f"Metrics plot saved to {save_path}")

class DocumentClassifierEvaluator(BaseEvaluator):
    """Evaluator for document classification models"""
    
    def __init__(self, config: EvaluationConfig):
        super().__init__(config)
        self.class_names = None
    
    def load_model(self, model_path: str) -> nn.Module:
        """Load document classification model"""
        if TORCH_AVAILABLE:
            checkpoint = torch.load(model_path, map_location=self.device)
            
            # Load model architecture
            from document_parser.models.document_classifier import DocumentClassifier, ClassifierConfig
            
            # Get config from checkpoint or use default
            if 'config' in checkpoint:
                model_config = ClassifierConfig.from_dict(checkpoint['config'])
            else:
                # Infer from model state dict
                model_config = ClassifierConfig()
            
            classifier = DocumentClassifier(model_config)
            model = classifier.model
            
            # Load state dict
            if 'model_state_dict' in checkpoint:
                model.load_state_dict(checkpoint['model_state_dict'])
            else:
                model.load_state_dict(checkpoint)
            
            model.to(self.device)
            model.eval()
            
            self.logger.info(f"Model loaded from {model_path}")
            return model
        else:
            raise RuntimeError("PyTorch not available for model loading")
    
    def prepare_data(self) -> DataLoader:
        """Prepare evaluation dataset"""
        from document_parser.models.document_classifier import DocumentDataset
        import torchvision.transforms as transforms
        
        # Define transforms
        transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        
        # Load dataset
        dataset = DocumentDataset(
            data_dir=self.config.test_data_path,
            transform=transform
        )
        
        self.class_names = dataset.classes
        
        # Create data loader
        data_loader = DataLoader(
            dataset,
            batch_size=self.config.batch_size,
            shuffle=False,
            num_workers=self.config.num_workers
        )
        
        self.logger.info(f"Test dataset prepared: {len(dataset)} samples, {len(self.class_names)} classes")
        
        return data_loader
    
    def evaluate_model(self, model: nn.Module, data_loader: DataLoader) -> EvaluationResult:
        """Evaluate document classification model"""
        start_time = datetime.now()
        
        model.eval()
        all_predictions = []
        all_targets = []
        all_probabilities = []
        misclassifications = []
        
        with torch.no_grad():
            for batch_idx, (data, target) in enumerate(tqdm(data_loader, desc="Evaluating")):
                data, target = data.to(self.device), target.to(self.device)
                
                output = model(data)
                probabilities = torch.softmax(output, dim=1)
                predictions = output.argmax(dim=1)
                
                # Store results
                all_predictions.extend(predictions.cpu().numpy())
                all_targets.extend(target.cpu().numpy())
                all_probabilities.extend(probabilities.cpu().numpy())
                
                # Collect misclassifications
                if self.config.save_misclassifications:
                    incorrect_mask = predictions != target
                    if incorrect_mask.any():
                        for i in range(len(data)):
                            if incorrect_mask[i] and len(misclassifications) < self.config.max_samples_to_save:
                                misclassifications.append({
                                    'true_label': int(target[i].cpu().item()),
                                    'predicted_label': int(predictions[i].cpu().item()),
                                    'confidence': float(probabilities[i].max().cpu().item()),
                                    'batch_idx': batch_idx,
                                    'sample_idx': i
                                })
        
        # Calculate metrics
        metrics = self._calculate_metrics(all_targets, all_predictions, all_probabilities)
        
        # Generate confusion matrix
        cm = confusion_matrix(all_targets, all_predictions)
        
        # Generate classification report
        if SKLEARN_AVAILABLE:
            class_report = classification_report(
                all_targets, all_predictions, 
                target_names=self.class_names if self.class_names else None
            )
        else:
            class_report = None
        
        # Benchmark inference speed if requested
        performance_stats = {}
        if self.config.inference_speed_test:
            performance_stats = self.benchmark_inference_speed(model, data_loader)
        
        # Prepare predictions for saving
        predictions_to_save = None
        if self.config.save_predictions:
            predictions_to_save = [
                {
                    'true_label': int(true),
                    'predicted_label': int(pred),
                    'probabilities': prob.tolist()
                }
                for true, pred, prob in zip(all_targets[:self.config.max_samples_to_save], 
                                          all_predictions[:self.config.max_samples_to_save],
                                          all_probabilities[:self.config.max_samples_to_save])
            ]
        
        evaluation_time = (datetime.now() - start_time).total_seconds()
        
        result = EvaluationResult(
            model_type=self.config.model_type,
            metrics=metrics,
            confusion_matrix=cm,
            classification_report=class_report,
            predictions=predictions_to_save,
            misclassifications=misclassifications,
            performance_stats=performance_stats,
            evaluation_time=evaluation_time,
            timestamp=datetime.now().isoformat()
        )
        
        return result
    
    def _calculate_metrics(self, y_true: List[int], y_pred: List[int], y_prob: List[np.ndarray]) -> Dict[str, float]:
        """Calculate classification metrics"""
        metrics = {}
        
        if SKLEARN_AVAILABLE:
            # Basic metrics
            metrics['accuracy'] = accuracy_score(y_true, y_pred)
            metrics['precision'] = precision_score(y_true, y_pred, average='weighted', zero_division=0)
            metrics['recall'] = recall_score(y_true, y_pred, average='weighted', zero_division=0)
            metrics['f1'] = f1_score(y_true, y_pred, average='weighted', zero_division=0)
            
            # AUC metrics (for multi-class)
            try:
                n_classes = len(np.unique(y_true))
                if n_classes > 2:
                    # Multi-class AUC
                    y_true_bin = label_binarize(y_true, classes=range(n_classes))
                    y_prob_array = np.array(y_prob)
                    metrics['auc'] = roc_auc_score(y_true_bin, y_prob_array, average='weighted', multi_class='ovr')
                else:
                    # Binary AUC
                    y_prob_pos = np.array(y_prob)[:, 1] if len(y_prob[0]) > 1 else np.array(y_prob).flatten()
                    metrics['auc'] = roc_auc_score(y_true, y_prob_pos)
            except Exception as e:
                self.logger.warning(f"Could not calculate AUC: {e}")
                metrics['auc'] = 0.0
            
            # Per-class metrics
            precision_per_class = precision_score(y_true, y_pred, average=None, zero_division=0)
            recall_per_class = recall_score(y_true, y_pred, average=None, zero_division=0)
            f1_per_class = f1_score(y_true, y_pred, average=None, zero_division=0)
            
            for i, (p, r, f) in enumerate(zip(precision_per_class, recall_per_class, f1_per_class)):
                class_name = self.class_names[i] if self.class_names and i < len(self.class_names) else f'class_{i}'
                metrics[f'precision_{class_name}'] = p
                metrics[f'recall_{class_name}'] = r
                metrics[f'f1_{class_name}'] = f
        
        else:
            # Basic accuracy calculation without sklearn
            correct = sum(1 for true, pred in zip(y_true, y_pred) if true == pred)
            metrics['accuracy'] = correct / len(y_true)
        
        return metrics

class ModelEvaluator:
    """Main model evaluation manager"""
    
    def __init__(self, config_path: Optional[str] = None):
        if config_path:
            self.config = EvaluationConfig.from_yaml(config_path)
        else:
            self.config = EvaluationConfig()
        
        self.logger = self._setup_logging()
        self.evaluator = None
        
        self.logger.info("Model Evaluator initialized")
    
    def _setup_logging(self) -> logging.Logger:
        """Setup logging configuration"""
        logger = logging.getLogger('ModelEvaluator')
        logger.setLevel(logging.INFO)
        
        # Create console handler
        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.INFO)
        
        # Create formatter
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        console_handler.setFormatter(formatter)
        
        # Add handler to logger
        logger.addHandler(console_handler)
        
        return logger
    
    def create_evaluator(self) -> BaseEvaluator:
        """Create appropriate evaluator based on model type"""
        if self.config.model_type == 'document_classifier':
            self.evaluator = DocumentClassifierEvaluator(self.config)
        else:
            raise ValueError(f"Unsupported model type: {self.config.model_type}")
        
        return self.evaluator
    
    def evaluate_model(self, model_path: str) -> EvaluationResult:
        """Evaluate a trained model"""
        self.logger.info(f"Starting evaluation for {self.config.model_type}")
        
        # Create evaluator
        evaluator = self.create_evaluator()
        
        # Load model
        model = evaluator.load_model(model_path)
        
        # Prepare data
        data_loader = evaluator.prepare_data()
        
        # Run evaluation
        result = evaluator.evaluate_model(model, data_loader)
        
        # Save results
        self._save_results(result, evaluator)
        
        self.logger.info("Evaluation completed successfully")
        
        return result
    
    def _save_results(self, result: EvaluationResult, evaluator: BaseEvaluator):
        """Save evaluation results"""
        # Save JSON results
        json_path = os.path.join(self.config.output_dir, 'evaluation_results.json')
        result.save_json(json_path)
        
        # Generate and save report
        report = evaluator.generate_evaluation_report(result)
        report_path = os.path.join(self.config.output_dir, 'evaluation_report.md')
        with open(report_path, 'w') as f:
            f.write(report)
        
        # Save plots
        if hasattr(evaluator, 'class_names'):
            evaluator.save_evaluation_plots(result, evaluator.class_names)
        else:
            evaluator.save_evaluation_plots(result)
        
        self.logger.info(f"Results saved to {self.config.output_dir}")

def main():
    """Main function for standalone execution"""
    print("📊 Document Parser Model Evaluator")
    print("=" * 50)
    
    # Check dependencies
    print(f"\n📦 Dependencies:")
    print(f"   PyTorch: {'✅' if TORCH_AVAILABLE else '❌'}")
    print(f"   Scikit-learn: {'✅' if SKLEARN_AVAILABLE else '❌'}")
    print(f"   OpenCV: {'✅' if CV2_AVAILABLE else '❌'}")
    
    if not TORCH_AVAILABLE:
        print("\n⚠️  PyTorch not available. Install with: pip install torch torchvision")
        return 1
    
    if not SKLEARN_AVAILABLE:
        print("\n⚠️  Scikit-learn not available. Install with: pip install scikit-learn")
        return 1
    
    # Example configuration
    config = EvaluationConfig(
        model_type='document_classifier',
        test_data_path='data/test',
        batch_size=32,
        generate_plots=True,
        save_predictions=True
    )
    
    print(f"\n⚙️ Configuration:")
    print(f"   Model type: {config.model_type}")
    print(f"   Test data path: {config.test_data_path}")
    print(f"   Batch size: {config.batch_size}")
    print(f"   Metrics: {config.metrics}")
    print(f"   Generate plots: {config.generate_plots}")
    
    # Initialize evaluator
    evaluator = ModelEvaluator()
    evaluator.config = config
    
    print("\n📋 Usage Examples:")
    print("1. evaluator = ModelEvaluator('config.yaml')")
    print("2. result = evaluator.evaluate_model('model.pth')")
    print("3. print(result.metrics)")
    
    return 0

if __name__ == "__main__":
    exit(main())