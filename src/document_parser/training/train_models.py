"""Comprehensive Training Scripts Module

This module provides comprehensive training scripts for all document parser model components
with logging, checkpointing, and monitoring capabilities, following the autocorrect model's
organizational patterns.
"""

import os
import sys
import json
import time
import argparse
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any, Union
from dataclasses import dataclass, field
from datetime import datetime
import logging
import traceback
from collections import defaultdict

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix
import wandb
from tqdm import tqdm

# Import training modules
from .data_preparation import DataPreparationPipeline, DataPreparationConfig
from .classifier_training import ClassifierTrainer, TrainingConfig as ClassifierConfig
from .ocr_integration import OCRIntegrationManager, OCRIntegrationConfig
from .field_extraction_training import FieldExtractionTrainer, FieldExtractionConfig
from .integration_pipeline import ModelIntegrationPipeline, PipelineConfig

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('training.log'),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

class TrainingMode:
    """Training mode constants"""
    DATA_PREPARATION = "data_preparation"
    CLASSIFIER = "classifier"
    OCR_INTEGRATION = "ocr_integration"
    FIELD_EXTRACTION = "field_extraction"
    FULL_PIPELINE = "full_pipeline"
    ALL = "all"

@dataclass
class TrainingScriptConfig:
    """Configuration for training scripts"""
    
    # General configuration
    project_name: str = "document_parser_training"
    experiment_name: str = "default_experiment"
    random_seed: int = 42
    device: str = "auto"  # auto, cpu, cuda
    
    # Training modes
    training_modes: List[str] = field(default_factory=lambda: [TrainingMode.ALL])
    
    # Data paths
    data_root: str = "data_collection"
    output_root: str = "model_artifacts/document_parser"
    logs_root: str = "logs/document_parser"
    
    # Checkpointing
    enable_checkpointing: bool = True
    checkpoint_frequency: int = 5  # epochs
    max_checkpoints: int = 5
    resume_from_checkpoint: Optional[str] = None
    
    # Logging and monitoring
    enable_tensorboard: bool = True
    enable_wandb: bool = False
    wandb_project: str = "document-parser"
    log_frequency: int = 10  # batches
    
    # Performance monitoring
    enable_profiling: bool = False
    memory_monitoring: bool = True
    gpu_monitoring: bool = True
    
    # Early stopping
    enable_early_stopping: bool = True
    early_stopping_patience: int = 10
    early_stopping_metric: str = "val_loss"
    early_stopping_mode: str = "min"  # min or max
    
    # Model saving
    save_best_model: bool = True
    save_final_model: bool = True
    model_format: str = "pytorch"  # pytorch, onnx, both
    
    # Evaluation
    enable_evaluation: bool = True
    evaluation_frequency: int = 1  # epochs
    test_split_ratio: float = 0.2
    
    # Resource management
    max_memory_gb: float = 8.0
    max_training_time_hours: float = 24.0
    
    # Component-specific configs
    data_prep_config: Optional[DataPreparationConfig] = None
    classifier_config: Optional[ClassifierConfig] = None
    ocr_config: Optional[OCRIntegrationConfig] = None
    field_extraction_config: Optional[FieldExtractionConfig] = None
    pipeline_config: Optional[PipelineConfig] = None

class TrainingLogger:
    """Enhanced training logger with multiple backends"""
    
    def __init__(self, config: TrainingScriptConfig):
        self.config = config
        self.setup_logging_backends()
        
        # Metrics storage
        self.metrics_history = defaultdict(list)
        self.current_epoch = 0
        self.current_step = 0
        
        # Performance tracking
        self.start_time = time.time()
        self.epoch_start_time = None
        
    def setup_logging_backends(self):
        """Setup logging backends"""
        # Create log directories
        log_dir = Path(self.config.logs_root)
        log_dir.mkdir(parents=True, exist_ok=True)
        
        # TensorBoard
        if self.config.enable_tensorboard:
            tb_log_dir = log_dir / "tensorboard" / self.config.experiment_name
            self.tb_writer = SummaryWriter(tb_log_dir)
            logger.info(f"TensorBoard logging enabled: {tb_log_dir}")
        else:
            self.tb_writer = None
        
        # Weights & Biases
        if self.config.enable_wandb:
            wandb.init(
                project=self.config.wandb_project,
                name=self.config.experiment_name,
                config=self.config.__dict__
            )
            logger.info("Weights & Biases logging enabled")
        
        # CSV logger
        self.csv_log_path = log_dir / f"{self.config.experiment_name}_metrics.csv"
        
    def log_metrics(self, metrics: Dict[str, float], step: Optional[int] = None):
        """Log metrics to all backends"""
        if step is None:
            step = self.current_step
        
        # Store in history
        for key, value in metrics.items():
            self.metrics_history[key].append((step, value))
        
        # TensorBoard
        if self.tb_writer:
            for key, value in metrics.items():
                self.tb_writer.add_scalar(key, value, step)
        
        # Weights & Biases
        if self.config.enable_wandb:
            wandb.log(metrics, step=step)
        
        # Console logging
        metrics_str = ", ".join([f"{k}: {v:.4f}" for k, v in metrics.items()])
        logger.info(f"Step {step} - {metrics_str}")
    
    def log_epoch_start(self, epoch: int):
        """Log epoch start"""
        self.current_epoch = epoch
        self.epoch_start_time = time.time()
        logger.info(f"Starting epoch {epoch}")
    
    def log_epoch_end(self, epoch: int, metrics: Dict[str, float]):
        """Log epoch end"""
        epoch_time = time.time() - self.epoch_start_time if self.epoch_start_time else 0
        total_time = time.time() - self.start_time
        
        # Add timing metrics
        metrics.update({
            'epoch_time': epoch_time,
            'total_time': total_time,
            'epoch': epoch
        })
        
        self.log_metrics(metrics, step=epoch)
        
        # Save metrics to CSV
        self.save_metrics_csv()
        
        logger.info(f"Epoch {epoch} completed in {epoch_time:.2f}s")
    
    def save_metrics_csv(self):
        """Save metrics to CSV file"""
        if not self.metrics_history:
            return
        
        # Convert to DataFrame
        data = []
        for metric_name, values in self.metrics_history.items():
            for step, value in values:
                data.append({
                    'step': step,
                    'metric': metric_name,
                    'value': value
                })
        
        df = pd.DataFrame(data)
        df.to_csv(self.csv_log_path, index=False)
    
    def close(self):
        """Close logging backends"""
        if self.tb_writer:
            self.tb_writer.close()
        
        if self.config.enable_wandb:
            wandb.finish()
        
        self.save_metrics_csv()

class CheckpointManager:
    """Checkpoint management for training"""
    
    def __init__(self, config: TrainingScriptConfig):
        self.config = config
        self.checkpoint_dir = Path(config.output_root) / "checkpoints"
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        
        self.best_metric = float('inf') if config.early_stopping_mode == 'min' else float('-inf')
        self.checkpoints = []
    
    def save_checkpoint(self, model: nn.Module, optimizer: optim.Optimizer, 
                      epoch: int, metrics: Dict[str, float], 
                      is_best: bool = False, additional_data: Dict[str, Any] = None):
        """Save training checkpoint"""
        if not self.config.enable_checkpointing:
            return
        
        checkpoint_data = {
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'metrics': metrics,
            'config': self.config.__dict__,
            'timestamp': datetime.now().isoformat()
        }
        
        if additional_data:
            checkpoint_data.update(additional_data)
        
        # Save regular checkpoint
        checkpoint_path = self.checkpoint_dir / f"checkpoint_epoch_{epoch}.pt"
        torch.save(checkpoint_data, checkpoint_path)
        
        # Track checkpoints
        self.checkpoints.append((epoch, checkpoint_path))
        
        # Save best model
        if is_best and self.config.save_best_model:
            best_path = self.checkpoint_dir / "best_model.pt"
            torch.save(checkpoint_data, best_path)
            logger.info(f"Best model saved: {best_path}")
        
        # Clean old checkpoints
        self.cleanup_checkpoints()
        
        logger.info(f"Checkpoint saved: {checkpoint_path}")
    
    def load_checkpoint(self, checkpoint_path: str) -> Dict[str, Any]:
        """Load training checkpoint"""
        if not os.path.exists(checkpoint_path):
            raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
        
        checkpoint = torch.load(checkpoint_path, map_location='cpu')
        logger.info(f"Checkpoint loaded: {checkpoint_path}")
        return checkpoint
    
    def cleanup_checkpoints(self):
        """Remove old checkpoints"""
        if len(self.checkpoints) > self.config.max_checkpoints:
            # Sort by epoch and remove oldest
            self.checkpoints.sort(key=lambda x: x[0])
            
            while len(self.checkpoints) > self.config.max_checkpoints:
                epoch, path = self.checkpoints.pop(0)
                if path.exists():
                    path.unlink()
                    logger.info(f"Removed old checkpoint: {path}")
    
    def is_best_model(self, metric_value: float) -> bool:
        """Check if current model is the best"""
        if self.config.early_stopping_mode == 'min':
            is_best = metric_value < self.best_metric
        else:
            is_best = metric_value > self.best_metric
        
        if is_best:
            self.best_metric = metric_value
        
        return is_best

class EarlyStoppingManager:
    """Early stopping management"""
    
    def __init__(self, config: TrainingScriptConfig):
        self.config = config
        self.patience = config.early_stopping_patience
        self.mode = config.early_stopping_mode
        self.metric_name = config.early_stopping_metric
        
        self.best_metric = float('inf') if self.mode == 'min' else float('-inf')
        self.patience_counter = 0
        self.should_stop = False
    
    def update(self, metrics: Dict[str, float]) -> bool:
        """Update early stopping state"""
        if not self.config.enable_early_stopping:
            return False
        
        if self.metric_name not in metrics:
            logger.warning(f"Early stopping metric '{self.metric_name}' not found in metrics")
            return False
        
        current_metric = metrics[self.metric_name]
        
        if self.mode == 'min':
            improved = current_metric < self.best_metric
        else:
            improved = current_metric > self.best_metric
        
        if improved:
            self.best_metric = current_metric
            self.patience_counter = 0
            logger.info(f"Early stopping metric improved: {self.metric_name} = {current_metric:.4f}")
        else:
            self.patience_counter += 1
            logger.info(f"Early stopping patience: {self.patience_counter}/{self.patience}")
        
        if self.patience_counter >= self.patience:
            self.should_stop = True
            logger.info(f"Early stopping triggered after {self.patience} epochs without improvement")
        
        return self.should_stop

class ResourceMonitor:
    """Resource usage monitoring"""
    
    def __init__(self, config: TrainingScriptConfig):
        self.config = config
        self.start_time = time.time()
        
        # Import optional dependencies
        try:
            import psutil
            self.psutil = psutil
        except ImportError:
            self.psutil = None
            logger.warning("psutil not available, memory monitoring disabled")
        
        try:
            import GPUtil
            self.gputil = GPUtil
        except ImportError:
            self.gputil = None
            logger.warning("GPUtil not available, GPU monitoring disabled")
    
    def get_resource_usage(self) -> Dict[str, float]:
        """Get current resource usage"""
        usage = {
            'training_time_hours': (time.time() - self.start_time) / 3600
        }
        
        # Memory usage
        if self.psutil and self.config.memory_monitoring:
            memory = self.psutil.virtual_memory()
            usage.update({
                'memory_used_gb': memory.used / (1024**3),
                'memory_percent': memory.percent
            })
        
        # GPU usage
        if self.gputil and self.config.gpu_monitoring:
            try:
                gpus = self.gputil.getGPUs()
                if gpus:
                    gpu = gpus[0]  # Use first GPU
                    usage.update({
                        'gpu_memory_used_gb': gpu.memoryUsed / 1024,
                        'gpu_memory_percent': gpu.memoryUtil * 100,
                        'gpu_utilization_percent': gpu.load * 100
                    })
            except Exception as e:
                logger.warning(f"GPU monitoring error: {e}")
        
        return usage
    
    def check_resource_limits(self) -> bool:
        """Check if resource limits are exceeded"""
        usage = self.get_resource_usage()
        
        # Check memory limit
        if 'memory_used_gb' in usage and usage['memory_used_gb'] > self.config.max_memory_gb:
            logger.error(f"Memory limit exceeded: {usage['memory_used_gb']:.2f}GB > {self.config.max_memory_gb}GB")
            return True
        
        # Check time limit
        if usage['training_time_hours'] > self.config.max_training_time_hours:
            logger.error(f"Time limit exceeded: {usage['training_time_hours']:.2f}h > {self.config.max_training_time_hours}h")
            return True
        
        return False

class ComprehensiveTrainer:
    """Main comprehensive trainer for all components"""
    
    def __init__(self, config: TrainingScriptConfig):
        self.config = config
        self.setup_environment()
        
        # Initialize managers
        self.logger = TrainingLogger(config)
        self.checkpoint_manager = CheckpointManager(config)
        self.early_stopping = EarlyStoppingManager(config)
        self.resource_monitor = ResourceMonitor(config)
        
        # Initialize trainers
        self.trainers = {}
        self.setup_trainers()
    
    def setup_environment(self):
        """Setup training environment"""
        # Set random seeds
        np.random.seed(self.config.random_seed)
        torch.manual_seed(self.config.random_seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(self.config.random_seed)
        
        # Setup device
        if self.config.device == "auto":
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(self.config.device)
        
        logger.info(f"Using device: {self.device}")
        
        # Create output directories
        Path(self.config.output_root).mkdir(parents=True, exist_ok=True)
        Path(self.config.logs_root).mkdir(parents=True, exist_ok=True)
    
    def setup_trainers(self):
        """Setup component trainers"""
        # Data preparation
        if self.config.data_prep_config:
            self.data_prep = DataPreparationPipeline(self.config.data_prep_config)
        
        # Classifier trainer
        if self.config.classifier_config:
            self.trainers[TrainingMode.CLASSIFIER] = ClassifierTrainer(self.config.classifier_config)
        
        # OCR integration
        if self.config.ocr_config:
            self.trainers[TrainingMode.OCR_INTEGRATION] = OCRIntegrationManager(self.config.ocr_config)
        
        # Field extraction trainer
        if self.config.field_extraction_config:
            self.trainers[TrainingMode.FIELD_EXTRACTION] = FieldExtractionTrainer(self.config.field_extraction_config)
        
        # Full pipeline
        if self.config.pipeline_config:
            self.trainers[TrainingMode.FULL_PIPELINE] = ModelIntegrationPipeline(self.config.pipeline_config)
    
    def train_component(self, mode: str) -> Dict[str, Any]:
        """Train specific component"""
        logger.info(f"Starting training for component: {mode}")
        
        try:
            if mode == TrainingMode.DATA_PREPARATION:
                return self.train_data_preparation()
            elif mode == TrainingMode.CLASSIFIER:
                return self.train_classifier()
            elif mode == TrainingMode.OCR_INTEGRATION:
                return self.train_ocr_integration()
            elif mode == TrainingMode.FIELD_EXTRACTION:
                return self.train_field_extraction()
            elif mode == TrainingMode.FULL_PIPELINE:
                return self.train_full_pipeline()
            else:
                raise ValueError(f"Unknown training mode: {mode}")
        
        except Exception as e:
            logger.error(f"Training failed for {mode}: {e}")
            logger.error(traceback.format_exc())
            return {'success': False, 'error': str(e)}
    
    def train_data_preparation(self) -> Dict[str, Any]:
        """Train data preparation component"""
        if not hasattr(self, 'data_prep'):
            return {'success': False, 'error': 'Data preparation not configured'}
        
        try:
            # Prepare datasets
            train_dataset, val_dataset, test_dataset = self.data_prep.prepare_datasets(
                data_path=self.config.data_root
            )
            
            # Save dataset information
            dataset_info = {
                'train_size': len(train_dataset),
                'val_size': len(val_dataset),
                'test_size': len(test_dataset),
                'preparation_config': self.config.data_prep_config.__dict__
            }
            
            # Save to file
            info_path = Path(self.config.output_root) / "dataset_info.json"
            with open(info_path, 'w') as f:
                json.dump(dataset_info, f, indent=2, default=str)
            
            logger.info(f"Data preparation completed: {dataset_info}")
            
            return {
                'success': True,
                'dataset_info': dataset_info,
                'train_dataset': train_dataset,
                'val_dataset': val_dataset,
                'test_dataset': test_dataset
            }
        
        except Exception as e:
            return {'success': False, 'error': str(e)}
    
    def train_classifier(self) -> Dict[str, Any]:
        """Train document classifier"""
        if TrainingMode.CLASSIFIER not in self.trainers:
            return {'success': False, 'error': 'Classifier trainer not configured'}
        
        trainer = self.trainers[TrainingMode.CLASSIFIER]
        
        try:
            # Train classifier
            results = trainer.train(
                data_path=self.config.data_root,
                output_path=self.config.output_root
            )
            
            logger.info(f"Classifier training completed: {results}")
            return {'success': True, 'results': results}
        
        except Exception as e:
            return {'success': False, 'error': str(e)}
    
    def train_ocr_integration(self) -> Dict[str, Any]:
        """Train OCR integration"""
        if TrainingMode.OCR_INTEGRATION not in self.trainers:
            return {'success': False, 'error': 'OCR integration not configured'}
        
        manager = self.trainers[TrainingMode.OCR_INTEGRATION]
        
        try:
            # Setup OCR integration
            setup_results = manager.setup_integration(
                data_path=self.config.data_root,
                output_path=self.config.output_root
            )
            
            logger.info(f"OCR integration setup completed: {setup_results}")
            return {'success': True, 'results': setup_results}
        
        except Exception as e:
            return {'success': False, 'error': str(e)}
    
    def train_field_extraction(self) -> Dict[str, Any]:
        """Train field extraction models"""
        if TrainingMode.FIELD_EXTRACTION not in self.trainers:
            return {'success': False, 'error': 'Field extraction trainer not configured'}
        
        trainer = self.trainers[TrainingMode.FIELD_EXTRACTION]
        
        try:
            # Train field extraction models
            results = trainer.train_all_models(
                data_path=self.config.data_root,
                output_path=self.config.output_root
            )
            
            logger.info(f"Field extraction training completed: {results}")
            return {'success': True, 'results': results}
        
        except Exception as e:
            return {'success': False, 'error': str(e)}
    
    def train_full_pipeline(self) -> Dict[str, Any]:
        """Train full pipeline integration"""
        if TrainingMode.FULL_PIPELINE not in self.trainers:
            return {'success': False, 'error': 'Full pipeline not configured'}
        
        pipeline = self.trainers[TrainingMode.FULL_PIPELINE]
        
        try:
            # Test pipeline with sample data
            test_results = self.test_pipeline_integration(pipeline)
            
            logger.info(f"Full pipeline training completed: {test_results}")
            return {'success': True, 'results': test_results}
        
        except Exception as e:
            return {'success': False, 'error': str(e)}
    
    def test_pipeline_integration(self, pipeline: ModelIntegrationPipeline) -> Dict[str, Any]:
        """Test pipeline integration"""
        # Find test images
        test_images = []
        data_path = Path(self.config.data_root)
        
        for ext in ['*.jpg', '*.jpeg', '*.png', '*.pdf']:
            test_images.extend(data_path.rglob(ext))
        
        if not test_images:
            return {'success': False, 'error': 'No test images found'}
        
        # Test with first few images
        test_sample = test_images[:5]
        results = []
        
        for img_path in test_sample:
            try:
                result = pipeline.process_single(str(img_path), f"test_{img_path.stem}")
                results.append({
                    'input_path': str(img_path),
                    'success': result.success,
                    'processing_time': result.processing_time,
                    'confidence': result.overall_confidence
                })
            except Exception as e:
                results.append({
                    'input_path': str(img_path),
                    'success': False,
                    'error': str(e)
                })
        
        # Calculate summary statistics
        successful = [r for r in results if r.get('success', False)]
        success_rate = len(successful) / len(results) if results else 0
        avg_processing_time = np.mean([r.get('processing_time', 0) for r in successful]) if successful else 0
        avg_confidence = np.mean([r.get('confidence', 0) for r in successful]) if successful else 0
        
        return {
            'success': True,
            'test_results': results,
            'summary': {
                'total_tests': len(results),
                'successful_tests': len(successful),
                'success_rate': success_rate,
                'average_processing_time': avg_processing_time,
                'average_confidence': avg_confidence
            }
        }
    
    def train_all(self) -> Dict[str, Any]:
        """Train all components"""
        logger.info("Starting comprehensive training for all components")
        
        all_results = {}
        
        # Determine training order
        training_order = [
            TrainingMode.DATA_PREPARATION,
            TrainingMode.CLASSIFIER,
            TrainingMode.OCR_INTEGRATION,
            TrainingMode.FIELD_EXTRACTION,
            TrainingMode.FULL_PIPELINE
        ]
        
        for mode in training_order:
            if TrainingMode.ALL in self.config.training_modes or mode in self.config.training_modes:
                logger.info(f"Training component: {mode}")
                
                # Check resource limits
                if self.resource_monitor.check_resource_limits():
                    logger.error("Resource limits exceeded, stopping training")
                    break
                
                # Train component
                result = self.train_component(mode)
                all_results[mode] = result
                
                # Log resource usage
                resource_usage = self.resource_monitor.get_resource_usage()
                self.logger.log_metrics({
                    f"{mode}_completed": 1,
                    **{f"resource_{k}": v for k, v in resource_usage.items()}
                })
                
                # Stop if component failed and fail_fast is enabled
                if not result.get('success', False):
                    logger.error(f"Component {mode} failed: {result.get('error', 'Unknown error')}")
                    if hasattr(self.config, 'fail_fast') and self.config.fail_fast:
                        break
        
        # Generate final report
        final_report = self.generate_training_report(all_results)
        
        # Save final report
        report_path = Path(self.config.output_root) / "training_report.json"
        with open(report_path, 'w') as f:
            json.dump(final_report, f, indent=2, default=str)
        
        logger.info(f"Comprehensive training completed. Report saved: {report_path}")
        
        return final_report
    
    def generate_training_report(self, results: Dict[str, Any]) -> Dict[str, Any]:
        """Generate comprehensive training report"""
        report = {
            'experiment_name': self.config.experiment_name,
            'timestamp': datetime.now().isoformat(),
            'configuration': self.config.__dict__,
            'results': results,
            'resource_usage': self.resource_monitor.get_resource_usage(),
            'summary': {
                'total_components': len(results),
                'successful_components': sum(1 for r in results.values() if r.get('success', False)),
                'failed_components': sum(1 for r in results.values() if not r.get('success', False))
            }
        }
        
        # Add success rate
        if report['summary']['total_components'] > 0:
            report['summary']['success_rate'] = (
                report['summary']['successful_components'] / report['summary']['total_components']
            )
        
        return report
    
    def cleanup(self):
        """Cleanup training resources"""
        self.logger.close()
        logger.info("Training cleanup completed")

def create_default_configs() -> TrainingScriptConfig:
    """Create default training configuration"""
    # Data preparation config
    data_prep_config = DataPreparationConfig(
        train_split=0.7,
        val_split=0.15,
        test_split=0.15,
        augmentation_enabled=True,
        balance_dataset=True
    )
    
    # Classifier config
    classifier_config = ClassifierConfig(
        model_name="efficientnet_b0",
        num_epochs=50,
        batch_size=16,
        learning_rate=0.001,
        early_stopping_patience=10
    )
    
    # OCR integration config
    ocr_config = OCRIntegrationConfig(
        engines=["tesseract", "easyocr"],
        languages=["en", "ms"],
        consensus_method="weighted_average"
    )
    
    # Field extraction config
    field_extraction_config = FieldExtractionConfig(
        extraction_methods=["template", "ner", "coordinate"],
        ner_model="bert-base-multilingual-cased",
        confidence_threshold=0.7
    )
    
    # Pipeline config
    pipeline_config = PipelineConfig(
        processing_mode="hybrid",
        max_workers=4,
        enable_caching=True
    )
    
    # Main training config
    config = TrainingScriptConfig(
        experiment_name=f"document_parser_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
        training_modes=[TrainingMode.ALL],
        data_prep_config=data_prep_config,
        classifier_config=classifier_config,
        ocr_config=ocr_config,
        field_extraction_config=field_extraction_config,
        pipeline_config=pipeline_config
    )
    
    return config

def main():
    """Main function for standalone execution"""
    parser = argparse.ArgumentParser(description="Comprehensive Document Parser Training")
    parser.add_argument("--config", type=str, help="Path to configuration file")
    parser.add_argument("--mode", type=str, choices=[
        TrainingMode.DATA_PREPARATION,
        TrainingMode.CLASSIFIER,
        TrainingMode.OCR_INTEGRATION,
        TrainingMode.FIELD_EXTRACTION,
        TrainingMode.FULL_PIPELINE,
        TrainingMode.ALL
    ], default=TrainingMode.ALL, help="Training mode")
    parser.add_argument("--experiment-name", type=str, help="Experiment name")
    parser.add_argument("--data-root", type=str, default="data_collection", help="Data root directory")
    parser.add_argument("--output-root", type=str, default="model_artifacts/document_parser", help="Output root directory")
    parser.add_argument("--device", type=str, choices=["auto", "cpu", "cuda"], default="auto", help="Training device")
    parser.add_argument("--resume", type=str, help="Resume from checkpoint")
    parser.add_argument("--wandb", action="store_true", help="Enable Weights & Biases logging")
    
    args = parser.parse_args()
    
    # Load or create configuration
    if args.config and os.path.exists(args.config):
        with open(args.config, 'r') as f:
            config_dict = json.load(f)
        # Convert to config object (simplified)
        config = create_default_configs()
        config.__dict__.update(config_dict)
    else:
        config = create_default_configs()
    
    # Override with command line arguments
    if args.experiment_name:
        config.experiment_name = args.experiment_name
    if args.data_root:
        config.data_root = args.data_root
    if args.output_root:
        config.output_root = args.output_root
    if args.device:
        config.device = args.device
    if args.resume:
        config.resume_from_checkpoint = args.resume
    if args.wandb:
        config.enable_wandb = True
    
    config.training_modes = [args.mode]
    
    # Create trainer
    trainer = ComprehensiveTrainer(config)
    
    try:
        # Start training
        logger.info(f"Starting training with configuration: {config.experiment_name}")
        
        if args.mode == TrainingMode.ALL:
            results = trainer.train_all()
        else:
            results = trainer.train_component(args.mode)
        
        # Print results
        print("\n" + "="*60)
        print("TRAINING COMPLETED")
        print("="*60)
        
        if isinstance(results, dict) and 'summary' in results:
            summary = results['summary']
            print(f"Total Components: {summary.get('total_components', 0)}")
            print(f"Successful: {summary.get('successful_components', 0)}")
            print(f"Failed: {summary.get('failed_components', 0)}")
            print(f"Success Rate: {summary.get('success_rate', 0):.2%}")
        
        print("="*60)
        
    except KeyboardInterrupt:
        logger.info("Training interrupted by user")
    except Exception as e:
        logger.error(f"Training failed: {e}")
        logger.error(traceback.format_exc())
    finally:
        trainer.cleanup()

if __name__ == "__main__":
    main()