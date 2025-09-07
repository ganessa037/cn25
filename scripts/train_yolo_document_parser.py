#!/usr/bin/env python3
"""
YOLO Document Parser Training Script

Trains a YOLO model for Malaysian Identity Card detection and field extraction
using the Roboflow dataset.
"""

import sys
import os
import argparse
import json
import time
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Any, Optional

import torch
import yaml
from ultralytics import YOLO
import cv2
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import seaborn as sns

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent))

class YOLODocumentTrainer:
    """YOLO-based document parser trainer."""
    
    def __init__(self, dataset_path: str, model_size: str = 'n'):
        """Initialize the YOLO trainer.
        
        Args:
            dataset_path: Path to the dataset directory
            model_size: YOLO model size ('n', 's', 'm', 'l', 'x')
        """
        self.dataset_path = Path(dataset_path)
        self.model_size = model_size
        self.model = None
        self.results = None
        
        # Create output directories
        self.output_dir = Path('model_artifacts/yolo_document_parser')
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Validate dataset
        self._validate_dataset()
        
    def _validate_dataset(self):
        """Validate the dataset structure."""
        required_files = ['data.yaml', 'train/images', 'train/labels', 'valid/images', 'valid/labels']
        
        for file_path in required_files:
            full_path = self.dataset_path / file_path
            if not full_path.exists():
                raise FileNotFoundError(f"Required dataset file/directory not found: {full_path}")
        
        print(f"✅ Dataset validation passed: {self.dataset_path}")
        
    def prepare_data_config(self) -> str:
        """Prepare the data configuration file for training."""
        data_yaml_path = self.dataset_path / 'data.yaml'
        
        # Read the original data.yaml
        with open(data_yaml_path, 'r') as f:
            data_config = yaml.safe_load(f)
        
        # Update paths to be absolute
        data_config['train'] = str(self.dataset_path / 'train' / 'images')
        data_config['val'] = str(self.dataset_path / 'valid' / 'images')
        
        # Create updated config file
        updated_config_path = self.output_dir / 'data_config.yaml'
        with open(updated_config_path, 'w') as f:
            yaml.dump(data_config, f, default_flow_style=False)
        
        print(f"📝 Data configuration saved: {updated_config_path}")
        return str(updated_config_path)
    
    def analyze_dataset(self):
        """Analyze the dataset and print statistics."""
        print("\n📊 Dataset Analysis:")
        
        # Count training images
        train_images = list((self.dataset_path / 'train' / 'images').glob('*.jpg'))
        valid_images = list((self.dataset_path / 'valid' / 'images').glob('*.jpg'))
        
        print(f"   Training images: {len(train_images)}")
        print(f"   Validation images: {len(valid_images)}")
        print(f"   Total images: {len(train_images) + len(valid_images)}")
        
        # Analyze image sizes
        if train_images:
            sample_img = cv2.imread(str(train_images[0]))
            print(f"   Sample image shape: {sample_img.shape}")
        
        # Analyze labels
        train_labels = list((self.dataset_path / 'train' / 'labels').glob('*.txt'))
        valid_labels = list((self.dataset_path / 'valid' / 'labels').glob('*.txt'))
        
        print(f"   Training labels: {len(train_labels)}")
        print(f"   Validation labels: {len(valid_labels)}")
        
        # Check class distribution
        class_counts = {}
        for label_file in train_labels + valid_labels:
            with open(label_file, 'r') as f:
                for line in f:
                    if line.strip():
                        class_id = int(line.split()[0])
                        class_counts[class_id] = class_counts.get(class_id, 0) + 1
        
        print(f"   Class distribution: {class_counts}")
        
    def train(self, epochs: int = 100, imgsz: int = 640, batch_size: int = 16, 
              patience: int = 50, save_period: int = 10) -> str:
        """Train the YOLO model.
        
        Args:
            epochs: Number of training epochs
            imgsz: Image size for training
            batch_size: Batch size
            patience: Early stopping patience
            save_period: Save model every N epochs
            
        Returns:
            Path to the best trained model
        """
        print(f"\n🚀 Starting YOLO training with {self.model_size} model...")
        
        # Initialize model
        model_name = f'yolov8{self.model_size}.pt'
        self.model = YOLO(model_name)
        
        # Prepare data config
        data_config_path = self.prepare_data_config()
        
        # Training parameters
        train_params = {
            'data': data_config_path,
            'epochs': epochs,
            'imgsz': imgsz,
            'batch': batch_size,
            'patience': patience,
            'save_period': save_period,
            'project': str(self.output_dir),
            'name': f'train_{datetime.now().strftime("%Y%m%d_%H%M%S")}',
            'exist_ok': True,
            'pretrained': True,
            'optimizer': 'AdamW',
            'lr0': 0.01,
            'lrf': 0.01,
            'momentum': 0.937,
            'weight_decay': 0.0005,
            'warmup_epochs': 3,
            'warmup_momentum': 0.8,
            'warmup_bias_lr': 0.1,
            'box': 7.5,
            'cls': 0.5,
            'dfl': 1.5,
            'pose': 12.0,
            'kobj': 1.0,
            'label_smoothing': 0.0,
            'nbs': 64,
            'hsv_h': 0.015,
            'hsv_s': 0.7,
            'hsv_v': 0.4,
            'degrees': 0.0,
            'translate': 0.1,
            'scale': 0.5,
            'shear': 0.0,
            'perspective': 0.0,
            'flipud': 0.0,
            'fliplr': 0.5,
            'mosaic': 1.0,
            'mixup': 0.0,
            'copy_paste': 0.0
        }
        
        print(f"📋 Training parameters:")
        for key, value in train_params.items():
            print(f"   {key}: {value}")
        
        # Start training
        start_time = time.time()
        self.results = self.model.train(**train_params)
        training_time = time.time() - start_time
        
        print(f"\n✅ Training completed in {training_time:.2f} seconds")
        
        # Get best model path
        best_model_path = self.results.save_dir / 'weights' / 'best.pt'
        # Copy to models directory with descriptive name
        final_model_path = Path('models/document_parser/yolo_document_classifier_v1.pt')
        final_model_path.parent.mkdir(parents=True, exist_ok=True)
        if best_model_path.exists():
            import shutil
            shutil.copy2(best_model_path, final_model_path)
            print(f"Model saved to: {final_model_path}")
        print(f"🏆 Best model saved: {best_model_path}")
        
        return str(best_model_path)
    
    def validate(self, model_path: str = None) -> Dict[str, Any]:
        """Validate the trained model.
        
        Args:
            model_path: Path to the model to validate (uses best if None)
            
        Returns:
            Validation metrics
        """
        print("\n🔍 Validating model...")
        
        if model_path:
            model = YOLO(model_path)
        elif self.model:
            model = self.model
        else:
            raise ValueError("No model available for validation")
        
        # Prepare data config
        data_config_path = self.prepare_data_config()
        
        # Run validation
        val_results = model.val(data=data_config_path)
        
        # Extract metrics
        metrics = {
            'mAP50': val_results.box.map50,
            'mAP50-95': val_results.box.map,
            'precision': val_results.box.mp,
            'recall': val_results.box.mr,
            'fitness': val_results.fitness
        }
        
        print(f"📊 Validation Results:")
        for metric, value in metrics.items():
            print(f"   {metric}: {value:.4f}")
        
        return metrics
    
    def export_model(self, model_path: str, formats: List[str] = ['onnx', 'torchscript']) -> Dict[str, str]:
        """Export the trained model to different formats.
        
        Args:
            model_path: Path to the trained model
            formats: List of export formats
            
        Returns:
            Dictionary of format -> exported_path
        """
        print(f"\n📦 Exporting model to formats: {formats}")
        
        model = YOLO(model_path)
        exported_paths = {}
        
        for format_name in formats:
            try:
                export_path = model.export(format=format_name)
                exported_paths[format_name] = str(export_path)
                print(f"   ✅ {format_name.upper()}: {export_path}")
            except Exception as e:
                print(f"   ❌ {format_name.upper()}: Failed - {e}")
        
        return exported_paths
    
    def create_training_report(self, model_path: str, metrics: Dict[str, Any]) -> str:
        """Create a comprehensive training report.
        
        Args:
            model_path: Path to the trained model
            metrics: Validation metrics
            
        Returns:
            Path to the generated report
        """
        report_path = self.output_dir / f'training_report_{datetime.now().strftime("%Y%m%d_%H%M%S")}.json'
        
        report = {
            'timestamp': datetime.now().isoformat(),
            'dataset_path': str(self.dataset_path),
            'model_size': self.model_size,
            'model_path': model_path,
            'metrics': metrics,
            'training_config': {
                'framework': 'YOLOv8',
                'task': 'Malaysian Identity Card Detection',
                'classes': ['MY Identity Card']
            }
        }
        
        with open(report_path, 'w') as f:
            json.dump(report, f, indent=2)
        
        print(f"📄 Training report saved: {report_path}")
        return str(report_path)

def main():
    """Main training function."""
    parser = argparse.ArgumentParser(description='Train YOLO Document Parser')
    parser.add_argument('--dataset', type=str, default='document_parser_dataset',
                       help='Path to dataset directory')
    parser.add_argument('--model-size', type=str, default='n', choices=['n', 's', 'm', 'l', 'x'],
                       help='YOLO model size')
    parser.add_argument('--epochs', type=int, default=100,
                       help='Number of training epochs')
    parser.add_argument('--batch-size', type=int, default=16,
                       help='Batch size')
    parser.add_argument('--imgsz', type=int, default=640,
                       help='Image size')
    parser.add_argument('--patience', type=int, default=50,
                       help='Early stopping patience')
    parser.add_argument('--export', nargs='+', default=['onnx'],
                       help='Export formats')
    
    args = parser.parse_args()
    
    try:
        # Initialize trainer
        trainer = YOLODocumentTrainer(
            dataset_path=args.dataset,
            model_size=args.model_size
        )
        
        # Analyze dataset
        trainer.analyze_dataset()
        
        # Train model
        best_model_path = trainer.train(
            epochs=args.epochs,
            batch_size=args.batch_size,
            imgsz=args.imgsz,
            patience=args.patience
        )
        
        # Validate model
        metrics = trainer.validate(best_model_path)
        
        # Export model
        exported_paths = trainer.export_model(best_model_path, args.export)
        
        # Create report
        report_path = trainer.create_training_report(best_model_path, metrics)
        
        print(f"\n🎉 Training pipeline completed successfully!")
        print(f"   Best model: {best_model_path}")
        print(f"   Report: {report_path}")
        print(f"   Exported formats: {list(exported_paths.keys())}")
        
        return 0
        
    except Exception as e:
        print(f"❌ Training failed: {e}")
        return 1

if __name__ == '__main__':
    exit(main())