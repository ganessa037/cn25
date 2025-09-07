#!/usr/bin/env python3
"""
License Plate Detection Training Script

This script provides a command-line interface for training YOLOv8 models
for license plate detection with dataset validation and model management.
"""

import argparse
import os
import sys
import logging
from pathlib import Path
from datetime import datetime
import yaml
import json
from typing import Dict, Any, Optional

# Add src to path for imports
sys.path.append(str(Path(__file__).parent.parent / 'src'))

from license_plate.core.detector import LicensePlateDetector
from license_plate.config.settings import (
    TRAINED_MODELS_DIR, DATA_DIR, TRAINING_CONFIG,
    YOLO_CONFIG, LOGGING_CONFIG
)

# Setup logging
logging.basicConfig(
    level=getattr(logging, LOGGING_CONFIG['level']),
    format=LOGGING_CONFIG['format'],
    handlers=[
        logging.FileHandler(LOGGING_CONFIG['log_file']),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class LicensePlateTrainer:
    """License plate detection model trainer with CLI interface."""
    
    def __init__(self):
        self.detector = None
        self.training_config = TRAINING_CONFIG.copy()
        
    def validate_dataset(self, data_path: str) -> bool:
        """Validate dataset structure and files.
        
        Args:
            data_path: Path to dataset directory
            
        Returns:
            bool: True if dataset is valid
        """
        logger.info(f"Validating dataset at: {data_path}")
        
        data_dir = Path(data_path)
        if not data_dir.exists():
            logger.error(f"Dataset directory does not exist: {data_path}")
            return False
            
        # Check for required directories
        required_dirs = ['train', 'valid']
        for dir_name in required_dirs:
            dir_path = data_dir / dir_name
            if not dir_path.exists():
                logger.error(f"Required directory missing: {dir_path}")
                return False
                
            # Check for images and labels
            images_dir = dir_path / 'images'
            labels_dir = dir_path / 'labels'
            
            if not images_dir.exists():
                logger.error(f"Images directory missing: {images_dir}")
                return False
                
            if not labels_dir.exists():
                logger.error(f"Labels directory missing: {labels_dir}")
                return False
                
            # Count files
            image_files = list(images_dir.glob('*.jpg')) + list(images_dir.glob('*.png'))
            label_files = list(labels_dir.glob('*.txt'))
            
            logger.info(f"{dir_name}: {len(image_files)} images, {len(label_files)} labels")
            
            if len(image_files) == 0:
                logger.error(f"No image files found in {images_dir}")
                return False
                
            if len(label_files) == 0:
                logger.warning(f"No label files found in {labels_dir}")
                
        # Check for data.yaml
        yaml_path = data_dir / 'data.yaml'
        if yaml_path.exists():
            try:
                with open(yaml_path, 'r') as f:
                    data_config = yaml.safe_load(f)
                logger.info(f"Dataset config loaded: {data_config.get('nc', 'unknown')} classes")
            except Exception as e:
                logger.warning(f"Could not load data.yaml: {e}")
        else:
            logger.warning("data.yaml not found - will use default configuration")
            
        logger.info("Dataset validation completed successfully")
        return True
        
    def create_data_yaml(self, data_path: str, output_path: Optional[str] = None) -> str:
        """Create data.yaml configuration file.
        
        Args:
            data_path: Path to dataset directory
            output_path: Optional output path for data.yaml
            
        Returns:
            str: Path to created data.yaml file
        """
        data_dir = Path(data_path)
        
        if output_path is None:
            yaml_path = data_dir / 'data.yaml'
        else:
            yaml_path = Path(output_path)
            
        # Create data configuration
        data_config = {
            'path': str(data_dir.absolute()),
            'train': 'train/images',
            'val': 'valid/images',
            'test': 'test/images' if (data_dir / 'test').exists() else None,
            'nc': 1,  # Number of classes (license plate)
            'names': ['license_plate']
        }
        
        # Remove None values
        data_config = {k: v for k, v in data_config.items() if v is not None}
        
        # Write YAML file
        with open(yaml_path, 'w') as f:
            yaml.dump(data_config, f, default_flow_style=False)
            
        logger.info(f"Created data.yaml at: {yaml_path}")
        return str(yaml_path)
        
    def setup_training_config(self, args: argparse.Namespace) -> Dict[str, Any]:
        """Setup training configuration from arguments.
        
        Args:
            args: Command line arguments
            
        Returns:
            dict: Training configuration
        """
        config = self.training_config.copy()
        
        # Update from command line arguments
        if args.epochs:
            config['epochs'] = args.epochs
        if args.batch_size:
            config['batch_size'] = args.batch_size
        if args.learning_rate:
            config['learning_rate'] = args.learning_rate
        if args.image_size:
            config['image_size'] = args.image_size
            
        # Set device
        config['device'] = args.device if args.device else 'auto'
        
        # Set workers
        config['workers'] = args.workers if args.workers else config.get('workers', 8)
        
        # Set patience for early stopping
        config['patience'] = args.patience if args.patience else config.get('patience', 50)
        
        logger.info(f"Training configuration: {config}")
        return config
        
    def train_model(self, data_path: str, config: Dict[str, Any], 
                   model_name: str = '../models/yolov8n.pt') -> str:
        """Train the license plate detection model.
        
        Args:
            data_path: Path to dataset
            config: Training configuration
            model_name: Base model name
            
        Returns:
            str: Path to trained model
        """
        logger.info(f"Starting training with model: {model_name}")
        
        # Initialize detector
        self.detector = LicensePlateDetector(device=config['device'])
        
        # Create data.yaml if it doesn't exist
        data_dir = Path(data_path)
        yaml_path = data_dir / 'data.yaml'
        if not yaml_path.exists():
            self.create_data_yaml(data_path)
            
        # Train model
        try:
            results = self.detector.train(
                data_yaml=str(yaml_path),
                epochs=config['epochs'],
                batch_size=config['batch_size'],
                imgsz=config['image_size'],
                workers=config['workers'],
                patience=config['patience']
            )
            
            logger.info("Training completed successfully")
            logger.info(f"Training results: {results}")
            
            return results.get('model_path', '')
            
        except Exception as e:
            logger.error(f"Training failed: {e}")
            raise
            
    def save_training_info(self, model_path: str, config: Dict[str, Any], 
                          results: Dict[str, Any]):
        """Save training information and metadata.
        
        Args:
            model_path: Path to trained model
            config: Training configuration
            results: Training results
        """
        model_dir = Path(model_path).parent
        info_path = model_dir / 'training_info.json'
        
        training_info = {
            'timestamp': datetime.now().isoformat(),
            'model_path': str(model_path),
            'config': config,
            'results': results,
            'dataset_info': {
                'path': config.get('data_path', ''),
                'classes': ['license_plate']
            }
        }
        
        with open(info_path, 'w') as f:
            json.dump(training_info, f, indent=2)
            
        logger.info(f"Training info saved to: {info_path}")
        
def main():
    """Main training function with CLI interface."""
    parser = argparse.ArgumentParser(
        description='Train YOLOv8 model for license plate detection'
    )
    
    # Required arguments
    parser.add_argument(
        'data_path',
        help='Path to dataset directory (should contain train/valid folders)'
    )
    
    # Optional arguments
    parser.add_argument(
        '--model', '-m',
        default='../models/yolov8n.pt',
        help='Base model to use (default: ../models/yolov8n.pt)'
    )
    
    parser.add_argument(
        '--epochs', '-e',
        type=int,
        help=f'Number of training epochs (default: {TRAINING_CONFIG["epochs"]})'
    )
    
    parser.add_argument(
        '--batch-size', '-b',
        type=int,
        help=f'Batch size (default: {TRAINING_CONFIG["batch_size"]})'
    )
    
    parser.add_argument(
        '--learning-rate', '-lr',
        type=float,
        help=f'Learning rate (default: {TRAINING_CONFIG["learning_rate"]})'
    )
    
    parser.add_argument(
        '--image-size', '-img',
        type=int,
        help=f'Image size (default: {TRAINING_CONFIG["image_size"]})'
    )
    
    parser.add_argument(
        '--device', '-d',
        help='Device to use (cpu, 0, 1, etc.) (default: auto)'
    )
    
    parser.add_argument(
        '--workers', '-w',
        type=int,
        help='Number of worker processes (default: 8)'
    )
    
    parser.add_argument(
        '--patience', '-p',
        type=int,
        help='Early stopping patience (default: 50)'
    )
    
    parser.add_argument(
        '--validate-only',
        action='store_true',
        help='Only validate dataset without training'
    )
    
    parser.add_argument(
        '--output-dir', '-o',
        help='Output directory for trained model (default: models/license_plate_detection/trained_models)'
    )
    
    parser.add_argument(
        '--verbose', '-v',
        action='store_true',
        help='Enable verbose logging'
    )
    
    args = parser.parse_args()
    
    # Set logging level
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
        
    # Initialize trainer
    trainer = LicensePlateTrainer()
    
    try:
        # Validate dataset
        if not trainer.validate_dataset(args.data_path):
            logger.error("Dataset validation failed")
            sys.exit(1)
            
        if args.validate_only:
            logger.info("Dataset validation completed. Exiting.")
            return
            
        # Setup training configuration
        config = trainer.setup_training_config(args)
        config['data_path'] = args.data_path
        
        # Train model
        logger.info("Starting model training...")
        model_path = trainer.train_model(
            data_path=args.data_path,
            config=config,
            model_name=args.model
        )
        
        if model_path:
            logger.info(f"Training completed successfully!")
            logger.info(f"Trained model saved to: {model_path}")
            
            # Save training information
            trainer.save_training_info(model_path, config, {'model_path': model_path})
            
        else:
            logger.error("Training failed - no model path returned")
            sys.exit(1)
            
    except KeyboardInterrupt:
        logger.info("Training interrupted by user")
        sys.exit(1)
    except Exception as e:
        logger.error(f"Training failed with error: {e}")
        sys.exit(1)
        
if __name__ == '__main__':
    main()