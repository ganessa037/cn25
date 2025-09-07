"""Data Preparation Pipeline for Document Parser

This module provides comprehensive data preparation functionality for document processing models,
including dataset creation, augmentation, validation, and preprocessing following the autocorrect
model's organizational patterns.
"""

import os
import json
import random
import shutil
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass, field
from datetime import datetime
import logging

import cv2
import numpy as np
import pandas as pd
from PIL import Image, ImageEnhance, ImageFilter
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.preprocessing import LabelEncoder
import albumentations as A
from albumentations.pytorch import ToTensorV2

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class DataPreparationConfig:
    """Configuration for data preparation pipeline"""
    
    # Dataset paths
    raw_data_path: str = "data_collection/real"
    synthetic_data_path: str = "data_collection/synthetic"
    output_path: str = "model_artifacts/document_parser/datasets"
    
    # Dataset splits
    train_ratio: float = 0.7
    val_ratio: float = 0.15
    test_ratio: float = 0.15
    
    # Image preprocessing
    target_size: Tuple[int, int] = (224, 224)
    normalize_mean: List[float] = field(default_factory=lambda: [0.485, 0.456, 0.406])
    normalize_std: List[float] = field(default_factory=lambda: [0.229, 0.224, 0.225])
    
    # Data augmentation
    augmentation_factor: int = 3
    rotation_range: int = 10
    brightness_range: Tuple[float, float] = (0.8, 1.2)
    contrast_range: Tuple[float, float] = (0.8, 1.2)
    noise_probability: float = 0.3
    blur_probability: float = 0.2
    
    # Quality control
    min_image_size: Tuple[int, int] = (100, 100)
    max_file_size_mb: float = 10.0
    supported_formats: List[str] = field(default_factory=lambda: ['.jpg', '.jpeg', '.png', '.tiff', '.bmp'])
    
    # Validation
    cross_validation_folds: int = 5
    stratify_by_class: bool = True
    balance_classes: bool = True
    
    # Processing
    batch_size: int = 32
    num_workers: int = 4
    random_seed: int = 42

class ImageAugmentator:
    """Advanced image augmentation for document images"""
    
    def __init__(self, config: DataPreparationConfig):
        self.config = config
        self.setup_augmentation_pipeline()
    
    def setup_augmentation_pipeline(self):
        """Setup albumentations augmentation pipeline"""
        self.train_transforms = A.Compose([
            A.Resize(height=self.config.target_size[0], width=self.config.target_size[1]),
            A.Rotate(limit=self.config.rotation_range, p=0.5),
            A.RandomBrightnessContrast(
                brightness_limit=0.2, 
                contrast_limit=0.2, 
                p=0.5
            ),
            A.OneOf([
                A.GaussNoise(var_limit=(10.0, 50.0), p=1.0),
                A.ISONoise(color_shift=(0.01, 0.05), intensity=(0.1, 0.5), p=1.0),
            ], p=self.config.noise_probability),
            A.OneOf([
                A.MotionBlur(blur_limit=3, p=1.0),
                A.GaussianBlur(blur_limit=3, p=1.0),
            ], p=self.config.blur_probability),
            A.ShiftScaleRotate(shift_limit=0.05, scale_limit=0.05, rotate_limit=5, p=0.3),
            A.Perspective(scale=(0.05, 0.1), p=0.3),
            A.GridDistortion(num_steps=5, distort_limit=0.1, p=0.2),
            A.Normalize(
                mean=self.config.normalize_mean,
                std=self.config.normalize_std
            ),
            ToTensorV2()
        ])
        
        self.val_transforms = A.Compose([
            A.Resize(height=self.config.target_size[0], width=self.config.target_size[1]),
            A.Normalize(
                mean=self.config.normalize_mean,
                std=self.config.normalize_std
            ),
            ToTensorV2()
        ])
    
    def augment_image(self, image: np.ndarray, is_training: bool = True) -> np.ndarray:
        """Apply augmentation to a single image"""
        transforms = self.train_transforms if is_training else self.val_transforms
        augmented = transforms(image=image)
        return augmented['image']
    
    def create_augmented_dataset(self, images: List[np.ndarray], labels: List[str]) -> Tuple[List[np.ndarray], List[str]]:
        """Create augmented dataset with specified factor"""
        augmented_images = []
        augmented_labels = []
        
        for image, label in zip(images, labels):
            # Add original image
            augmented_images.append(image)
            augmented_labels.append(label)
            
            # Add augmented versions
            for _ in range(self.config.augmentation_factor):
                aug_image = self.augment_image(image, is_training=True)
                augmented_images.append(aug_image)
                augmented_labels.append(label)
        
        return augmented_images, augmented_labels

class DatasetValidator:
    """Validate dataset quality and integrity"""
    
    def __init__(self, config: DataPreparationConfig):
        self.config = config
        self.validation_results = {}
    
    def validate_image(self, image_path: str) -> Dict[str, Any]:
        """Validate a single image file"""
        validation_result = {
            'path': image_path,
            'valid': True,
            'issues': []
        }
        
        try:
            # Check file existence
            if not os.path.exists(image_path):
                validation_result['valid'] = False
                validation_result['issues'].append('File does not exist')
                return validation_result
            
            # Check file size
            file_size_mb = os.path.getsize(image_path) / (1024 * 1024)
            if file_size_mb > self.config.max_file_size_mb:
                validation_result['valid'] = False
                validation_result['issues'].append(f'File too large: {file_size_mb:.2f}MB')
            
            # Check file format
            file_ext = Path(image_path).suffix.lower()
            if file_ext not in self.config.supported_formats:
                validation_result['valid'] = False
                validation_result['issues'].append(f'Unsupported format: {file_ext}')
            
            # Check image properties
            image = cv2.imread(image_path)
            if image is None:
                validation_result['valid'] = False
                validation_result['issues'].append('Cannot read image')
                return validation_result
            
            height, width = image.shape[:2]
            if height < self.config.min_image_size[0] or width < self.config.min_image_size[1]:
                validation_result['valid'] = False
                validation_result['issues'].append(f'Image too small: {width}x{height}')
            
            # Check for corruption
            if np.all(image == 0) or np.all(image == 255):
                validation_result['valid'] = False
                validation_result['issues'].append('Image appears corrupted')
            
            validation_result['width'] = width
            validation_result['height'] = height
            validation_result['channels'] = image.shape[2] if len(image.shape) == 3 else 1
            validation_result['file_size_mb'] = file_size_mb
            
        except Exception as e:
            validation_result['valid'] = False
            validation_result['issues'].append(f'Validation error: {str(e)}')
        
        return validation_result
    
    def validate_dataset(self, dataset_info: Dict[str, List]) -> Dict[str, Any]:
        """Validate entire dataset"""
        logger.info("Starting dataset validation...")
        
        validation_summary = {
            'total_images': 0,
            'valid_images': 0,
            'invalid_images': 0,
            'class_distribution': {},
            'issues_summary': {},
            'detailed_results': []
        }
        
        for class_name, image_paths in dataset_info.items():
            validation_summary['class_distribution'][class_name] = len(image_paths)
            
            for image_path in image_paths:
                validation_summary['total_images'] += 1
                result = self.validate_image(image_path)
                validation_summary['detailed_results'].append(result)
                
                if result['valid']:
                    validation_summary['valid_images'] += 1
                else:
                    validation_summary['invalid_images'] += 1
                    for issue in result['issues']:
                        if issue not in validation_summary['issues_summary']:
                            validation_summary['issues_summary'][issue] = 0
                        validation_summary['issues_summary'][issue] += 1
        
        self.validation_results = validation_summary
        logger.info(f"Validation complete: {validation_summary['valid_images']}/{validation_summary['total_images']} valid images")
        
        return validation_summary

class DatasetSplitter:
    """Handle dataset splitting with stratification and cross-validation"""
    
    def __init__(self, config: DataPreparationConfig):
        self.config = config
        random.seed(config.random_seed)
        np.random.seed(config.random_seed)
    
    def create_train_val_test_split(self, dataset_info: Dict[str, List]) -> Dict[str, Dict[str, List]]:
        """Create stratified train/validation/test splits"""
        logger.info("Creating train/validation/test splits...")
        
        all_paths = []
        all_labels = []
        
        for class_name, image_paths in dataset_info.items():
            all_paths.extend(image_paths)
            all_labels.extend([class_name] * len(image_paths))
        
        # First split: separate test set
        train_val_paths, test_paths, train_val_labels, test_labels = train_test_split(
            all_paths, all_labels,
            test_size=self.config.test_ratio,
            stratify=all_labels if self.config.stratify_by_class else None,
            random_state=self.config.random_seed
        )
        
        # Second split: separate train and validation
        val_ratio_adjusted = self.config.val_ratio / (self.config.train_ratio + self.config.val_ratio)
        train_paths, val_paths, train_labels, val_labels = train_test_split(
            train_val_paths, train_val_labels,
            test_size=val_ratio_adjusted,
            stratify=train_val_labels if self.config.stratify_by_class else None,
            random_state=self.config.random_seed
        )
        
        splits = {
            'train': {'paths': train_paths, 'labels': train_labels},
            'val': {'paths': val_paths, 'labels': val_labels},
            'test': {'paths': test_paths, 'labels': test_labels}
        }
        
        # Log split statistics
        for split_name, split_data in splits.items():
            class_counts = pd.Series(split_data['labels']).value_counts()
            logger.info(f"{split_name.upper()} set: {len(split_data['paths'])} images")
            for class_name, count in class_counts.items():
                logger.info(f"  {class_name}: {count} images")
        
        return splits
    
    def create_cross_validation_folds(self, train_paths: List[str], train_labels: List[str]) -> List[Dict[str, List]]:
        """Create cross-validation folds"""
        logger.info(f"Creating {self.config.cross_validation_folds}-fold cross-validation splits...")
        
        skf = StratifiedKFold(
            n_splits=self.config.cross_validation_folds,
            shuffle=True,
            random_state=self.config.random_seed
        )
        
        cv_folds = []
        for fold_idx, (train_idx, val_idx) in enumerate(skf.split(train_paths, train_labels)):
            fold_train_paths = [train_paths[i] for i in train_idx]
            fold_val_paths = [train_paths[i] for i in val_idx]
            fold_train_labels = [train_labels[i] for i in train_idx]
            fold_val_labels = [train_labels[i] for i in val_idx]
            
            cv_folds.append({
                'fold': fold_idx + 1,
                'train': {'paths': fold_train_paths, 'labels': fold_train_labels},
                'val': {'paths': fold_val_paths, 'labels': fold_val_labels}
            })
            
            logger.info(f"Fold {fold_idx + 1}: {len(fold_train_paths)} train, {len(fold_val_paths)} val")
        
        return cv_folds

class DataPreparationPipeline:
    """Main data preparation pipeline manager"""
    
    def __init__(self, config: DataPreparationConfig):
        self.config = config
        self.augmentator = ImageAugmentator(config)
        self.validator = DatasetValidator(config)
        self.splitter = DatasetSplitter(config)
        
        # Create output directories
        self.setup_output_directories()
    
    def setup_output_directories(self):
        """Create necessary output directories"""
        output_path = Path(self.config.output_path)
        output_path.mkdir(parents=True, exist_ok=True)
        
        for subdir in ['train', 'val', 'test', 'metadata', 'cross_validation']:
            (output_path / subdir).mkdir(exist_ok=True)
    
    def discover_dataset(self) -> Dict[str, List[str]]:
        """Discover and catalog all available images"""
        logger.info("Discovering dataset...")
        
        dataset_info = {}
        
        # Scan real data
        real_data_path = Path(self.config.raw_data_path)
        if real_data_path.exists():
            for class_dir in real_data_path.iterdir():
                if class_dir.is_dir():
                    class_name = class_dir.name
                    image_paths = []
                    
                    for ext in self.config.supported_formats:
                        image_paths.extend(list(class_dir.glob(f'*{ext}')))
                        image_paths.extend(list(class_dir.glob(f'*{ext.upper()}')))
                    
                    if image_paths:
                        dataset_info[class_name] = [str(p) for p in image_paths]
        
        # Scan synthetic data
        synthetic_data_path = Path(self.config.synthetic_data_path)
        if synthetic_data_path.exists():
            for class_dir in synthetic_data_path.iterdir():
                if class_dir.is_dir():
                    class_name = f"synthetic_{class_dir.name}"
                    image_paths = []
                    
                    for ext in self.config.supported_formats:
                        image_paths.extend(list(class_dir.glob(f'*{ext}')))
                        image_paths.extend(list(class_dir.glob(f'*{ext.upper()}')))
                    
                    if image_paths:
                        if class_name in dataset_info:
                            dataset_info[class_name].extend([str(p) for p in image_paths])
                        else:
                            dataset_info[class_name] = [str(p) for p in image_paths]
        
        # Log discovery results
        total_images = sum(len(paths) for paths in dataset_info.values())
        logger.info(f"Discovered {total_images} images across {len(dataset_info)} classes:")
        for class_name, paths in dataset_info.items():
            logger.info(f"  {class_name}: {len(paths)} images")
        
        return dataset_info
    
    def prepare_dataset(self) -> Dict[str, Any]:
        """Execute complete data preparation pipeline"""
        logger.info("Starting data preparation pipeline...")
        
        # Step 1: Discover dataset
        dataset_info = self.discover_dataset()
        
        # Step 2: Validate dataset
        validation_results = self.validator.validate_dataset(dataset_info)
        
        # Step 3: Filter valid images
        valid_dataset_info = {}
        for class_name, image_paths in dataset_info.items():
            valid_paths = []
            for result in validation_results['detailed_results']:
                if result['valid'] and any(path in result['path'] for path in image_paths):
                    valid_paths.append(result['path'])
            if valid_paths:
                valid_dataset_info[class_name] = valid_paths
        
        # Step 4: Create train/val/test splits
        splits = self.splitter.create_train_val_test_split(valid_dataset_info)
        
        # Step 5: Create cross-validation folds
        cv_folds = self.splitter.create_cross_validation_folds(
            splits['train']['paths'], 
            splits['train']['labels']
        )
        
        # Step 6: Save metadata
        metadata = {
            'config': self.config.__dict__,
            'dataset_info': valid_dataset_info,
            'validation_results': validation_results,
            'splits': splits,
            'cross_validation_folds': cv_folds,
            'preparation_timestamp': datetime.now().isoformat(),
            'total_images': sum(len(paths) for paths in valid_dataset_info.values()),
            'num_classes': len(valid_dataset_info)
        }
        
        metadata_path = Path(self.config.output_path) / 'metadata' / 'dataset_metadata.json'
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2, default=str)
        
        logger.info(f"Data preparation complete. Metadata saved to {metadata_path}")
        
        return metadata
    
    def create_balanced_dataset(self, dataset_info: Dict[str, List[str]]) -> Dict[str, List[str]]:
        """Create balanced dataset through augmentation or subsampling"""
        if not self.config.balance_classes:
            return dataset_info
        
        logger.info("Creating balanced dataset...")
        
        # Find target size (max class size)
        class_sizes = {class_name: len(paths) for class_name, paths in dataset_info.items()}
        target_size = max(class_sizes.values())
        
        balanced_dataset = {}
        
        for class_name, image_paths in dataset_info.items():
            current_size = len(image_paths)
            
            if current_size < target_size:
                # Augment smaller classes
                needed_samples = target_size - current_size
                augmented_paths = image_paths.copy()
                
                # Repeat existing images with augmentation
                for _ in range(needed_samples):
                    random_path = random.choice(image_paths)
                    augmented_paths.append(random_path)  # Will be augmented during training
                
                balanced_dataset[class_name] = augmented_paths
                logger.info(f"Augmented {class_name}: {current_size} -> {len(augmented_paths)} images")
            else:
                # Keep original size for larger classes
                balanced_dataset[class_name] = image_paths
                logger.info(f"Kept {class_name}: {current_size} images")
        
        return balanced_dataset

def main():
    """Main function for standalone execution"""
    # Initialize configuration
    config = DataPreparationConfig()
    
    # Create pipeline
    pipeline = DataPreparationPipeline(config)
    
    # Execute preparation
    metadata = pipeline.prepare_dataset()
    
    # Print summary
    print("\n" + "="*50)
    print("DATA PREPARATION SUMMARY")
    print("="*50)
    print(f"Total images processed: {metadata['total_images']}")
    print(f"Number of classes: {metadata['num_classes']}")
    print(f"Train images: {len(metadata['splits']['train']['paths'])}")
    print(f"Validation images: {len(metadata['splits']['val']['paths'])}")
    print(f"Test images: {len(metadata['splits']['test']['paths'])}")
    print(f"Cross-validation folds: {len(metadata['cross_validation_folds'])}")
    print(f"Output directory: {config.output_path}")
    print("="*50)

if __name__ == "__main__":
    main()