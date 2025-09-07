#!/usr/bin/env python3
"""
Image Preprocessing Pipeline for Document Parser

This module provides comprehensive image preprocessing capabilities including
enhancement, augmentation, and dataset preparation, following the organizational
patterns established by the autocorrect model.

Author: AI Assistant
Date: 2024
"""

import os
import sys
import json
import random
from pathlib import Path
from typing import Dict, List, Tuple, Any, Optional, Union
from datetime import datetime
import numpy as np
import cv2
from PIL import Image, ImageEnhance, ImageFilter
import albumentations as A
from sklearn.model_selection import train_test_split
import logging
from dataclasses import dataclass, asdict

# Add src directory to path for imports
sys.path.append(str(Path(__file__).parent.parent.parent / 'src'))

@dataclass
class PreprocessingConfig:
    """Configuration for preprocessing pipeline"""
    target_size: Tuple[int, int] = (1024, 768)
    normalize: bool = True
    enhance_contrast: bool = True
    denoise: bool = True
    augmentation_probability: float = 0.8
    train_split: float = 0.7
    val_split: float = 0.15
    test_split: float = 0.15
    random_seed: int = 42
    
    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'PreprocessingConfig':
        return cls(**data)

class ImagePreprocessor:
    """Comprehensive image preprocessing pipeline"""
    
    def __init__(self, config: PreprocessingConfig = None, 
                 output_path: str = "./preprocessed_data"):
        self.config = config or PreprocessingConfig()
        self.output_path = Path(output_path)
        self.output_path.mkdir(parents=True, exist_ok=True)
        
        # Setup logging
        self.logger = self._setup_logging()
        
        # Initialize augmentation pipeline
        self.augmentation_pipeline = self._create_augmentation_pipeline()
        
        # Processing statistics
        self.stats = {
            'total_processed': 0,
            'enhanced_images': 0,
            'augmented_images': 0,
            'failed_processing': 0
        }
        
        # Set random seed for reproducibility
        random.seed(self.config.random_seed)
        np.random.seed(self.config.random_seed)
    
    def _setup_logging(self) -> logging.Logger:
        """Setup logging configuration"""
        logger = logging.getLogger('ImagePreprocessor')
        logger.setLevel(logging.INFO)
        
        # Create file handler
        log_file = self.output_path / 'preprocessing.log'
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
    
    def _create_augmentation_pipeline(self) -> A.Compose:
        """Create augmentation pipeline using Albumentations"""
        return A.Compose([
            # Geometric transformations
            A.OneOf([
                A.Rotate(limit=5, p=0.5),
                A.Affine(scale=(0.95, 1.05), translate_percent=0.02, p=0.5),
                A.Perspective(scale=(0.02, 0.05), p=0.3)
            ], p=0.7),
            
            # Optical distortions
            A.OneOf([
                A.OpticalDistortion(distort_limit=0.1, p=0.3),
                A.GridDistortion(num_steps=5, distort_limit=0.1, p=0.3),
                A.ElasticTransform(alpha=1, sigma=50, alpha_affine=50, p=0.2)
            ], p=0.4),
            
            # Lighting and color
            A.OneOf([
                A.RandomBrightnessContrast(brightness_limit=0.1, contrast_limit=0.1, p=0.5),
                A.CLAHE(clip_limit=2.0, tile_grid_size=(8, 8), p=0.3),
                A.RandomGamma(gamma_limit=(90, 110), p=0.3)
            ], p=0.6),
            
            # Noise and blur
            A.OneOf([
                A.GaussNoise(var_limit=(10, 50), p=0.3),
                A.ISONoise(color_shift=(0.01, 0.05), intensity=(0.1, 0.5), p=0.2),
                A.MultiplicativeNoise(multiplier=(0.9, 1.1), p=0.2)
            ], p=0.4),
            
            A.OneOf([
                A.MotionBlur(blur_limit=3, p=0.2),
                A.GaussianBlur(blur_limit=3, p=0.2),
                A.Defocus(radius=(1, 3), alias_blur=(0.1, 0.5), p=0.1)
            ], p=0.3),
            
            # Quality degradation
            A.OneOf([
                A.ImageCompression(quality_lower=75, quality_upper=95, p=0.3),
                A.Downscale(scale_min=0.8, scale_max=0.95, p=0.2)
            ], p=0.3)
        ], p=self.config.augmentation_probability)
    
    def enhance_image(self, image: np.ndarray) -> np.ndarray:
        """Apply image enhancement techniques"""
        enhanced = image.copy()
        
        # Convert to PIL for some operations
        pil_image = Image.fromarray(cv2.cvtColor(enhanced, cv2.COLOR_BGR2RGB))
        
        # Contrast enhancement
        if self.config.enhance_contrast:
            enhancer = ImageEnhance.Contrast(pil_image)
            pil_image = enhancer.enhance(1.2)
        
        # Sharpening
        pil_image = pil_image.filter(ImageFilter.UnsharpMask(radius=1, percent=150, threshold=3))
        
        # Convert back to OpenCV format
        enhanced = cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2BGR)
        
        # Denoising
        if self.config.denoise:
            enhanced = cv2.fastNlMeansDenoisingColored(enhanced, None, 10, 10, 7, 21)
        
        # Adaptive histogram equalization
        lab = cv2.cvtColor(enhanced, cv2.COLOR_BGR2LAB)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        lab[:, :, 0] = clahe.apply(lab[:, :, 0])
        enhanced = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)
        
        return enhanced
    
    def resize_image(self, image: np.ndarray, target_size: Tuple[int, int] = None) -> np.ndarray:
        """Resize image while maintaining aspect ratio"""
        if target_size is None:
            target_size = self.config.target_size
        
        height, width = image.shape[:2]
        target_width, target_height = target_size
        
        # Calculate scaling factor
        scale = min(target_width / width, target_height / height)
        
        # Calculate new dimensions
        new_width = int(width * scale)
        new_height = int(height * scale)
        
        # Resize image
        resized = cv2.resize(image, (new_width, new_height), interpolation=cv2.INTER_AREA)
        
        # Create canvas with target size
        canvas = np.zeros((target_height, target_width, 3), dtype=np.uint8)
        canvas.fill(255)  # White background
        
        # Center the resized image on canvas
        y_offset = (target_height - new_height) // 2
        x_offset = (target_width - new_width) // 2
        
        canvas[y_offset:y_offset + new_height, x_offset:x_offset + new_width] = resized
        
        return canvas
    
    def normalize_image(self, image: np.ndarray) -> np.ndarray:
        """Normalize image pixel values"""
        if self.config.normalize:
            # Convert to float and normalize to [0, 1]
            normalized = image.astype(np.float32) / 255.0
            return normalized
        return image
    
    def apply_augmentation(self, image: np.ndarray, annotations: Dict[str, Any] = None) -> Tuple[np.ndarray, Dict[str, Any]]:
        """Apply augmentation to image and adjust annotations accordingly"""
        # Convert BGR to RGB for Albumentations
        rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Prepare bounding boxes if annotations provided
        bboxes = []
        bbox_labels = []
        
        if annotations and 'fields' in annotations:
            for field in annotations['fields']:
                if 'bbox' in field:
                    bbox = field['bbox']
                    # Convert to Albumentations format (x_min, y_min, x_max, y_max, normalized)
                    x_min = bbox['x'] / image.shape[1]
                    y_min = bbox['y'] / image.shape[0]
                    x_max = (bbox['x'] + bbox['width']) / image.shape[1]
                    y_max = (bbox['y'] + bbox['height']) / image.shape[0]
                    
                    bboxes.append([x_min, y_min, x_max, y_max])
                    bbox_labels.append(field.get('field_type', 'unknown'))
        
        # Apply augmentation
        if bboxes:
            augmented = self.augmentation_pipeline(
                image=rgb_image,
                bboxes=bboxes,
                bbox_labels=bbox_labels
            )
            
            # Update annotations with new bounding boxes
            if annotations and 'fields' in annotations:
                for i, field in enumerate(annotations['fields']):
                    if i < len(augmented['bboxes']):
                        new_bbox = augmented['bboxes'][i]
                        # Convert back to pixel coordinates
                        field['bbox'] = {
                            'x': int(new_bbox[0] * augmented['image'].shape[1]),
                            'y': int(new_bbox[1] * augmented['image'].shape[0]),
                            'width': int((new_bbox[2] - new_bbox[0]) * augmented['image'].shape[1]),
                            'height': int((new_bbox[3] - new_bbox[1]) * augmented['image'].shape[0])
                        }
        else:
            augmented = self.augmentation_pipeline(image=rgb_image)
        
        # Convert back to BGR
        augmented_image = cv2.cvtColor(augmented['image'], cv2.COLOR_RGB2BGR)
        
        return augmented_image, annotations
    
    def process_single_image(self, image_path: str, annotations: Dict[str, Any] = None,
                           apply_augmentation: bool = True) -> Dict[str, Any]:
        """Process a single image through the complete pipeline"""
        try:
            # Load image
            image = cv2.imread(image_path)
            if image is None:
                raise ValueError(f"Cannot load image: {image_path}")
            
            original_shape = image.shape
            
            # Enhancement
            enhanced_image = self.enhance_image(image)
            
            # Resize
            resized_image = self.resize_image(enhanced_image)
            
            # Normalization
            normalized_image = self.normalize_image(resized_image)
            
            # Augmentation (if requested)
            augmented_image = normalized_image
            updated_annotations = annotations
            
            if apply_augmentation:
                # Convert back to uint8 for augmentation
                uint8_image = (normalized_image * 255).astype(np.uint8) if self.config.normalize else normalized_image
                augmented_image, updated_annotations = self.apply_augmentation(uint8_image, annotations)
                
                # Re-normalize if needed
                if self.config.normalize:
                    augmented_image = self.normalize_image(augmented_image)
            
            result = {
                'original_path': image_path,
                'original_shape': original_shape,
                'processed_shape': augmented_image.shape,
                'processed_image': augmented_image,
                'annotations': updated_annotations,
                'processing_steps': {
                    'enhanced': True,
                    'resized': True,
                    'normalized': self.config.normalize,
                    'augmented': apply_augmentation
                },
                'processing_date': datetime.now().isoformat()
            }
            
            self.stats['total_processed'] += 1
            if apply_augmentation:
                self.stats['augmented_images'] += 1
            self.stats['enhanced_images'] += 1
            
            return result
            
        except Exception as e:
            self.logger.error(f"Failed to process image {image_path}: {str(e)}")
            self.stats['failed_processing'] += 1
            return {
                'original_path': image_path,
                'error': str(e),
                'processing_date': datetime.now().isoformat()
            }
    
    def process_dataset(self, dataset_path: str, annotation_file: str = None) -> Dict[str, Any]:
        """Process entire dataset through preprocessing pipeline"""
        dataset_path = Path(dataset_path)
        
        if not dataset_path.exists():
            raise FileNotFoundError(f"Dataset path not found: {dataset_path}")
        
        # Load annotations if provided
        annotations_dict = {}
        if annotation_file and Path(annotation_file).exists():
            with open(annotation_file, 'r') as f:
                annotations_data = json.load(f)
                
            # Create mapping from image filename to annotations
            for annotation in annotations_data.get('annotations', []):
                image_filename = annotation.get('image_filename', '')
                if image_filename:
                    annotations_dict[image_filename] = annotation
        
        # Find all image files
        image_extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif']
        image_files = []
        
        for ext in image_extensions:
            image_files.extend(dataset_path.rglob(f"*{ext}"))
            image_files.extend(dataset_path.rglob(f"*{ext.upper()}"))
        
        self.logger.info(f"Found {len(image_files)} images to process")
        
        processed_data = {
            'images': [],
            'annotations': [],
            'metadata': {
                'total_images': len(image_files),
                'processing_config': self.config.to_dict(),
                'processing_date': datetime.now().isoformat()
            }
        }
        
        # Process each image
        for i, image_path in enumerate(image_files):
            image_filename = image_path.name
            annotations = annotations_dict.get(image_filename)
            
            # Process original image
            result = self.process_single_image(str(image_path), annotations, apply_augmentation=False)
            
            if 'error' not in result:
                # Save processed image
                processed_filename = f"processed_{image_filename}"
                processed_path = self.output_path / 'images' / processed_filename
                processed_path.parent.mkdir(parents=True, exist_ok=True)
                
                # Save image
                if self.config.normalize:
                    # Convert back to uint8 for saving
                    save_image = (result['processed_image'] * 255).astype(np.uint8)
                else:
                    save_image = result['processed_image']
                
                cv2.imwrite(str(processed_path), save_image)
                
                processed_data['images'].append({
                    'original_path': str(image_path),
                    'processed_path': str(processed_path),
                    'filename': processed_filename,
                    'original_shape': result['original_shape'],
                    'processed_shape': result['processed_shape']
                })
                
                if result['annotations']:
                    processed_data['annotations'].append(result['annotations'])
                
                # Generate augmented versions
                num_augmentations = 3  # Generate 3 augmented versions per image
                for aug_idx in range(num_augmentations):
                    aug_result = self.process_single_image(str(image_path), annotations, apply_augmentation=True)
                    
                    if 'error' not in aug_result:
                        aug_filename = f"aug_{aug_idx}_{image_filename}"
                        aug_path = self.output_path / 'images' / aug_filename
                        
                        # Save augmented image
                        if self.config.normalize:
                            save_image = (aug_result['processed_image'] * 255).astype(np.uint8)
                        else:
                            save_image = aug_result['processed_image']
                        
                        cv2.imwrite(str(aug_path), save_image)
                        
                        processed_data['images'].append({
                            'original_path': str(image_path),
                            'processed_path': str(aug_path),
                            'filename': aug_filename,
                            'original_shape': aug_result['original_shape'],
                            'processed_shape': aug_result['processed_shape'],
                            'augmented': True,
                            'augmentation_index': aug_idx
                        })
                        
                        if aug_result['annotations']:
                            processed_data['annotations'].append(aug_result['annotations'])
            
            if (i + 1) % 50 == 0:
                self.logger.info(f"Processed {i + 1}/{len(image_files)} images")
        
        # Save processing metadata
        metadata_path = self.output_path / 'processing_metadata.json'
        with open(metadata_path, 'w') as f:
            json.dump(processed_data, f, indent=2, default=str)
        
        self.logger.info(f"Dataset processing completed. Processed {len(processed_data['images'])} images")
        
        return processed_data
    
    def split_dataset(self, processed_data: Dict[str, Any]) -> Dict[str, Any]:
        """Split processed dataset into train/validation/test sets"""
        images = processed_data['images']
        annotations = processed_data['annotations']
        
        # Group by original image to ensure augmented versions stay together
        image_groups = {}
        for i, img_data in enumerate(images):
            original_path = img_data['original_path']
            if original_path not in image_groups:
                image_groups[original_path] = []
            image_groups[original_path].append(i)
        
        # Split original images
        original_images = list(image_groups.keys())
        
        # First split: train + val vs test
        train_val_images, test_images = train_test_split(
            original_images,
            test_size=self.config.test_split,
            random_state=self.config.random_seed
        )
        
        # Second split: train vs val
        val_size = self.config.val_split / (self.config.train_split + self.config.val_split)
        train_images, val_images = train_test_split(
            train_val_images,
            test_size=val_size,
            random_state=self.config.random_seed
        )
        
        # Create splits
        splits = {
            'train': {'images': [], 'annotations': []},
            'val': {'images': [], 'annotations': []},
            'test': {'images': [], 'annotations': []}
        }
        
        # Assign images to splits
        for split_name, split_images in [('train', train_images), ('val', val_images), ('test', test_images)]:
            for original_path in split_images:
                for img_idx in image_groups[original_path]:
                    splits[split_name]['images'].append(images[img_idx])
                    if img_idx < len(annotations):
                        splits[split_name]['annotations'].append(annotations[img_idx])
        
        # Save splits
        for split_name, split_data in splits.items():
            split_dir = self.output_path / 'splits' / split_name
            split_dir.mkdir(parents=True, exist_ok=True)
            
            # Save split metadata
            split_metadata = {
                'split': split_name,
                'num_images': len(split_data['images']),
                'num_annotations': len(split_data['annotations']),
                'split_date': datetime.now().isoformat(),
                'config': self.config.to_dict()
            }
            
            with open(split_dir / 'metadata.json', 'w') as f:
                json.dump(split_metadata, f, indent=2)
            
            # Save image list
            with open(split_dir / 'images.json', 'w') as f:
                json.dump(split_data['images'], f, indent=2, default=str)
            
            # Save annotations
            if split_data['annotations']:
                with open(split_dir / 'annotations.json', 'w') as f:
                    json.dump(split_data['annotations'], f, indent=2, default=str)
        
        split_summary = {
            'train': len(splits['train']['images']),
            'val': len(splits['val']['images']),
            'test': len(splits['test']['images']),
            'total': len(images)
        }
        
        self.logger.info(f"Dataset split completed: {split_summary}")
        
        return splits
    
    def get_processing_statistics(self) -> Dict[str, Any]:
        """Get comprehensive processing statistics"""
        return {
            'processing_stats': self.stats,
            'config': self.config.to_dict(),
            'output_path': str(self.output_path),
            'timestamp': datetime.now().isoformat()
        }

def main():
    """Main function for standalone execution"""
    print("🖼️ Image Preprocessing Pipeline")
    print("=" * 40)
    
    # Initialize preprocessor
    config = PreprocessingConfig(
        target_size=(1024, 768),
        normalize=True,
        enhance_contrast=True,
        denoise=True,
        augmentation_probability=0.8
    )
    
    preprocessor = ImagePreprocessor(config, "./preprocessed_data")
    
    # Display configuration
    print(f"\n⚙️ Configuration:")
    print(f"   Target size: {config.target_size}")
    print(f"   Normalize: {config.normalize}")
    print(f"   Enhance contrast: {config.enhance_contrast}")
    print(f"   Denoise: {config.denoise}")
    print(f"   Augmentation probability: {config.augmentation_probability}")
    
    # Example usage
    print("\n📋 Usage Examples:")
    print("1. result = preprocessor.process_single_image('/path/to/image.jpg')")
    print("2. dataset = preprocessor.process_dataset('/path/to/dataset')")
    print("3. splits = preprocessor.split_dataset(dataset)")
    print("4. stats = preprocessor.get_processing_statistics()")
    
    print(f"\n📂 Output directory: {preprocessor.output_path}")
    
    return 0

if __name__ == "__main__":
    exit(main())