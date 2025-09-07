#!/usr/bin/env python3
"""
Dataset Management System for Document Parser

This module provides comprehensive dataset management capabilities including
organization, validation, format conversion, and quality control, following
the organizational patterns established by the autocorrect model.

Author: AI Assistant
Date: 2024
"""

import os
import sys
import json
import shutil
import hashlib
from pathlib import Path
from typing import Dict, List, Tuple, Any, Optional, Union, Set
from datetime import datetime
import numpy as np
import cv2
from PIL import Image
import pandas as pd
from sklearn.model_selection import train_test_split
import logging
from dataclasses import dataclass, asdict
from collections import defaultdict, Counter
import yaml

# Add src directory to path for imports
sys.path.append(str(Path(__file__).parent.parent.parent / 'src'))

@dataclass
class DatasetConfig:
    """Configuration for dataset management"""
    name: str = "document_parser_dataset"
    version: str = "1.0.0"
    description: str = "Document parsing dataset with MyKad and SPK samples"
    supported_formats: List[str] = None
    quality_thresholds: Dict[str, float] = None
    validation_rules: Dict[str, Any] = None
    export_formats: List[str] = None
    
    def __post_init__(self):
        if self.supported_formats is None:
            self.supported_formats = ['.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif']
        
        if self.quality_thresholds is None:
            self.quality_thresholds = {
                'min_resolution': 300,  # minimum DPI
                'min_width': 200,       # minimum width in pixels
                'min_height': 200,      # minimum height in pixels
                'max_file_size': 50,    # maximum file size in MB
                'min_contrast': 0.1,    # minimum contrast ratio
                'max_blur': 100         # maximum blur threshold
            }
        
        if self.validation_rules is None:
            self.validation_rules = {
                'require_annotations': True,
                'min_fields_per_document': 3,
                'allowed_document_types': ['mykad', 'spk'],
                'required_field_types': ['name', 'ic_number', 'address']
            }
        
        if self.export_formats is None:
            self.export_formats = ['coco', 'yolo', 'pascal_voc', 'tensorflow']
    
    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'DatasetConfig':
        return cls(**data)

@dataclass
class DatasetStats:
    """Dataset statistics and metrics"""
    total_images: int = 0
    total_annotations: int = 0
    document_types: Dict[str, int] = None
    field_types: Dict[str, int] = None
    image_resolutions: Dict[str, int] = None
    file_sizes: Dict[str, float] = None
    quality_scores: Dict[str, float] = None
    validation_errors: List[str] = None
    
    def __post_init__(self):
        if self.document_types is None:
            self.document_types = {}
        if self.field_types is None:
            self.field_types = {}
        if self.image_resolutions is None:
            self.image_resolutions = {}
        if self.file_sizes is None:
            self.file_sizes = {}
        if self.quality_scores is None:
            self.quality_scores = {}
        if self.validation_errors is None:
            self.validation_errors = []

class DatasetManager:
    """Comprehensive dataset management system"""
    
    def __init__(self, config: DatasetConfig = None, 
                 base_path: str = "./managed_datasets"):
        self.config = config or DatasetConfig()
        self.base_path = Path(base_path)
        self.base_path.mkdir(parents=True, exist_ok=True)
        
        # Setup logging
        self.logger = self._setup_logging()
        
        # Dataset registry
        self.registry_path = self.base_path / 'registry.json'
        self.registry = self._load_registry()
        
        # Current dataset path
        self.dataset_path = self.base_path / self.config.name / self.config.version
        self.dataset_path.mkdir(parents=True, exist_ok=True)
        
        # Initialize dataset structure
        self._initialize_dataset_structure()
    
    def _setup_logging(self) -> logging.Logger:
        """Setup logging configuration"""
        logger = logging.getLogger('DatasetManager')
        logger.setLevel(logging.INFO)
        
        # Create file handler
        log_file = self.base_path / 'dataset_management.log'
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
    
    def _load_registry(self) -> Dict[str, Any]:
        """Load dataset registry"""
        if self.registry_path.exists():
            with open(self.registry_path, 'r') as f:
                return json.load(f)
        return {'datasets': {}, 'created_at': datetime.now().isoformat()}
    
    def _save_registry(self):
        """Save dataset registry"""
        self.registry['updated_at'] = datetime.now().isoformat()
        with open(self.registry_path, 'w') as f:
            json.dump(self.registry, f, indent=2)
    
    def _initialize_dataset_structure(self):
        """Initialize dataset directory structure"""
        directories = [
            'raw/images',
            'raw/annotations',
            'processed/images',
            'processed/annotations',
            'splits/train',
            'splits/val',
            'splits/test',
            'exports',
            'metadata',
            'quality_reports'
        ]
        
        for directory in directories:
            (self.dataset_path / directory).mkdir(parents=True, exist_ok=True)
    
    def calculate_image_hash(self, image_path: str) -> str:
        """Calculate MD5 hash of image file"""
        hash_md5 = hashlib.md5()
        with open(image_path, "rb") as f:
            for chunk in iter(lambda: f.read(4096), b""):
                hash_md5.update(chunk)
        return hash_md5.hexdigest()
    
    def assess_image_quality(self, image_path: str) -> Dict[str, Any]:
        """Assess image quality metrics"""
        try:
            # Load image
            image = cv2.imread(image_path)
            if image is None:
                return {'error': 'Cannot load image'}
            
            height, width = image.shape[:2]
            file_size = os.path.getsize(image_path) / (1024 * 1024)  # MB
            
            # Convert to grayscale for analysis
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            
            # Calculate contrast (standard deviation)
            contrast = np.std(gray) / 255.0
            
            # Calculate blur (Laplacian variance)
            blur_score = cv2.Laplacian(gray, cv2.CV_64F).var()
            
            # Calculate brightness
            brightness = np.mean(gray) / 255.0
            
            # Estimate resolution (assuming 96 DPI as baseline)
            estimated_dpi = max(width, height) / 8.5 * 96  # Assuming letter size
            
            quality_metrics = {
                'width': width,
                'height': height,
                'file_size_mb': file_size,
                'contrast': contrast,
                'blur_score': blur_score,
                'brightness': brightness,
                'estimated_dpi': estimated_dpi,
                'aspect_ratio': width / height
            }
            
            # Quality assessment
            quality_issues = []
            
            if width < self.config.quality_thresholds['min_width']:
                quality_issues.append(f"Width too small: {width}px")
            
            if height < self.config.quality_thresholds['min_height']:
                quality_issues.append(f"Height too small: {height}px")
            
            if file_size > self.config.quality_thresholds['max_file_size']:
                quality_issues.append(f"File too large: {file_size:.1f}MB")
            
            if contrast < self.config.quality_thresholds['min_contrast']:
                quality_issues.append(f"Low contrast: {contrast:.3f}")
            
            if blur_score < self.config.quality_thresholds['max_blur']:
                quality_issues.append(f"Image too blurry: {blur_score:.1f}")
            
            if estimated_dpi < self.config.quality_thresholds['min_resolution']:
                quality_issues.append(f"Low resolution: {estimated_dpi:.0f} DPI")
            
            quality_metrics['issues'] = quality_issues
            quality_metrics['quality_score'] = max(0, 1 - len(quality_issues) * 0.2)
            
            return quality_metrics
            
        except Exception as e:
            return {'error': str(e)}
    
    def validate_annotation(self, annotation: Dict[str, Any]) -> List[str]:
        """Validate annotation against rules"""
        errors = []
        
        # Check required fields
        if 'document_type' not in annotation:
            errors.append("Missing document_type")
        elif annotation['document_type'] not in self.config.validation_rules['allowed_document_types']:
            errors.append(f"Invalid document_type: {annotation['document_type']}")
        
        if 'fields' not in annotation:
            errors.append("Missing fields")
            return errors
        
        fields = annotation['fields']
        
        # Check minimum number of fields
        if len(fields) < self.config.validation_rules['min_fields_per_document']:
            errors.append(f"Too few fields: {len(fields)}")
        
        # Check required field types
        field_types = {field.get('field_type') for field in fields}
        required_types = set(self.config.validation_rules['required_field_types'])
        missing_types = required_types - field_types
        
        if missing_types:
            errors.append(f"Missing required field types: {missing_types}")
        
        # Validate individual fields
        for i, field in enumerate(fields):
            field_errors = []
            
            if 'field_type' not in field:
                field_errors.append("Missing field_type")
            
            if 'bbox' not in field:
                field_errors.append("Missing bbox")
            else:
                bbox = field['bbox']
                required_bbox_keys = ['x', 'y', 'width', 'height']
                missing_bbox_keys = [key for key in required_bbox_keys if key not in bbox]
                if missing_bbox_keys:
                    field_errors.append(f"Missing bbox keys: {missing_bbox_keys}")
            
            if 'text' not in field or not field['text'].strip():
                field_errors.append("Missing or empty text")
            
            if field_errors:
                errors.append(f"Field {i}: {'; '.join(field_errors)}")
        
        return errors
    
    def import_dataset(self, source_path: str, annotation_file: str = None,
                      validate_quality: bool = True) -> Dict[str, Any]:
        """Import dataset from source directory"""
        source_path = Path(source_path)
        
        if not source_path.exists():
            raise FileNotFoundError(f"Source path not found: {source_path}")
        
        self.logger.info(f"Importing dataset from: {source_path}")
        
        # Load annotations if provided
        annotations_dict = {}
        if annotation_file and Path(annotation_file).exists():
            with open(annotation_file, 'r') as f:
                annotations_data = json.load(f)
            
            for annotation in annotations_data.get('annotations', []):
                image_filename = annotation.get('image_filename', '')
                if image_filename:
                    annotations_dict[image_filename] = annotation
        
        # Find all image files
        image_files = []
        for ext in self.config.supported_formats:
            image_files.extend(source_path.rglob(f"*{ext}"))
            image_files.extend(source_path.rglob(f"*{ext.upper()}"))
        
        self.logger.info(f"Found {len(image_files)} images to import")
        
        # Import statistics
        import_stats = DatasetStats()
        imported_files = []
        duplicate_hashes = set()
        
        for i, image_path in enumerate(image_files):
            try:
                # Calculate hash to detect duplicates
                image_hash = self.calculate_image_hash(str(image_path))
                
                if image_hash in duplicate_hashes:
                    self.logger.warning(f"Duplicate image detected: {image_path.name}")
                    continue
                
                duplicate_hashes.add(image_hash)
                
                # Assess image quality
                quality_metrics = {}
                if validate_quality:
                    quality_metrics = self.assess_image_quality(str(image_path))
                    
                    if 'error' in quality_metrics:
                        import_stats.validation_errors.append(
                            f"{image_path.name}: {quality_metrics['error']}"
                        )
                        continue
                
                # Copy image to raw directory
                target_filename = f"{image_hash}_{image_path.name}"
                target_path = self.dataset_path / 'raw' / 'images' / target_filename
                shutil.copy2(image_path, target_path)
                
                # Process annotation if available
                annotation = annotations_dict.get(image_path.name)
                annotation_errors = []
                
                if annotation:
                    if self.config.validation_rules['require_annotations']:
                        annotation_errors = self.validate_annotation(annotation)
                    
                    if not annotation_errors:
                        # Save annotation
                        annotation_filename = f"{image_hash}_{image_path.stem}.json"
                        annotation_path = self.dataset_path / 'raw' / 'annotations' / annotation_filename
                        
                        annotation_data = {
                            'image_filename': target_filename,
                            'image_hash': image_hash,
                            'original_path': str(image_path),
                            'import_date': datetime.now().isoformat(),
                            **annotation
                        }
                        
                        with open(annotation_path, 'w') as f:
                            json.dump(annotation_data, f, indent=2)
                        
                        import_stats.total_annotations += 1
                        
                        # Update statistics
                        doc_type = annotation.get('document_type', 'unknown')
                        import_stats.document_types[doc_type] = import_stats.document_types.get(doc_type, 0) + 1
                        
                        for field in annotation.get('fields', []):
                            field_type = field.get('field_type', 'unknown')
                            import_stats.field_types[field_type] = import_stats.field_types.get(field_type, 0) + 1
                    else:
                        import_stats.validation_errors.extend(
                            [f"{image_path.name}: {error}" for error in annotation_errors]
                        )
                
                elif self.config.validation_rules['require_annotations']:
                    import_stats.validation_errors.append(
                        f"{image_path.name}: Missing annotation"
                    )
                    continue
                
                # Record imported file
                file_info = {
                    'original_path': str(image_path),
                    'imported_path': str(target_path),
                    'filename': target_filename,
                    'hash': image_hash,
                    'import_date': datetime.now().isoformat(),
                    'has_annotation': annotation is not None,
                    'quality_metrics': quality_metrics
                }
                
                imported_files.append(file_info)
                import_stats.total_images += 1
                
                # Update resolution statistics
                if quality_metrics and 'width' in quality_metrics:
                    resolution = f"{quality_metrics['width']}x{quality_metrics['height']}"
                    import_stats.image_resolutions[resolution] = import_stats.image_resolutions.get(resolution, 0) + 1
                
                if (i + 1) % 100 == 0:
                    self.logger.info(f"Imported {i + 1}/{len(image_files)} images")
                    
            except Exception as e:
                self.logger.error(f"Failed to import {image_path}: {str(e)}")
                import_stats.validation_errors.append(f"{image_path.name}: {str(e)}")
        
        # Save import metadata
        import_metadata = {
            'source_path': str(source_path),
            'import_date': datetime.now().isoformat(),
            'config': self.config.to_dict(),
            'statistics': asdict(import_stats),
            'imported_files': imported_files
        }
        
        metadata_path = self.dataset_path / 'metadata' / 'import_metadata.json'
        with open(metadata_path, 'w') as f:
            json.dump(import_metadata, f, indent=2, default=str)
        
        # Update registry
        dataset_key = f"{self.config.name}_{self.config.version}"
        self.registry['datasets'][dataset_key] = {
            'name': self.config.name,
            'version': self.config.version,
            'path': str(self.dataset_path),
            'created_at': datetime.now().isoformat(),
            'total_images': import_stats.total_images,
            'total_annotations': import_stats.total_annotations
        }
        self._save_registry()
        
        self.logger.info(f"Import completed: {import_stats.total_images} images, {import_stats.total_annotations} annotations")
        
        return import_metadata
    
    def export_dataset(self, export_format: str, output_path: str = None,
                      split: str = None) -> str:
        """Export dataset in specified format"""
        if export_format not in self.config.export_formats:
            raise ValueError(f"Unsupported export format: {export_format}")
        
        if output_path is None:
            output_path = self.dataset_path / 'exports' / f"{self.config.name}_{export_format}"
        
        output_path = Path(output_path)
        output_path.mkdir(parents=True, exist_ok=True)
        
        self.logger.info(f"Exporting dataset in {export_format} format to: {output_path}")
        
        # Determine source directory
        if split:
            source_dir = self.dataset_path / 'splits' / split
            if not source_dir.exists():
                raise FileNotFoundError(f"Split not found: {split}")
        else:
            source_dir = self.dataset_path / 'processed'
        
        # Load images and annotations
        images_dir = source_dir / 'images'
        annotations_dir = source_dir / 'annotations'
        
        if export_format == 'coco':
            return self._export_coco_format(images_dir, annotations_dir, output_path)
        elif export_format == 'yolo':
            return self._export_yolo_format(images_dir, annotations_dir, output_path)
        elif export_format == 'pascal_voc':
            return self._export_pascal_voc_format(images_dir, annotations_dir, output_path)
        elif export_format == 'tensorflow':
            return self._export_tensorflow_format(images_dir, annotations_dir, output_path)
        else:
            raise NotImplementedError(f"Export format {export_format} not implemented")
    
    def _export_coco_format(self, images_dir: Path, annotations_dir: Path, output_path: Path) -> str:
        """Export dataset in COCO format"""
        # Implementation for COCO format export
        coco_data = {
            'info': {
                'description': self.config.description,
                'version': self.config.version,
                'year': datetime.now().year,
                'contributor': 'Document Parser Dataset Manager',
                'date_created': datetime.now().isoformat()
            },
            'licenses': [],
            'images': [],
            'annotations': [],
            'categories': []
        }
        
        # Add categories (field types)
        category_map = {}
        category_id = 1
        
        for field_type in self.config.validation_rules['required_field_types']:
            coco_data['categories'].append({
                'id': category_id,
                'name': field_type,
                'supercategory': 'document_field'
            })
            category_map[field_type] = category_id
            category_id += 1
        
        # Process images and annotations
        image_id = 1
        annotation_id = 1
        
        for annotation_file in annotations_dir.glob('*.json'):
            with open(annotation_file, 'r') as f:
                annotation_data = json.load(f)
            
            image_filename = annotation_data.get('image_filename')
            image_path = images_dir / image_filename
            
            if not image_path.exists():
                continue
            
            # Load image to get dimensions
            image = cv2.imread(str(image_path))
            height, width = image.shape[:2]
            
            # Add image info
            coco_data['images'].append({
                'id': image_id,
                'width': width,
                'height': height,
                'file_name': image_filename,
                'license': 0,
                'flickr_url': '',
                'coco_url': '',
                'date_captured': annotation_data.get('import_date', '')
            })
            
            # Add annotations
            for field in annotation_data.get('fields', []):
                field_type = field.get('field_type')
                if field_type in category_map:
                    bbox = field.get('bbox', {})
                    
                    coco_data['annotations'].append({
                        'id': annotation_id,
                        'image_id': image_id,
                        'category_id': category_map[field_type],
                        'segmentation': [],
                        'area': bbox.get('width', 0) * bbox.get('height', 0),
                        'bbox': [bbox.get('x', 0), bbox.get('y', 0), 
                                bbox.get('width', 0), bbox.get('height', 0)],
                        'iscrowd': 0,
                        'attributes': {
                            'text': field.get('text', ''),
                            'confidence': field.get('confidence', 1.0)
                        }
                    })
                    annotation_id += 1
            
            image_id += 1
        
        # Save COCO format file
        coco_file = output_path / 'annotations.json'
        with open(coco_file, 'w') as f:
            json.dump(coco_data, f, indent=2)
        
        # Copy images
        images_output = output_path / 'images'
        images_output.mkdir(exist_ok=True)
        
        for image_file in images_dir.glob('*'):
            if image_file.suffix.lower() in self.config.supported_formats:
                shutil.copy2(image_file, images_output / image_file.name)
        
        self.logger.info(f"COCO format export completed: {coco_file}")
        return str(coco_file)
    
    def _export_yolo_format(self, images_dir: Path, annotations_dir: Path, output_path: Path) -> str:
        """Export dataset in YOLO format"""
        # Create YOLO directory structure
        (output_path / 'images').mkdir(exist_ok=True)
        (output_path / 'labels').mkdir(exist_ok=True)
        
        # Create class mapping
        class_names = self.config.validation_rules['required_field_types']
        class_map = {name: i for i, name in enumerate(class_names)}
        
        # Save class names
        with open(output_path / 'classes.txt', 'w') as f:
            for class_name in class_names:
                f.write(f"{class_name}\n")
        
        # Process annotations
        for annotation_file in annotations_dir.glob('*.json'):
            with open(annotation_file, 'r') as f:
                annotation_data = json.load(f)
            
            image_filename = annotation_data.get('image_filename')
            image_path = images_dir / image_filename
            
            if not image_path.exists():
                continue
            
            # Load image to get dimensions
            image = cv2.imread(str(image_path))
            height, width = image.shape[:2]
            
            # Copy image
            shutil.copy2(image_path, output_path / 'images' / image_filename)
            
            # Create YOLO annotation
            label_filename = Path(image_filename).stem + '.txt'
            label_path = output_path / 'labels' / label_filename
            
            with open(label_path, 'w') as f:
                for field in annotation_data.get('fields', []):
                    field_type = field.get('field_type')
                    if field_type in class_map:
                        bbox = field.get('bbox', {})
                        
                        # Convert to YOLO format (normalized center coordinates)
                        x_center = (bbox.get('x', 0) + bbox.get('width', 0) / 2) / width
                        y_center = (bbox.get('y', 0) + bbox.get('height', 0) / 2) / height
                        norm_width = bbox.get('width', 0) / width
                        norm_height = bbox.get('height', 0) / height
                        
                        f.write(f"{class_map[field_type]} {x_center} {y_center} {norm_width} {norm_height}\n")
        
        self.logger.info(f"YOLO format export completed: {output_path}")
        return str(output_path)
    
    def _export_pascal_voc_format(self, images_dir: Path, annotations_dir: Path, output_path: Path) -> str:
        """Export dataset in Pascal VOC format"""
        # Implementation placeholder for Pascal VOC format
        self.logger.info(f"Pascal VOC format export not yet implemented")
        return str(output_path)
    
    def _export_tensorflow_format(self, images_dir: Path, annotations_dir: Path, output_path: Path) -> str:
        """Export dataset in TensorFlow format"""
        # Implementation placeholder for TensorFlow format
        self.logger.info(f"TensorFlow format export not yet implemented")
        return str(output_path)
    
    def generate_quality_report(self) -> Dict[str, Any]:
        """Generate comprehensive quality report"""
        report = {
            'dataset_info': {
                'name': self.config.name,
                'version': self.config.version,
                'path': str(self.dataset_path),
                'report_date': datetime.now().isoformat()
            },
            'statistics': {},
            'quality_metrics': {},
            'recommendations': []
        }
        
        # Analyze raw images
        raw_images_dir = self.dataset_path / 'raw' / 'images'
        if raw_images_dir.exists():
            image_files = list(raw_images_dir.glob('*'))
            
            quality_scores = []
            resolution_counts = Counter()
            file_size_stats = []
            
            for image_file in image_files:
                if image_file.suffix.lower() in self.config.supported_formats:
                    quality_metrics = self.assess_image_quality(str(image_file))
                    
                    if 'quality_score' in quality_metrics:
                        quality_scores.append(quality_metrics['quality_score'])
                    
                    if 'width' in quality_metrics and 'height' in quality_metrics:
                        resolution = f"{quality_metrics['width']}x{quality_metrics['height']}"
                        resolution_counts[resolution] += 1
                    
                    if 'file_size_mb' in quality_metrics:
                        file_size_stats.append(quality_metrics['file_size_mb'])
            
            report['statistics'] = {
                'total_images': len(image_files),
                'avg_quality_score': np.mean(quality_scores) if quality_scores else 0,
                'min_quality_score': np.min(quality_scores) if quality_scores else 0,
                'max_quality_score': np.max(quality_scores) if quality_scores else 0,
                'common_resolutions': dict(resolution_counts.most_common(5)),
                'avg_file_size_mb': np.mean(file_size_stats) if file_size_stats else 0
            }
            
            # Generate recommendations
            if quality_scores:
                low_quality_count = sum(1 for score in quality_scores if score < 0.7)
                if low_quality_count > len(quality_scores) * 0.1:
                    report['recommendations'].append(
                        f"Consider reviewing {low_quality_count} low-quality images"
                    )
            
            if file_size_stats:
                large_files = sum(1 for size in file_size_stats if size > 10)
                if large_files > 0:
                    report['recommendations'].append(
                        f"Consider compressing {large_files} large image files"
                    )
        
        # Save report
        report_path = self.dataset_path / 'quality_reports' / f"quality_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        with open(report_path, 'w') as f:
            json.dump(report, f, indent=2, default=str)
        
        self.logger.info(f"Quality report generated: {report_path}")
        
        return report
    
    def get_dataset_info(self) -> Dict[str, Any]:
        """Get comprehensive dataset information"""
        info = {
            'config': self.config.to_dict(),
            'path': str(self.dataset_path),
            'structure': {},
            'statistics': {},
            'registry': self.registry
        }
        
        # Analyze directory structure
        for subdir in ['raw', 'processed', 'splits', 'exports']:
            subdir_path = self.dataset_path / subdir
            if subdir_path.exists():
                info['structure'][subdir] = {
                    'exists': True,
                    'subdirectories': [d.name for d in subdir_path.iterdir() if d.is_dir()],
                    'file_count': len([f for f in subdir_path.rglob('*') if f.is_file()])
                }
            else:
                info['structure'][subdir] = {'exists': False}
        
        return info

def main():
    """Main function for standalone execution"""
    print("📊 Dataset Management System")
    print("=" * 40)
    
    # Initialize dataset manager
    config = DatasetConfig(
        name="document_parser_dataset",
        version="1.0.0",
        description="Document parsing dataset with MyKad and SPK samples"
    )
    
    manager = DatasetManager(config, "./managed_datasets")
    
    # Display configuration
    print(f"\n⚙️ Configuration:")
    print(f"   Dataset: {config.name} v{config.version}")
    print(f"   Supported formats: {config.supported_formats}")
    print(f"   Export formats: {config.export_formats}")
    
    # Display dataset info
    info = manager.get_dataset_info()
    print(f"\n📂 Dataset path: {info['path']}")
    
    # Example usage
    print("\n📋 Usage Examples:")
    print("1. metadata = manager.import_dataset('/path/to/source', '/path/to/annotations.json')")
    print("2. export_path = manager.export_dataset('coco', '/path/to/output')")
    print("3. report = manager.generate_quality_report()")
    print("4. info = manager.get_dataset_info()")
    
    return 0

if __name__ == "__main__":
    exit(main())