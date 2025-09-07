#!/usr/bin/env python3
"""
Document Annotation System

This module provides comprehensive annotation tools for document parser training data,
following the organizational patterns established by the autocorrect model.

Author: AI Assistant
Date: 2024
"""

import os
import sys
import json
import uuid
from pathlib import Path
from typing import Dict, List, Tuple, Any, Optional, Union
from datetime import datetime
import cv2
import numpy as np
from dataclasses import dataclass, asdict
import logging

# Add src directory to path for imports
sys.path.append(str(Path(__file__).parent.parent.parent / 'src'))

@dataclass
class BoundingBox:
    """Bounding box coordinates"""
    x: int
    y: int
    width: int
    height: int
    
    def to_dict(self) -> Dict[str, int]:
        return asdict(self)
    
    @classmethod
    def from_dict(cls, data: Dict[str, int]) -> 'BoundingBox':
        return cls(**data)
    
    def area(self) -> int:
        return self.width * self.height
    
    def center(self) -> Tuple[int, int]:
        return (self.x + self.width // 2, self.y + self.height // 2)

@dataclass
class FieldAnnotation:
    """Annotation for a document field"""
    field_name: str
    field_type: str
    value: str
    bbox: BoundingBox
    confidence: float
    annotator_id: str
    annotation_date: str
    validation_status: str = 'pending'  # pending, validated, rejected
    notes: str = ''
    
    def to_dict(self) -> Dict[str, Any]:
        data = asdict(self)
        data['bbox'] = self.bbox.to_dict()
        return data
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'FieldAnnotation':
        bbox_data = data.pop('bbox')
        bbox = BoundingBox.from_dict(bbox_data)
        return cls(bbox=bbox, **data)

@dataclass
class DocumentAnnotation:
    """Complete annotation for a document"""
    annotation_id: str
    image_path: str
    document_type: str
    fields: List[FieldAnnotation]
    image_dimensions: Tuple[int, int]
    annotator_id: str
    annotation_date: str
    review_status: str = 'pending'  # pending, approved, rejected, needs_revision
    quality_score: float = 0.0
    metadata: Dict[str, Any] = None
    
    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}
    
    def to_dict(self) -> Dict[str, Any]:
        data = asdict(self)
        data['fields'] = [field.to_dict() for field in self.fields]
        return data
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'DocumentAnnotation':
        fields_data = data.pop('fields')
        fields = [FieldAnnotation.from_dict(field_data) for field_data in fields_data]
        return cls(fields=fields, **data)

class AnnotationSystem:
    """Comprehensive annotation system for document parser training"""
    
    def __init__(self, workspace_path: str = "./annotation_workspace"):
        self.workspace_path = Path(workspace_path)
        self.workspace_path.mkdir(parents=True, exist_ok=True)
        
        # Setup logging
        self.logger = self._setup_logging()
        
        # Annotation configuration
        self.field_types = self._load_field_types()
        self.document_types = ['mykad', 'spk', 'passport', 'license', 'unknown']
        
        # Quality control settings
        self.quality_thresholds = {
            'min_bbox_area': 100,
            'max_bbox_overlap': 0.3,
            'min_confidence': 0.7,
            'required_fields_per_type': {
                'mykad': ['name', 'ic_number', 'address'],
                'spk': ['name', 'ic_number', 'address'],
                'passport': ['name', 'passport_number', 'nationality'],
                'license': ['name', 'license_number', 'class']
            }
        }
        
        # Annotation statistics
        self.stats = {
            'total_annotations': 0,
            'validated_annotations': 0,
            'pending_annotations': 0,
            'rejected_annotations': 0
        }
        
        # Load existing annotations
        self._load_existing_annotations()
    
    def _setup_logging(self) -> logging.Logger:
        """Setup logging configuration"""
        logger = logging.getLogger('AnnotationSystem')
        logger.setLevel(logging.INFO)
        
        # Create file handler
        log_file = self.workspace_path / 'annotation.log'
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
    
    def _load_field_types(self) -> Dict[str, Dict[str, Any]]:
        """Load field type configurations"""
        return {
            'name': {
                'data_type': 'text',
                'validation_pattern': r'^[A-Za-z\s]+$',
                'max_length': 100,
                'required': True
            },
            'ic_number': {
                'data_type': 'text',
                'validation_pattern': r'^\d{6}-\d{2}-\d{4}$|^\d{12}$',
                'max_length': 14,
                'required': True
            },
            'address': {
                'data_type': 'text',
                'validation_pattern': None,
                'max_length': 500,
                'required': True
            },
            'birth_date': {
                'data_type': 'date',
                'validation_pattern': r'^\d{2}/\d{2}/\d{4}$',
                'max_length': 10,
                'required': False
            },
            'gender': {
                'data_type': 'categorical',
                'allowed_values': ['LELAKI', 'PEREMPUAN', 'MALE', 'FEMALE'],
                'required': False
            },
            'passport_number': {
                'data_type': 'text',
                'validation_pattern': r'^[A-Z]\d{8}$',
                'max_length': 9,
                'required': True
            },
            'license_number': {
                'data_type': 'text',
                'validation_pattern': r'^[A-Z]\d{7,8}$',
                'max_length': 9,
                'required': True
            }
        }
    
    def _load_existing_annotations(self):
        """Load existing annotations from workspace"""
        annotations_dir = self.workspace_path / 'annotations'
        if annotations_dir.exists():
            annotation_files = list(annotations_dir.glob('*.json'))
            self.stats['total_annotations'] = len(annotation_files)
            
            # Count by status
            for file_path in annotation_files:
                try:
                    with open(file_path, 'r') as f:
                        data = json.load(f)
                        status = data.get('review_status', 'pending')
                        if status == 'approved':
                            self.stats['validated_annotations'] += 1
                        elif status == 'rejected':
                            self.stats['rejected_annotations'] += 1
                        else:
                            self.stats['pending_annotations'] += 1
                except Exception as e:
                    self.logger.warning(f"Failed to load annotation {file_path}: {e}")
    
    def create_annotation(self, image_path: str, document_type: str, 
                         annotator_id: str) -> DocumentAnnotation:
        """Create a new document annotation"""
        # Load image to get dimensions
        image = cv2.imread(image_path)
        if image is None:
            raise ValueError(f"Cannot load image: {image_path}")
        
        height, width = image.shape[:2]
        
        annotation = DocumentAnnotation(
            annotation_id=str(uuid.uuid4()),
            image_path=image_path,
            document_type=document_type,
            fields=[],
            image_dimensions=(width, height),
            annotator_id=annotator_id,
            annotation_date=datetime.now().isoformat()
        )
        
        self.logger.info(f"Created new annotation: {annotation.annotation_id}")
        return annotation
    
    def add_field_annotation(self, doc_annotation: DocumentAnnotation,
                           field_name: str, field_type: str, value: str,
                           bbox: BoundingBox, confidence: float,
                           annotator_id: str, notes: str = '') -> FieldAnnotation:
        """Add a field annotation to a document"""
        # Validate field type
        if field_type not in self.field_types:
            raise ValueError(f"Unknown field type: {field_type}")
        
        # Validate bounding box
        if not self._validate_bbox(bbox, doc_annotation.image_dimensions):
            raise ValueError("Invalid bounding box coordinates")
        
        # Validate field value
        validation_result = self._validate_field_value(field_type, value)
        if not validation_result['valid']:
            self.logger.warning(f"Field validation warning: {validation_result['message']}")
        
        field_annotation = FieldAnnotation(
            field_name=field_name,
            field_type=field_type,
            value=value,
            bbox=bbox,
            confidence=confidence,
            annotator_id=annotator_id,
            annotation_date=datetime.now().isoformat(),
            notes=notes
        )
        
        doc_annotation.fields.append(field_annotation)
        
        self.logger.info(f"Added field annotation: {field_name} = {value}")
        return field_annotation
    
    def _validate_bbox(self, bbox: BoundingBox, image_dims: Tuple[int, int]) -> bool:
        """Validate bounding box coordinates"""
        width, height = image_dims
        
        # Check if bbox is within image bounds
        if bbox.x < 0 or bbox.y < 0:
            return False
        if bbox.x + bbox.width > width or bbox.y + bbox.height > height:
            return False
        
        # Check minimum area
        if bbox.area() < self.quality_thresholds['min_bbox_area']:
            return False
        
        return True
    
    def _validate_field_value(self, field_type: str, value: str) -> Dict[str, Any]:
        """Validate field value against type constraints"""
        field_config = self.field_types[field_type]
        
        # Check length
        if len(value) > field_config['max_length']:
            return {
                'valid': False,
                'message': f"Value too long (max {field_config['max_length']} chars)"
            }
        
        # Check pattern if specified
        if field_config.get('validation_pattern'):
            import re
            if not re.match(field_config['validation_pattern'], value):
                return {
                    'valid': False,
                    'message': f"Value doesn't match expected pattern"
                }
        
        # Check categorical values
        if field_config.get('allowed_values'):
            if value not in field_config['allowed_values']:
                return {
                    'valid': False,
                    'message': f"Value not in allowed list: {field_config['allowed_values']}"
                }
        
        return {'valid': True, 'message': 'Valid'}
    
    def save_annotation(self, annotation: DocumentAnnotation) -> str:
        """Save annotation to workspace"""
        annotations_dir = self.workspace_path / 'annotations'
        annotations_dir.mkdir(exist_ok=True)
        
        filename = f"{annotation.annotation_id}.json"
        file_path = annotations_dir / filename
        
        with open(file_path, 'w') as f:
            json.dump(annotation.to_dict(), f, indent=2)
        
        self.logger.info(f"Saved annotation: {file_path}")
        return str(file_path)
    
    def load_annotation(self, annotation_id: str) -> Optional[DocumentAnnotation]:
        """Load annotation from workspace"""
        annotations_dir = self.workspace_path / 'annotations'
        file_path = annotations_dir / f"{annotation_id}.json"
        
        if not file_path.exists():
            return None
        
        try:
            with open(file_path, 'r') as f:
                data = json.load(f)
            return DocumentAnnotation.from_dict(data)
        except Exception as e:
            self.logger.error(f"Failed to load annotation {annotation_id}: {e}")
            return None
    
    def validate_annotation_quality(self, annotation: DocumentAnnotation) -> Dict[str, Any]:
        """Validate annotation quality and completeness"""
        quality_report = {
            'overall_score': 0.0,
            'issues': [],
            'warnings': [],
            'field_coverage': {},
            'bbox_quality': {}
        }
        
        doc_type = annotation.document_type
        required_fields = self.quality_thresholds['required_fields_per_type'].get(doc_type, [])
        
        # Check field coverage
        annotated_fields = {field.field_name for field in annotation.fields}
        missing_fields = set(required_fields) - annotated_fields
        
        if missing_fields:
            quality_report['issues'].append(f"Missing required fields: {missing_fields}")
        
        quality_report['field_coverage'] = {
            'required': len(required_fields),
            'annotated': len(annotated_fields),
            'coverage_ratio': len(annotated_fields) / max(len(required_fields), 1)
        }
        
        # Check bounding box quality
        bbox_issues = []
        for i, field1 in enumerate(annotation.fields):
            # Check bbox area
            if field1.bbox.area() < self.quality_thresholds['min_bbox_area']:
                bbox_issues.append(f"Field '{field1.field_name}' has very small bounding box")
            
            # Check overlaps with other fields
            for j, field2 in enumerate(annotation.fields[i+1:], i+1):
                overlap = self._calculate_bbox_overlap(field1.bbox, field2.bbox)
                if overlap > self.quality_thresholds['max_bbox_overlap']:
                    bbox_issues.append(
                        f"High overlap between '{field1.field_name}' and '{field2.field_name}': {overlap:.2f}"
                    )
        
        quality_report['bbox_quality']['issues'] = bbox_issues
        
        # Check confidence scores
        low_confidence_fields = [
            field.field_name for field in annotation.fields 
            if field.confidence < self.quality_thresholds['min_confidence']
        ]
        
        if low_confidence_fields:
            quality_report['warnings'].append(
                f"Low confidence fields: {low_confidence_fields}"
            )
        
        # Calculate overall score
        score_components = [
            quality_report['field_coverage']['coverage_ratio'],
            1.0 - (len(bbox_issues) / max(len(annotation.fields), 1)),
            1.0 - (len(low_confidence_fields) / max(len(annotation.fields), 1))
        ]
        
        quality_report['overall_score'] = sum(score_components) / len(score_components)
        
        return quality_report
    
    def _calculate_bbox_overlap(self, bbox1: BoundingBox, bbox2: BoundingBox) -> float:
        """Calculate overlap ratio between two bounding boxes"""
        # Calculate intersection
        x1 = max(bbox1.x, bbox2.x)
        y1 = max(bbox1.y, bbox2.y)
        x2 = min(bbox1.x + bbox1.width, bbox2.x + bbox2.width)
        y2 = min(bbox1.y + bbox1.height, bbox2.y + bbox2.height)
        
        if x2 <= x1 or y2 <= y1:
            return 0.0
        
        intersection_area = (x2 - x1) * (y2 - y1)
        union_area = bbox1.area() + bbox2.area() - intersection_area
        
        return intersection_area / union_area if union_area > 0 else 0.0
    
    def batch_validate_annotations(self, annotation_ids: List[str] = None) -> Dict[str, Any]:
        """Validate multiple annotations in batch"""
        if annotation_ids is None:
            # Validate all annotations
            annotations_dir = self.workspace_path / 'annotations'
            annotation_files = list(annotations_dir.glob('*.json'))
            annotation_ids = [f.stem for f in annotation_files]
        
        batch_results = {
            'total_validated': 0,
            'passed': 0,
            'failed': 0,
            'average_score': 0.0,
            'results': []
        }
        
        total_score = 0.0
        
        for annotation_id in annotation_ids:
            annotation = self.load_annotation(annotation_id)
            if annotation is None:
                continue
            
            quality_report = self.validate_annotation_quality(annotation)
            
            result = {
                'annotation_id': annotation_id,
                'score': quality_report['overall_score'],
                'issues': quality_report['issues'],
                'warnings': quality_report['warnings']
            }
            
            batch_results['results'].append(result)
            batch_results['total_validated'] += 1
            total_score += quality_report['overall_score']
            
            if quality_report['overall_score'] >= 0.8 and not quality_report['issues']:
                batch_results['passed'] += 1
            else:
                batch_results['failed'] += 1
        
        if batch_results['total_validated'] > 0:
            batch_results['average_score'] = total_score / batch_results['total_validated']
        
        # Save batch validation report
        report_path = self.workspace_path / f"batch_validation_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        with open(report_path, 'w') as f:
            json.dump(batch_results, f, indent=2)
        
        self.logger.info(f"Batch validation completed: {batch_results['passed']}/{batch_results['total_validated']} passed")
        
        return batch_results
    
    def export_training_data(self, output_format: str = 'coco') -> str:
        """Export annotations in training-ready format"""
        if output_format not in ['coco', 'yolo', 'custom']:
            raise ValueError(f"Unsupported output format: {output_format}")
        
        export_dir = self.workspace_path / 'exports' / output_format
        export_dir.mkdir(parents=True, exist_ok=True)
        
        # Load all validated annotations
        annotations_dir = self.workspace_path / 'annotations'
        annotation_files = list(annotations_dir.glob('*.json'))
        
        exported_data = {
            'info': {
                'description': 'Document Parser Training Data',
                'version': '1.0',
                'year': datetime.now().year,
                'contributor': 'Document Parser Team',
                'date_created': datetime.now().isoformat()
            },
            'images': [],
            'annotations': [],
            'categories': []
        }
        
        # Create category mapping
        field_types = list(self.field_types.keys())
        for i, field_type in enumerate(field_types):
            exported_data['categories'].append({
                'id': i + 1,
                'name': field_type,
                'supercategory': 'document_field'
            })
        
        category_map = {field_type: i + 1 for i, field_type in enumerate(field_types)}
        
        annotation_id_counter = 1
        
        for i, file_path in enumerate(annotation_files):
            try:
                with open(file_path, 'r') as f:
                    data = json.load(f)
                
                annotation = DocumentAnnotation.from_dict(data)
                
                # Skip non-validated annotations
                if annotation.review_status != 'approved':
                    continue
                
                # Add image info
                image_info = {
                    'id': i + 1,
                    'width': annotation.image_dimensions[0],
                    'height': annotation.image_dimensions[1],
                    'file_name': Path(annotation.image_path).name
                }
                exported_data['images'].append(image_info)
                
                # Add field annotations
                for field in annotation.fields:
                    if field.field_type in category_map:
                        field_annotation = {
                            'id': annotation_id_counter,
                            'image_id': i + 1,
                            'category_id': category_map[field.field_type],
                            'bbox': [field.bbox.x, field.bbox.y, field.bbox.width, field.bbox.height],
                            'area': field.bbox.area(),
                            'iscrowd': 0,
                            'text': field.value,
                            'confidence': field.confidence
                        }
                        exported_data['annotations'].append(field_annotation)
                        annotation_id_counter += 1
                
            except Exception as e:
                self.logger.error(f"Failed to process annotation {file_path}: {e}")
        
        # Save exported data
        export_file = export_dir / f"annotations_{output_format}.json"
        with open(export_file, 'w') as f:
            json.dump(exported_data, f, indent=2)
        
        self.logger.info(f"Exported {len(exported_data['images'])} images and {len(exported_data['annotations'])} annotations to {export_file}")
        
        return str(export_file)
    
    def get_annotation_statistics(self) -> Dict[str, Any]:
        """Get comprehensive annotation statistics"""
        # Refresh stats
        self._load_existing_annotations()
        
        stats = self.stats.copy()
        
        # Add detailed breakdown
        annotations_dir = self.workspace_path / 'annotations'
        if annotations_dir.exists():
            doc_type_counts = {}
            field_type_counts = {}
            annotator_counts = {}
            
            for file_path in annotations_dir.glob('*.json'):
                try:
                    with open(file_path, 'r') as f:
                        data = json.load(f)
                    
                    # Count by document type
                    doc_type = data.get('document_type', 'unknown')
                    doc_type_counts[doc_type] = doc_type_counts.get(doc_type, 0) + 1
                    
                    # Count by annotator
                    annotator = data.get('annotator_id', 'unknown')
                    annotator_counts[annotator] = annotator_counts.get(annotator, 0) + 1
                    
                    # Count field types
                    for field in data.get('fields', []):
                        field_type = field.get('field_type', 'unknown')
                        field_type_counts[field_type] = field_type_counts.get(field_type, 0) + 1
                
                except Exception as e:
                    self.logger.warning(f"Failed to analyze annotation {file_path}: {e}")
            
            stats.update({
                'document_type_distribution': doc_type_counts,
                'field_type_distribution': field_type_counts,
                'annotator_distribution': annotator_counts
            })
        
        return stats

def main():
    """Main function for standalone execution"""
    print("📝 Document Annotation System")
    print("=" * 40)
    
    # Initialize annotation system
    annotation_system = AnnotationSystem("./annotation_workspace")
    
    # Display statistics
    stats = annotation_system.get_annotation_statistics()
    print(f"\n📊 Annotation Statistics:")
    print(f"   Total annotations: {stats['total_annotations']}")
    print(f"   Validated: {stats['validated_annotations']}")
    print(f"   Pending: {stats['pending_annotations']}")
    print(f"   Rejected: {stats['rejected_annotations']}")
    
    # Example usage
    print("\n📋 Usage Examples:")
    print("1. annotation = system.create_annotation('/path/to/image.jpg', 'mykad', 'annotator_1')")
    print("2. system.add_field_annotation(annotation, 'name', 'name', 'John Doe', bbox, 0.95, 'annotator_1')")
    print("3. system.save_annotation(annotation)")
    print("4. system.validate_annotation_quality(annotation)")
    print("5. system.export_training_data('coco')")
    
    print(f"\n📂 Workspace created at: {annotation_system.workspace_path}")
    
    return 0

if __name__ == "__main__":
    exit(main())