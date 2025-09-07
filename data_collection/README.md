# Document Parser Data Collection and Preparation

This directory contains tools and utilities for Phase 2 of the document parser project: data collection, annotation, and preprocessing.

## Directory Structure

```
data_collection/
├── README.md                    # This file
├── synthetic/                   # Synthetic data generation
│   ├── __init__.py
│   ├── document_generator.py    # Template-based document generation
│   ├── variation_engine.py      # Apply realistic variations
│   └── templates/               # Document templates
│       ├── mykad_template.json
│       └── vehicle_cert_template.json
├── real_data/                   # Real data collection tools
│   ├── __init__.py
│   ├── data_collector.py        # Data collection utilities
│   ├── privacy_anonymizer.py    # Privacy compliance tools
│   └── quality_checker.py       # Data quality validation
├── annotation/                  # Data annotation tools
│   ├── __init__.py
│   ├── annotation_tool.py       # Custom annotation interface
│   ├── quality_control.py       # Multi-annotator validation
│   ├── schema_validator.py      # Annotation schema validation
│   └── guidelines/              # Annotation guidelines
│       ├── mykad_guidelines.md
│       └── vehicle_cert_guidelines.md
├── preprocessing/               # Data preprocessing pipeline
│   ├── __init__.py
│   ├── image_enhancer.py        # Image enhancement utilities
│   ├── augmentation.py          # Data augmentation pipeline
│   ├── dataset_splitter.py      # Train/validation/test splitting
│   └── format_converter.py      # Format conversion utilities
├── datasets/                    # Dataset organization
│   ├── raw/                     # Raw collected data
│   ├── annotated/               # Annotated datasets
│   ├── processed/               # Preprocessed datasets
│   └── splits/                  # Train/val/test splits
└── utils/                       # Common utilities
    ├── __init__.py
    ├── file_manager.py          # File management utilities
    ├── metadata_tracker.py      # Dataset metadata tracking
    └── validation.py            # Data validation utilities
```

## Phase 2 Implementation Plan

### 2.1 Data Collection Strategy
- **Synthetic Data**: Template-based generation with realistic variations
- **Real Data**: Privacy-compliant collection with anonymization
- **Target Volume**: 10,000+ samples per document type
- **Diversity**: Multiple conditions (lighting, angles, quality, wear)

### 2.2 Data Annotation
- **Custom Annotation Tools**: Web-based annotation interface
- **Quality Control**: Multi-annotator validation and consistency checks
- **Schema Validation**: Standardized annotation formats
- **Guidelines**: Comprehensive annotation documentation

### 2.3 Data Preprocessing
- **Image Enhancement**: Noise reduction, contrast adjustment, deskewing
- **Standardization**: Consistent image formats and sizes
- **Augmentation**: Rotation, scaling, color variations, blur, noise
- **Dataset Splitting**: 70% training, 15% validation, 15% testing

## Usage

### Synthetic Data Generation
```python
from data_collection.synthetic import DocumentGenerator

generator = DocumentGenerator()
mykad_samples = generator.generate_mykad_dataset(count=1000)
vehicle_samples = generator.generate_vehicle_cert_dataset(count=1000)
```

### Data Annotation
```python
from data_collection.annotation import AnnotationTool

tool = AnnotationTool()
tool.start_annotation_session(dataset_path="datasets/raw/mykad")
```

### Data Preprocessing
```python
from data_collection.preprocessing import DataPreprocessor

processor = DataPreprocessor()
processor.enhance_images(input_dir="datasets/raw", output_dir="datasets/processed")
processor.split_dataset(dataset_path="datasets/annotated", splits=[0.7, 0.15, 0.15])
```

## Data Privacy and Compliance

- All real data collection follows privacy regulations (GDPR, PDPA)
- Automatic anonymization of sensitive information
- Secure storage and access controls
- Data retention and deletion policies

## Quality Standards

- Minimum 95% annotation accuracy
- Multi-annotator agreement threshold: 90%
- Image quality standards: minimum 300 DPI, clear text visibility
- Comprehensive validation and quality control processes