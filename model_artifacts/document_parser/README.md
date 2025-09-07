# Document Parser Model Artifacts

This directory contains trained models, configurations, and metadata for the document parser system.

## Directory Structure

```
model_artifacts/document_parser/
├── README.md                    # This file
├── models/                      # Trained model files
│   ├── classification/          # Document classification models
│   │   ├── model.pkl           # Main classification model
│   │   ├── model_config.json   # Model configuration
│   │   └── training_log.json   # Training history
│   ├── field_extraction/       # Field extraction models
│   │   ├── mykad/              # MyKad-specific models
│   │   ├── spk/                # SPK-specific models
│   │   └── general/            # General extraction models
│   └── ocr/                    # OCR models (if custom)
│       ├── text_detection/     # Text detection models
│       └── text_recognition/   # Text recognition models
├── configs/                     # Model configurations
│   ├── classification_config.json
│   ├── extraction_config.json
│   └── training_config.json
├── metadata/                    # Model metadata and performance
│   ├── model_registry.json     # Model version registry
│   ├── performance_metrics.json # Performance benchmarks
│   └── validation_results.json # Validation test results
└── checkpoints/                 # Training checkpoints
    ├── classification/
    └── field_extraction/
```

## Model Versioning

Models are versioned using semantic versioning (MAJOR.MINOR.PATCH):
- **MAJOR**: Incompatible API changes or major architecture changes
- **MINOR**: New features or improvements while maintaining compatibility
- **PATCH**: Bug fixes and minor improvements

## Model Registry

The `metadata/model_registry.json` file tracks all model versions:

```json
{
  "classification": {
    "current_version": "1.0.0",
    "versions": {
      "1.0.0": {
        "path": "models/classification/v1.0.0/",
        "created_at": "2024-01-15T10:30:00Z",
        "accuracy": 0.95,
        "f1_score": 0.94,
        "training_data_size": 10000,
        "description": "Initial production model"
      }
    }
  },
  "field_extraction": {
    "mykad": {
      "current_version": "1.0.0",
      "versions": {...}
    },
    "spk": {
      "current_version": "1.0.0",
      "versions": {...}
    }
  }
}
```

## Performance Benchmarks

The `metadata/performance_metrics.json` file contains performance benchmarks:

```json
{
  "classification": {
    "accuracy": 0.95,
    "precision": 0.94,
    "recall": 0.96,
    "f1_score": 0.95,
    "confusion_matrix": [[850, 50], [30, 870]],
    "processing_time_ms": 150
  },
  "field_extraction": {
    "mykad": {
      "field_accuracy": {
        "name": 0.98,
        "ic_number": 0.99,
        "address": 0.92,
        "date_of_birth": 0.97
      },
      "overall_accuracy": 0.96
    }
  }
}
```

## Usage

### Loading Models

```python
from src.document_parser.core.document_processor import DocumentProcessor

# Load with default models
processor = DocumentProcessor()

# Load specific model version
processor = DocumentProcessor(
    classification_model_path="model_artifacts/document_parser/models/classification/v1.0.0/model.pkl",
    extraction_model_path="model_artifacts/document_parser/models/field_extraction/mykad/v1.0.0/"
)
```

### Model Training

```bash
# Train classification model
python scripts/train_document_parser.py --task classification --output-dir model_artifacts/document_parser/models/classification/

# Train field extraction model
python scripts/train_document_parser.py --task field_extraction --document-type mykad --output-dir model_artifacts/document_parser/models/field_extraction/mykad/
```

### Model Evaluation

```bash
# Evaluate model performance
python scripts/evaluate_model.py --model-path model_artifacts/document_parser/models/classification/v1.0.0/ --test-data data/test/
```

## Model Deployment

1. **Development**: Models in `checkpoints/` directory
2. **Staging**: Validated models in `models/` directory with version tags
3. **Production**: Current production models referenced in `model_registry.json`

## Backup and Recovery

- Models are automatically backed up during training
- Checkpoints are saved every epoch during training
- Model registry maintains history of all versions
- Performance metrics are tracked for rollback decisions

## Security

- Model files should be integrity-checked before deployment
- Access to model artifacts should be restricted
- Model provenance should be tracked for audit purposes