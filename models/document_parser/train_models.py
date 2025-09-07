#!/usr/bin/env python3
"""
Model Training Script

Training pipeline for document parser models including classification,
field extraction, and OCR models.
"""

import logging
import json
import time
from pathlib import Path
from typing import Dict, List, Any, Optional
import argparse
from datetime import datetime

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Import model classes
from extraction_models import (
    ClassificationModel, FieldExtractionModel, OCRModel,
    ModelRegistry, ModelConfig, ModelMetrics,
    create_classification_model, create_field_extraction_model, create_ocr_model
)

class ModelTrainer:
    """Main training orchestrator for document parser models."""
    
    def __init__(self, config_path: str, models_dir: str, data_dir: str):
        self.config_path = Path(config_path)
        self.models_dir = Path(models_dir)
        self.data_dir = Path(data_dir)
        
        # Create directories
        self.models_dir.mkdir(parents=True, exist_ok=True)
        self.data_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize model registry
        self.registry = ModelRegistry(self.models_dir)
        
        # Load training configuration
        self.config = self._load_config()
        
        logger.info(f"ModelTrainer initialized")
        logger.info(f"Models directory: {self.models_dir}")
        logger.info(f"Data directory: {self.data_dir}")
    
    def _load_config(self) -> Dict:
        """Load training configuration."""
        if self.config_path.exists():
            with open(self.config_path, 'r') as f:
                config = json.load(f)
            logger.info(f"Configuration loaded from {self.config_path}")
        else:
            # Default configuration
            config = self._get_default_config()
            self._save_config(config)
            logger.info(f"Default configuration created at {self.config_path}")
        
        return config
    
    def _get_default_config(self) -> Dict:
        """Get default training configuration."""
        return {
            "models": {
                "classification": {
                    "enabled": True,
                    "model_name": "document_classifier",
                    "parameters": {
                        "max_features": 5000,
                        "ngram_range": [1, 2]
                    },
                    "hyperparameters": {
                        "n_estimators": 100,
                        "max_depth": None,
                        "min_samples_split": 2,
                        "min_samples_leaf": 1
                    },
                    "training": {
                        "test_size": 0.2,
                        "validation_size": 0.1,
                        "cross_validation_folds": 5
                    }
                },
                "field_extraction": {
                    "enabled": True,
                    "model_name": "field_extractor",
                    "parameters": {
                        "supported_fields": [
                            "ic_number", "name", "address", "phone", "email",
                            "date_of_birth", "place_of_birth", "nationality"
                        ]
                    },
                    "training": {
                        "min_examples_per_field": 10,
                        "pattern_confidence_threshold": 0.7
                    }
                },
                "ocr": {
                    "enabled": True,
                    "engines": [
                        {
                            "name": "tesseract",
                            "languages": ["eng", "msa"],
                            "config": "--psm 6"
                        },
                        {
                            "name": "easyocr",
                            "languages": ["en", "ms"],
                            "gpu": False
                        }
                    ]
                }
            },
            "data": {
                "classification_data_file": "classification_training_data.json",
                "field_extraction_data_file": "field_extraction_training_data.json",
                "validation_split": 0.2,
                "augmentation": {
                    "enabled": True,
                    "techniques": ["rotation", "noise", "blur"]
                }
            },
            "output": {
                "save_models": True,
                "save_metrics": True,
                "generate_reports": True,
                "model_format": "pickle"
            },
            "logging": {
                "level": "INFO",
                "save_logs": True,
                "log_file": "training.log"
            }
        }
    
    def _save_config(self, config: Dict):
        """Save configuration to file."""
        self.config_path.parent.mkdir(parents=True, exist_ok=True)
        with open(self.config_path, 'w') as f:
            json.dump(config, f, indent=2)
    
    def load_training_data(self) -> Dict[str, Any]:
        """Load training data for all models."""
        training_data = {}
        
        # Load classification data
        if self.config["models"]["classification"]["enabled"]:
            classification_file = self.data_dir / self.config["data"]["classification_data_file"]
            if classification_file.exists():
                with open(classification_file, 'r') as f:
                    training_data["classification"] = json.load(f)
                logger.info(f"Classification data loaded: {len(training_data['classification']['documents'])} documents")
            else:
                training_data["classification"] = self._generate_sample_classification_data()
                logger.warning(f"No classification data found, using sample data")
        
        # Load field extraction data
        if self.config["models"]["field_extraction"]["enabled"]:
            field_extraction_file = self.data_dir / self.config["data"]["field_extraction_data_file"]
            if field_extraction_file.exists():
                with open(field_extraction_file, 'r') as f:
                    training_data["field_extraction"] = json.load(f)
                logger.info(f"Field extraction data loaded: {len(training_data['field_extraction']['documents'])} documents")
            else:
                training_data["field_extraction"] = self._generate_sample_field_data()
                logger.warning(f"No field extraction data found, using sample data")
        
        return training_data
    
    def _generate_sample_classification_data(self) -> Dict:
        """Generate sample classification training data."""
        return {
            "documents": [
                {"text": "MyKad Identity Card Number 123456-12-1234 Name AHMAD BIN ALI", "label": "mykad"},
                {"text": "Sijil Pelajaran Malaysia SPM Certificate Student Name SITI AMINAH", "label": "spk"},
                {"text": "Passport Republic of Malaysia Passport No A12345678", "label": "passport"},
                {"text": "Driver License Lesen Memandu Class D License No D1234567", "label": "license"},
                {"text": "Birth Certificate Sijil Kelahiran Child Name MUHAMMAD HASSAN", "label": "birth_cert"},
                {"text": "Unknown document type with random text content", "label": "unknown"}
            ]
        }
    
    def _generate_sample_field_data(self) -> Dict:
        """Generate sample field extraction training data."""
        return {
            "documents": [
                {
                    "text": "MyKad 123456-12-1234 AHMAD BIN ALI Jalan Bukit Bintang Kuala Lumpur",
                    "fields": {
                        "ic_number": "123456-12-1234",
                        "name": "AHMAD BIN ALI",
                        "address": "Jalan Bukit Bintang Kuala Lumpur"
                    }
                },
                {
                    "text": "Name SITI AMINAH IC 987654-32-9876 Phone 012-3456789",
                    "fields": {
                        "ic_number": "987654-32-9876",
                        "name": "SITI AMINAH",
                        "phone": "012-3456789"
                    }
                }
            ]
        }
    
    def train_classification_model(self, training_data: Dict) -> ClassificationModel:
        """Train document classification model."""
        logger.info("Training classification model...")
        
        # Create model
        model_config = self.config["models"]["classification"]
        model = create_classification_model(model_config["model_name"])
        
        # Update model configuration
        model.config.parameters.update(model_config["parameters"])
        model.config.hyperparameters.update(model_config["hyperparameters"])
        
        # Prepare training data
        documents = training_data["documents"]
        texts = [doc["text"] for doc in documents]
        labels = [doc["label"] for doc in documents]
        
        # Train model
        training_config = model_config["training"]
        metrics = model.train(texts, labels, **training_config)
        
        # Register and save model
        self.registry.register_model(model, "classification")
        if self.config["output"]["save_models"]:
            self.registry.save_model("classification")
        
        logger.info(f"Classification model trained - Accuracy: {metrics.accuracy:.3f}")
        return model
    
    def train_field_extraction_model(self, training_data: Dict) -> FieldExtractionModel:
        """Train field extraction model."""
        logger.info("Training field extraction model...")
        
        # Create model
        model_config = self.config["models"]["field_extraction"]
        model = create_field_extraction_model(model_config["model_name"])
        
        # Update model configuration
        model.config.parameters.update(model_config["parameters"])
        
        # Train model
        metrics = model.train(training_data)
        
        # Register and save model
        self.registry.register_model(model, "field_extraction")
        if self.config["output"]["save_models"]:
            self.registry.save_model("field_extraction")
        
        logger.info(f"Field extraction model trained")
        return model
    
    def initialize_ocr_models(self) -> List[OCRModel]:
        """Initialize OCR models."""
        logger.info("Initializing OCR models...")
        
        ocr_models = []
        ocr_config = self.config["models"]["ocr"]
        
        for engine_config in ocr_config["engines"]:
            try:
                model = create_ocr_model(
                    engine=engine_config["name"],
                    languages=engine_config["languages"]
                )
                
                # Initialize (no training required for OCR)
                metrics = model.train()
                
                # Register model
                model_id = f"ocr_{engine_config['name']}"
                self.registry.register_model(model, model_id)
                
                if self.config["output"]["save_models"]:
                    self.registry.save_model(model_id)
                
                ocr_models.append(model)
                logger.info(f"OCR model '{engine_config['name']}' initialized")
                
            except Exception as e:
                logger.error(f"Failed to initialize OCR engine '{engine_config['name']}': {e}")
        
        return ocr_models
    
    def train_all_models(self) -> Dict[str, Any]:
        """Train all enabled models."""
        logger.info("Starting model training pipeline...")
        start_time = time.time()
        
        results = {
            "training_start": datetime.now().isoformat(),
            "models_trained": [],
            "metrics": {},
            "errors": []
        }
        
        try:
            # Load training data
            training_data = self.load_training_data()
            
            # Train classification model
            if self.config["models"]["classification"]["enabled"]:
                try:
                    model = self.train_classification_model(training_data["classification"])
                    results["models_trained"].append("classification")
                    results["metrics"]["classification"] = model.metrics.to_dict()
                except Exception as e:
                    error_msg = f"Classification model training failed: {e}"
                    logger.error(error_msg)
                    results["errors"].append(error_msg)
            
            # Train field extraction model
            if self.config["models"]["field_extraction"]["enabled"]:
                try:
                    model = self.train_field_extraction_model(training_data["field_extraction"])
                    results["models_trained"].append("field_extraction")
                    results["metrics"]["field_extraction"] = model.metrics.to_dict()
                except Exception as e:
                    error_msg = f"Field extraction model training failed: {e}"
                    logger.error(error_msg)
                    results["errors"].append(error_msg)
            
            # Initialize OCR models
            if self.config["models"]["ocr"]["enabled"]:
                try:
                    ocr_models = self.initialize_ocr_models()
                    for i, model in enumerate(ocr_models):
                        model_name = f"ocr_{i}"
                        results["models_trained"].append(model_name)
                        results["metrics"][model_name] = model.metrics.to_dict()
                except Exception as e:
                    error_msg = f"OCR model initialization failed: {e}"
                    logger.error(error_msg)
                    results["errors"].append(error_msg)
            
            # Calculate total training time
            total_time = time.time() - start_time
            results["training_end"] = datetime.now().isoformat()
            results["total_training_time"] = total_time
            
            # Save training results
            if self.config["output"]["save_metrics"]:
                results_file = self.models_dir / "training_results.json"
                with open(results_file, 'w') as f:
                    json.dump(results, f, indent=2)
                logger.info(f"Training results saved to {results_file}")
            
            logger.info(f"Training pipeline completed in {total_time:.2f} seconds")
            logger.info(f"Models trained: {', '.join(results['models_trained'])}")
            
            if results["errors"]:
                logger.warning(f"Training completed with {len(results['errors'])} errors")
            
        except Exception as e:
            error_msg = f"Training pipeline failed: {e}"
            logger.error(error_msg)
            results["errors"].append(error_msg)
            raise
        
        return results
    
    def evaluate_models(self) -> Dict[str, Any]:
        """Evaluate trained models."""
        logger.info("Evaluating trained models...")
        
        evaluation_results = {
            "evaluation_timestamp": datetime.now().isoformat(),
            "model_evaluations": {},
            "summary": {}
        }
        
        # Get all registered models
        models_info = self.registry.list_models()
        
        for model_id, model_info in models_info.items():
            try:
                model = self.registry.get_model(model_id)
                if model and model.metrics:
                    evaluation_results["model_evaluations"][model_id] = {
                        "model_info": model_info,
                        "metrics": model.metrics.to_dict(),
                        "status": "evaluated"
                    }
                else:
                    evaluation_results["model_evaluations"][model_id] = {
                        "model_info": model_info,
                        "status": "no_metrics"
                    }
            except Exception as e:
                evaluation_results["model_evaluations"][model_id] = {
                    "status": "error",
                    "error": str(e)
                }
        
        # Generate summary
        total_models = len(models_info)
        evaluated_models = sum(1 for eval_data in evaluation_results["model_evaluations"].values() 
                             if eval_data["status"] == "evaluated")
        
        evaluation_results["summary"] = {
            "total_models": total_models,
            "evaluated_models": evaluated_models,
            "evaluation_success_rate": evaluated_models / total_models if total_models > 0 else 0
        }
        
        # Save evaluation results
        if self.config["output"]["save_metrics"]:
            eval_file = self.models_dir / "evaluation_results.json"
            with open(eval_file, 'w') as f:
                json.dump(evaluation_results, f, indent=2)
            logger.info(f"Evaluation results saved to {eval_file}")
        
        logger.info(f"Model evaluation completed: {evaluated_models}/{total_models} models evaluated")
        return evaluation_results
    
    def generate_training_report(self) -> str:
        """Generate comprehensive training report."""
        logger.info("Generating training report...")
        
        # Load results
        training_results_file = self.models_dir / "training_results.json"
        evaluation_results_file = self.models_dir / "evaluation_results.json"
        
        report_lines = [
            "# Document Parser Model Training Report",
            f"Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
            "",
            "## Training Configuration",
            f"- Models Directory: {self.models_dir}",
            f"- Data Directory: {self.data_dir}",
            f"- Configuration File: {self.config_path}",
            ""
        ]
        
        # Training results
        if training_results_file.exists():
            with open(training_results_file, 'r') as f:
                training_results = json.load(f)
            
            report_lines.extend([
                "## Training Results",
                f"- Training Start: {training_results.get('training_start', 'N/A')}",
                f"- Training End: {training_results.get('training_end', 'N/A')}",
                f"- Total Training Time: {training_results.get('total_training_time', 0):.2f} seconds",
                f"- Models Trained: {', '.join(training_results.get('models_trained', []))}",
                f"- Errors: {len(training_results.get('errors', []))}",
                ""
            ])
            
            # Model metrics
            if training_results.get("metrics"):
                report_lines.append("### Model Performance")
                for model_name, metrics in training_results["metrics"].items():
                    report_lines.extend([
                        f"#### {model_name.title()} Model",
                        f"- Accuracy: {metrics.get('accuracy', 0):.3f}",
                        f"- Precision: {metrics.get('precision', 0):.3f}",
                        f"- Recall: {metrics.get('recall', 0):.3f}",
                        f"- F1 Score: {metrics.get('f1_score', 0):.3f}",
                        f"- Training Time: {metrics.get('training_time', 0):.2f} seconds",
                        ""
                    ])
        
        # Registry information
        registry_info = self.registry.get_registry_info()
        report_lines.extend([
            "## Model Registry",
            f"- Registry Directory: {registry_info['models_dir']}",
            f"- Registered Models: {registry_info['model_count']}",
            f"- Available Model Files: {len(registry_info['available_files'])}",
            ""
        ])
        
        # Model list
        models_info = self.registry.list_models()
        if models_info:
            report_lines.append("### Registered Models")
            for model_id, model_info in models_info.items():
                report_lines.extend([
                    f"#### {model_id}",
                    f"- Type: {model_info['config']['model_type']}",
                    f"- Version: {model_info['config']['version']}",
                    f"- Trained: {model_info['is_trained']}",
                    f"- Features: {model_info['feature_count']}",
                    ""
                ])
        
        report_content = "\n".join(report_lines)
        
        # Save report
        if self.config["output"]["generate_reports"]:
            report_file = self.models_dir / "training_report.md"
            with open(report_file, 'w') as f:
                f.write(report_content)
            logger.info(f"Training report saved to {report_file}")
        
        return report_content

def main():
    """Main training script entry point."""
    parser = argparse.ArgumentParser(description="Train document parser models")
    parser.add_argument("--config", default="training_config.json", help="Training configuration file")
    parser.add_argument("--models-dir", default="./models", help="Models output directory")
    parser.add_argument("--data-dir", default="./data", help="Training data directory")
    parser.add_argument("--train", action="store_true", help="Train models")
    parser.add_argument("--evaluate", action="store_true", help="Evaluate models")
    parser.add_argument("--report", action="store_true", help="Generate training report")
    parser.add_argument("--all", action="store_true", help="Run all steps (train, evaluate, report)")
    
    args = parser.parse_args()
    
    # Initialize trainer
    trainer = ModelTrainer(
        config_path=args.config,
        models_dir=args.models_dir,
        data_dir=args.data_dir
    )
    
    try:
        if args.all or args.train:
            # Train models
            training_results = trainer.train_all_models()
            print(f"Training completed. Models trained: {', '.join(training_results['models_trained'])}")
        
        if args.all or args.evaluate:
            # Evaluate models
            evaluation_results = trainer.evaluate_models()
            print(f"Evaluation completed. {evaluation_results['summary']['evaluated_models']} models evaluated.")
        
        if args.all or args.report:
            # Generate report
            report = trainer.generate_training_report()
            print("Training report generated.")
            print("\n" + "="*50)
            print(report)
    
    except Exception as e:
        logger.error(f"Training script failed: {e}")
        raise

if __name__ == "__main__":
    main()