#!/usr/bin/env python3
"""
Google Colab Training Script for YOLO Document Parser

This script is designed to run in Google Colab with GPU acceleration
for training Malaysian Identity Card detection models.
"""

# Install required packages (uncomment when running in Colab)
# !pip install ultralytics roboflow
# !pip install -q torch torchvision torchaudio

import os
import sys
import json
import time
import zipfile
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Any

import torch
import yaml
from ultralytics import YOLO
import cv2
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import seaborn as sns
from google.colab import files, drive
import requests

class ColabYOLOTrainer:
    """Google Colab optimized YOLO trainer for document parsing."""
    
    def __init__(self, use_drive: bool = True):
        """Initialize the Colab trainer.
        
        Args:
            use_drive: Whether to mount Google Drive for persistent storage
        """
        self.use_drive = use_drive
        self.drive_path = None
        self.dataset_path = None
        self.model = None
        self.results = None
        
        # Setup environment
        self._setup_environment()
        
    def _setup_environment(self):
        """Setup the Colab environment."""
        print("🔧 Setting up Colab environment...")
        
        # Check GPU availability
        if torch.cuda.is_available():
            gpu_name = torch.cuda.get_device_name(0)
            print(f"✅ GPU available: {gpu_name}")
            print(f"   CUDA version: {torch.version.cuda}")
            print(f"   PyTorch version: {torch.__version__}")
        else:
            print("⚠️ No GPU available. Training will be slower.")
        
        # Mount Google Drive if requested
        if self.use_drive:
            try:
                drive.mount('/content/drive')
                self.drive_path = Path('/content/drive/MyDrive/document_parser_training')
                self.drive_path.mkdir(parents=True, exist_ok=True)
                print(f"✅ Google Drive mounted: {self.drive_path}")
            except Exception as e:
                print(f"⚠️ Failed to mount Google Drive: {e}")
                self.use_drive = False
        
        # Set working directory
        os.chdir('/content')
        
    def download_dataset_from_roboflow(self, api_key: str, workspace: str, 
                                      project: str, version: int = 2) -> str:
        """Download dataset from Roboflow.
        
        Args:
            api_key: Roboflow API key
            workspace: Roboflow workspace name
            project: Roboflow project name
            version: Dataset version
            
        Returns:
            Path to downloaded dataset
        """
        print(f"📥 Downloading dataset from Roboflow...")
        
        try:
            from roboflow import Roboflow
            
            rf = Roboflow(api_key=api_key)
            project = rf.workspace(workspace).project(project)
            dataset = project.version(version).download("yolov8")
            
            self.dataset_path = Path(dataset.location)
            print(f"✅ Dataset downloaded: {self.dataset_path}")
            
            return str(self.dataset_path)
            
        except ImportError:
            print("❌ Roboflow package not installed. Installing...")
            os.system("pip install roboflow")
            return self.download_dataset_from_roboflow(api_key, workspace, project, version)
        except Exception as e:
            print(f"❌ Failed to download dataset: {e}")
            raise
    
    def download_dataset_from_url(self, url: str) -> str:
        """Download dataset from a direct URL.
        
        Args:
            url: Direct download URL for the dataset
            
        Returns:
            Path to downloaded dataset
        """
        print(f"📥 Downloading dataset from URL...")
        
        try:
            # Download the dataset
            response = requests.get(url, stream=True)
            response.raise_for_status()
            
            zip_path = '/content/dataset.zip'
            with open(zip_path, 'wb') as f:
                for chunk in response.iter_content(chunk_size=8192):
                    f.write(chunk)
            
            # Extract the dataset
            with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                zip_ref.extractall('/content')
            
            # Find the extracted dataset directory
            extracted_dirs = [d for d in os.listdir('/content') if os.path.isdir(d) and 'train' in os.listdir(d)]
            if extracted_dirs:
                self.dataset_path = Path('/content') / extracted_dirs[0]
            else:
                # Look for data.yaml file
                yaml_files = list(Path('/content').glob('**/data.yaml'))
                if yaml_files:
                    self.dataset_path = yaml_files[0].parent
                else:
                    raise FileNotFoundError("Could not find dataset structure")
            
            # Clean up
            os.remove(zip_path)
            
            print(f"✅ Dataset extracted: {self.dataset_path}")
            return str(self.dataset_path)
            
        except Exception as e:
            print(f"❌ Failed to download dataset: {e}")
            raise
    
    def upload_dataset_from_local(self) -> str:
        """Upload dataset from local machine.
        
        Returns:
            Path to uploaded dataset
        """
        print("📤 Upload your dataset zip file:")
        
        uploaded = files.upload()
        
        if not uploaded:
            raise ValueError("No file uploaded")
        
        # Get the uploaded file
        zip_filename = list(uploaded.keys())[0]
        
        # Extract the dataset
        with zipfile.ZipFile(zip_filename, 'r') as zip_ref:
            zip_ref.extractall('/content')
        
        # Find the extracted dataset directory
        extracted_dirs = [d for d in os.listdir('/content') if os.path.isdir(d) and d != 'sample_data']
        if extracted_dirs:
            self.dataset_path = Path('/content') / extracted_dirs[-1]  # Get the most recent
        else:
            raise FileNotFoundError("Could not find dataset structure")
        
        # Clean up
        os.remove(zip_filename)
        
        print(f"✅ Dataset uploaded: {self.dataset_path}")
        return str(self.dataset_path)
    
    def analyze_dataset(self):
        """Analyze the dataset and display statistics."""
        if not self.dataset_path:
            raise ValueError("No dataset loaded")
        
        print("\n📊 Dataset Analysis:")
        
        # Count images
        train_images = list((self.dataset_path / 'train' / 'images').glob('*.jpg'))
        valid_images = list((self.dataset_path / 'valid' / 'images').glob('*.jpg'))
        
        print(f"   Training images: {len(train_images)}")
        print(f"   Validation images: {len(valid_images)}")
        print(f"   Total images: {len(train_images) + len(valid_images)}")
        
        # Display sample images
        if train_images:
            fig, axes = plt.subplots(2, 3, figsize=(15, 10))
            fig.suptitle('Sample Training Images', fontsize=16)
            
            for i, ax in enumerate(axes.flat):
                if i < len(train_images) and i < 6:
                    img = cv2.imread(str(train_images[i]))
                    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                    ax.imshow(img_rgb)
                    ax.set_title(f'Image {i+1}: {img.shape}')
                    ax.axis('off')
                else:
                    ax.axis('off')
            
            plt.tight_layout()
            plt.show()
        
        # Analyze class distribution
        train_labels = list((self.dataset_path / 'train' / 'labels').glob('*.txt'))
        valid_labels = list((self.dataset_path / 'valid' / 'labels').glob('*.txt'))
        
        class_counts = {}
        for label_file in train_labels + valid_labels:
            with open(label_file, 'r') as f:
                for line in f:
                    if line.strip():
                        class_id = int(line.split()[0])
                        class_counts[class_id] = class_counts.get(class_id, 0) + 1
        
        print(f"   Class distribution: {class_counts}")
        
        # Plot class distribution
        if class_counts:
            plt.figure(figsize=(8, 6))
            plt.bar(class_counts.keys(), class_counts.values())
            plt.title('Class Distribution')
            plt.xlabel('Class ID')
            plt.ylabel('Number of Instances')
            plt.show()
    
    def train_model(self, model_size: str = 'n', epochs: int = 100, 
                   batch_size: int = 16, imgsz: int = 640) -> str:
        """Train the YOLO model with Colab optimizations.
        
        Args:
            model_size: YOLO model size ('n', 's', 'm', 'l', 'x')
            epochs: Number of training epochs
            batch_size: Batch size (auto-adjusted for GPU memory)
            imgsz: Image size
            
        Returns:
            Path to the best trained model
        """
        if not self.dataset_path:
            raise ValueError("No dataset loaded")
        
        print(f"\n🚀 Starting YOLO training with {model_size} model...")
        
        # Auto-adjust batch size based on GPU memory
        if torch.cuda.is_available():
            gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1e9
            print(f"   GPU Memory: {gpu_memory:.1f} GB")
            
            if gpu_memory < 8:
                batch_size = min(batch_size, 8)
                print(f"   Adjusted batch size to {batch_size} for GPU memory")
        
        # Initialize model
        model_name = f'yolov8{model_size}.pt'
        self.model = YOLO(model_name)
        
        # Prepare data config
        data_yaml_path = self.dataset_path / 'data.yaml'
        
        # Update data.yaml with absolute paths
        with open(data_yaml_path, 'r') as f:
            data_config = yaml.safe_load(f)
        
        data_config['train'] = str(self.dataset_path / 'train' / 'images')
        data_config['val'] = str(self.dataset_path / 'valid' / 'images')
        
        updated_yaml_path = self.dataset_path / 'data_updated.yaml'
        with open(updated_yaml_path, 'w') as f:
            yaml.dump(data_config, f, default_flow_style=False)
        
        # Training parameters optimized for Colab
        train_params = {
            'data': str(updated_yaml_path),
            'epochs': epochs,
            'imgsz': imgsz,
            'batch': batch_size,
            'patience': 50,
            'save_period': 10,
            'project': '/content/runs',
            'name': f'train_{datetime.now().strftime("%Y%m%d_%H%M%S")}',
            'exist_ok': True,
            'pretrained': True,
            'optimizer': 'AdamW',
            'lr0': 0.01,
            'lrf': 0.01,
            'momentum': 0.937,
            'weight_decay': 0.0005,
            'warmup_epochs': 3,
            'cache': True,  # Cache images for faster training
            'device': 0 if torch.cuda.is_available() else 'cpu',
            'workers': 2,  # Reduced for Colab
            'amp': True,  # Automatic Mixed Precision
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
        
        # Save to Google Drive if available
        if self.use_drive and self.drive_path:
            drive_model_path = self.drive_path / f'best_model_{datetime.now().strftime("%Y%m%d_%H%M%S")}.pt'
            import shutil
            shutil.copy2(best_model_path, drive_model_path)
            print(f"💾 Model saved to Google Drive: {drive_model_path}")
        
        print(f"🏆 Best model: {best_model_path}")
        return str(best_model_path)
    
    def validate_model(self, model_path: str = None) -> Dict[str, Any]:
        """Validate the trained model.
        
        Args:
            model_path: Path to model (uses current if None)
            
        Returns:
            Validation metrics
        """
        print("\n🔍 Validating model...")
        
        if model_path:
            model = YOLO(model_path)
        elif self.model:
            model = self.model
        else:
            raise ValueError("No model available")
        
        # Run validation
        data_yaml_path = self.dataset_path / 'data_updated.yaml'
        val_results = model.val(data=str(data_yaml_path))
        
        # Extract metrics
        metrics = {
            'mAP50': float(val_results.box.map50),
            'mAP50-95': float(val_results.box.map),
            'precision': float(val_results.box.mp),
            'recall': float(val_results.box.mr),
            'fitness': float(val_results.fitness)
        }
        
        print(f"📊 Validation Results:")
        for metric, value in metrics.items():
            print(f"   {metric}: {value:.4f}")
        
        # Plot validation results
        if hasattr(val_results, 'plots'):
            plt.figure(figsize=(12, 8))
            # Display confusion matrix if available
            if hasattr(val_results, 'confusion_matrix'):
                plt.subplot(2, 2, 1)
                sns.heatmap(val_results.confusion_matrix.matrix, annot=True, fmt='d')
                plt.title('Confusion Matrix')
        
        return metrics
    
    def test_inference(self, model_path: str, test_images: List[str] = None):
        """Test model inference on sample images.
        
        Args:
            model_path: Path to trained model
            test_images: List of test image paths (uses validation set if None)
        """
        print("\n🧪 Testing model inference...")
        
        model = YOLO(model_path)
        
        if test_images is None:
            # Use validation images
            valid_images = list((self.dataset_path / 'valid' / 'images').glob('*.jpg'))
            test_images = [str(img) for img in valid_images[:6]]  # Test on first 6
        
        if not test_images:
            print("No test images available")
            return
        
        # Run inference
        results = model(test_images)
        
        # Display results
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        fig.suptitle('Model Inference Results', fontsize=16)
        
        for i, (result, ax) in enumerate(zip(results, axes.flat)):
            if i >= len(test_images):
                ax.axis('off')
                continue
            
            # Get original image
            img = cv2.imread(test_images[i])
            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            
            # Draw predictions
            if result.boxes is not None:
                for box in result.boxes:
                    x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                    conf = box.conf[0].cpu().numpy()
                    
                    # Draw bounding box
                    cv2.rectangle(img_rgb, (int(x1), int(y1)), (int(x2), int(y2)), (255, 0, 0), 2)
                    cv2.putText(img_rgb, f'{conf:.2f}', (int(x1), int(y1-10)), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
            
            ax.imshow(img_rgb)
            ax.set_title(f'Test Image {i+1}')
            ax.axis('off')
        
        plt.tight_layout()
        plt.show()
    
    def download_model_to_local(self, model_path: str) -> str:
        """Download the trained model to local device.
        
        Args:
            model_path: Path to the trained model
            
        Returns:
            Local filename of downloaded model
        """
        print("\n📥 Downloading trained model to local device...")
        
        try:
            # Create a timestamped filename
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            local_filename = f'document_parser_model_{timestamp}.pt'
            
            # Download the model file
            files.download(model_path)
            
            print(f"✅ Model downloaded as: {local_filename}")
            print("   You can now use this model in your local environment!")
            
            return local_filename
            
        except Exception as e:
            print(f"❌ Failed to download model: {e}")
            print("   You can manually download from the files panel on the left")
            raise
    
    def create_training_summary(self, model_path: str, metrics: Dict[str, Any]) -> str:
        """Create a comprehensive training summary.
        
        Args:
            model_path: Path to trained model
            metrics: Validation metrics
            
        Returns:
            Path to summary file
        """
        summary = {
            'timestamp': datetime.now().isoformat(),
            'environment': 'Google Colab',
            'gpu_available': torch.cuda.is_available(),
            'gpu_name': torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
            'dataset_path': str(self.dataset_path),
            'model_path': model_path,
            'metrics': metrics,
            'framework': 'YOLOv8',
            'task': 'Malaysian Identity Card Detection'
        }
        
        summary_path = '/content/training_summary.json'
        with open(summary_path, 'w') as f:
            json.dump(summary, f, indent=2)
        
        # Save to Google Drive if available
        if self.use_drive and self.drive_path:
            drive_summary_path = self.drive_path / f'training_summary_{datetime.now().strftime("%Y%m%d_%H%M%S")}.json'
            import shutil
            shutil.copy2(summary_path, drive_summary_path)
            print(f"📄 Summary saved to Google Drive: {drive_summary_path}")
        
        print(f"📄 Training summary: {summary_path}")
        return summary_path

# Example usage for Colab
def run_colab_training():
    """Complete training pipeline for Google Colab."""
    print("🎯 Starting Document Parser Training in Google Colab")
    
    # Initialize trainer
    trainer = ColabYOLOTrainer(use_drive=True)
    
    # Option 1: Download from Roboflow (replace with your credentials)
    # dataset_path = trainer.download_dataset_from_roboflow(
    #     api_key="your_api_key",
    #     workspace="jinendra",
    #     project="malaysia-id-sfzs6",
    #     version=2
    # )
    
    # Option 2: Download from URL
    dataset_url = "https://app.roboflow.com/ds/KUwV8k7Fw9?key=UbgXm5O3pJ"
    dataset_path = trainer.download_dataset_from_url(dataset_url)
    
    # Option 3: Upload from local (uncomment to use)
    # dataset_path = trainer.upload_dataset_from_local()
    
    # Analyze dataset
    trainer.analyze_dataset()
    
    # Train model
    best_model_path = trainer.train_model(
        model_size='n',  # Start with nano for faster training
        epochs=50,       # Reduced for Colab time limits
        batch_size=16,
        imgsz=640
    )
    
    # Validate model
    metrics = trainer.validate_model(best_model_path)
    
    # Test inference
    trainer.test_inference(best_model_path)
    
    # Create summary
    summary_path = trainer.create_training_summary(best_model_path, metrics)
    
    # Download model to local device
    try:
        local_model_filename = trainer.download_model_to_local(best_model_path)
        print(f"\n🎉 Training completed successfully!")
        print(f"   Best model: {best_model_path}")
        print(f"   Downloaded as: {local_model_filename}")
        print(f"   Summary: {summary_path}")
    except Exception as e:
        print(f"\n🎉 Training completed successfully!")
        print(f"   Best model: {best_model_path}")
        print(f"   Summary: {summary_path}")
        print(f"   ⚠️ Model download failed: {e}")
    
    return best_model_path, metrics

if __name__ == '__main__':
    # Run the complete training pipeline
    model_path, metrics = run_colab_training()