#!/usr/bin/env python3
"""
Document Classification Models for Document Parser

This module provides CNN-based document classification capabilities using
ResNet, EfficientNet, and Vision Transformer architectures, following
the organizational patterns established by the autocorrect model.

Author: AI Assistant
Date: 2024
"""

import os
import sys
import json
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
from torchvision import models
import timm
import numpy as np
from PIL import Image
from pathlib import Path
from typing import Dict, List, Tuple, Any, Optional, Union
from datetime import datetime
import logging
from dataclasses import dataclass, asdict
from collections import defaultdict
import cv2

# Add src directory to path for imports
sys.path.append(str(Path(__file__).parent.parent.parent))

@dataclass
class ClassifierConfig:
    """Configuration for document classifier"""
    model_name: str = "efficientnet_b0"
    num_classes: int = 4  # mykad, spk, vehicle_cert, other
    input_size: Tuple[int, int] = (224, 224)
    pretrained: bool = True
    dropout_rate: float = 0.3
    learning_rate: float = 0.001
    batch_size: int = 32
    num_epochs: int = 50
    early_stopping_patience: int = 10
    class_names: List[str] = None
    data_augmentation: bool = True
    mixed_precision: bool = True
    
    def __post_init__(self):
        if self.class_names is None:
            self.class_names = ['mykad', 'spk', 'vehicle_cert', 'other']
    
    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'ClassifierConfig':
        return cls(**data)

class DocumentDataset(Dataset):
    """Dataset class for document classification"""
    
    def __init__(self, image_paths: List[str], labels: List[int], 
                 transform: Optional[transforms.Compose] = None,
                 config: ClassifierConfig = None):
        self.image_paths = image_paths
        self.labels = labels
        self.transform = transform
        self.config = config or ClassifierConfig()
        
        if len(image_paths) != len(labels):
            raise ValueError("Number of images and labels must match")
    
    def __len__(self) -> int:
        return len(self.image_paths)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        image_path = self.image_paths[idx]
        label = self.labels[idx]
        
        # Load image
        try:
            image = Image.open(image_path).convert('RGB')
        except Exception as e:
            # Return a blank image if loading fails
            image = Image.new('RGB', self.config.input_size, color='white')
        
        # Apply transforms
        if self.transform:
            image = self.transform(image)
        
        return image, label

class ResNetClassifier(nn.Module):
    """ResNet-based document classifier"""
    
    def __init__(self, config: ClassifierConfig):
        super(ResNetClassifier, self).__init__()
        self.config = config
        
        # Load pretrained ResNet
        if config.model_name == "resnet18":
            self.backbone = models.resnet18(pretrained=config.pretrained)
            feature_dim = 512
        elif config.model_name == "resnet34":
            self.backbone = models.resnet34(pretrained=config.pretrained)
            feature_dim = 512
        elif config.model_name == "resnet50":
            self.backbone = models.resnet50(pretrained=config.pretrained)
            feature_dim = 2048
        elif config.model_name == "resnet101":
            self.backbone = models.resnet101(pretrained=config.pretrained)
            feature_dim = 2048
        else:
            raise ValueError(f"Unsupported ResNet model: {config.model_name}")
        
        # Remove the final classification layer
        self.backbone = nn.Sequential(*list(self.backbone.children())[:-1])
        
        # Add custom classifier head
        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Dropout(config.dropout_rate),
            nn.Linear(feature_dim, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(config.dropout_rate),
            nn.Linear(256, config.num_classes)
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = self.backbone(x)
        output = self.classifier(features)
        return output

class EfficientNetClassifier(nn.Module):
    """EfficientNet-based document classifier"""
    
    def __init__(self, config: ClassifierConfig):
        super(EfficientNetClassifier, self).__init__()
        self.config = config
        
        # Load pretrained EfficientNet using timm
        self.backbone = timm.create_model(
            config.model_name,
            pretrained=config.pretrained,
            num_classes=0,  # Remove classifier head
            global_pool='avg'
        )
        
        # Get feature dimension
        feature_dim = self.backbone.num_features
        
        # Add custom classifier head
        self.classifier = nn.Sequential(
            nn.Dropout(config.dropout_rate),
            nn.Linear(feature_dim, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(config.dropout_rate),
            nn.Linear(256, config.num_classes)
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = self.backbone(x)
        output = self.classifier(features)
        return output

class VisionTransformerClassifier(nn.Module):
    """Vision Transformer-based document classifier"""
    
    def __init__(self, config: ClassifierConfig):
        super(VisionTransformerClassifier, self).__init__()
        self.config = config
        
        # Load pretrained Vision Transformer using timm
        self.backbone = timm.create_model(
            config.model_name,
            pretrained=config.pretrained,
            num_classes=0,  # Remove classifier head
            global_pool='token'
        )
        
        # Get feature dimension
        feature_dim = self.backbone.num_features
        
        # Add custom classifier head
        self.classifier = nn.Sequential(
            nn.LayerNorm(feature_dim),
            nn.Dropout(config.dropout_rate),
            nn.Linear(feature_dim, 256),
            nn.GELU(),
            nn.Dropout(config.dropout_rate),
            nn.Linear(256, config.num_classes)
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = self.backbone(x)
        output = self.classifier(features)
        return output

class DocumentClassifier:
    """Main document classifier with multiple architecture support"""
    
    def __init__(self, config: ClassifierConfig = None, device: str = None):
        self.config = config or ClassifierConfig()
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Setup logging
        self.logger = self._setup_logging()
        
        # Initialize model
        self.model = self._create_model()
        self.model.to(self.device)
        
        # Initialize transforms
        self.train_transform = self._create_train_transform()
        self.val_transform = self._create_val_transform()
        
        # Training components
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=self.config.learning_rate,
            weight_decay=0.01
        )
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode='min', patience=5, factor=0.5
        )
        
        # Training history
        self.training_history = {
            'train_loss': [],
            'val_loss': [],
            'train_acc': [],
            'val_acc': [],
            'learning_rates': []
        }
        
        # Mixed precision training
        self.scaler = torch.cuda.amp.GradScaler() if self.config.mixed_precision else None
    
    def _setup_logging(self) -> logging.Logger:
        """Setup logging configuration"""
        logger = logging.getLogger('DocumentClassifier')
        logger.setLevel(logging.INFO)
        
        # Create console handler
        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.INFO)
        
        # Create formatter
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        console_handler.setFormatter(formatter)
        
        # Add handler to logger
        logger.addHandler(console_handler)
        
        return logger
    
    def _create_model(self) -> nn.Module:
        """Create model based on configuration"""
        model_name = self.config.model_name.lower()
        
        if 'resnet' in model_name:
            return ResNetClassifier(self.config)
        elif 'efficientnet' in model_name:
            return EfficientNetClassifier(self.config)
        elif 'vit' in model_name or 'vision_transformer' in model_name:
            return VisionTransformerClassifier(self.config)
        else:
            # Default to EfficientNet
            self.logger.warning(f"Unknown model {model_name}, defaulting to EfficientNet")
            self.config.model_name = "efficientnet_b0"
            return EfficientNetClassifier(self.config)
    
    def _create_train_transform(self) -> transforms.Compose:
        """Create training data transforms"""
        transform_list = [
            transforms.Resize((256, 256)),
            transforms.RandomCrop(self.config.input_size),
        ]
        
        if self.config.data_augmentation:
            transform_list.extend([
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.RandomRotation(degrees=5),
                transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
                transforms.RandomAffine(degrees=0, translate=(0.1, 0.1), scale=(0.9, 1.1)),
            ])
        
        transform_list.extend([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        
        return transforms.Compose(transform_list)
    
    def _create_val_transform(self) -> transforms.Compose:
        """Create validation data transforms"""
        return transforms.Compose([
            transforms.Resize(self.config.input_size),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
    
    def prepare_data(self, data_path: str, split_ratios: Tuple[float, float, float] = (0.7, 0.15, 0.15)) -> Dict[str, DataLoader]:
        """Prepare data loaders for training"""
        data_path = Path(data_path)
        
        # Collect images and labels
        image_paths = []
        labels = []
        
        for class_idx, class_name in enumerate(self.config.class_names):
            class_dir = data_path / class_name
            if class_dir.exists():
                for img_file in class_dir.glob('*'):
                    if img_file.suffix.lower() in ['.jpg', '.jpeg', '.png', '.bmp']:
                        image_paths.append(str(img_file))
                        labels.append(class_idx)
        
        if not image_paths:
            raise ValueError(f"No images found in {data_path}")
        
        self.logger.info(f"Found {len(image_paths)} images across {len(self.config.class_names)} classes")
        
        # Split data
        from sklearn.model_selection import train_test_split
        
        # First split: train+val vs test
        train_val_paths, test_paths, train_val_labels, test_labels = train_test_split(
            image_paths, labels, test_size=split_ratios[2], stratify=labels, random_state=42
        )
        
        # Second split: train vs val
        val_ratio = split_ratios[1] / (split_ratios[0] + split_ratios[1])
        train_paths, val_paths, train_labels, val_labels = train_test_split(
            train_val_paths, train_val_labels, test_size=val_ratio, stratify=train_val_labels, random_state=42
        )
        
        # Create datasets
        train_dataset = DocumentDataset(train_paths, train_labels, self.train_transform, self.config)
        val_dataset = DocumentDataset(val_paths, val_labels, self.val_transform, self.config)
        test_dataset = DocumentDataset(test_paths, test_labels, self.val_transform, self.config)
        
        # Create data loaders
        dataloaders = {
            'train': DataLoader(
                train_dataset, batch_size=self.config.batch_size, shuffle=True, num_workers=4
            ),
            'val': DataLoader(
                val_dataset, batch_size=self.config.batch_size, shuffle=False, num_workers=4
            ),
            'test': DataLoader(
                test_dataset, batch_size=self.config.batch_size, shuffle=False, num_workers=4
            )
        }
        
        self.logger.info(f"Data split - Train: {len(train_dataset)}, Val: {len(val_dataset)}, Test: {len(test_dataset)}")
        
        return dataloaders
    
    def train_epoch(self, dataloader: DataLoader) -> Tuple[float, float]:
        """Train for one epoch"""
        self.model.train()
        running_loss = 0.0
        correct_predictions = 0
        total_samples = 0
        
        for batch_idx, (images, labels) in enumerate(dataloader):
            images = images.to(self.device)
            labels = labels.to(self.device)
            
            self.optimizer.zero_grad()
            
            if self.config.mixed_precision and self.scaler:
                with torch.cuda.amp.autocast():
                    outputs = self.model(images)
                    loss = self.criterion(outputs, labels)
                
                self.scaler.scale(loss).backward()
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                outputs = self.model(images)
                loss = self.criterion(outputs, labels)
                loss.backward()
                self.optimizer.step()
            
            # Statistics
            running_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total_samples += labels.size(0)
            correct_predictions += (predicted == labels).sum().item()
            
            if batch_idx % 50 == 0:
                self.logger.info(f"Batch {batch_idx}/{len(dataloader)}, Loss: {loss.item():.4f}")
        
        epoch_loss = running_loss / len(dataloader)
        epoch_acc = correct_predictions / total_samples
        
        return epoch_loss, epoch_acc
    
    def validate_epoch(self, dataloader: DataLoader) -> Tuple[float, float]:
        """Validate for one epoch"""
        self.model.eval()
        running_loss = 0.0
        correct_predictions = 0
        total_samples = 0
        
        with torch.no_grad():
            for images, labels in dataloader:
                images = images.to(self.device)
                labels = labels.to(self.device)
                
                if self.config.mixed_precision:
                    with torch.cuda.amp.autocast():
                        outputs = self.model(images)
                        loss = self.criterion(outputs, labels)
                else:
                    outputs = self.model(images)
                    loss = self.criterion(outputs, labels)
                
                # Statistics
                running_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                total_samples += labels.size(0)
                correct_predictions += (predicted == labels).sum().item()
        
        epoch_loss = running_loss / len(dataloader)
        epoch_acc = correct_predictions / total_samples
        
        return epoch_loss, epoch_acc
    
    def train(self, dataloaders: Dict[str, DataLoader], save_path: str = "./model_checkpoints") -> Dict[str, Any]:
        """Train the model"""
        save_path = Path(save_path)
        save_path.mkdir(parents=True, exist_ok=True)
        
        best_val_acc = 0.0
        patience_counter = 0
        
        self.logger.info(f"Starting training for {self.config.num_epochs} epochs")
        
        for epoch in range(self.config.num_epochs):
            self.logger.info(f"Epoch {epoch+1}/{self.config.num_epochs}")
            
            # Training phase
            train_loss, train_acc = self.train_epoch(dataloaders['train'])
            
            # Validation phase
            val_loss, val_acc = self.validate_epoch(dataloaders['val'])
            
            # Update learning rate
            self.scheduler.step(val_loss)
            current_lr = self.optimizer.param_groups[0]['lr']
            
            # Record history
            self.training_history['train_loss'].append(train_loss)
            self.training_history['val_loss'].append(val_loss)
            self.training_history['train_acc'].append(train_acc)
            self.training_history['val_acc'].append(val_acc)
            self.training_history['learning_rates'].append(current_lr)
            
            self.logger.info(
                f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}, "
                f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}, LR: {current_lr:.6f}"
            )
            
            # Save best model
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                patience_counter = 0
                
                # Save model checkpoint
                checkpoint = {
                    'epoch': epoch + 1,
                    'model_state_dict': self.model.state_dict(),
                    'optimizer_state_dict': self.optimizer.state_dict(),
                    'scheduler_state_dict': self.scheduler.state_dict(),
                    'config': self.config.to_dict(),
                    'training_history': self.training_history,
                    'best_val_acc': best_val_acc
                }
                
                torch.save(checkpoint, save_path / 'best_model.pth')
                self.logger.info(f"New best model saved with validation accuracy: {best_val_acc:.4f}")
            else:
                patience_counter += 1
            
            # Early stopping
            if patience_counter >= self.config.early_stopping_patience:
                self.logger.info(f"Early stopping triggered after {epoch + 1} epochs")
                break
        
        # Save final model
        final_checkpoint = {
            'epoch': epoch + 1,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'config': self.config.to_dict(),
            'training_history': self.training_history,
            'final_val_acc': val_acc
        }
        
        torch.save(final_checkpoint, save_path / 'final_model.pth')
        
        # Save training history
        with open(save_path / 'training_history.json', 'w') as f:
            json.dump(self.training_history, f, indent=2)
        
        training_summary = {
            'best_val_acc': best_val_acc,
            'final_val_acc': val_acc,
            'total_epochs': epoch + 1,
            'early_stopped': patience_counter >= self.config.early_stopping_patience
        }
        
        self.logger.info(f"Training completed. Best validation accuracy: {best_val_acc:.4f}")
        
        return training_summary
    
    def predict(self, image: Union[str, np.ndarray, Image.Image]) -> Dict[str, Any]:
        """Predict document type for a single image"""
        self.model.eval()
        
        # Preprocess image
        if isinstance(image, str):
            image = Image.open(image).convert('RGB')
        elif isinstance(image, np.ndarray):
            image = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        
        # Apply transforms
        input_tensor = self.val_transform(image).unsqueeze(0).to(self.device)
        
        # Predict
        with torch.no_grad():
            if self.config.mixed_precision:
                with torch.cuda.amp.autocast():
                    outputs = self.model(input_tensor)
            else:
                outputs = self.model(input_tensor)
            
            probabilities = F.softmax(outputs, dim=1)
            confidence, predicted_class = torch.max(probabilities, 1)
        
        result = {
            'predicted_class': self.config.class_names[predicted_class.item()],
            'confidence': confidence.item(),
            'probabilities': {
                class_name: prob.item() 
                for class_name, prob in zip(self.config.class_names, probabilities[0])
            },
            'prediction_date': datetime.now().isoformat()
        }
        
        return result
    
    def load_model(self, checkpoint_path: str):
        """Load model from checkpoint"""
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        
        # Update config if available
        if 'config' in checkpoint:
            self.config = ClassifierConfig.from_dict(checkpoint['config'])
            # Recreate model with loaded config
            self.model = self._create_model()
            self.model.to(self.device)
        
        # Load model state
        self.model.load_state_dict(checkpoint['model_state_dict'])
        
        # Load training history if available
        if 'training_history' in checkpoint:
            self.training_history = checkpoint['training_history']
        
        self.logger.info(f"Model loaded from {checkpoint_path}")
    
    def save_model(self, save_path: str):
        """Save current model"""
        checkpoint = {
            'model_state_dict': self.model.state_dict(),
            'config': self.config.to_dict(),
            'training_history': self.training_history,
            'save_date': datetime.now().isoformat()
        }
        
        torch.save(checkpoint, save_path)
        self.logger.info(f"Model saved to {save_path}")

def main():
    """Main function for standalone execution"""
    print("🔍 Document Classification Models")
    print("=" * 40)
    
    # Initialize classifier
    config = ClassifierConfig(
        model_name="efficientnet_b0",
        num_classes=4,
        input_size=(224, 224),
        batch_size=16,
        num_epochs=10
    )
    
    classifier = DocumentClassifier(config)
    
    # Display configuration
    print(f"\n⚙️ Configuration:")
    print(f"   Model: {config.model_name}")
    print(f"   Classes: {config.class_names}")
    print(f"   Input size: {config.input_size}")
    print(f"   Device: {classifier.device}")
    
    # Example usage
    print("\n📋 Usage Examples:")
    print("1. dataloaders = classifier.prepare_data('/path/to/dataset')")
    print("2. summary = classifier.train(dataloaders)")
    print("3. result = classifier.predict('/path/to/image.jpg')")
    print("4. classifier.save_model('/path/to/model.pth')")
    
    return 0

if __name__ == "__main__":
    exit(main())